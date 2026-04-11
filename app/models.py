from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel

from app.config import settings
from app.diarization import SpeakerTurn
from app.segmentation import Window, build_windows, normalized_entropy, smooth_and_merge


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


class Pooling(nn.Module):
    def compute_length_from_mask(self, mask: torch.Tensor) -> list[int]:
        wav_lens = torch.sum(mask, dim=1)
        feat_lens = torch.div(wav_lens - 1, 320, rounding_mode="floor") + 1
        return feat_lens.int().tolist()


class AttentiveStatisticsPooling(Pooling):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.empty(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0.0, std=1.0)

    def forward(self, xs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens, strict=True):
            x = x[:feat_len].unsqueeze(0)
            hidden = torch.tanh(self.sap_linear(x))
            weights = torch.matmul(hidden, self.attention).squeeze(dim=2)
            weights = functional.softmax(weights, dim=1).view(x.size(0), x.size(1), 1)
            mean = torch.sum(x * weights, dim=1)
            std = torch.sqrt((torch.sum((x**2) * weights, dim=1) - mean**2).clamp(min=1e-5))
            pooled_list.append(torch.cat((mean, std), 1).squeeze(0))
        return torch.stack(pooled_list)


class EmotionRegression(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            ]
        )
        for _ in range(num_layers - 1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.out = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.inp_drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.inp_drop(inputs)
        for layer in self.fc:
            hidden = layer(hidden)
        return self.out(hidden)


class SERModel(nn.Module):
    def __init__(self, config: dict[str, Any], token: str | None) -> None:
        super().__init__()
        self.ssl_model = AutoModel.from_pretrained(config["ssl_type"], token=token)
        if hasattr(self.ssl_model, "freeze_feature_encoder"):
            self.ssl_model.freeze_feature_encoder()
        self.pool_model = AttentiveStatisticsPooling(config["hidden_size"])
        self.ser_model = EmotionRegression(
            input_dim=config["hidden_size"] * 2,
            hidden_dim=config["hidden_size"],
            num_layers=config["classifier_hidden_layers"],
            output_dim=config["num_classes"],
            dropout=config["classifier_dropout_prob"],
        )

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ssl = self.ssl_model(inputs, attention_mask=attention_mask).last_hidden_state
        pooled = self.pool_model(ssl, attention_mask)
        return self.ser_model(pooled)


@dataclass
class LoadedModel:
    model_id: str
    snapshot_path: Path
    config: dict[str, Any]
    id2label: list[str]
    mean: float
    std: float
    model: SERModel


@dataclass
class ReadyState:
    ready: bool
    device: str
    detail: str
    categorical_model: str
    avd_model: str
    categorical_labels: list[str]


class EmotionSegmenter:
    def __init__(self) -> None:
        self.settings = settings
        self.ready_detail = "initializing"
        self.inference_lock = threading.Lock()
        self.device = torch.device(self.settings.device)
        self._dtype = (
            torch.float16
            if self.settings.use_half_precision and self.device.type == "cuda"
            else torch.float32
        )
        self.categorical: LoadedModel | None = None
        self.avd: LoadedModel | None = None

    def load(self) -> None:
        os.environ.setdefault("HF_HOME", str(self.settings.hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.settings.hf_home / "transformers"))
        self.settings.hf_home.mkdir(parents=True, exist_ok=True)
        if self.device.type != "cuda":
            raise RuntimeError(f"Configured device must be CUDA, received {self.device}.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside the container.")

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.ready_detail = "downloading checkpoints"
        self.categorical = self._load_model(self.settings.categorical_model_id)
        self.avd = self._load_model(self.settings.avd_model_id)
        self.ready_detail = "warming up gpu"
        self._warmup()
        self.ready_detail = "ready"

    def _load_model(self, model_id: str) -> LoadedModel:
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_id,
                token=self.settings.huggingface_token,
                allow_patterns=["config.json", "*.safetensors"],
            )
        )
        config = _load_json(snapshot_path / "config.json")
        model = SERModel(config=config, token=self.settings.huggingface_token)
        state_dict = load_file(snapshot_path / "model.safetensors", device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device=self.device, dtype=self._dtype)
        labels = [config["id2label"][str(index)] for index in range(config["num_classes"])]
        return LoadedModel(
            model_id=model_id,
            snapshot_path=snapshot_path,
            config=config,
            id2label=labels,
            mean=float(config.get("mean", 0.0)),
            std=float(config.get("std", 1.0)),
            model=model,
        )

    def _warmup(self) -> None:
        samples = self.settings.window_samples
        dummy_audio = torch.zeros((1, samples), device=self.device, dtype=self._dtype)
        dummy_mask = torch.ones((1, samples), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            assert self.categorical is not None
            assert self.avd is not None
            self.categorical.model(dummy_audio, dummy_mask)
            self.avd.model(dummy_audio, dummy_mask)
        torch.cuda.synchronize(self.device)

    @property
    def is_ready(self) -> bool:
        return self.ready_detail == "ready"

    def readiness(self) -> ReadyState:
        categorical_labels = self.categorical.id2label if self.categorical else []
        return ReadyState(
            ready=self.is_ready,
            device=str(self.device),
            detail=self.ready_detail,
            categorical_model=self.settings.categorical_model_id,
            avd_model=self.settings.avd_model_id,
            categorical_labels=categorical_labels,
        )

    def infer(
        self,
        audio: np.ndarray,
        diarization: list[SpeakerTurn],
        speaker_metadata: dict[str, dict[str, Any]],
        filename: str,
    ) -> dict[str, Any]:
        if not self.is_ready or self.categorical is None or self.avd is None:
            raise RuntimeError("Service is not ready.")
        windows = build_windows(
            diarization=diarization,
            sample_rate_hz=self.settings.model_sample_rate_hz,
            window_seconds=self.settings.window_seconds,
            hop_seconds=self.settings.hop_seconds,
            audio_duration_seconds=audio.shape[0] / self.settings.model_sample_rate_hz,
        )
        if not windows:
            raise ValueError("No inference windows were produced from the provided diarization.")

        window_predictions = self._predict_windows(audio=audio, windows=windows)
        segments = smooth_and_merge(
            diarization=diarization,
            window_predictions=window_predictions,
            label_names=self.categorical.id2label,
            resolution_seconds=self.settings.segment_resolution_seconds,
            smoothing_seconds=self.settings.smoothing_seconds,
            min_segment_seconds=self.settings.min_segment_seconds,
            merge_gap_seconds=self.settings.merge_gap_seconds,
        )

        public_windows = [
            {
                key: value
                for key, value in window.items()
                if not key.startswith("_")
            }
            for window in window_predictions
        ]

        return {
            "audio": {
                "filename": filename,
                "duration_seconds": round(audio.shape[0] / self.settings.model_sample_rate_hz, 3),
                "sample_rate_hz": self.settings.model_sample_rate_hz,
            },
            "config": {
                "window_seconds": self.settings.window_seconds,
                "hop_seconds": self.settings.hop_seconds,
                "smoothing_seconds": self.settings.smoothing_seconds,
                "segment_resolution_seconds": self.settings.segment_resolution_seconds,
                "device": str(self.device),
            },
            "models": {
                "categorical": self.categorical.model_id,
                "avd": self.avd.model_id,
                "categorical_labels": self.categorical.id2label,
            },
            "diarization": [
                {
                    "speaker_id": turn.speaker_id,
                    "start_seconds": round(turn.start_seconds, 3),
                    "end_seconds": round(turn.end_seconds, 3),
                    "metadata": speaker_metadata.get(turn.speaker_id, {}),
                }
                for turn in diarization
            ],
            "window_predictions": public_windows,
            "segments": segments,
        }

    def _predict_windows(self, audio: np.ndarray, windows: list[Window]) -> list[dict[str, Any]]:
        categorical_outputs: list[np.ndarray] = []
        avd_outputs: list[np.ndarray] = []
        actual_lengths: list[int] = []
        with torch.inference_mode():
            for offset in range(0, len(windows), self.settings.batch_size):
                batch_windows = windows[offset : offset + self.settings.batch_size]
                batch_audio = []
                batch_masks = []
                for window in batch_windows:
                    snippet = audio[window.audio_start_sample : window.audio_end_sample]
                    actual_length = int(snippet.shape[0])
                    actual_lengths.append(actual_length)
                    padded = np.zeros((self.settings.window_samples,), dtype=np.float32)
                    padded[:actual_length] = snippet[: self.settings.window_samples]
                    batch_audio.append(padded)
                    mask = np.zeros((self.settings.window_samples,), dtype=np.int64)
                    mask[:actual_length] = 1
                    batch_masks.append(mask)

                audio_tensor = torch.from_numpy(np.stack(batch_audio)).to(self.device, dtype=self._dtype)
                mask_tensor = torch.from_numpy(np.stack(batch_masks)).to(self.device)

                categorical_logits = self.categorical.model(
                    self._normalize(audio_tensor, self.categorical),
                    mask_tensor,
                )
                avd_values = self.avd.model(
                    self._normalize(audio_tensor, self.avd),
                    mask_tensor,
                )

                categorical_outputs.extend(torch.softmax(categorical_logits.float(), dim=-1).cpu().numpy())
                avd_outputs.extend(avd_values.float().cpu().numpy())

        predictions: list[dict[str, Any]] = []
        for window, probabilities, avd_vector, actual_length in zip(
            windows,
            categorical_outputs,
            avd_outputs,
            actual_lengths,
            strict=True,
        ):
            confidence = float(np.max(probabilities))
            label_index = int(np.argmax(probabilities))
            predictions.append(
                {
                    "speaker_id": window.speaker_id,
                    "start_seconds": round(window.start_seconds, 3),
                    "end_seconds": round(window.end_seconds, 3),
                    "speaker_overlap_start_seconds": round(window.coverage_start_seconds, 3),
                    "speaker_overlap_end_seconds": round(window.coverage_end_seconds, 3),
                    "predicted_label": self.categorical.id2label[label_index],
                    "categorical_probabilities": {
                        label: round(float(value), 6)
                        for label, value in zip(self.categorical.id2label, probabilities, strict=True)
                    },
                    "avd_scores": {
                        "arousal": round(float(avd_vector[0]), 6),
                        "valence": round(float(avd_vector[2]), 6),
                        "dominance": round(float(avd_vector[1]), 6),
                    },
                    "confidence": round(confidence, 6),
                    "uncertainty": round(normalized_entropy(probabilities), 6),
                    "coverage_ratio": round(actual_length / float(self.settings.window_samples), 6),
                    "_coverage_start_seconds": window.coverage_start_seconds,
                    "_coverage_end_seconds": window.coverage_end_seconds,
                    "_categorical_probabilities_vector": [float(value) for value in probabilities],
                    "_avd_vector": [float(avd_vector[0]), float(avd_vector[1]), float(avd_vector[2])],
                }
            )
        return predictions

    @staticmethod
    def _normalize(audio_tensor: torch.Tensor, loaded_model: LoadedModel) -> torch.Tensor:
        std = loaded_model.std if loaded_model.std != 0 else 1.0
        return (audio_tensor - loaded_model.mean) / std
