from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    service_port: int = _env_int("SERVICE_PORT", 8000)
    log_level: str = _env("LOG_LEVEL", "info")
    model_sample_rate_hz: int = _env_int("MODEL_SAMPLE_RATE_HZ", 16000)
    categorical_model_id: str = _env(
        "CATEGORICAL_MODEL_ID",
        "3loi/SER-Odyssey-Baseline-WavLM-Categorical",
    )
    avd_model_id: str = _env(
        "AVD_MODEL_ID",
        "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
    )
    huggingface_token: str | None = os.getenv("HF_TOKEN") or None
    hf_home: Path = Path(_env("HF_HOME", "/models/cache/huggingface"))
    device: str = _env("MODEL_DEVICE", "cuda:0")
    use_half_precision: bool = _env_bool("MODEL_USE_HALF_PRECISION", False)
    batch_size: int = _env_int("MODEL_BATCH_SIZE", 8)
    window_seconds: float = _env_float("WINDOW_SECONDS", 4.0)
    hop_seconds: float = _env_float("HOP_SECONDS", 1.0)
    smoothing_seconds: float = _env_float("SMOOTHING_SECONDS", 1.5)
    segment_resolution_seconds: float = _env_float("SEGMENT_RESOLUTION_SECONDS", 0.5)
    min_segment_seconds: float = _env_float("MIN_SEGMENT_SECONDS", 1.0)
    merge_gap_seconds: float = _env_float("MERGE_GAP_SECONDS", 0.25)
    startup_timeout_seconds: int = _env_int("STARTUP_TIMEOUT_SECONDS", 1800)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.model_sample_rate_hz))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.model_sample_rate_hz))

    @property
    def segment_resolution_samples(self) -> int:
        return max(1, int(round(self.segment_resolution_seconds * self.model_sample_rate_hz)))


settings = Settings()
