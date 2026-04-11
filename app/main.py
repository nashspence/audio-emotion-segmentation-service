from __future__ import annotations

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.audio import decode_audio
from app.config import settings
from app.diarization import parse_diarization_payload, parse_speaker_metadata
from app.models import EmotionSegmenter

segmenter = EmotionSegmenter()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(segmenter.load)
    yield


app = FastAPI(
    title="emotion-diarization-service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict:
    state = segmenter.readiness()
    payload = {
        "ready": state.ready,
        "detail": state.detail,
        "device": state.device,
        "models": {
            "categorical": state.categorical_model,
            "avd": state.avd_model,
            "categorical_labels": state.categorical_labels,
        },
    }
    if not state.ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.post("/v1/segment")
async def segment_audio(
    audio: UploadFile = File(...),
    diarization_json: str | None = Form(default=None),
    speaker_metadata_json: str | None = Form(default=None),
    diarization_file: UploadFile | None = File(default=None),
) -> dict:
    if not segmenter.is_ready:
        raise HTTPException(status_code=503, detail="Model service is still warming up.")

    with tempfile.TemporaryDirectory(prefix="emotion-diarization-") as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / (audio.filename or "audio.bin")
        audio_path.write_bytes(await audio.read())

        if diarization_file is not None:
            diarization_json = (await diarization_file.read()).decode("utf-8")

        try:
            waveform = await asyncio.to_thread(decode_audio, audio_path, settings.model_sample_rate_hz)
            duration_seconds = waveform.shape[0] / settings.model_sample_rate_hz
            diarization = parse_diarization_payload(diarization_json, duration_seconds)
            speaker_metadata = parse_speaker_metadata(speaker_metadata_json)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - surfaced in smoke test
            raise HTTPException(status_code=400, detail=f"Audio decode failed: {exc}") from exc

        with segmenter.inference_lock:
            try:
                return await asyncio.to_thread(
                    segmenter.infer,
                    waveform,
                    diarization,
                    speaker_metadata,
                    audio.filename or "audio.bin",
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
