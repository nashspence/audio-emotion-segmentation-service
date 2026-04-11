from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start_seconds: float
    end_seconds: float


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc


def _coerce_segment(item: dict[str, Any]) -> SpeakerTurn:
    speaker_id = str(
        item.get("speaker_id")
        or item.get("speaker")
        or item.get("speakerId")
        or item.get("id")
        or "speaker_0"
    )
    start_seconds = _as_float(
        item.get("start_seconds", item.get("start", item.get("offset_seconds", 0.0))),
        "segment start",
    )
    end_seconds = _as_float(
        item.get("end_seconds", item.get("end", item.get("stop_seconds", 0.0))),
        "segment end",
    )
    if end_seconds <= start_seconds:
        raise ValueError(f"Invalid diarization segment for {speaker_id}: end must exceed start.")
    return SpeakerTurn(
        speaker_id=speaker_id,
        start_seconds=max(0.0, start_seconds),
        end_seconds=end_seconds,
    )


def parse_diarization_payload(payload: str | None, duration_seconds: float) -> list[SpeakerTurn]:
    if not payload:
        return [SpeakerTurn(speaker_id="speaker_0", start_seconds=0.0, end_seconds=duration_seconds)]

    parsed = json.loads(payload)
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = parsed.get("segments") or parsed.get("diarization") or parsed.get("speakers")
        if not isinstance(items, list):
            raise ValueError("Diarization JSON must contain a list or a 'segments' array.")
    else:
        raise ValueError("Unsupported diarization JSON payload.")

    segments = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each diarization segment must be an object.")
        segment = _coerce_segment(item)
        if segment.start_seconds >= duration_seconds:
            continue
        clipped_end = min(duration_seconds, segment.end_seconds)
        if clipped_end <= segment.start_seconds:
            continue
        segments.append(
            SpeakerTurn(
                speaker_id=segment.speaker_id,
                start_seconds=segment.start_seconds,
                end_seconds=clipped_end,
            )
        )

    if not segments:
        raise ValueError("No usable diarization segments remained after clipping to the audio duration.")

    return sorted(segments, key=lambda segment: (segment.start_seconds, segment.end_seconds, segment.speaker_id))


def parse_speaker_metadata(payload: str | None) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("Speaker metadata must be a JSON object keyed by speaker id.")
    metadata: dict[str, dict[str, Any]] = {}
    for key, value in parsed.items():
        metadata[str(key)] = value if isinstance(value, dict) else {"value": value}
    return metadata
