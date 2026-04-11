from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def decode_audio(path: Path, sample_rate_hz: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Decoded audio was empty.")
    return audio
