from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log

import numpy as np

from app.diarization import SpeakerTurn


@dataclass(frozen=True)
class Window:
    speaker_id: str
    start_seconds: float
    end_seconds: float
    audio_start_sample: int
    audio_end_sample: int
    coverage_start_seconds: float
    coverage_end_seconds: float


def build_windows(
    diarization: list[SpeakerTurn],
    sample_rate_hz: int,
    window_seconds: float,
    hop_seconds: float,
    audio_duration_seconds: float,
) -> list[Window]:
    windows: list[Window] = []
    for turn in diarization:
        turn_duration = turn.end_seconds - turn.start_seconds
        if turn_duration <= 0:
            continue

        if turn_duration <= window_seconds:
            starts = [turn.start_seconds]
        else:
            steps = int(ceil((turn_duration - window_seconds) / hop_seconds)) + 1
            starts = [turn.start_seconds + step * hop_seconds for step in range(steps)]
            last_start = max(turn.start_seconds, turn.end_seconds - window_seconds)
            starts.append(last_start)

        seen: set[int] = set()
        for start_seconds in starts:
            start_seconds = min(start_seconds, max(turn.start_seconds, turn.end_seconds - window_seconds))
            start_sample = int(round(start_seconds * sample_rate_hz))
            if start_sample in seen:
                continue
            seen.add(start_sample)
            end_seconds = min(audio_duration_seconds, start_seconds + window_seconds)
            end_sample = int(round(end_seconds * sample_rate_hz))
            windows.append(
                Window(
                    speaker_id=turn.speaker_id,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    audio_start_sample=start_sample,
                    audio_end_sample=end_sample,
                    coverage_start_seconds=max(turn.start_seconds, start_seconds),
                    coverage_end_seconds=min(turn.end_seconds, end_seconds),
                )
            )
    return sorted(windows, key=lambda window: (window.start_seconds, window.end_seconds, window.speaker_id))


def _moving_average(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return values
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    padded = np.pad(values, ((kernel_size // 2, kernel_size // 2), (0, 0)), mode="edge")
    return np.vstack(
        [
            np.convolve(padded[:, index], kernel, mode="valid")
            for index in range(values.shape[1])
        ]
    ).T


def normalized_entropy(probabilities: np.ndarray) -> float:
    epsilon = 1e-8
    safe = np.clip(probabilities, epsilon, 1.0)
    return float(-np.sum(safe * np.log(safe)) / log(probabilities.size))


def smooth_and_merge(
    diarization: list[SpeakerTurn],
    window_predictions: list[dict],
    label_names: list[str],
    resolution_seconds: float,
    smoothing_seconds: float,
    min_segment_seconds: float,
    merge_gap_seconds: float,
) -> list[dict]:
    resolution_seconds = max(0.05, resolution_seconds)
    kernel_size = max(1, int(round(smoothing_seconds / resolution_seconds)))
    min_bins = max(1, int(round(min_segment_seconds / resolution_seconds)))
    grouped: dict[str, list[SpeakerTurn]] = {}
    for turn in diarization:
        grouped.setdefault(turn.speaker_id, []).append(turn)

    by_speaker_window: dict[str, list[dict]] = {}
    for window in window_predictions:
        by_speaker_window.setdefault(window["speaker_id"], []).append(window)

    segments: list[dict] = []
    for speaker_id, turns in grouped.items():
        speaker_windows = by_speaker_window.get(speaker_id, [])
        if not speaker_windows:
            continue
        for turn in turns:
            bin_count = max(1, int(ceil((turn.end_seconds - turn.start_seconds) / resolution_seconds)))
            probs = np.zeros((bin_count, len(label_names)), dtype=np.float32)
            avd = np.zeros((bin_count, 3), dtype=np.float32)
            weights = np.zeros(bin_count, dtype=np.float32)

            for window in speaker_windows:
                overlap_start = max(turn.start_seconds, window["_coverage_start_seconds"])
                overlap_end = min(turn.end_seconds, window["_coverage_end_seconds"])
                if overlap_end <= overlap_start:
                    continue
                start_index = max(0, int((overlap_start - turn.start_seconds) // resolution_seconds))
                end_index = min(bin_count, int(ceil((overlap_end - turn.start_seconds) / resolution_seconds)))
                for index in range(start_index, end_index):
                    bin_start = turn.start_seconds + index * resolution_seconds
                    bin_end = min(turn.end_seconds, bin_start + resolution_seconds)
                    weight = max(
                        0.0,
                        min(bin_end, overlap_end) - max(bin_start, overlap_start),
                    )
                    if weight <= 0.0:
                        continue
                    probs[index] += np.asarray(window["_categorical_probabilities_vector"], dtype=np.float32) * weight
                    if window.get("_avd_vector") is not None:
                        avd[index] += np.asarray(window["_avd_vector"], dtype=np.float32) * weight
                    weights[index] += weight

            weights = np.where(weights <= 0.0, 1.0, weights)
            probs = probs / weights[:, None]
            avd = avd / weights[:, None]
            probs = _moving_average(probs, kernel_size)
            avd = _moving_average(avd, kernel_size)

            labels = np.argmax(probs, axis=1)
            gap_bins = max(0, int(round(merge_gap_seconds / resolution_seconds)))
            if gap_bins > 0:
                for index in range(1, bin_count - 1):
                    if int(labels[index - 1]) == int(labels[index + 1]) != int(labels[index]):
                        labels[index] = labels[index - 1]

            start_index = 0
            while start_index < bin_count:
                current_label = int(labels[start_index])
                end_index = start_index + 1
                while end_index < bin_count and int(labels[end_index]) == current_label:
                    end_index += 1

                if end_index - start_index < min_bins:
                    if start_index > 0:
                        current_label = int(labels[start_index - 1])
                        labels[start_index:end_index] = current_label
                    elif end_index < bin_count:
                        current_label = int(labels[end_index])
                        labels[start_index:end_index] = current_label
                    else:
                        labels[start_index:end_index] = current_label
                start_index = end_index

            start_index = 0
            while start_index < bin_count:
                current_label = int(labels[start_index])
                end_index = start_index + 1
                while end_index < bin_count and int(labels[end_index]) == current_label:
                    end_index += 1

                segment_probs = probs[start_index:end_index].mean(axis=0)
                segment_avd = avd[start_index:end_index].mean(axis=0)
                confidence = float(np.max(segment_probs))
                segments.append(
                    {
                        "speaker_id": speaker_id,
                        "start_seconds": round(turn.start_seconds + start_index * resolution_seconds, 3),
                        "end_seconds": round(
                            min(turn.end_seconds, turn.start_seconds + end_index * resolution_seconds),
                            3,
                        ),
                        "label": label_names[current_label],
                        "categorical_probabilities": {
                            label: round(float(value), 6)
                            for label, value in zip(label_names, segment_probs, strict=True)
                        },
                        "avd_scores": {
                            "arousal": round(float(segment_avd[0]), 6),
                            "valence": round(float(segment_avd[2]), 6),
                            "dominance": round(float(segment_avd[1]), 6),
                        },
                        "confidence": round(confidence, 6),
                        "uncertainty": round(normalized_entropy(segment_probs), 6),
                    }
                )
                start_index = end_index

    return segments
