"""Onset decoding helpers shared by predict.py and evaluate.py."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from maestro import HOP, MIDI_LOW, SR


def peak_pick(
    probs: np.ndarray,
    threshold: float = 0.5,
    refractory_frames: int = 5,
) -> List[Tuple[int, int]]:
    """Per-pitch local-maximum peak picking with a refractory window.

    Args:
        probs: [T, K] sigmoid probabilities, K=88.
        threshold: minimum probability to consider.
        refractory_frames: minimum gap (in frames) between consecutive picks per pitch.

    Returns: list of (frame_idx, pitch_idx) sorted by frame, then pitch.
    """
    T, K = probs.shape
    if T < 3:
        return []
    is_peak = np.zeros_like(probs, dtype=bool)
    middle = probs[1:-1]
    is_peak[1:-1] = (
        (middle >= threshold)
        & (middle > probs[:-2])
        & (middle >= probs[2:])
    )
    events: List[Tuple[int, int]] = []
    for k in range(K):
        candidates = np.where(is_peak[:, k])[0]
        if candidates.size == 0:
            continue
        last = -refractory_frames
        for t in candidates:
            t_int = int(t)
            if t_int - last >= refractory_frames:
                events.append((t_int, k))
                last = t_int
    events.sort()
    return events


def events_to_onsets(
    events: List[Tuple[int, int]],
    hop: int = HOP,
    sr: int = SR,
    midi_low: int = MIDI_LOW,
) -> List[Tuple[float, int]]:
    """Convert (frame_idx, pitch_idx) events to (seconds, midi_pitch)."""
    return [(t * hop / sr, midi_low + k) for t, k in events]


def refractory_frames_from_ms(ms: float, hop: int = HOP, sr: int = SR) -> int:
    return max(1, int(round(ms * sr / (hop * 1000.0))))
