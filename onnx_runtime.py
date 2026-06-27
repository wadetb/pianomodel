"""Reference on-device inference pipeline using ONLY numpy + onnxruntime.

This mirrors exactly what the mobile app must implement: raw 16 kHz mono audio in,
onset events (time, midi, confidence) out. It is the executable spec — the parity
test (test_onnx_parity.py) asserts it matches the PyTorch streaming path bit-for-bit
(events) on real audio.

Pipeline:
    audio (16 kHz mono) -> log-mel frames -> 22-frame windows + GRU state -> ONNX core
        -> per-frame onset probs -> stateful peak-pick -> (frame, pitch) events

No torch, no torchaudio — everything here is portable DSP that an app can reimplement
in Swift/Kotlin/C++.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort

# Reuse the (already pure-numpy) streaming peak picker — the app ports this ~30-line class.
from stream import StreamingPeakPicker


def hann_periodic(n: int) -> np.ndarray:
    """torch.hann_window(n, periodic=True) == np.hanning(n+1)[:-1]."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def compute_logmel(
    audio: np.ndarray,
    filterbank: np.ndarray,
    *,
    n_fft: int,
    hop: int,
    mel_mean: float,
    mel_std: float,
) -> np.ndarray:
    """Whole-signal log-mel matching torchaudio MelSpectrogram(center=True, power=2,
    pad_mode='reflect', hann window) + log1p + fixed normalization.

    audio: [num_samples] float32 mono.  filterbank: [n_freqs, n_mels] float32.
    Returns: [T, n_mels] float32 normalized log-mel.

    The app computes this incrementally over a rolling audio buffer; steady-state
    frame values are identical to this whole-signal version (they differ only within
    n_fft/2 of a buffer edge, which the rolling buffer keeps as carried context).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pad = n_fft // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    win = hann_periodic(n_fft)
    n_frames = 1 + (len(padded) - n_fft) // hop
    # Frame the signal: [n_frames, n_fft]
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = padded[idx] * win[None, :]                      # [n_frames, n_fft]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)              # [n_frames, n_freqs]
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)  # power=2.0
    mel = power @ filterbank                                  # [n_frames, n_mels]
    mel = np.log1p(mel).astype(np.float32)                   # log1p
    mel = (mel - np.float32(mel_mean)) / np.float32(mel_std)  # fixed normalize
    return mel


class OnnxStreamingDetector:
    """Streaming onset detector driven by the two exported ONNX graphs.

    Reproduces stream.StreamingOnsetDetector *exactly* (and therefore whole-piece
    offline inference, which it matches to IoU 1.0):

      - First chunk: a one-time `first` graph over a 19-frame window with no left
        context (left_trim=0), zero GRU state. Covers mel frames 0..chunk-1.
      - Steady state: the `steady` graph over a 22-frame window (3 left-ctx + 16 +
        3 right-ctx, left_trim=right_trim=3), carrying GRU state. Slides by `chunk`.

    Why two graphs: the first chunk's conv-internal edge padding (left_trim=0) is NOT
    reproducible by prepending zero mel frames (conv(zeros)!=0 via BN/bias), and the
    recurrent GRU propagates any first-chunk error through the whole stream. Matching
    the first chunk exactly is what keeps streaming == whole-piece.

    Event frame indices returned are TRUE global mel-frame indices (chunk 0 -> 0..15).
    Returns events as (global_mel_frame, pitch_index).
    """

    def __init__(
        self,
        config_path_or_dict,
        threshold: Optional[float] = None,
        refractory_frames: Optional[int] = None,
        asset_dir: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        if isinstance(config_path_or_dict, dict):
            cfg = config_path_or_dict
            asset_dir = asset_dir or "."
        else:
            cfg = json.load(open(config_path_or_dict))
            asset_dir = asset_dir or os.path.dirname(config_path_or_dict)
        io = cfg["io"]
        self.chunk = io["chunk_frames"]
        self.ctx = io["context_frames"]
        self.n_mels = io["n_mels"]
        self.hidden = io["hidden"]
        self.steady_window = io["steady"]["mel_window_frames"]
        self.first_window = io["first"]["mel_window_frames"]
        thr = cfg["decode"]["default_threshold"] if threshold is None else threshold
        ref = cfg["decode"]["refractory_frames"] if refractory_frames is None else refractory_frames
        prov = providers or ["CPUExecutionProvider"]
        self.steady = ort.InferenceSession(os.path.join(asset_dir, io["steady"]["onnx"]), providers=prov)
        self.first = ort.InferenceSession(os.path.join(asset_dir, io["first"]["onnx"]), providers=prov)
        self.picker = StreamingPeakPicker(thr, ref)
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros((0, self.n_mels), dtype=np.float32)
        self._fp = 0                 # mel rows already fed to the GRU (== StreamingOnsetDetector)
        self._h = np.zeros((1, 1, self.hidden), dtype=np.float32)
        self._first_done = False
        self.picker.reset()

    def _run_chunk(self, win_start: int, win_end: int) -> List[Tuple[int, int]]:
        window = self._buffer[win_start:win_end]
        x = window[None, :, :].astype(np.float32)
        sess = self.steady if self._first_done else self.first
        probs, self._h = sess.run(["probs", "h_out"], {"mel_window": x, "h_in": self._h})
        self._first_done = True
        self._fp += self.chunk
        return self.picker.process(probs[0])  # picker frame == true mel frame

    def process(self, mel_frames: np.ndarray) -> List[Tuple[int, int]]:
        if mel_frames.shape[0] == 0:
            return []
        self._buffer = np.concatenate([self._buffer, mel_frames.astype(np.float32)], axis=0)
        events: List[Tuple[int, int]] = []
        while True:
            need_end = self._fp + self.chunk + self.ctx
            if self._buffer.shape[0] < need_end:
                break
            win_start = max(0, self._fp - self.ctx)
            events.extend(self._run_chunk(win_start, need_end))
        keep_from = max(0, self._fp - self.ctx)
        if keep_from > 0:
            self._buffer = self._buffer[keep_from:]
            self._fp -= keep_from
        return events


def detect_from_audio(
    audio: np.ndarray,
    config: dict,
    filterbank: np.ndarray,
    asset_dir: str = ".",
    threshold: Optional[float] = None,
) -> List[Tuple[float, int]]:
    """End-to-end: 16 kHz mono audio -> list of (time_seconds, midi_pitch).

    Convenience wrapper that runs the whole-signal mel then streams it. (The app
    does this incrementally over a rolling audio buffer; steady-state frames match.)
    """
    pre = config["preproc"]
    mel = compute_logmel(
        audio, filterbank,
        n_fft=pre["n_fft"], hop=pre["hop"],
        mel_mean=pre["mel_mean"], mel_std=pre["mel_std"],
    )
    det = OnnxStreamingDetector(config, asset_dir=asset_dir, threshold=threshold)
    raw = det.process(mel)
    hop, sr, midi_low = pre["hop"], pre["sample_rate"], config["decode"]["midi_low"]
    return [((fr * hop) / sr, midi_low + k) for fr, k in raw]
