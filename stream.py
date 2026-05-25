#!/usr/bin/env python3
"""Streaming inference primitives for live/mic and large-scale evaluation.

`StreamingOnsetDetector` is the shared core: it owns the model + GRU state
and a peak-picker that tracks state across chunk boundaries. Both `live.py`
(mic input) and `evaluate_streaming.py` (offline) feed it mel chunks.

`StreamingMelExtractor` turns raw audio chunks into mel frames; it's only
used by the live demo. The offline evaluator operates on already-preprocessed
mel files.

For a quick end-to-end test, run with --wav to stream a WAV file through both
the mel extractor and the detector and print detected onsets.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

from maestro import (
    HOP,
    MIDI_LOW,
    N_FFT,
    N_KEYS,
    N_MELS,
    SR,
    MelSpecProcessor,
    load_mono_wav,
)
from model import OnsetCRNN


C_NOTE_NAMES = {
    21: "A0",
    24: "C1", 36: "C2", 48: "C3", 60: "C4",
    72: "C5", 84: "C6", 96: "C7", 108: "C8",
}

# Conv receptive field on the time axis (3 SeparableConv2d blocks, 3x3, padding 1, stride 1).
# Each chunk needs this many frames of context on each side of its convs to avoid
# zero-padding artifacts (which otherwise drop F1 ~50% on small chunks).
CONV_TIME_CONTEXT = 3


class StreamingPeakPicker:
    """Peak picker that tracks state across chunk boundaries.

    Call `process(prob_chunk)` with consecutive [C, 88] probability chunks.
    Returns a list of (global_frame_idx, pitch_idx) onset events emitted
    during this chunk. Frame indices are global from the last `reset()`.

    Decision rule matches `decode.peak_pick`:
        is_peak[t] = (p[t] >= threshold) & (p[t] > p[t-1]) & (p[t] >= p[t+1])
    plus a per-pitch refractory window. Because the rule needs p[t+1], the
    last frame of each chunk is held back and decided on the next call (so
    onset events are emitted with one-frame look-ahead delay).
    """

    def __init__(self, threshold: float = 0.65, refractory_frames: int = 5):
        self.threshold = threshold
        self.refractory_frames = refractory_frames
        self.reset()

    def reset(self) -> None:
        self._tail = np.zeros((0, N_KEYS), dtype=np.float32)
        self._tail_global_offset = 0  # global frame index of self._tail[0]
        self._last_emit_frame = np.full(N_KEYS, -(10**9), dtype=np.int64)

    def process(self, chunk_probs: np.ndarray) -> List[Tuple[int, int]]:
        extended = np.concatenate([self._tail, chunk_probs], axis=0)
        T = extended.shape[0]
        if T < 3:
            self._tail = extended  # offset unchanged; we haven't dropped frames
            return []

        middle = extended[1:-1]
        is_peak = (
            (middle >= self.threshold)
            & (middle > extended[:-2])
            & (middle >= extended[2:])
        )  # [T-2, 88] — is_peak[i] decides extended[i+1]

        events: List[Tuple[int, int]] = []
        for k in range(N_KEYS):
            for i in np.where(is_peak[:, k])[0]:
                global_frame = self._tail_global_offset + int(i) + 1
                if global_frame - self._last_emit_frame[k] >= self.refractory_frames:
                    events.append((global_frame, int(k)))
                    self._last_emit_frame[k] = global_frame
        events.sort()

        # Keep last 2 frames: extended[T-2] is the latest decided frame (needed as
        # predecessor for extended[T-1]'s decision next call), and extended[T-1] is
        # the undecided latest frame (needs a successor from the next chunk).
        self._tail = extended[-2:].copy()
        self._tail_global_offset = self._tail_global_offset + T - 2
        return events


class StreamingOnsetDetector:
    """Live streaming detector: model with GRU state + conv-context overlap + peak picker.

    Internally buffers mel frames so each model call gets `context_frames` of
    real left/right context on its convs (avoids the conv-padding artifacts
    that drop F1 ~50% on short chunks). This adds `context_frames * 10 ms` of
    end-to-end latency on top of `chunk_frames * 10 ms` (so chunk=16 + ctx=3
    = 190 ms total).

    Push frames via `process(mel_frames)` as audio arrives; returns events
    that can now be emitted (some may be delayed by the right-context wait).
    Call `flush()` at end-of-stream to drain the final partial chunk.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        chunk_frames: int = 16,
        context_frames: int = CONV_TIME_CONTEXT,
        threshold: float = 0.65,
        refractory_frames: int = 5,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.chunk_frames = chunk_frames
        self.context_frames = context_frames
        self.picker = StreamingPeakPicker(threshold, refractory_frames)
        self.reset()

    def reset(self) -> None:
        self.h: Optional[torch.Tensor] = None
        self._buffer = np.zeros((0, N_MELS), dtype=np.float32)
        self._frames_processed = 0  # mel rows in buffer already fed to GRU
        self.total_frames_in = 0
        self.picker.reset()

    @torch.no_grad()
    def _run_chunk(self, win_start: int, win_end: int, chunk_size: int) -> List[Tuple[int, int]]:
        left_trim = self._frames_processed - win_start
        right_trim = win_end - (self._frames_processed + chunk_size)
        window = self._buffer[win_start:win_end]
        x = (
            torch.from_numpy(np.ascontiguousarray(window))
            .unsqueeze(0)
            .to(self.device)
        )
        logits, self.h = self.model.forward_with_context(
            x, self.h, left_trim=left_trim, right_trim=right_trim
        )
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [chunk_size, 88]
        self._frames_processed += chunk_size
        return self.picker.process(probs)

    def process(self, mel_frames: np.ndarray) -> List[Tuple[int, int]]:
        """Append mel frames; emit events for any chunks that have full right context."""
        if mel_frames.shape[0] == 0:
            return []
        self._buffer = np.concatenate([self._buffer, mel_frames], axis=0)
        self.total_frames_in += mel_frames.shape[0]

        events: List[Tuple[int, int]] = []
        while True:
            # Need at least chunk_frames + context_frames unprocessed frames so we have
            # enough right context to run a full chunk cleanly.
            need_end = self._frames_processed + self.chunk_frames + self.context_frames
            if self._buffer.shape[0] < need_end:
                break
            win_start = max(0, self._frames_processed - self.context_frames)
            win_end = need_end
            events.extend(self._run_chunk(win_start, win_end, self.chunk_frames))

        # Trim already-processed history, keeping context_frames as left context for next chunk.
        keep_from = max(0, self._frames_processed - self.context_frames)
        if keep_from > 0:
            self._buffer = self._buffer[keep_from:]
            self._frames_processed -= keep_from
        return events

    @torch.no_grad()
    def flush(self) -> List[Tuple[int, int]]:
        """Drain the final partial chunk at end-of-stream (no right context available)."""
        events: List[Tuple[int, int]] = []
        remaining = self._buffer.shape[0] - self._frames_processed
        if remaining <= 0:
            return events
        win_start = max(0, self._frames_processed - self.context_frames)
        win_end = self._buffer.shape[0]
        events.extend(self._run_chunk(win_start, win_end, remaining))
        return events


@torch.no_grad()
def chunked_inference(
    model: torch.nn.Module,
    mel: np.ndarray,
    chunk_frames: int,
    device: Optional[torch.device] = None,
    context_frames: int = CONV_TIME_CONTEXT,
) -> np.ndarray:
    """Run model on mel in fixed-size chunks with conv-context overlap.

    Each chunk's convs see `context_frames` extra frames on each side (when
    available) to avoid zero-padding artifacts at chunk boundaries; the GRU
    state advances by exactly `chunk_frames` per call so it matches what
    whole-piece inference would produce.

    Returns: [T, 88] sigmoid probabilities. T = mel.shape[0].
    """
    if device is None:
        device = next(model.parameters()).device
    T = int(mel.shape[0])
    if T == 0:
        return np.zeros((0, N_KEYS), dtype=np.float32)
    parts: List[np.ndarray] = []
    h: Optional[torch.Tensor] = None
    for start in range(0, T, chunk_frames):
        end = min(T, start + chunk_frames)
        win_start = max(0, start - context_frames)
        win_end = min(T, end + context_frames)
        left_trim = start - win_start
        right_trim = win_end - end
        x = (
            torch.from_numpy(np.ascontiguousarray(mel[win_start:win_end]))
            .unsqueeze(0)
            .to(device)
        )
        logits, h = model.forward_with_context(x, h, left_trim=left_trim, right_trim=right_trim)
        parts.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
    return np.concatenate(parts, axis=0)


class StreamingMelExtractor:
    """Buffer raw audio and emit new mel frames as they become available.

    Simple sliding-buffer approach: append incoming audio, compute mel on
    the (windowed) buffer, return only mel frames that haven't been emitted
    yet. Keeps the audio buffer bounded to a few seconds.
    """

    def __init__(
        self,
        sr: int = SR,
        n_fft: int = N_FFT,
        hop: int = HOP,
        n_mels: int = N_MELS,
        buffer_seconds: float = 4.0,
    ):
        self.processor = MelSpecProcessor(sample_rate=sr, n_fft=n_fft, hop=hop, n_mels=n_mels)
        self.sr = sr
        self.hop = hop
        self.n_fft = n_fft
        self.max_buffer_samples = int(buffer_seconds * sr)
        self.audio_buf = torch.zeros((1, 0), dtype=torch.float32)
        self.input_sr: Optional[int] = None
        # samples_already_emitted: how many input-sr samples have been "covered"
        # by mel frames we've already returned. Each emitted frame covers `hop`
        # input-sr samples (mel hop is at sr, the processor handles resampling).
        self.frames_emitted = 0

    def push(self, audio: torch.Tensor, input_sr: int) -> np.ndarray:
        """audio: [1, T] or [T] float32. Returns [n_new, n_mels] mel frames."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if self.input_sr is None:
            self.input_sr = input_sr
        elif self.input_sr != input_sr:
            raise ValueError(
                f"input_sr changed mid-stream: was {self.input_sr}, got {input_sr}"
            )

        self.audio_buf = torch.cat([self.audio_buf, audio], dim=1)
        if self.audio_buf.shape[1] < self.n_fft:
            return np.zeros((0, self.processor.melspec.n_mels), dtype=np.float32)

        # Trim buffer to bounded length (drop oldest, but keep enough context).
        if self.audio_buf.shape[1] > self.max_buffer_samples:
            drop = self.audio_buf.shape[1] - self.max_buffer_samples
            # Round drop down to a multiple of hop (in input_sr) so emitted-frame
            # counting stays sample-aligned.
            drop -= drop % self.hop
            if drop > 0:
                self.audio_buf = self.audio_buf[:, drop:]
                self.frames_emitted -= drop // self.hop

        with torch.no_grad():
            mel = self.processor(self.audio_buf, self.input_sr, normalize="streaming")
        n_total = int(mel.shape[0])
        if n_total <= self.frames_emitted:
            return np.zeros((0, self.processor.melspec.n_mels), dtype=np.float32)

        new = mel[self.frames_emitted : n_total].cpu().numpy()
        self.frames_emitted = n_total
        return new


def make_pianoroll_header(width_cols: int = N_KEYS) -> str:
    """Header row labelling octaves above the pianoroll column space."""
    chars = ["."] * width_cols
    for midi, label in C_NOTE_NAMES.items():
        idx = midi - MIDI_LOW
        if 0 <= idx < width_cols:
            # Place the first char of the label at idx if it fits.
            for j, c in enumerate(label):
                if 0 <= idx + j < width_cols:
                    chars[idx + j] = c
    return "".join(chars)


def pianoroll_row(events: List[Tuple[int, int]], width_cols: int = N_KEYS) -> str:
    """Render onsets in one chunk as an 88-char row; '.' inactive, 'X' onset."""
    row = ["."] * width_cols
    for _, k in events:
        if 0 <= k < width_cols:
            row[k] = "X"
    return "".join(row)


def stream_wav(
    wav_path: str,
    model_path: str,
    chunk_frames: int = 16,
    threshold: float = 0.65,
    refractory_frames: int = 5,
    device: Optional[str] = None,
    show_pianoroll: bool = True,
) -> List[Tuple[float, int]]:
    """End-to-end streaming over a WAV file, useful for testing without a mic.

    Returns: list of (time_seconds, midi_pitch) onset events.
    """
    device_t = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = (
        OnsetCRNN.from_checkpoint(model_path, map_location=device_t).to(device_t).eval()
    )

    mel_ext = StreamingMelExtractor()
    detector = StreamingOnsetDetector(
        model,
        chunk_frames=chunk_frames,
        threshold=threshold,
        refractory_frames=refractory_frames,
    )

    wav, sr = load_mono_wav(wav_path)
    chunk_samples = chunk_frames * HOP  # in target SR; we'll size audio chunks at input_sr
    # Convert chunk_samples to input_sr (since the extractor will resample internally).
    audio_chunk_samples = max(1, int(round(chunk_samples * sr / SR)))

    if show_pianoroll:
        print(
            f"# streaming {wav_path}  model={os.path.basename(model_path)} "
            f"chunk={chunk_frames}f ({chunk_frames * HOP * 1000 / SR:.0f}ms)  "
            f"thresh={threshold}  refractory={refractory_frames}f"
        )
        print(f"# {make_pianoroll_header()}")

    events_out: List[Tuple[float, int]] = []
    total_samples = wav.shape[-1]

    def emit(events):
        if not events:
            return
        for fr, k in events:
            events_out.append(((fr * HOP) / SR, MIDI_LOW + k))
        if show_pianoroll:
            t_sec = (events[0][0] * HOP) / SR
            print(f"{t_sec:6.2f}s {pianoroll_row(events)}")

    for start in range(0, total_samples, audio_chunk_samples):
        end = min(total_samples, start + audio_chunk_samples)
        audio = wav[:, start:end]
        new_mel = mel_ext.push(audio, sr)
        if new_mel.size == 0:
            continue
        emit(detector.process(new_mel))

    emit(detector.flush())
    return events_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True, help="Input WAV path.")
    p.add_argument("--model", default="output/train.pt")
    p.add_argument("--chunk_frames", type=int, default=16,
                   help="Frames per streaming chunk (default 16 = 160 ms latency).")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--refractory_frames", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--no_pianoroll", action="store_true",
                   help="Suppress the ASCII pianoroll output; just count events.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = stream_wav(
        wav_path=args.wav,
        model_path=args.model,
        chunk_frames=args.chunk_frames,
        threshold=args.threshold,
        refractory_frames=args.refractory_frames,
        device=args.device,
        show_pianoroll=not args.no_pianoroll,
    )
    print(f"# detected {len(events)} onsets")
    for t, p in events[:10]:
        print(f"#   t={t:.3f}s pitch={p}")


if __name__ == "__main__":
    main()
