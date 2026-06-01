#!/usr/bin/env python3
"""Live microphone -> streaming onset detection -> ASCII pianoroll.

Run on a machine with a working audio input (i.e. *not* over plain SSH).
Each row of output is one streaming chunk; the position of 'X' marks the
MIDI pitch of detected onsets in that chunk. 'C' letters in the header
mark octaves.

Defaults: 16 kHz mono mic, 16-frame chunks (160 ms latency), threshold 0.65.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np
import torch

try:
    import sounddevice as sd
except OSError as exc:  # PortAudio missing
    raise SystemExit(
        "sounddevice failed to import. On Linux this usually means PortAudio "
        "isn't installed: `sudo apt-get install libportaudio2`. Original error:\n"
        f"  {exc}"
    )

from maestro import HOP, MIDI_LOW, N_FFT, N_KEYS, SR
from model import OnsetCRNN
from stream import (
    StreamingMelExtractor,
    StreamingOnsetDetector,
    make_pianoroll_header,
    pianoroll_row,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="output/train.pt")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--refractory_frames", type=int, default=5)
    p.add_argument(
        "--chunk_frames",
        type=int,
        default=16,
        help="Mel frames per inference call. 16 = 160 ms inference latency.",
    )
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    p.add_argument(
        "--input_device",
        default=None,
        help="sounddevice input device index or name substring. None = system default.",
    )
    p.add_argument(
        "--input_sr",
        type=int,
        default=SR,
        help=f"Microphone sample rate (default {SR}). Higher rates get resampled to {SR}.",
    )
    p.add_argument(
        "--list_devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    p.add_argument("--n_fft", type=int, default=N_FFT,
                   help="STFT window. Set 2048 for mel-229/logfreq-256 models.")
    p.add_argument("--hop", type=int, default=HOP)
    p.add_argument("--sample_rate", type=int, default=SR)
    p.add_argument("--repr", choices=["mel", "logfreq"], default="mel")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    device_t = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = (
        OnsetCRNN.from_checkpoint(args.model, map_location=device_t).to(device_t).eval()
    )
    print(
        f"# model={args.model}  hidden={model.hidden}  conv={model.conv_channels}  "
        f"device={device_t}",
        file=sys.stderr,
    )

    mel_mean, mel_std = model.get_mel_stats()
    pre = model.get_preproc_config()
    n_fft = args.n_fft if args.n_fft != N_FFT else pre.get("n_fft", args.n_fft)
    hop = args.hop if args.hop != HOP else pre.get("hop", args.hop)
    sample_rate = args.sample_rate if args.sample_rate != SR else pre.get("sample_rate", args.sample_rate)
    repr_type = args.repr if args.repr != "mel" else pre.get("repr_type", args.repr)
    print(
        f"# n_mels={model.n_mels} n_fft={n_fft} hop={hop} sr={sample_rate} "
        f"repr={repr_type} mel_mean={mel_mean} mel_std={mel_std}",
        file=sys.stderr,
    )
    extractor = StreamingMelExtractor(
        sr=sample_rate, n_fft=n_fft, hop=hop, n_mels=model.n_mels,
        repr_type=repr_type, mel_mean=mel_mean, mel_std=mel_std,
    )
    detector = StreamingOnsetDetector(
        model,
        threshold=args.threshold,
        refractory_frames=args.refractory_frames,
    )

    # Hardware chunk size: at least one mel chunk's worth of input samples.
    audio_block = int(args.chunk_frames * hop * args.input_sr / sample_rate)
    print(
        f"# input_sr={args.input_sr}  audio_block={audio_block} samples  "
        f"chunk_frames={args.chunk_frames}  threshold={args.threshold}  refractory={args.refractory_frames}",
        file=sys.stderr,
    )
    print(f"# {make_pianoroll_header()}")

    start = time.perf_counter()
    last_print = start

    def on_audio(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal last_print
        if status:
            print(f"# audio status: {status}", file=sys.stderr)
        mono = indata[:, 0] if indata.ndim == 2 else indata
        # sounddevice gives float32 in [-1, 1] for the default float dtype.
        wav = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
        new_mel = extractor.push(wav, args.input_sr)
        if new_mel.size == 0:
            return
        events = detector.process(new_mel)
        now = time.perf_counter() - start
        if events:
            print(f"{now:6.2f}s {pianoroll_row(events)}", flush=True)
            last_print = time.perf_counter()
        elif time.perf_counter() - last_print > 1.0:
            # Print a blank row every second of silence so the user knows it's alive.
            print(f"{now:6.2f}s {'.' * N_KEYS}", flush=True)
            last_print = time.perf_counter()

    try:
        with sd.InputStream(
            samplerate=args.input_sr,
            channels=1,
            blocksize=audio_block,
            dtype="float32",
            callback=on_audio,
            device=args.input_device,
        ):
            print("# listening — Ctrl+C to stop", file=sys.stderr)
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start
        print(
            f"\n# stopped after {elapsed:.1f}s  "
            f"detector processed {detector.total_frames_in} frames",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
