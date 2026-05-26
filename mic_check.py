#!/usr/bin/env python3
"""Record a few seconds from the mic and report what the model would see.

Useful for diagnosing spurious detections: run in a quiet room, check whether
the mic's noise floor is hot enough to trigger the onset detector.

Usage:
    uv run python mic_check.py [--seconds 5] [--model output/train.pt]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

try:
    import sounddevice as sd
except OSError as exc:
    raise SystemExit(
        "sounddevice failed to import. Install PortAudio: "
        f"`sudo apt install libportaudio2` or `brew install portaudio`.\n  {exc}"
    )

from maestro import SR, MelSpecProcessor
from model import OnsetCRNN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--model", default=None, help="Run the model and report max onset probabilities.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--input_device", default=None, help="sounddevice input device index or name.")
    p.add_argument("--list_devices", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"Recording {args.seconds:.1f}s of audio at {SR} Hz ...", file=sys.stderr)
    x = sd.rec(int(args.seconds * SR), samplerate=SR, channels=1, dtype="float32",
               device=args.input_device)
    sd.wait()
    mono = x[:, 0]

    rms = float(np.sqrt((mono ** 2).mean()))
    peak = float(np.abs(mono).max())
    print(f"audio:  rms={rms:.6f}  peak={peak:.4f}  samples={len(mono)}")

    proc = MelSpecProcessor()
    wav_t = torch.from_numpy(mono).unsqueeze(0)
    mel = proc(wav_t, SR, normalize="fixed")
    print(f"mel (fixed norm): shape={tuple(mel.shape)}  min={mel.min():.3f}  max={mel.max():.3f}  mean={mel.mean():.3f}")

    if rms > 0.001:
        print("WARNING: RMS > 0.001 in a quiet room — mic gain is probably too high.", file=sys.stderr)
    if peak > 0.05:
        print("WARNING: peak > 0.05 — check for clipping or AGC boost.", file=sys.stderr)

    if args.model is None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OnsetCRNN.from_checkpoint(args.model, map_location=device).to(device).eval()
    with torch.no_grad():
        mel_t = mel.unsqueeze(0).to(device)
        logits, _ = model(mel_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    max_per_key = probs.max(axis=0)
    hot_keys = np.where(max_per_key > args.threshold)[0]
    print(f"model:  max_prob={probs.max():.3f}  keys_above_{args.threshold}={len(hot_keys)}")
    if len(hot_keys) > 0:
        from maestro import MIDI_LOW
        for k in hot_keys:
            print(f"  key {k} (MIDI {k + MIDI_LOW}): max_prob={max_per_key[k]:.3f}")


if __name__ == "__main__":
    main()
