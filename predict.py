#!/usr/bin/env python3
"""Visualize model onset predictions on an arbitrary WAV slice.

Top:    log-mel spectrogram of the slice.
Bottom: per-frame onset probabilities from the model (88 keys x time, dark = high).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from decode import peak_pick, refractory_frames_from_ms
from maestro import (
    HOP,
    N_FFT,
    N_MELS,
    SR,
    MelSpecProcessor,
    load_mono_wav,
)
from model import OnsetCRNN
from viz import render_panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True, help="Input WAV path.")
    p.add_argument(
        "--model",
        default="output/train.pt",
        help="Model checkpoint (.pt). Default: output/train.pt.",
    )
    p.add_argument("--start", type=float, default=0.0, help="Slice start in seconds.")
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Slice duration in seconds (max 600 to avoid huge PNGs).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: output/predict/<wav_basename>_<start>-<end>s.png.",
    )
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto).")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the peak-pick count shown in panel info.",
    )
    p.add_argument(
        "--refractory_ms",
        type=float,
        default=50.0,
        help="Per-pitch refractory window for peak picking.",
    )
    p.add_argument("--n_fft", type=int, default=N_FFT,
                   help="STFT window. Set 2048 for mel-229/logfreq-256 models.")
    p.add_argument("--hop", type=int, default=HOP)
    p.add_argument("--sample_rate", type=int, default=SR)
    p.add_argument("--repr", choices=["mel", "logfreq"], default="mel")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.duration > 600:
        raise SystemExit("--duration must be in (0, 600] seconds")
    if not os.path.exists(args.wav):
        raise SystemExit(f"wav not found: {args.wav}")
    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    wav, sr = load_mono_wav(args.wav)
    total_sec = wav.shape[-1] / sr
    if args.start >= total_sec:
        raise SystemExit(
            f"--start ({args.start}s) is past wav duration ({total_sec:.2f}s)"
        )
    end_sec = float(min(total_sec, args.start + args.duration))
    start_samp = int(round(args.start * sr))
    end_samp = int(round(end_sec * sr))
    wav_slice = wav[:, start_samp:end_samp]
    print(
        f"[predict] wav={args.wav} sr={sr} total={total_sec:.2f}s "
        f"slice=[{args.start:.2f}, {end_sec:.2f}]s "
        f"({wav_slice.shape[-1]} samples) device={device}"
    )

    model = OnsetCRNN.from_checkpoint(args.model, map_location=device).to(device).eval()
    mel_mean, mel_std = model.get_mel_stats()
    proc_kwargs = dict(
        sample_rate=args.sample_rate, n_fft=args.n_fft, hop=args.hop,
        n_mels=model.n_mels, repr_type=args.repr,
    )
    if mel_mean is not None:
        proc_kwargs["mel_mean"] = mel_mean
    if mel_std is not None:
        proc_kwargs["mel_std"] = mel_std
    processor = MelSpecProcessor(**proc_kwargs)
    print(f"[predict] processor: n_mels={model.n_mels} n_fft={args.n_fft} repr={args.repr} "
          f"mel_mean={mel_mean} mel_std={mel_std} freq_pool={model.freq_pool}")
    with torch.no_grad():
        mel = processor(wav_slice, sr, normalize="fixed")  # [T, n_mels]
    with torch.no_grad():
        x = mel.unsqueeze(0).to(device)
        logits, _ = model(x)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()  # [T, 88]

    refractory_frames = refractory_frames_from_ms(args.refractory_ms)
    events = peak_pick(probs, threshold=args.threshold, refractory_frames=refractory_frames)
    n_peaks = len(events)
    print(
        f"[predict] model_hidden={model.hidden} model_conv={model.conv_channels} "
        f"mel={tuple(mel.shape)} probs={probs.shape} "
        f"max={probs.max():.3f} peaks(p>={args.threshold})={n_peaks}"
    )

    if args.out is None:
        wav_base = os.path.splitext(os.path.basename(args.wav))[0]
        out_dir = "./output/predict"
        out_path = os.path.join(
            out_dir, f"{wav_base}_{int(round(args.start)):04d}-{int(round(end_sec)):04d}s.png"
        )
    else:
        out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"{os.path.basename(args.wav)}  [{args.start:.2f}-{end_sec:.2f}s]  "
        f"pred via {os.path.basename(args.model)}"
    )
    render_panel(
        mel_view=mel.cpu().numpy().T,
        roll_view=probs.T,
        start_sec=args.start,
        end_sec=end_sec,
        title=title,
        out_path=out_path,
        info_text=f"{n_peaks} peaks @ p>={args.threshold:.2f}",
    )
    print(f"[predict] wrote {out_path}")


if __name__ == "__main__":
    main()
