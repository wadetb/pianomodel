#!/usr/bin/env python3
"""Print onset events for a WAV via the exported ONNX pipeline.

The host-side half of the on-device parity check (MOBILE.md §7): run this on
the same WAV the app's debugRunWavAsset feeds through its pipeline and compare
the printed (time, midi) lines. Output format matches the app's debugPrint.

    uv run python onnx_events.py --wav ../musml-client/assets/test/parity.wav
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from maestro import load_mono_wav
from onnx_runtime import OnnxStreamingDetector, compute_logmel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True)
    p.add_argument("--config", default="output/mel229_fixed_aug_2x_chunk16.json")
    p.add_argument("--threshold", type=float, default=None,
                   help="Default: the config's bundled threshold.")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    pre = cfg["preproc"]
    asset_dir = args.config.rsplit("/", 1)[0] if "/" in args.config else "."
    fb = np.load(f"{asset_dir}/{pre['mel_filterbank_npy']}").astype(np.float32)

    wav, sr = load_mono_wav(args.wav)
    audio = wav.squeeze(0).numpy().astype(np.float32)
    if sr != pre["sample_rate"]:
        import torchaudio
        import torch
        audio = (
            torchaudio.functional.resample(
                torch.from_numpy(audio).unsqueeze(0), sr, pre["sample_rate"])
            .squeeze(0).numpy().astype(np.float32)
        )

    mel = compute_logmel(
        audio, fb,
        n_fft=pre["n_fft"], hop=pre["hop"],
        mel_mean=pre["mel_mean"], mel_std=pre["mel_std"],
    )
    # Match the app's live extractor: no end-padded frames (a stream has no end).
    live_frames = max(0, (audio.shape[0] - pre["n_fft"] // 2) // pre["hop"] + 1)
    mel = mel[:live_frames]

    det = OnnxStreamingDetector(cfg, asset_dir=asset_dir, threshold=args.threshold)
    events = det.process(mel)

    thr = cfg["decode"]["default_threshold"] if args.threshold is None else args.threshold
    hop, srate = pre["hop"], pre["sample_rate"]
    for frame, pitch in events:
        t = frame * hop / srate
        print(f"onset t={t:.2f}s midi={pitch + cfg['decode']['midi_low']}")
    print(f"{args.wav}: {len(events)} onsets (threshold {thr})")


if __name__ == "__main__":
    main()
