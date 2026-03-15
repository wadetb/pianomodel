#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from maestro import (
    FMIN,
    HOP,
    N_FFT,
    N_MELS,
    SR,
    MelSpecProcessor,
    load_mono_wav,
    load_split_items,
    midi_onset_frames,
    preproc_base_dir,
)


def save_mel_png(
    mel: np.ndarray,
    png_path: str,
    sr: int = SR,
    hop: int = HOP,
    fmin: float = FMIN,
) -> None:
    """Save a mel spectrogram as a PNG image.

    Args:
        mel: Mel spectrogram array of shape (T, n_mels).
        png_path: Output path for the PNG file.
        sr: Sample rate used to compute time axis.
        hop: Hop length used to compute time axis.
        fmin: Minimum frequency for the y-axis label.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    duration_s = mel.shape[0] * hop / sr
    ax.imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        extent=[0, duration_s, 0, mel.shape[1]],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title("Mel Spectrogram")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def preprocess_split(
    data_root: str,
    split: str,
    preproc_root: str,
    processor: MelSpecProcessor,
    overwrite: bool = False,
    limit_items: Optional[int] = None,
    save_png: bool = True,
) -> None:
    items = load_split_items(data_root=data_root, split=split, limit_items=limit_items)
    print(
        f"Preprocessing {len(items)} items for split '{split}' -> {preproc_root} (npy)"
    )
    os.makedirs(os.path.join(preproc_root, split), exist_ok=True)

    for i, item in enumerate(items, start=1):
        base_dir = preproc_base_dir(preproc_root, item)
        mel_path = os.path.join(base_dir, "mel.npy")
        labels_path = os.path.join(base_dir, "labels.npy")
        png_path = os.path.join(base_dir, "mel.png")

        need_npy = overwrite or not (
            os.path.exists(mel_path) and os.path.exists(labels_path)
        )
        need_png = save_png and (overwrite or not os.path.exists(png_path))

        if not need_npy and not need_png:
            if i % 10 == 0:
                print(f"  [{i}/{len(items)}] exists, skipping: {mel_path}")
            continue

        wav, sr = load_mono_wav(item.audio_path)
        with np.errstate(all="raise"):
            mel = processor(wav, sr)
        mel_np = mel.cpu().numpy().astype(np.float32)
        total_frames = mel_np.shape[0]
        labels = midi_onset_frames(
            item.midi_path,
            total_frames=total_frames,
            sr=processor.sample_rate,
            hop=processor.hop,
            widen=1,
        )

        os.makedirs(base_dir, exist_ok=True)

        if need_npy:
            np.save(mel_path, mel_np, allow_pickle=False)
            np.save(labels_path, labels.astype(np.float32), allow_pickle=False)

        if need_png:
            save_mel_png(
                mel_np, png_path, sr=processor.sample_rate, hop=processor.hop
            )

        if i % 10 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] wrote {mel_path} & {labels_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess MAESTRO into mel.npy + labels.npy"
    )
    p.add_argument("--data_root", required=True, help="Path to MAESTRO root")
    p.add_argument(
        "--preproc_root", required=True, help="Output directory for preprocessed files"
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="Comma-separated splits: train,validation,test",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument(
        "--limit_per_split",
        type=int,
        default=None,
        help="Optional item cap per split",
    )
    p.add_argument(
        "--no_png",
        action="store_true",
        help="Skip generating mel spectrum PNG images",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.preproc_root, exist_ok=True)
    processor = MelSpecProcessor(sample_rate=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)
    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        preprocess_split(
            data_root=args.data_root,
            split=split,
            preproc_root=args.preproc_root,
            processor=processor,
            overwrite=args.overwrite,
            limit_items=args.limit_per_split,
            save_png=not args.no_png,
        )


if __name__ == "__main__":
    main()
