#!/usr/bin/env python3
import argparse
import os
from typing import Optional

import numpy as np

from maestro import (
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


def preprocess_split(
    data_root: str,
    split: str,
    preproc_root: str,
    processor: MelSpecProcessor,
    overwrite: bool = False,
    limit_items: Optional[int] = None,
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

        if not overwrite and os.path.exists(mel_path) and os.path.exists(labels_path):
            if i % 10 == 0:
                print(f"  [{i}/{len(items)}] exists, skipping: {mel_path}")
            continue

        wav, sr = load_mono_wav(item.audio_path)
        with np.errstate(all="raise"):
            mel = processor(wav, sr)
        total_frames = mel.shape[0]
        labels = midi_onset_frames(
            item.midi_path,
            total_frames=total_frames,
            sr=processor.sample_rate,
            hop=processor.hop,
            widen=1,
        )

        os.makedirs(base_dir, exist_ok=True)
        np.save(mel_path, mel.cpu().numpy().astype(np.float32), allow_pickle=False)
        np.save(labels_path, labels.astype(np.float32), allow_pickle=False)

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
        )


if __name__ == "__main__":
    main()
