#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from typing import Dict, Optional

import numpy as np

from maestro import (
    HOP,
    N_FFT,
    N_MELS,
    SR,
    MaestroItem,
    MelSpecProcessor,
    find_maestro_csv,
    load_mono_wav,
    load_split_items,
    midi_onset_frames,
    preproc_base_dir,
)
from viz import render_piece_pages


def _load_metadata_map(data_root: str) -> Dict[str, Dict[str, str]]:
    """Build {audio_basename: csv_row_dict} for fast per-item metadata lookup."""
    csv_path = find_maestro_csv(data_root)
    out: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            audio_rel = (
                row.get("audio_filename") or row.get("audio_path") or row.get("audio")
            )
            if audio_rel:
                out[os.path.basename(audio_rel)] = dict(row)
    return out


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_source_assets(
    item: MaestroItem,
    base_dir: str,
    metadata_map: Dict[str, Dict[str, str]],
    overwrite: bool,
) -> None:
    for src in (item.audio_path, item.midi_path):
        if not os.path.exists(src):
            continue
        dst = os.path.join(base_dir, os.path.basename(src))
        if os.path.exists(dst):
            if not overwrite:
                continue
            os.remove(dst)
        _link_or_copy(src, dst)

    md = metadata_map.get(os.path.basename(item.audio_path))
    if md is None:
        return
    md_path = os.path.join(base_dir, "metadata.json")
    if os.path.exists(md_path) and not overwrite:
        return
    with open(md_path, "w") as f:
        json.dump(md, f, indent=2, sort_keys=True)


def preprocess_split(
    data_root: str,
    split: str,
    preproc_root: str,
    processor: MelSpecProcessor,
    overwrite: bool = False,
    limit_items: Optional[int] = None,
    viz: bool = False,
    viz_page_seconds: float = 30.0,
    normalize: str = "per_piece",
) -> None:
    items = load_split_items(data_root=data_root, split=split, limit_items=limit_items)
    print(
        f"Preprocessing {len(items)} items for split '{split}' -> {preproc_root} (npy)"
    )
    os.makedirs(os.path.join(preproc_root, split), exist_ok=True)
    metadata_map = _load_metadata_map(data_root)

    for i, item in enumerate(items, start=1):
        base_dir = preproc_base_dir(preproc_root, item)
        mel_path = os.path.join(base_dir, "mel.npy")
        labels_path = os.path.join(base_dir, "labels.npy")

        if not overwrite and os.path.exists(mel_path) and os.path.exists(labels_path):
            if i % 10 == 0:
                print(f"  [{i}/{len(items)}] exists, skipping: {mel_path}")
            os.makedirs(base_dir, exist_ok=True)
            _copy_source_assets(item, base_dir, metadata_map, overwrite)
            if viz:
                render_piece_pages(
                    mel_path=mel_path,
                    labels_path=labels_path,
                    out_dir=base_dir,
                    title_prefix=f"{split} | {os.path.basename(item.audio_path)}",
                    page_seconds=viz_page_seconds,
                    overwrite=overwrite,
                )
            continue

        wav, sr = load_mono_wav(item.audio_path)
        with np.errstate(all="raise"):
            mel = processor(wav, sr, normalize=normalize)
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
        _copy_source_assets(item, base_dir, metadata_map, overwrite)

        if viz:
            render_piece_pages(
                mel_path=mel_path,
                labels_path=labels_path,
                out_dir=base_dir,
                title_prefix=f"{split} | {os.path.basename(item.audio_path)}",
                page_seconds=viz_page_seconds,
                overwrite=overwrite,
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
        "--viz",
        action="store_true",
        help="Also emit paginated mel+pianoroll PNGs next to each piece's npy files.",
    )
    p.add_argument(
        "--viz_page_seconds",
        type=float,
        default=30.0,
        help="Seconds per visualization page (default 30).",
    )
    p.add_argument(
        "--normalize",
        choices=["per_piece", "fixed"],
        default="per_piece",
        help="Mel normalization mode. 'fixed' uses global training-set stats.",
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
            viz=args.viz,
            viz_page_seconds=args.viz_page_seconds,
            normalize=args.normalize,
        )


if __name__ == "__main__":
    main()
