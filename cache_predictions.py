#!/usr/bin/env python3
"""Cache model probability outputs per piece for fast decoder iteration.

For each piece in a split, runs the model once and saves a `[T, 88]` sigmoid
probability tensor to `output/probs_cache/<model_stem>/<piece_stem>.npy` (float16
by default to halve disk usage; 0.0005 precision is more than enough for
threshold sweeps).

Once cached, evaluate.py with `--use_cache` skips the model forward pass entirely
and threshold/refractory sweeps cost milliseconds instead of minutes.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from evaluate import predict_piece
from maestro import MaestroItem, load_split_items, preproc_paths
from model import OnsetCRNN
from stream import chunked_inference


DEFAULT_CHUNK_FRAMES = 6000  # matches evaluate.predict_piece default


def cache_dir_for(
    model_path: str,
    root: str = "output/probs_cache",
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> Path:
    base = Path(root) / Path(model_path).stem
    if chunk_frames != DEFAULT_CHUNK_FRAMES:
        base = base / f"c{chunk_frames}"
    return base


def cache_path_for(cache_dir: Path, item: MaestroItem) -> Path:
    return cache_dir / (Path(item.audio_path).stem + ".npy")


@torch.no_grad()
def predict_pieces_batched(
    model: torch.nn.Module,
    mel_paths: List[str],
    device: torch.device,
    batch_size: int,
) -> List[np.ndarray]:
    """Length-bucketed batched inference.

    Sorts the input mels by length so each padded batch wastes as little
    compute as possible. GRU starts from h=0 per piece (no carryover from
    previous batch), matching `predict_piece` behavior at chunk boundaries.

    Returns probs in the same order as mel_paths.
    """
    mels = [np.load(p).astype(np.float32) for p in mel_paths]
    order = sorted(range(len(mels)), key=lambda i: mels[i].shape[0], reverse=True)
    out: List[np.ndarray] = [None] * len(mels)  # type: ignore

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        Tmax = max(mels[i].shape[0] for i in idxs)
        n_mels = mels[idxs[0]].shape[1]
        batch = np.zeros((len(idxs), Tmax, n_mels), dtype=np.float32)
        lengths = []
        for k, i in enumerate(idxs):
            T = mels[i].shape[0]
            batch[k, :T] = mels[i]
            lengths.append(T)
        x = torch.from_numpy(batch).to(device)
        logits, _ = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()  # [B, Tmax, 88]
        for k, i in enumerate(idxs):
            out[i] = probs[k, : lengths[k]].copy()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--preproc_root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--limit_items", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype. float16 saves 50%% disk; precision is ~0.0005 in [0,1].",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batched inference: pieces are length-sorted then padded into batches "
        "of this size. 1 = original per-piece chunked behavior with GRU state carryover.",
    )
    p.add_argument(
        "--cache_root",
        default="output/probs_cache",
        help="Cache root; per-model dir is <cache_root>/<model_stem>/.",
    )
    p.add_argument(
        "--chunk_frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help=f"Chunk size for streaming inference (default {DEFAULT_CHUNK_FRAMES}). "
        "Smaller = closer to deployment behavior. Non-default values write to "
        "<cache_dir>/c<chunk_frames>/ to keep caches distinct.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = OnsetCRNN.from_checkpoint(args.model, map_location=device).to(device).eval()
    items = load_split_items(args.data_root, args.split, limit_items=args.limit_items)

    cache_dir = cache_dir_for(args.model, args.cache_root, args.chunk_frames)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    print(
        f"[cache] model={args.model} hidden={model.hidden} conv={model.conv_channels} "
        f"split={args.split} pieces={len(items)} batch_size={args.batch_size} dtype={args.dtype} "
        f"chunk_frames={args.chunk_frames}"
    )
    print(f"[cache] -> {cache_dir}/")

    # Filter out pieces that already have a cache file (unless --overwrite).
    pending: List[Tuple[MaestroItem, str, Path]] = []
    for item in items:
        out_path = cache_path_for(cache_dir, item)
        if out_path.exists() and not args.overwrite:
            continue
        mel_path, _ = preproc_paths(args.preproc_root, item)
        if not os.path.exists(mel_path):
            print(f"  missing mel, skipping: {mel_path}")
            continue
        pending.append((item, mel_path, out_path))

    if not pending:
        print("[cache] nothing to do (all cached, use --overwrite to refresh)")
        return

    print(f"[cache] {len(pending)} pieces to predict")

    if args.batch_size <= 1:
        for item, mel_path, out_path in tqdm(pending, desc="cache"):
            mel = np.load(mel_path).astype(np.float32)
            probs = chunked_inference(model, mel, args.chunk_frames, device).astype(
                np_dtype, copy=False
            )
            np.save(out_path, probs, allow_pickle=False)
    else:
        # Process in batches over the pending list.
        # We chunk into groups of (e.g.) 64 pieces to keep memory bounded across the run,
        # then within each group do length-bucketed batched inference of `batch_size`.
        group_size = max(args.batch_size * 8, 32)
        for grp_start in tqdm(range(0, len(pending), group_size), desc="groups"):
            group = pending[grp_start : grp_start + group_size]
            mel_paths = [g[1] for g in group]
            probs_list = predict_pieces_batched(model, mel_paths, device, args.batch_size)
            for (item, _, out_path), probs in zip(group, probs_list):
                np.save(out_path, probs.astype(np_dtype, copy=False), allow_pickle=False)

    total_bytes = sum(out_path.stat().st_size for _, _, out_path in pending)
    print(
        f"[cache] wrote {len(pending)} files, total {total_bytes / (1024**2):.1f} MiB "
        f"(avg {total_bytes / max(len(pending),1) / 1024:.1f} KiB/piece)"
    )


if __name__ == "__main__":
    main()
