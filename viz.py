#!/usr/bin/env python3
"""Render mel + onset-label PNG pages for preprocessed MAESTRO pieces.

Each piece is split into fixed-length pages (default 30 seconds). Per page:
  Top:    log-mel spectrogram (low frequency at bottom).
  Bottom: ground-truth onset piano-roll (low pitch at bottom, dark = onset).

Time axes are shared and labelled in absolute seconds so you can eyeball
audio/label alignment for quality review.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maestro import (
    HOP,
    MIDI_LOW,
    N_KEYS,
    SR,
    MaestroItem,
    load_split_items,
    preproc_base_dir,
    preproc_paths,
)


C_NOTE_NAMES = {
    24: "C1",
    36: "C2",
    48: "C3",
    60: "C4",
    72: "C5",
    84: "C6",
    96: "C7",
    108: "C8",
}


def render_panel(
    mel_view: np.ndarray,
    roll_view: np.ndarray,
    start_sec: float,
    end_sec: float,
    title: str,
    out_path: str,
    info_text: Optional[str] = None,
    dpi: int = 110,
) -> None:
    """Two-panel figure: mel on top, 88-key piano-roll on bottom.

    `roll_view` is `[88, T]`, values in [0, 1] (binary onsets or continuous
    probabilities both work with the Greys cmap).
    """
    duration = end_sec - start_sec
    width = float(np.clip(duration * 0.4, 8.0, 24.0))
    fig, (ax_mel, ax_roll) = plt.subplots(
        2, 1, figsize=(width, 6.5), sharex=True, constrained_layout=True
    )

    vmin = float(np.percentile(mel_view, 1.0))
    vmax = float(np.percentile(mel_view, 99.0))
    ax_mel.imshow(
        mel_view,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=(start_sec, end_sec, 0.0, mel_view.shape[0]),
        interpolation="nearest",
    )
    ax_mel.set_ylabel("mel bin")
    ax_mel.set_title(title, fontsize=10)

    ax_roll.imshow(
        roll_view,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=(start_sec, end_sec, MIDI_LOW - 0.5, MIDI_LOW + N_KEYS - 0.5),
        interpolation="nearest",
    )
    c_pitches = [p for p in C_NOTE_NAMES if MIDI_LOW <= p <= MIDI_LOW + N_KEYS - 1]
    ax_roll.set_yticks(c_pitches)
    ax_roll.set_yticklabels([C_NOTE_NAMES[p] for p in c_pitches])
    ax_roll.set_ylabel("MIDI pitch")
    ax_roll.set_xlabel("time (s)")
    ax_roll.grid(True, axis="y", color="0.85", linewidth=0.5)
    if info_text:
        ax_roll.text(
            0.995,
            0.97,
            info_text,
            transform=ax_roll.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.35",
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def render_piece_pages(
    mel_path: str,
    labels_path: str,
    out_dir: str,
    title_prefix: str,
    page_seconds: float = 30.0,
    overwrite: bool = False,
    dpi: int = 110,
) -> List[str]:
    mel = np.load(mel_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    total_frames = min(int(mel.shape[0]), int(labels.shape[0]))
    if total_frames <= 0:
        return []

    frames_per_page = max(1, int(round(page_seconds * SR / HOP)))
    n_pages = (total_frames + frames_per_page - 1) // frames_per_page

    written: List[str] = []
    for pi in range(n_pages):
        start = pi * frames_per_page
        end = min(total_frames, start + frames_per_page)
        start_sec = start * HOP / SR
        end_sec = end * HOP / SR
        out_path = os.path.join(out_dir, f"viz_{int(round(start_sec)):04d}s.png")
        written.append(out_path)
        if not overwrite and os.path.exists(out_path):
            continue

        mel_view = np.asarray(mel[start:end], dtype=np.float32).T
        labels_view = np.asarray(labels[start:end], dtype=np.float32).T
        title = f"{title_prefix}  [page {pi + 1}/{n_pages}, {start_sec:.0f}-{end_sec:.0f}s]"
        n_onsets = int((np.diff(labels_view, axis=1) > 0).sum())
        render_panel(
            mel_view=mel_view,
            roll_view=labels_view,
            start_sec=start_sec,
            end_sec=end_sec,
            title=title,
            out_path=out_path,
            info_text=f"{n_onsets} onsets",
            dpi=dpi,
        )

    return written


def render_item(
    item: MaestroItem,
    preproc_root: str,
    page_seconds: float = 30.0,
    overwrite: bool = False,
) -> List[str]:
    mel_path, labels_path = preproc_paths(preproc_root, item)
    base_dir = preproc_base_dir(preproc_root, item)
    if not (os.path.exists(mel_path) and os.path.exists(labels_path)):
        return []
    title_prefix = f"{item.split} | {os.path.basename(item.audio_path)}"
    return render_piece_pages(
        mel_path=mel_path,
        labels_path=labels_path,
        out_dir=base_dir,
        title_prefix=title_prefix,
        page_seconds=page_seconds,
        overwrite=overwrite,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--preproc_root", required=True)
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--page_seconds", type=float, default=30.0)
    p.add_argument("--limit_per_split", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes over pieces (default 1 = serial).",
    )
    args = p.parse_args()

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    grand_pages = 0
    for split in splits:
        items = load_split_items(args.data_root, split, limit_items=args.limit_per_split)
        print(f"[viz] split={split}: {len(items)} pieces, workers={args.workers}")
        split_pages = 0
        if args.workers <= 1:
            iter_results = (
                (i, item, render_item(item, args.preproc_root, args.page_seconds, args.overwrite))
                for i, item in enumerate(items, start=1)
            )
        else:
            ex = ProcessPoolExecutor(max_workers=args.workers)
            futures = {
                ex.submit(
                    render_item,
                    item,
                    args.preproc_root,
                    args.page_seconds,
                    args.overwrite,
                ): item
                for item in items
            }

            def _drain():
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    item = futures[fut]
                    try:
                        yield done, item, fut.result()
                    except Exception as exc:
                        print(f"  [{done}/{len(items)}] FAILED {item.audio_path}: {exc}")
                        yield done, item, []

            iter_results = _drain()

        for done, item, paths in iter_results:
            split_pages += len(paths)
            if not paths:
                print(f"  [{done}/{len(items)}] missing npy, skipped {item.audio_path}")
            elif done % 25 == 0 or done == len(items) or done <= 3:
                base_dir = preproc_base_dir(args.preproc_root, item)
                print(f"  [{done}/{len(items)}] {len(paths)} pages -> {base_dir}/")

        if args.workers > 1:
            ex.shutdown(wait=True)
        grand_pages += split_pages
        print(f"[viz] split={split}: {split_pages} pages total")
    print(f"[viz] done: {grand_pages} pages across {len(splits)} split(s)")


if __name__ == "__main__":
    main()
