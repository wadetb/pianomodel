#!/usr/bin/env python3
"""Teaching-app-focused evaluation: measures how well the model supports
a "did the student play the right note?" use case.

For each ground-truth onset in the test set, checks:
1. Is the correct pitch detected (above threshold) within a time window?
2. How many other pitches are also detected (candidate set size)?
3. What is the probability rank of the correct pitch among all 88?

Reports metrics at multiple thresholds to find the sweet spot between
catching correct notes (recall) and keeping the candidate set tight
(so wrong-note identification works).

Uses cached predictions from cache_predictions.py.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from evaluate import midi_onsets
from maestro import HOP, MIDI_LOW, SR, load_split_items, preproc_paths


@dataclass
class TeachingJob:
    audio_basename: str
    cache_path: str
    midi_path: str
    thresholds: Tuple[float, ...]
    window_frames: int


def score_one_piece_teaching(job: TeachingJob) -> Dict:
    probs = np.load(job.cache_path).astype(np.float32, copy=False)
    T, K = probs.shape
    ref_onsets = midi_onsets(job.midi_path)
    if not ref_onsets:
        return {"audio": job.audio_basename, "n_ref": 0, "per_t": {}}

    frame_rate = SR / HOP

    per_t: Dict[float, Dict] = {}
    for thresh in job.thresholds:
        correct_detected = 0
        total_candidates = 0
        total_onsets = 0
        rank_sum = 0
        prob_sum = 0.0
        wrong_note_identifiable = 0

        for onset_time, midi_pitch in ref_onsets:
            pitch_idx = midi_pitch - MIDI_LOW
            if pitch_idx < 0 or pitch_idx >= K:
                continue

            center_frame = int(round(onset_time * frame_rate))
            lo = max(0, center_frame - job.window_frames)
            hi = min(T, center_frame + job.window_frames + 1)
            if lo >= hi:
                continue

            total_onsets += 1

            window = probs[lo:hi]  # [W, 88]
            correct_prob = float(window[:, pitch_idx].max())
            prob_sum += correct_prob

            max_probs = window.max(axis=0)  # [88] - max prob per pitch in window
            candidates_above = int((max_probs >= thresh).sum())
            total_candidates += candidates_above

            rank = int((max_probs > correct_prob).sum()) + 1
            rank_sum += rank

            if correct_prob >= thresh:
                correct_detected += 1
            else:
                non_correct_max = float(np.delete(max_probs, pitch_idx).max())
                if non_correct_max >= thresh:
                    wrong_note_identifiable += 1

        if total_onsets == 0:
            continue

        per_t[thresh] = {
            "onset_recall": correct_detected / total_onsets,
            "mean_candidates": total_candidates / total_onsets,
            "mean_rank": rank_sum / total_onsets,
            "mean_correct_prob": prob_sum / total_onsets,
            "n_correct": correct_detected,
            "n_missed": total_onsets - correct_detected,
            "n_missed_with_wrong": wrong_note_identifiable,
            "n_onsets": total_onsets,
        }

    return {
        "audio": job.audio_basename,
        "n_ref": len(ref_onsets),
        "per_t": per_t,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--preproc_root", required=True)
    p.add_argument("--model", required=True, help="Model checkpoint (used to locate cache)")
    p.add_argument("--split", default="test")
    p.add_argument(
        "--threshold",
        type=str,
        default="0.20,0.30,0.40,0.50,0.60,0.70,0.80",
    )
    p.add_argument("--window_ms", type=float, default=50.0, help="Onset matching window (ms)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = tuple(float(t) for t in args.threshold.split(","))
    window_frames = max(1, int(round(args.window_ms / 1000.0 * SR / HOP)))

    model_stem = Path(args.model).stem
    cache_root = Path("output/probs_cache") / model_stem

    items = load_split_items(data_root=args.data_root, split=args.split)
    jobs: List[TeachingJob] = []
    for item in items:
        basename = os.path.splitext(os.path.basename(item.audio_path))[0]
        cache_path = cache_root / f"{basename}.npy"
        if not cache_path.exists():
            continue
        jobs.append(TeachingJob(
            audio_basename=basename,
            cache_path=str(cache_path),
            midi_path=item.midi_path,
            thresholds=thresholds,
            window_frames=window_frames,
        ))

    print(f"[teaching-eval] model={args.model} split={args.split} pieces={len(jobs)} "
          f"window={args.window_ms}ms ({window_frames}f) thresholds={list(thresholds)}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(score_one_piece_teaching, j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"teaching-eval {args.split}"):
            results.append(fut.result())

    agg: Dict[float, Dict[str, float]] = {}
    for thresh in thresholds:
        total_onsets = 0
        total_correct = 0
        total_candidates = 0
        total_rank = 0
        total_prob = 0.0
        total_missed = 0
        total_missed_with_wrong = 0
        pieces_with_data = 0

        for r in results:
            if thresh not in r["per_t"]:
                continue
            s = r["per_t"][thresh]
            n = s["n_onsets"]
            total_onsets += n
            total_correct += s["n_correct"]
            total_candidates += s["mean_candidates"] * n
            total_rank += s["mean_rank"] * n
            total_prob += s["mean_correct_prob"] * n
            total_missed += s["n_missed"]
            total_missed_with_wrong += s["n_missed_with_wrong"]
            pieces_with_data += 1

        if total_onsets == 0:
            continue
        agg[thresh] = {
            "onset_recall": total_correct / total_onsets,
            "mean_candidates": total_candidates / total_onsets,
            "mean_rank": total_rank / total_onsets,
            "mean_correct_prob": total_prob / total_onsets,
            "missed_with_wrong_note": total_missed_with_wrong / total_onsets,
            "total_onsets": total_onsets,
            "pieces": pieces_with_data,
        }

    print(f"\n{'thresh':>6s}  {'recall':>6s}  {'cands':>5s}  {'rank':>5s}  {'prob':>5s}  {'wrong':>6s}")
    print("-" * 48)
    for thresh in thresholds:
        if thresh not in agg:
            continue
        a = agg[thresh]
        print(f"  {thresh:4.2f}  {a['onset_recall']:6.3f}  {a['mean_candidates']:5.1f}  "
              f"{a['mean_rank']:5.2f}  {a['mean_correct_prob']:5.3f}  {a['missed_with_wrong_note']:6.3f}")

    if args.out:
        out_data = {
            "model": args.model,
            "split": args.split,
            "window_ms": args.window_ms,
            "summary": {str(k): v for k, v in agg.items()},
        }
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n[teaching-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
