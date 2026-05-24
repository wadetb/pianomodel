#!/usr/bin/env python3
"""Note-level mir_eval F1 for an OnsetCRNN checkpoint over a MAESTRO split.

Per piece: load mel.npy, run model, peak-pick to (time, pitch) events, then
compare to ground-truth MIDI onsets via mir_eval.transcription with a 50 ms
onset tolerance and 50-cent pitch tolerance.

Reports macro-averaged precision/recall/F1 (mean of per-piece scores) — the
standard headline metric in the piano transcription literature.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
import torch
from tqdm.auto import tqdm

from decode import events_to_onsets, peak_pick, refractory_frames_from_ms
from maestro import (
    HOP,
    MIDI_HIGH,
    MIDI_LOW,
    SR,
    load_split_items,
    preproc_paths,
)
from model import OnsetCRNN

import mir_eval
from mir_eval.transcription import precision_recall_f1_overlap


def midi_onsets(
    midi_path: str,
    midi_low: int = MIDI_LOW,
    midi_high: int = MIDI_HIGH,
) -> List[Tuple[float, int]]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    out: List[Tuple[float, int]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if midi_low <= note.pitch <= midi_high:
                out.append((float(note.start), int(note.pitch)))
    out.sort()
    return out


def to_mir_eval_arrays(
    onsets: List[Tuple[float, int]],
    fake_offset: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    if not onsets:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    times = np.array([o[0] for o in onsets], dtype=float)
    pitches = np.array([o[1] for o in onsets], dtype=float)
    intervals = np.stack([times, times + fake_offset], axis=1)
    hz = mir_eval.util.midi_to_hz(pitches)
    return intervals, hz


@dataclass
class ScoreJob:
    """Picklable per-piece work unit for the multiprocessing pool (cache mode)."""

    audio_basename: str
    cache_path: str
    midi_path: str
    thresholds: Tuple[float, ...]
    refractory_frames: int
    onset_tolerance_s: float
    pitch_tolerance_cents: float


def score_one_piece(job: ScoreJob) -> Dict:
    """Compute per-threshold P/R/F for one piece. Module-level for pickling."""
    probs = np.load(job.cache_path).astype(np.float32, copy=False)
    ref_onsets = midi_onsets(job.midi_path)
    ref_intervals, ref_hz = to_mir_eval_arrays(ref_onsets)

    per_t: Dict[float, Dict] = {}
    for t in job.thresholds:
        events = peak_pick(
            probs, threshold=t, refractory_frames=job.refractory_frames
        )
        est_onsets = events_to_onsets(events)
        est_intervals, est_hz = to_mir_eval_arrays(est_onsets)
        if len(ref_hz) == 0 and len(est_hz) == 0:
            p = r = f = 0.0
        else:
            p, r, f, _ = precision_recall_f1_overlap(
                ref_intervals,
                ref_hz,
                est_intervals,
                est_hz,
                onset_tolerance=job.onset_tolerance_s,
                pitch_tolerance=job.pitch_tolerance_cents,
                offset_ratio=None,
                strict=False,
            )
        per_t[t] = {
            "p": float(p),
            "r": float(r),
            "f": float(f),
            "n_est": int(len(est_onsets)),
        }
    return {
        "audio": job.audio_basename,
        "n_ref": int(len(ref_onsets)),
        "per_t": per_t,
    }


@torch.no_grad()
def predict_piece(
    model: torch.nn.Module,
    mel_path: str,
    device: torch.device,
    chunk_frames: int = 6000,
) -> np.ndarray:
    """Run model over a full piece in time-chunks, carrying GRU state across chunks.

    Chunking avoids cuDNN sequence-length limits on multi-minute pieces and keeps
    peak GPU memory bounded.
    """
    mel = np.load(mel_path).astype(np.float32)  # [T, n_mels]
    total = int(mel.shape[0])
    parts = []
    h = None
    for start in range(0, total, chunk_frames):
        end = min(total, start + chunk_frames)
        x = torch.from_numpy(mel[start:end]).unsqueeze(0).to(device).contiguous()
        logits, h = model(x, h)
        parts.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 88), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", required=True)
    p.add_argument("--preproc_root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument(
        "--threshold",
        type=str,
        default="0.5",
        help='Probability threshold(s). Comma-separated for sweep, e.g. "0.4,0.5,0.6". '
        "Top-level macro stats reflect the best-F1 threshold for back-compat.",
    )
    p.add_argument("--refractory_ms", type=float, default=50.0)
    p.add_argument("--onset_tolerance_ms", type=float, default=50.0)
    p.add_argument("--pitch_tolerance_cents", type=float, default=50.0)
    p.add_argument("--limit_items", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="Optional path for JSON output.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-piece tqdm bar.")
    p.add_argument(
        "--use_cache",
        action="store_true",
        help="Read probs from output/probs_cache/<model_stem>/<piece>.npy if present "
        "(populate via cache_predictions.py). Skips model forward entirely when the "
        "cache covers all pieces.",
    )
    p.add_argument(
        "--cache_root",
        default="output/probs_cache",
        help="Cache root used by --use_cache.",
    )
    p.add_argument(
        "--cache_chunk_frames",
        type=int,
        default=6000,
        help="Match cache_predictions.py --chunk_frames so eval finds the right subdir. "
        "Default 6000 = top-level <model_stem>/; non-default uses <model_stem>/c<n>/.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for scoring. >1 requires --use_cache "
        "(otherwise GPU contention). 1 = serial (default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    items = load_split_items(args.data_root, args.split, limit_items=args.limit_items)

    cache_dir = Path(args.cache_root) / Path(args.model).stem
    if args.cache_chunk_frames != 6000:
        cache_dir = cache_dir / f"c{args.cache_chunk_frames}"
    if args.use_cache:
        n_cached = sum(
            1
            for it in items
            if (cache_dir / (Path(it.audio_path).stem + ".npy")).exists()
        )
        cache_complete = n_cached == len(items)
        print(f"[eval] cache: {n_cached}/{len(items)} pieces in {cache_dir}/")
    else:
        cache_complete = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cache_complete:
        # Even with full cache, instantiate the model so we can report its config.
        model = OnsetCRNN.from_checkpoint(args.model, map_location="cpu").eval()
        print("[eval] cache covers all pieces; skipping model forward")
    else:
        model = OnsetCRNN.from_checkpoint(args.model, map_location=device).to(device).eval()

    refractory_frames = refractory_frames_from_ms(args.refractory_ms)
    thresholds = sorted({float(t) for t in args.threshold.split(",") if t.strip()})
    if not thresholds:
        raise SystemExit("--threshold must be one or more comma-separated floats")
    print(
        f"[eval] model={args.model} split={args.split} pieces={len(items)} "
        f"hidden={model.hidden} conv={model.conv_channels} "
        f"thresholds={thresholds} refractory={refractory_frames}f "
        f"({args.refractory_ms:.0f}ms) tol={args.onset_tolerance_ms:.0f}ms"
    )

    per_t = {
        t: {"sum_p": 0.0, "sum_r": 0.0, "sum_f": 0.0, "n_est": 0, "pieces": []}
        for t in thresholds
    }
    n_scored = 0
    n_ref_total = 0

    use_workers = args.workers > 1 and cache_complete
    if args.workers > 1 and not cache_complete:
        print("[eval] --workers >1 requires --use_cache with full coverage; falling back to serial")

    if use_workers:
        # Build picklable jobs and dispatch.
        jobs: List[ScoreJob] = []
        for item in items:
            mel_path, _ = preproc_paths(args.preproc_root, item)
            if not os.path.exists(mel_path):
                continue
            jobs.append(
                ScoreJob(
                    audio_basename=os.path.basename(item.audio_path),
                    cache_path=str(cache_dir / (Path(item.audio_path).stem + ".npy")),
                    midi_path=item.midi_path,
                    thresholds=tuple(thresholds),
                    refractory_frames=refractory_frames,
                    onset_tolerance_s=args.onset_tolerance_ms / 1000.0,
                    pitch_tolerance_cents=args.pitch_tolerance_cents,
                )
            )
        print(f"[eval] dispatching {len(jobs)} pieces across {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(score_one_piece, j): j for j in jobs}
            it = as_completed(futures)
            if not args.quiet:
                it = tqdm(it, total=len(jobs), desc=f"eval {args.split}", dynamic_ncols=True)
            for fut in it:
                res = fut.result()
                n_scored += 1
                n_ref_total += res["n_ref"]
                for t in thresholds:
                    pt = res["per_t"][t]
                    per_t[t]["sum_p"] += pt["p"]
                    per_t[t]["sum_r"] += pt["r"]
                    per_t[t]["sum_f"] += pt["f"]
                    per_t[t]["n_est"] += pt["n_est"]
                    per_t[t]["pieces"].append(
                        {
                            "audio": res["audio"],
                            "n_ref": res["n_ref"],
                            "n_est": pt["n_est"],
                            "p": pt["p"],
                            "r": pt["r"],
                            "f": pt["f"],
                        }
                    )
    else:
        iterable = items if args.quiet else tqdm(items, desc=f"eval {args.split}", dynamic_ncols=True)
        for item in iterable:
            mel_path, _ = preproc_paths(args.preproc_root, item)
            if not os.path.exists(mel_path):
                continue
            cache_path = cache_dir / (Path(item.audio_path).stem + ".npy")
            if args.use_cache and cache_path.exists():
                probs = np.load(cache_path).astype(np.float32, copy=False)
            else:
                probs = predict_piece(model, mel_path, device)
            ref_onsets = midi_onsets(item.midi_path)
            ref_intervals, ref_hz = to_mir_eval_arrays(ref_onsets)
            n_ref_total += len(ref_onsets)
            n_scored += 1

            for t in thresholds:
                events = peak_pick(probs, threshold=t, refractory_frames=refractory_frames)
                est_onsets = events_to_onsets(events)
                est_intervals, est_hz = to_mir_eval_arrays(est_onsets)
                if len(ref_hz) == 0 and len(est_hz) == 0:
                    p = r = f = 0.0
                else:
                    p, r, f, _ = precision_recall_f1_overlap(
                        ref_intervals,
                        ref_hz,
                        est_intervals,
                        est_hz,
                        onset_tolerance=args.onset_tolerance_ms / 1000.0,
                        pitch_tolerance=args.pitch_tolerance_cents,
                        offset_ratio=None,
                        strict=False,
                    )
                per_t[t]["sum_p"] += float(p)
                per_t[t]["sum_r"] += float(r)
                per_t[t]["sum_f"] += float(f)
                per_t[t]["n_est"] += len(est_onsets)
                per_t[t]["pieces"].append(
                    {
                        "audio": os.path.basename(item.audio_path),
                        "n_ref": len(ref_onsets),
                        "n_est": len(est_onsets),
                        "p": float(p),
                        "r": float(r),
                        "f": float(f),
                    }
                )

    if n_scored == 0:
        raise SystemExit("[eval] no pieces scored")

    threshold_results = []
    for t in thresholds:
        d = per_t[t]
        threshold_results.append(
            {
                "threshold": t,
                "n_est_onsets": int(d["n_est"]),
                "macro_p": d["sum_p"] / n_scored,
                "macro_r": d["sum_r"] / n_scored,
                "macro_f": d["sum_f"] / n_scored,
            }
        )
    best = max(threshold_results, key=lambda r: r["macro_f"])

    summary = {
        "split": args.split,
        "model": args.model,
        "model_hidden": int(model.hidden),
        "model_conv_channels": list(model.conv_channels),
        "n_pieces": int(n_scored),
        "n_ref_onsets": int(n_ref_total),
        "n_est_onsets": int(best["n_est_onsets"]),
        "config": {
            "threshold": args.threshold,
            "refractory_ms": args.refractory_ms,
            "refractory_frames": refractory_frames,
            "onset_tolerance_ms": args.onset_tolerance_ms,
            "pitch_tolerance_cents": args.pitch_tolerance_cents,
        },
        "best_threshold": best["threshold"],
        "macro_p": best["macro_p"],
        "macro_r": best["macro_r"],
        "macro_f": best["macro_f"],
        "thresholds": threshold_results,
    }
    print(
        f"[eval] split={args.split} pieces={n_scored} ref={n_ref_total}"
    )
    if len(thresholds) == 1:
        print(
            f"  macro: P={best['macro_p']:.4f} R={best['macro_r']:.4f} F={best['macro_f']:.4f} "
            f"est={best['n_est_onsets']}"
        )
    else:
        print(f"  {'thresh':>7} {'P':>7} {'R':>7} {'F':>7} {'est':>9} {'est/ref':>8}")
        for tr in threshold_results:
            mark = "  <- best" if tr["threshold"] == best["threshold"] else ""
            ratio = tr["n_est_onsets"] / max(n_ref_total, 1)
            print(
                f"  {tr['threshold']:>7.2f} {tr['macro_p']:>7.4f} {tr['macro_r']:>7.4f} "
                f"{tr['macro_f']:>7.4f} {tr['n_est_onsets']:>9d} {ratio:>8.2f}{mark}"
            )

    if args.out:
        out = {"summary": summary, "pieces": per_t[best["threshold"]]["pieces"]}
        if len(thresholds) > 1:
            out["pieces_per_threshold"] = {
                f"{t:.2f}": per_t[t]["pieces"] for t in thresholds
            }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
