#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required. Install it with `uv add matplotlib` or run "
        "`uv run --with matplotlib graph.py`."
    ) from exc


EPOCH_RE = re.compile(r"^\s*Epoch\s+(?P<epoch>\d+)(?:/\d+)?")
SUMMARY_RE = re.compile(
    r"train:\s*loss=(?P<train_loss>[-+0-9.eE]+)\s*"
    r"f1=(?P<train_f1>[-+0-9.eE]+)\s*\|\s*"
    r"val:\s*loss=(?P<val_loss>[-+0-9.eE]+)\s*"
    r"f1=(?P<val_f1>[-+0-9.eE]+)\s*"
    r"p=(?P<val_precision>[-+0-9.eE]+)\s*"
    r"r=(?P<val_recall>[-+0-9.eE]+)"
    r"(?:\s*\|\s*t=(?P<epoch_seconds>[-+0-9.eE]+)s\s*"
    r"fps=(?P<train_fps>[-+0-9.eE]+))?"
)

SKIP_DIRS = {"preproc", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training metrics from a JSON stats log or text train.log. "
            "If no --log is given, the newest parseable log under --root is used."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to a log file (json or log). Default: newest parseable log in --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output"),
        help="Search root for auto-discovery when --log is omitted (default: output).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <log_stem>_graph.png beside the log.",
    )
    parser.add_argument("--dpi", type=int, default=130, help="Output image DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively (in addition to saving).",
    )
    return parser.parse_args()


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_json_epochs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    epochs = data.get("epochs") if isinstance(data, dict) else None
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("json does not contain non-empty `epochs` list")
    if not all(isinstance(row, dict) for row in epochs):
        raise ValueError("`epochs` must be a list of objects")
    return epochs


def load_text_epochs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_epoch: int | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group("epoch"))
                continue

            summary_match = SUMMARY_RE.search(line)
            if not summary_match:
                continue

            if current_epoch is None:
                current_epoch = len(rows) + 1

            row: dict[str, Any] = {"epoch": current_epoch}
            for key, value in summary_match.groupdict().items():
                if value is not None:
                    row[key] = float(value)
            rows.append(row)

    if not rows:
        raise ValueError("text log does not contain parseable epoch summaries")
    return rows


def load_epochs(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json_epochs(path)
    return load_text_epochs(path)


def sorted_epochs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda it: to_float(it[1].get("epoch", it[0] + 1)))
    return [row for _, row in indexed]


def series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([to_float(row.get(key)) for row in rows], dtype=float)


def has_values(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).any())


def find_latest_parseable(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"search root does not exist: {root}")

    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            name = filename.lower()
            if not name.endswith((".json", ".log", ".txt")):
                continue
            if not any(token in name for token in ("train", "stats", "epoch")):
                continue
            candidates.append(Path(dirpath) / filename)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    errors: list[str] = []
    for candidate in candidates:
        try:
            return candidate, load_epochs(candidate)
        except Exception as exc:  # noqa: BLE001 - keep discovery robust.
            errors.append(f"{candidate}: {exc}")

    detail = "\n".join(errors[:5])
    raise FileNotFoundError(
        f"no parseable logs found under {root}. Checked {len(candidates)} candidates.\n{detail}"
    )


def default_output_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_graph.png")


def plot_epochs(
    rows: list[dict[str, Any]],
    source_path: Path,
    out_path: Path,
    dpi: int,
    show: bool = False,
) -> None:
    rows = sorted_epochs(rows)
    x = np.array(
        [to_float(row.get("epoch", idx + 1)) for idx, row in enumerate(rows)], dtype=float
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    fig.suptitle(f"Training Metrics: {source_path}", fontsize=13)

    loss_ax = axes[0]
    loss_series = [
        ("train_loss", "train loss"),
        ("val_loss", "val loss"),
    ]
    for key, label in loss_series:
        values = series(rows, key)
        if has_values(values):
            loss_ax.plot(x, values, marker="o", ms=3, linewidth=2, label=label)
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Loss")
    if loss_ax.lines:
        loss_ax.legend(loc="best")

    score_ax = axes[1]
    score_series = [
        ("train_f1", "train f1"),
        ("val_f1", "val f1"),
        ("val_precision", "val precision"),
        ("val_recall", "val recall"),
    ]
    for key, label in score_series:
        values = series(rows, key)
        if has_values(values):
            score_ax.plot(x, values, marker="o", ms=3, linewidth=2, label=label)
    score_ax.set_ylabel("score")
    score_ax.set_ylim(0.0, 1.0)
    score_ax.set_title("F1 / Precision / Recall")
    if score_ax.lines:
        score_ax.legend(loc="best")

    speed_ax = axes[2]
    time_series = [
        ("epoch_seconds", "epoch sec"),
        ("train_seconds", "train sec"),
        ("val_seconds", "val sec"),
    ]
    for key, label in time_series:
        values = series(rows, key)
        if has_values(values):
            speed_ax.plot(x, values, marker="o", ms=3, linewidth=2, label=label)
    speed_ax.set_ylabel("seconds")
    speed_ax.set_title("Runtime / Throughput")

    rate_ax = speed_ax.twinx()
    rate_series = [
        ("train_samples_per_sec", "samples/s", 1.0),
        ("train_fps", "fps", 1.0),
        ("train_frames_per_sec", "Mframes/s", 1_000_000.0),
    ]
    for key, label, scale in rate_series:
        values = series(rows, key)
        if has_values(values):
            rate_ax.plot(x, values / scale, linestyle="--", linewidth=2, label=label)
    rate_ax.set_ylabel("throughput")

    handles, labels = speed_ax.get_legend_handles_labels()
    handles2, labels2 = rate_ax.get_legend_handles_labels()
    if handles or handles2:
        speed_ax.legend(handles + handles2, labels + labels2, loc="best")

    axes[2].set_xlabel("epoch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.log is not None:
        log_path = args.log
        rows = load_epochs(log_path)
    else:
        log_path, rows = find_latest_parseable(args.root)

    out_path = args.out if args.out is not None else default_output_path(log_path)
    plot_epochs(
        rows=rows, source_path=log_path, out_path=out_path, dpi=args.dpi, show=args.show
    )

    print(f"log: {log_path}")
    print(f"epochs: {len(rows)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
