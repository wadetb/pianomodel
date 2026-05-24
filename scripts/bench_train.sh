#!/usr/bin/env bash
# Run a training command while sampling GPU utilization, then print a summary.
# Usage:
#   scripts/bench_train.sh <label> <stats_json_path> -- <train args...>
# Example:
#   scripts/bench_train.sh baseline /tmp/bench/baseline.json -- \
#     --epochs 3 --batch_size 40 --segment_frames 2048 ...

set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="$1"; shift
STATS_JSON="$1"; shift
[[ "$1" == "--" ]] || { echo "expected -- between stats path and train args" >&2; exit 2; }
shift

mkdir -p "$(dirname "$STATS_JSON")"
GPU_LOG="${STATS_JSON%.json}.gpu.csv"

# 4 Hz sampling of util + memory
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -lms 250 > "$GPU_LOG" &
GPU_PID=$!

start=$(date +%s.%N)
uv run python train.py \
  --data_root ./maestro-v3.0.0 --preproc_root ./maestro-preprocessed \
  --stats_json "$STATS_JSON" \
  "$@"
status=$?
end=$(date +%s.%N)

kill "$GPU_PID" 2>/dev/null || true
wait "$GPU_PID" 2>/dev/null || true

uv run python - "$LABEL" "$STATS_JSON" "$GPU_LOG" "$start" "$end" <<'PY'
import csv, json, statistics, sys
label, stats_path, gpu_log, t0, t1 = sys.argv[1:]
wall = float(t1) - float(t0)

with open(stats_path) as f:
    s = json.load(f)
epochs = s.get("epochs", [])

util, mem = [], []
with open(gpu_log) as f:
    for row in csv.reader(f):
        try:
            util.append(int(row[0])); mem.append(int(row[1]))
        except (ValueError, IndexError):
            pass

def pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(q * len(xs))))] if xs else 0

print()
print(f"=== bench: {label} ===")
print(f"wall:        {wall:.1f}s")
if epochs:
    fps = sum(e.get("train_frames_per_sec", 0) for e in epochs) / len(epochs)
    sps = sum(e.get("train_samples_per_sec", 0) for e in epochs) / len(epochs)
    print(f"throughput:  avg {sps:.0f} samples/s  {fps:.0f} frames/s")
    for e in epochs:
        print(f"  epoch {e['epoch']}: train_loss={e['train_loss']:.4f} train_f1={e['train_f1']:.3f} "
              f"val_loss={e['val_loss']:.4f} val_f1={e['val_f1']:.3f} "
              f"t={e['epoch_seconds']:.1f}s fps={e['train_frames_per_sec']:.0f}")
if util:
    print(f"gpu util:    mean={statistics.mean(util):.1f}%  p50={pct(util, 0.5)}%  p90={pct(util, 0.9)}%  max={max(util)}%  (n={len(util)})")
if mem:
    print(f"gpu memory:  max={max(mem)} MiB / {len(mem)} samples")
PY
