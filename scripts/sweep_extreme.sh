#!/usr/bin/env bash
# Production-budget retrain to probe the extreme small end of the architecture space.
# Pushes the GRU hidden size down and tests a no-recurrence baseline. Mirrors the
# train.sh hyperparameters so results are directly comparable to the baseline
# (output/train.pt -> F=0.624) and the sweep_long results.

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-128}
SEG=${SEG:-2048}
WORKERS=${WORKERS:-8}
SAMPLES=${SAMPLES:-100000}
LR=${LR:-9.6e-4}
OUT_DIR=${OUT_DIR:-output/sweep_extreme}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_THRESHOLDS=${EVAL_THRESHOLDS:-"0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"}
EVAL_WORKERS=${EVAL_WORKERS:-32}

mkdir -p "$OUT_DIR"

# (label hidden channels temporal_mode)
CONFIGS=(
  "h48_c32-64-96    48  32,64,96  gru"
  "h32_c32-64-96    32  32,64,96  gru"
  "h16_c32-64-96    16  32,64,96  gru"
  "nogru_c32-64-96   0  32,64,96  none"
)

for cfg in "${CONFIGS[@]}"; do
  read -r name hidden channels temporal <<< "$cfg"
  ckpt="$OUT_DIR/${name}.pt"
  train_json="$OUT_DIR/${name}.json"
  eval_json="$OUT_DIR/${name}_eval.json"

  if [[ -f "$eval_json" ]]; then
    echo "[extreme] $name: eval exists, skipping"
    continue
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[extreme] training $name (hidden=$hidden channels=$channels temporal=$temporal epochs=$EPOCHS samples=$SAMPLES)"
    uv run python train.py \
      --data_root ./maestro-v3.0.0 \
      --preproc_root ./maestro-preprocessed \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH" --segment_frames "$SEG" \
      --lr "$LR" \
      --num_workers "$WORKERS" --prefetch_factor 8 --persistent_workers \
      --fused_adamw --allow_tf32 --matmul_precision high \
      --random_item_sampling --train_samples_per_epoch "$SAMPLES" \
      --hidden "$hidden" \
      --conv_channels "$channels" \
      --temporal_mode "$temporal" \
      --tqdm \
      --out "$ckpt" \
      --stats_json "$train_json"
  else
    echo "[extreme] $name: checkpoint exists, skipping training"
  fi

  echo "[extreme] evaluating $name on $EVAL_SPLIT (thresholds=$EVAL_THRESHOLDS)"
  # Cache predictions first so the eval can use the parallel scoring path.
  uv run python cache_predictions.py \
    --data_root ./maestro-v3.0.0 \
    --preproc_root ./maestro-preprocessed \
    --model "$ckpt" \
    --split "$EVAL_SPLIT"
  uv run python evaluate.py \
    --data_root ./maestro-v3.0.0 \
    --preproc_root ./maestro-preprocessed \
    --model "$ckpt" \
    --split "$EVAL_SPLIT" \
    --threshold "$EVAL_THRESHOLDS" \
    --use_cache --workers "$EVAL_WORKERS" \
    --out "$eval_json"
done

echo
echo "[extreme] summary (sorted by best macro F1, includes prior sweep_long + baseline):"
uv run python - "$OUT_DIR" output/sweep_long output/eval_test_cached_parallel.json <<'PY'
import json, sys
from pathlib import Path
import torch

cur_dir, long_dir, baseline_path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []

def add(label, eval_path, ckpt_path):
    with open(eval_path) as f:
        s = json.load(f)["summary"]
    state = torch.load(ckpt_path, map_location="cpu")
    n_params = sum(t.numel() for t in state.values())
    rows.append((s["macro_f"], label, s["model_hidden"], tuple(s["model_conv_channels"]),
                 s["macro_p"], s["macro_r"], s["best_threshold"], n_params))

if Path(baseline_path).exists():
    add("baseline (h192_c32-64-96)", baseline_path, "output/train.pt")

for ev_path in sorted(Path(long_dir).glob("*_eval.json")):
    name = ev_path.stem.removesuffix("_eval")
    ckpt = Path(long_dir) / f"{name}.pt"
    if ckpt.exists():
        add(name, str(ev_path), str(ckpt))

for ev_path in sorted(Path(cur_dir).glob("*_eval.json")):
    name = ev_path.stem.removesuffix("_eval")
    ckpt = Path(cur_dir) / f"{name}.pt"
    if ckpt.exists():
        add(name, str(ev_path), str(ckpt))

rows.sort(reverse=True)
print(f"{'name':<27} {'hidden':>6} {'channels':>14} {'F':>5} {'P':>5} {'R':>5} {'t':>4} {'params':>8}")
print("-" * 85)
for f, name, h, ch, p, r, t, np_ in rows:
    print(f"{name:<27} {h:>6} {str(ch):>14} {f:>5.3f} {p:>5.3f} {r:>5.3f} {t:>4.2f} {np_:>8,}")
PY
