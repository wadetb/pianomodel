#!/usr/bin/env bash
# Production-budget retrain of small candidates from the architecture sweep.
# Mirrors scripts/train.sh hyperparams so results are directly comparable to
# the existing baseline (output/train.pt -> F=0.624 from eval_test_threshold_sweep.json).
#
# Override knobs via env, e.g. EPOCHS=60 SAMPLES=80000 bash scripts/sweep_long.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-128}
SEG=${SEG:-2048}
WORKERS=${WORKERS:-8}
SAMPLES=${SAMPLES:-100000}
LR=${LR:-9.6e-4}
OUT_DIR=${OUT_DIR:-output/sweep_long}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_THRESHOLDS=${EVAL_THRESHOLDS:-"0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"}

mkdir -p "$OUT_DIR"

CONFIGS=(
  "h96_c32-64-96    96 32,64,96"
  "h64_c32-64-96    64 32,64,96"
  "h128_c16-32-48  128 16,32,48"
)

for cfg in "${CONFIGS[@]}"; do
  read -r name hidden channels <<< "$cfg"
  ckpt="$OUT_DIR/${name}.pt"
  train_json="$OUT_DIR/${name}.json"
  eval_json="$OUT_DIR/${name}_eval.json"

  if [[ -f "$eval_json" ]]; then
    echo "[long] $name: eval exists, skipping"
    continue
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[long] training $name (hidden=$hidden channels=$channels epochs=$EPOCHS samples=$SAMPLES)"
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
      --tqdm \
      --out "$ckpt" \
      --stats_json "$train_json"
  else
    echo "[long] $name: checkpoint exists, skipping training"
  fi

  echo "[long] evaluating $name on $EVAL_SPLIT (thresholds=$EVAL_THRESHOLDS)"
  uv run python evaluate.py \
    --data_root ./maestro-v3.0.0 \
    --preproc_root ./maestro-preprocessed \
    --model "$ckpt" \
    --split "$EVAL_SPLIT" \
    --threshold "$EVAL_THRESHOLDS" \
    --out "$eval_json"
done

echo
echo "[long] summary (sorted by best macro F1, includes baseline for reference):"
uv run python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
rows = []

baseline_eval = Path("output/eval_test_threshold_sweep.json")
if baseline_eval.exists():
    with baseline_eval.open() as f:
        s = json.load(f)["summary"]
    state = torch.load(s["model"], map_location="cpu")
    n_params = sum(t.numel() for t in state.values())
    rows.append((s["macro_f"], "baseline (h192_c32-64-96)", s["model_hidden"],
                 tuple(s["model_conv_channels"]), s["macro_p"], s["macro_r"],
                 s["best_threshold"], n_params))

for ev_path in sorted(root.glob("*_eval.json")):
    with ev_path.open() as f:
        s = json.load(f)["summary"]
    name = ev_path.stem.removesuffix("_eval")
    state = torch.load(Path(f"{root}/{name}.pt"), map_location="cpu")
    n_params = sum(t.numel() for t in state.values())
    rows.append((s["macro_f"], name, s["model_hidden"],
                 tuple(s["model_conv_channels"]), s["macro_p"], s["macro_r"],
                 s["best_threshold"], n_params))

rows.sort(reverse=True)
print(f"{'name':<27} {'hidden':>6} {'channels':>14} {'F':>5} {'P':>5} {'R':>5} {'t':>4} {'params':>8}")
print("-" * 85)
for f, name, h, ch, p, r, t, np_ in rows:
    print(f"{name:<27} {h:>6} {str(ch):>14} {f:>5.3f} {p:>5.3f} {r:>5.3f} {t:>4.2f} {np_:>8,}")
PY
