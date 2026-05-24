#!/usr/bin/env bash
# Ablation sweep: train each (hidden, conv_channels) variant, then run mir_eval.
# Override knobs via env: EPOCHS=20 BATCH=40 SEG=2048 WORKERS=8 SAMPLES=50000

set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-40}
SEG=${SEG:-2048}
WORKERS=${WORKERS:-8}
SAMPLES=${SAMPLES:-50000}
OUT_DIR=${OUT_DIR:-output/sweep}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_THRESHOLDS=${EVAL_THRESHOLDS:-"0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"}

mkdir -p "$OUT_DIR"

# (label hidden conv_channels)
CONFIGS=(
  "h192_c32-64-96 192 32,64,96"
  "h128_c32-64-96 128 32,64,96"
  "h96_c32-64-96   96 32,64,96"
  "h64_c32-64-96   64 32,64,96"
  "h128_c16-32-48 128 16,32,48"
  "h96_c16-32-48   96 16,32,48"
  "h64_c16-32-48   64 16,32,48"
  "h64_c8-16-24    64 8,16,24"
  "h32_c8-16-24    32 8,16,24"
)

for cfg in "${CONFIGS[@]}"; do
  read -r name hidden channels <<< "$cfg"
  ckpt="$OUT_DIR/${name}.pt"
  train_json="$OUT_DIR/${name}.json"
  eval_json="$OUT_DIR/${name}_eval.json"

  if [[ -f "$eval_json" ]]; then
    echo "[sweep] $name: eval exists, skipping"
    continue
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[sweep] training $name (hidden=$hidden channels=$channels epochs=$EPOCHS samples=$SAMPLES)"
    uv run python train.py \
      --data_root ./maestro-v3.0.0 \
      --preproc_root ./maestro-preprocessed \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH" --segment_frames "$SEG" \
      --num_workers "$WORKERS" --prefetch_factor 8 --persistent_workers \
      --fused_adamw --allow_tf32 --matmul_precision high \
      --random_item_sampling --train_samples_per_epoch "$SAMPLES" \
      --hidden "$hidden" \
      --conv_channels "$channels" \
      --tqdm \
      --out "$ckpt" \
      --stats_json "$train_json"
  else
    echo "[sweep] $name: checkpoint exists, skipping training"
  fi

  echo "[sweep] evaluating $name on $EVAL_SPLIT (thresholds=$EVAL_THRESHOLDS)"
  uv run python evaluate.py \
    --data_root ./maestro-v3.0.0 \
    --preproc_root ./maestro-preprocessed \
    --model "$ckpt" \
    --split "$EVAL_SPLIT" \
    --threshold "$EVAL_THRESHOLDS" \
    --out "$eval_json"
done

echo
echo "[sweep] summary (sorted by macro F1):"
uv run python - "$OUT_DIR" <<'PY'
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for ev_path in sorted(root.glob("*_eval.json")):
    with ev_path.open() as f:
        s = json.load(f)["summary"]
    name = ev_path.stem.removesuffix("_eval")
    best_t = s.get("best_threshold", s.get("config", {}).get("threshold"))
    rows.append((s["macro_f"], s["macro_p"], s["macro_r"], name,
                 s["model_hidden"], tuple(s["model_conv_channels"]), best_t))
rows.sort(reverse=True)
print(f"{'name':<22} {'hidden':>6} {'channels':>14} {'F':>6} {'P':>6} {'R':>6} {'thresh':>7}")
for f, p, r, name, h, ch, t in rows:
    t_str = f"{t:.2f}" if isinstance(t, (int, float)) else str(t)
    print(f"{name:<22} {h:>6} {str(ch):>14} {f:>6.3f} {p:>6.3f} {r:>6.3f} {t_str:>7}")
PY
