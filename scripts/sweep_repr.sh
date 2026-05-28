#!/usr/bin/env bash
# Representation comparison sweep: vary input representation AND frequency
# aggregation (mean-pool vs flatten) to find the optimal MAESTRO baseline.
#
# All configs use per-piece normalization (representation-agnostic, fair) and
# NO augmentation (isolates the representation effect). Once a winner is chosen,
# retrain it with fixed-norm + augmentation for deployment.
#
# Resumable: skips preprocessing if the dir exists, skips a config if its eval
# json exists. Continues past per-config failures (e.g. OOM).

set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-128}
SEG=${SEG:-2048}
LR=${LR:-9.6e-4}
SAMPLES=${SAMPLES:-100000}
WORKERS=${WORKERS:-8}
EVAL_THRESHOLDS=${EVAL_THRESHOLDS:-"0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"}
EVAL_WORKERS=${EVAL_WORKERS:-32}
OUT_DIR=${OUT_DIR:-output/sweep_repr}
PRE_ROOT=${PRE_ROOT:-maestro-preproc-repr}

mkdir -p "$OUT_DIR"

# Unique preprocessing dirs: tag|repr|n_fft|n_mels|hop|sample_rate
PREPROCS=(
  "mel64_16k|mel|512|64|160|16000"
  "mel128_16k|mel|1024|128|160|16000"
  "mel229_16k|mel|2048|229|160|16000"
  "logfreq_16k|logfreq|2048|256|160|16000"
  "mel229_44k|mel|2048|229|441|44100"
)

# Training configs: name|preproc_tag|freq_pool
CONFIGS=(
  "mel64_mean|mel64_16k|mean"
  "mel64_flat|mel64_16k|flatten"
  "mel128_flat|mel128_16k|flatten"
  "mel229_flat|mel229_16k|flatten"
  "logfreq_flat|logfreq_16k|flatten"
  "mel229_mean|mel229_16k|mean"
  "mel229_flat_44k|mel229_44k|flatten"
)

# Process one representation at a time: preprocess -> run all its configs ->
# delete the preproc dir. This caps peak disk at one representation (~100 GB)
# instead of all five coexisting (~390 GB). Checkpoints + eval JSONs are kept.
# Set KEEP_PREPROC=1 to retain preproc dirs (needs ~390 GB free for the full run).
KEEP_PREPROC=${KEEP_PREPROC:-0}

for spec in "${PREPROCS[@]}"; do
  IFS='|' read -r tag repr nfft nmels hop sr <<< "$spec"
  pdir="$PRE_ROOT/$tag"

  # Which configs use this preproc, and are any of them not yet evaluated?
  pending=()
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name ptag fpool <<< "$cfg"
    [[ "$ptag" == "$tag" ]] || continue
    [[ -f "$OUT_DIR/${name}_teaching.json" ]] && continue
    pending+=("$cfg")
  done
  if [[ ${#pending[@]} -eq 0 ]]; then
    echo "[repr-sweep] $tag: all configs already evaluated, skipping"
    continue
  fi

  echo "[repr-sweep] === representation $tag ==="
  if [[ ! -d "$pdir/test" ]]; then
    echo "[repr-sweep] preprocessing $tag (repr=$repr n_fft=$nfft n_mels=$nmels hop=$hop sr=$sr)"
    uv run python preprocess.py \
      --data_root ./maestro-v3.0.0 \
      --preproc_root "$pdir" \
      --splits train,validation,test \
      --normalize per_piece \
      --repr "$repr" --n_fft "$nfft" --n_mels "$nmels" --hop "$hop" --sample_rate "$sr" \
      || { echo "[repr-sweep] PREPROC FAILED: $tag, skipping its configs"; continue; }
  else
    echo "[repr-sweep] preproc $tag exists, reusing"
  fi

  for cfg in "${pending[@]}"; do
    IFS='|' read -r name ptag fpool <<< "$cfg"
    ckpt="$OUT_DIR/${name}.pt"
    eval_json="$OUT_DIR/${name}_eval.json"
    teach_json="$OUT_DIR/${name}_teaching.json"

    if [[ ! -f "$ckpt" ]]; then
      echo "[repr-sweep] training $name (preproc=$ptag freq_pool=$fpool)"
      uv run python train.py \
        --data_root ./maestro-v3.0.0 \
        --preproc_root "$pdir" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH" --segment_frames "$SEG" \
        --lr "$LR" \
        --num_workers "$WORKERS" --prefetch_factor 8 --persistent_workers \
        --seed 42 \
        --fused_adamw --allow_tf32 --matmul_precision high \
        --random_item_sampling --train_samples_per_epoch "$SAMPLES" \
        --freq_pool "$fpool" \
        --tqdm \
        --stats_json "$OUT_DIR/${name}.json" \
        --out "$ckpt" \
        || { echo "[repr-sweep] TRAIN FAILED: $name"; continue; }
    fi

    echo "[repr-sweep] caching predictions for $name"
    uv run python cache_predictions.py \
      --data_root ./maestro-v3.0.0 \
      --preproc_root "$pdir" \
      --model "$ckpt" \
      --split test \
      || { echo "[repr-sweep] CACHE FAILED: $name"; continue; }

    echo "[repr-sweep] note-level eval for $name"
    uv run python evaluate.py \
      --data_root ./maestro-v3.0.0 \
      --preproc_root "$pdir" \
      --model "$ckpt" \
      --split test \
      --threshold "$EVAL_THRESHOLDS" \
      --use_cache --workers "$EVAL_WORKERS" \
      --out "$eval_json" \
      || echo "[repr-sweep] NOTE-EVAL FAILED: $name"

    echo "[repr-sweep] teaching-app eval for $name"
    uv run python evaluate_teaching.py \
      --data_root ./maestro-v3.0.0 \
      --preproc_root "$pdir" \
      --model "$ckpt" \
      --split test --workers "$EVAL_WORKERS" \
      --out "$teach_json" \
      || echo "[repr-sweep] TEACH-EVAL FAILED: $name"
  done

  if [[ "$KEEP_PREPROC" != "1" ]]; then
    echo "[repr-sweep] deleting preproc $tag to free disk"
    rm -rf "$pdir"
  fi
done

echo "[repr-sweep] === summary ==="
uv run python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
import torch

out_dir = Path(sys.argv[1])
rows = []
for ev in sorted(out_dir.glob("*_eval.json")):
    name = ev.stem.removesuffix("_eval")
    with open(ev) as f:
        s = json.load(f)["summary"]
    best_f = s["macro_f"]
    best_thr = float(s["best_threshold"])
    best_p = s["macro_p"]
    best_r = s["macro_r"]
    ckpt = out_dir / f"{name}.pt"
    n_params = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu")
        n_params = sum(t.numel() for t in st.values() if hasattr(t, "numel"))
    teach = out_dir / f"{name}_teaching.json"
    t_recall = t_cands = t_rank = float("nan")
    if teach.exists():
        with open(teach) as f:
            ts = json.load(f)["summary"]
        if "0.3" in ts:
            t_recall = ts["0.3"]["onset_recall"]
            t_cands = ts["0.3"]["mean_candidates"]
            t_rank = ts["0.3"]["mean_rank"]
    rows.append((best_f, name, best_thr, best_p, best_r,
                 n_params, t_recall, t_cands, t_rank))

rows.sort(reverse=True)
print(f"{'name':<18} {'F':>5} {'thr':>4} {'P':>5} {'R':>5} {'params':>9} "
      f"{'t_rec':>5} {'t_cnd':>5} {'t_rnk':>5}")
print("-" * 72)
for f, name, thr, p, r, npar, tr, tc, trk in rows:
    print(f"{name:<18} {f:>5.3f} {thr:>4.2f} {p:>5.3f} {r:>5.3f} {npar:>9,} "
          f"{tr:>5.3f} {tc:>5.1f} {trk:>5.2f}")
print()
print("t_rec/t_cnd/t_rnk = teaching-app recall / mean candidates / mean rank at threshold 0.30")
PY
