#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required but was not found in PATH" >&2
  exit 1
fi

if [[ ! -d maestro-v3.0.0 ]]; then
  echo "error: data root not found: maestro-v3.0.0" >&2
  exit 1
fi
if [[ ! -d maestro-preprocessed ]]; then
  echo "error: preprocessed root not found: maestro-preprocessed" >&2
  exit 1
fi

rm -rf output/smoke
mkdir -p output/smoke/preproc

echo "[smoke] run dir: output/smoke"
echo "[smoke] train.py (1 epoch, 1 train batch, 1 val batch)"
uv run train.py \
  --data_root maestro-v3.0.0 \
  --preproc_root maestro-preprocessed \
  --epochs 1 \
  --batch_size 2 \
  --segment_frames 128 \
  --limit_train_items 2 \
  --limit_val_items 2 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --num_workers 0 \
  --out output/smoke/smoke_train.pt

echo "[smoke] preprocess.py (1 validation item)"
uv run preprocess.py \
  --data_root maestro-v3.0.0 \
  --preproc_root output/smoke/preproc \
  --splits validation \
  --limit_per_split 1

echo "[smoke] generating 1s silent wav for stream smoke test"
uv run python - <<'PY'
import torch
import torchaudio

wav = torch.zeros(1, 16_000, dtype=torch.float32)
torchaudio.save("output/smoke/smoke_stream_1s.wav", wav, 16_000)
print("wrote output/smoke/smoke_stream_1s.wav")
PY

echo "[smoke] stream.py (cpu, short window)"
uv run stream.py \
  --wav output/smoke/smoke_stream_1s.wav \
  --model output/smoke/smoke_train.pt \
  --device cpu \
  --window_frames 16

echo "[smoke] complete"
echo "[smoke] artifacts:"
echo "  train checkpoint: output/smoke/smoke_train.pt"
echo "  stream wav:       output/smoke/smoke_stream_1s.wav"
echo "  preproc root:     output/smoke/preproc"
