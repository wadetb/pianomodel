#!/usr/bin/env bash
# Tuned for RTX 4090: batch_size=128 saturates GPU (p50 100%, mean ~80%);
# lr=9.6e-4 = 3e-4 * (128/40) preserves convergence math under linear-scaling rule.
# torch.compile and --preproc_cache ram tested, both made things slower at this model size.
uv run python train.py \
  --data_root ./maestro-v3.0.0 \
  --preproc_root ./maestro-preprocessed \
  --preproc_cache mmap \
  --epochs 100 \
  --batch_size 128 --segment_frames 2048 \
  --lr 9.6e-4 \
  --num_workers 8 --prefetch_factor 8 --persistent_workers \
  --seed 42 \
  --fused_adamw --allow_tf32 --matmul_precision high \
  --random_item_sampling --train_samples_per_epoch 100000 \
  --tqdm \
  --stats_json ./output/train.json \
  --out ./output/train.pt

  # --epochs 120 \

  #  --torch_compile default --torch_compile_backend inductor
  #  --torch_compile reduce-overhead --torch_compile_backend inductor
  #  --torch_compile max-autotune --torch_compile_backend inductor
  #  --torch_compile max-autotune-no-cudagraphs --torch_compile_backend inductor

  # --preproc_cache ram
