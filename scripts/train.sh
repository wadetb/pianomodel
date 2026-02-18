#!/usr/bin/env bash
uv run python train.py \
  --data_root ./maestro-v3.0.0 \
  --preproc_root ./maestro-preprocessed \
  --preproc_cache mmap \
  --epochs 100 \
  --batch_size 40 --segment_frames 2048 \
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
