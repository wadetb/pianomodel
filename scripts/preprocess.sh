#!/usr/bin/env bash
uv run python preprocess.py \
  --data_root ./maestro-v3.0.0 \
  --preproc_root ./maestro-preprocessed


# --limit_per_split 10 \
# --overwrite
