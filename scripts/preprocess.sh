#!/usr/bin/env bash
uv run python preprocess.py \
  --data_root ./maestro-v3.0.0 \
  --preproc_root ./maestro-preprocessed \
  --splits train,validation,test


# --limit_per_split 10 \
# --overwrite
# --viz             # also emit paginated mel+pianoroll PNGs (serial; for parallel run viz.py --workers N)
