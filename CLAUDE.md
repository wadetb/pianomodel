# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Piano onset detection on MAESTRO v3.0.0. A small CRNN predicts per-frame onset probabilities for the 88 piano keys from log-mel spectrograms, then `stream.py` simulates real-time decoding from a WAV.

## Toolchain

- Python 3.10, dependencies pinned in `pyproject.toml` / `uv.lock`. Always invoke via `uv run` (e.g. `uv run python train.py …`); do not assume a global venv.
- Run scripts from the repo root — `train.sh`, `preprocess.sh`, etc. all use paths relative to CWD (`./maestro-v3.0.0`, `./maestro-preprocessed`, `./output`).

## Common commands

- `bash scripts/preprocess.sh` — one-time preprocessing of MAESTRO into `maestro-preprocessed/{split}/{piece}/{mel,labels}.npy`. Must complete before training. Pass `--viz` to also emit paginated mel+pianoroll PNGs (`viz_NNNNs.png`, 30s/page) next to each piece.
- `uv run python viz.py --data_root ./maestro-v3.0.0 --preproc_root ./maestro-preprocessed` — render the same PNGs over already-preprocessed data, no re-preprocessing needed. Pages are skipped if they exist (use `--overwrite` to force).
- `uv run python predict.py --wav <path>.wav [--start S --duration D]` — render mel + model-probability piano-roll for an arbitrary WAV slice. Uses `output/train.pt` by default; auto-detects `hidden`/`conv_channels` from the checkpoint via `OnsetCRNN.from_checkpoint`.
- `uv run python evaluate.py --data_root ./maestro-v3.0.0 --preproc_root ./maestro-preprocessed --model <ckpt.pt> --split test --threshold "0.4,0.45,...,0.8" --out <eval.json>` — note-level mir_eval F1 over a split with peak-picked decoding (50 ms onset tol, 50-cent pitch tol). Macro-averaged P/R/F. Multi-threshold sweep does one forward pass per piece and peak-picks at each threshold. This is the headline metric, not the frame F1 the trainer prints.
- `uv run python cache_predictions.py --data_root ... --model <ckpt.pt> --split test` — one-time per-model: writes `output/probs_cache/<model_stem>/<piece>.npy` (float16) for every piece. ~60s for the test split.
- `uv run python evaluate.py ... --use_cache --workers 32` — eval reads the cache instead of running the model, then parallelizes the per-piece mir_eval matching across N workers. ~3 min for full test split with 9-threshold sweep (vs 4-7 min uncached, 22 min cached-but-serial). For decoder iteration (new threshold / refractory / peak-pick algo on the same model), this is the fast path — no GPU needed.
- `uv run python cache_predictions.py --chunk_frames 16 ...` + `evaluate.py --use_cache --cache_chunk_frames 16 ...` — validate the *streaming-deployment* F1: model runs in 16-frame chunks (160 ms) with conv-context overlap, GRU state carries between chunks. Matches canonical F=0.624 within 0.001.
- `uv run python stream.py --wav <path>` — end-to-end streaming demo from a WAV file. Mel extraction → conv-context streaming detector → ASCII pianoroll. Useful for testing without a mic.
- `uv run python live.py` — live mic capture → same streaming pipeline → ASCII pianoroll. Needs `libportaudio2` on the local machine; doesn't work over plain SSH.
- `bash scripts/sweep.sh` — train + evaluate a grid of `(hidden, conv_channels)` variants for ablation. Outputs go to `output/sweep/`. Override knobs via env (`EPOCHS=20 BATCH=40 SAMPLES=50000 bash scripts/sweep.sh`).
- `bash scripts/train.sh` — full training run; writes checkpoint to `output/train.pt` and stats to `output/train.json`.
- `bash scripts/smoke.sh` — fast end-to-end sanity check (1 epoch, 2 items, 1 batch each, then a 1s-silence stream test). Run this after any non-trivial change to `train.py`, `preprocess.py`, `stream.py`, `model.py`, or `maestro.py`.
- `uv run python graph.py` — auto-discovers the newest parseable log under `output/` (json or text) and writes `<stem>_graph.png` next to it; pass `--log` to target a specific run.
- `uv run python stream.py --wav <path> --model output/train.pt` — streaming inference with hidden-state continuity and on/off thresholds.

There is no test suite or linter beyond `ruff` being a dependency.

## Architecture

**Two-stage pipeline.** Audio decoding + mel computation are slow, so they're done once offline and the trainer only consumes `.npy` arrays:

1. `preprocess.py` walks the MAESTRO CSV, runs `MelSpecProcessor` on each WAV, and writes `mel.npy` (`[T, N_MELS]`, log1p+normalized) plus `labels.npy` (`[T, 88]`, onset frames widened by ±1) per piece.
2. `PreprocessedMaestroDataset` (in `maestro.py`) loads those `.npy` files, slicing a random `segment_frames` window per `__getitem__`. `--preproc_cache mmap` (default) maps from disk; `--preproc_cache ram` preloads everything (only viable for small subsets — full MAESTRO is ~100s of GiB raw audio).

**Audio constants live in `maestro.py`** and are imported everywhere: `SR=16000`, `HOP=160` (10 ms), `N_FFT=512`, `N_MELS=64`, `N_KEYS=88`, `MIDI_LOW=21`. Changing any of these invalidates the cached `.npy` files — re-run `preprocess.sh --overwrite`.

**Model (`model.py`).** `OnsetCRNN` = three `SeparableConv2d` blocks (depthwise + pointwise + BN + ReLU) that downsample only the mel axis (stride `(1,2)`) so the time axis is preserved frame-for-frame, → mean over remaining freq axis → 1-layer GRU → linear head to 88 logits. `fuse_for_eval()` folds BN into the pointwise conv for inference; `--fuse_eval` in `train.py` writes a fused checkpoint after the run.

**Training loop (`train.py`).** BCE-with-logits using `--pos_weight` (default 5) to compensate for sparse onsets; `frame_counts` accumulates TP/FP/FN on-GPU as tensors so per-batch metrics don't sync. AMP autocast + `GradScaler` on CUDA. Cosine LR over `--epochs`. The best `val_f1` checkpoint is saved.

**Virtual epochs.** `--random_item_sampling` + `--train_samples_per_epoch N` decouple "epoch length" from dataset size: the dataset reports `len()=N` and each `__getitem__` picks a random piece + random window. The default `train.sh` uses 100k virtual samples per epoch. When this is on, the DataLoader uses `shuffle=False` because sampling is already random.

**Performance knobs** (all optional, surface in `train.py` and the commented section of `scripts/train.sh`): `--fused_adamw`, `--allow_tf32`, `--matmul_precision`, `--torch_compile {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}`. `maybe_compile_model` wraps the model; `unwrap_compiled_model` is used before `state_dict()` so checkpoints stay compile-agnostic.

**Streaming (`stream.py`).** Provides the runtime primitives used by both the live demo (`live.py`) and the streaming evaluator path (`cache_predictions.py --chunk_frames N`):

- `StreamingMelExtractor` — buffers raw audio, emits mel frames incrementally.
- `StreamingPeakPicker` — peak-picks across chunk boundaries with a 1-frame look-ahead delay; bit-identical to `decode.peak_pick` at any chunk size.
- `StreamingOnsetDetector` — wraps the model with GRU state + a 3-frame **conv-context overlap** (`OnsetCRNN.forward_with_context`). The overlap is required: without it, naive chunking drops F1 from 0.624 → 0.328 at chunk=16, because each chunk's convs see zero-padding on ~37% of frames. With the overlap, F1 is bit-equivalent to whole-piece inference. Costs `context_frames * 10 ms` (=30 ms) of right-context look-ahead per chunk.
- `chunked_inference()` — the offline counterpart used by `cache_predictions.py` to generate streaming-equivalent probabilities for evaluation.

Total deployment latency at chunk=16: 16 frames (chunk) + 3 frames (right context) = 190 ms input-audio-to-detected-event.

**Decoder (`decode.py`).** Shared peak-picker used by `predict.py` and `evaluate.py`: per-pitch local maxima above `threshold` with a refractory window (default 50 ms / 5 frames). Returns `(frame_idx, pitch_idx)` events; `events_to_onsets` converts to `(seconds, midi)`. Note-level eval uses this; the streaming code in `stream.py` uses different on/off hysteresis (which conflates onsets with sustained activations and is the wrong shape for evaluation).

**Model is shape-introspected from checkpoints.** `OnsetCRNN.from_checkpoint(path)` reads `conv1/conv2/conv3.pw.weight` and `gru.weight_ih_l0` shapes to recover `(conv_channels, hidden, classes)` so callers don't need to know how a checkpoint was trained. Use this in any new tool that loads a checkpoint.

**Logging / graphing.** `--stats_json` writes a structured per-epoch log; `graph.py` accepts both that JSON and free-form stdout logs (parsed via the `Epoch …` / `train: loss=… | val: …` regexes), so capturing `train.sh` output to a `.log` file is also plottable.

## Notes

- `constants.py` is empty — ignore.
- `.vscode/launch.json` references a stale `piano_onset_crnn.py` entry point; the real entry is `train.py`.
- The 100 GB `maestro-v3.0.0.zip` and the extracted dataset are gitignored; never add them.
