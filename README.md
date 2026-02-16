# Piano Onset CRNN

This repo focuses on a compact, piano-only, onset-only CRNN in PyTorch. It trains on MAESTRO audio/MIDI pairs and supports optional preprocessing and a streaming inference simulation.

## Features
- MAESTRO dataset support (v3.x)
- Log-mel features via `torchaudio`
- CRNN with separable convs + GRU
- Optional preprocessing to NPY/NPZ for faster training
- Streaming inference simulation over WAV files

## Setup
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Download the MAESTRO dataset and note the root directory that contains the CSV and audio/midi files.

## Training
```bash
python piano_onset_crnn.py --data_root /path/to/maestro-v3.0.0 --epochs 50 --batch_size 16 --segment_frames 512 --out ./onset_crnn.pt
```

## Preprocessing (Optional)
```bash
python piano_onset_crnn.py --data_root /path/to/maestro-v3.0.0 --preprocess --preprocess_splits train,validation --preprocess_format npy
```

## Streaming (Simulated)
```bash
python piano_onset_crnn.py --stream_wav input.wav --model ./onset_crnn.pt
```

## Notes
- This model predicts onset frames only (no sustain/offset/velocity).
- The streaming mode uses a hysteresis threshold to reduce spurious triggers.
