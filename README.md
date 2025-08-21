# Piano Transformer MIDI Transcription

This project implements a simple transformer-based machine learning model in PyTorch to transcribe piano audio recordings into MIDI-like events. The model is inspired by Google's MT3 architecture, but is designed for real-time, piano-specific note onset detection and MIDI event emission.

## Features
- Uses the Maestro dataset for paired audio/MIDI training data
- PyTorch transformer model for sequence-to-sequence transcription
- Log-mel spectrogram audio features
- MIDI event vocabulary for piano (note on/off, velocity, time shift)

## Usage
1. Download the Maestro dataset and place it in `maestro-v3.0.0/`.
2. Install dependencies:
	```bash
	uv sync
	```
3. Run training:
	```bash
	python train.py
	```

## Development State
- Model, data pipeline, and training loop are implemented
- MIDI event extraction is a stub and should be improved for full event coverage
- No evaluation or inference scripts yet
- No automatic checkpointing or hyperparameter management

## Next Steps
- Improve MIDI event extraction to handle time shifts and full event vocabulary
- Add evaluation and inference scripts for real-time transcription
- Experiment with model architecture and training hyperparameters
- Integrate with a music app for live piano transcription

## References
- [Google Magenta MT3](https://magenta.tensorflow.org/mt3)
- [Maestro Dataset](https://magenta.tensorflow.org/datasets/maestro)
