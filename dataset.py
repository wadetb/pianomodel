import csv
import os

import librosa
import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset

# MIDI event vocabulary
MIDI_EVENT_TYPES = [
    "note_on",  # Note start
    "note_off",  # Note end
    "velocity",  # Velocity value
    "time_shift",  # Time shift event
]

# For piano, notes range from 21 (A0) to 108 (C8)
NOTE_RANGE = range(21, 109)
VELOCITY_BINS = range(1, 128)  # MIDI velocity values
TIME_SHIFT_BINS = [1, 2, 5, 10, 20, 50, 100]  # ms steps

# Build vocabulary
VOCAB = []
for note in NOTE_RANGE:
    VOCAB.append(f"note_on_{note}")
    VOCAB.append(f"note_off_{note}")
for vel in VELOCITY_BINS:
    VOCAB.append(f"velocity_{vel}")
for ts in TIME_SHIFT_BINS:
    VOCAB.append(f"time_shift_{ts}")

TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

# Audio feature extraction
def audio_to_logmel(audio_path, sr=16000, n_mels=128, hop_length=512):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.T  # shape: (frames, n_mels)


def encode_midi_events(events):
    return [TOKEN_TO_ID[e] for e in events if e in TOKEN_TO_ID]


def decode_midi_events(token_ids):
    return [ID_TO_TOKEN[i] for i in token_ids if i in ID_TO_TOKEN]


# Stub for extracting MIDI events from MIDI file
def midi_file_to_events(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for note in midi.instruments[0].notes:
        events.append(f"note_on_{note.pitch}")
        events.append(f"velocity_{note.velocity}")
        events.append(f"note_off_{note.pitch}")
    return events


# Maestro dataset
CSV_PATH = "maestro-v3.0.0/maestro-v3.0.0.csv"
BASE_DIR = "maestro-v3.0.0"


class MaestroDataset(Dataset):
    def __init__(self, split=None, num_samples=10000, segment_ms=2000, sr=16000, cache_size=32, cache_replace_interval=100):
        self.entries = []
        self.segment_ms = segment_ms
        self.sr = sr
        self.n_mels = 128
        self.hop_length = 512
        self.cache_size = cache_size
        self.cache_replace_interval = cache_replace_interval
        self.num_samples = num_samples

        with open(CSV_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is None or row["split"] == split:
                    audio_path = os.path.join(BASE_DIR, row["audio_filename"])
                    midi_path = os.path.join(BASE_DIR, row["midi_filename"])
                    self.entries.append(
                        {
                            "audio_path": audio_path,
                            "midi_path": midi_path,
                            "composer": row["canonical_composer"],
                            "title": row["canonical_title"],
                            "year": row["year"],
                            "duration": float(row["duration"]),
                            "split": row["split"],
                        }
                    )

        self.cache = []
        self.cache_indices = []
        self._fill_cache()
        self.sample_count = 0

    def _process_entry(self, idx):
        entry = self.entries[idx]
        y, _ = librosa.load(entry["audio_path"], sr=self.sr)
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        log_S = librosa.power_to_db(S, ref=np.max)
        features = log_S.T  # (frames, n_mels)
        midi = pretty_midi.PrettyMIDI(entry["midi_path"])
        return {"features": features, "midi": midi, "entry": entry}

    def _fill_cache(self):
        self.cache = []
        self.cache_indices = np.random.choice(len(self.entries), self.cache_size, replace=False).tolist()
        for idx in self.cache_indices:
            self.cache.append(self._process_entry(idx))

    def _replace_cache_entry(self):
        replace_idx = np.random.randint(0, self.cache_size)
        new_idx = np.random.randint(0, len(self.entries))
        self.cache[replace_idx] = self._process_entry(new_idx)
        self.cache_indices[replace_idx] = new_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cache_idx = np.random.randint(0, self.cache_size)
        cache_item = self.cache[cache_idx]
        features = cache_item["features"]
        midi = cache_item["midi"]
        total_frames = features.shape[0]
        segment_frames = int(self.segment_ms * self.sr / 1000 // self.hop_length)
        segment_frames = max(1, segment_frames)
        if total_frames <= segment_frames:
            start = 0
        else:
            start = np.random.randint(0, total_frames - segment_frames)
        end = start + segment_frames

        features_segment = features[start:end]
        segment_start_s = start * self.hop_length / self.sr
        segment_end_s = end * self.hop_length / self.sr
        
        events = []
        for note in midi.instruments[0].notes:
            if segment_start_s <= note.start < segment_end_s:
                events.append(f"note_on_{note.pitch}")
                events.append(f"velocity_{note.velocity}")
            if segment_start_s <= note.end < segment_end_s:
                events.append(f"note_off_{note.pitch}")
        tokens = encode_midi_events(events)

        self.sample_count += 1
        if self.sample_count % self.cache_replace_interval == 0:
            self._replace_cache_entry()

        return torch.tensor(features_segment, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)
