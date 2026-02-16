"""
Minimal PyTorch training + streaming inference for a small piano-only, onset-only model.

Assumptions
- Dataset: MAESTRO (v3.x). Provide --data_root pointing to the folder that contains
  the metadata CSV (e.g., maestro-v3.0.0/maestro-v3.0.0.csv) and the audio/midi files.
- Dependencies: torch, torchaudio, numpy, pretty_midi (for MIDI parsing).

Usage examples
- Train:  python piano_onset_crnn.py --data_root /path/to/maestro-v3.0.0 --epochs 50 \
          --batch_size 16 --segment_frames 512 --out ./onset_crnn.pt
- Stream (simulate from WAV):  python piano_onset_crnn.py --stream_wav input.wav --model ./onset_crnn.pt

Notes
- This is a compact, quantization-friendly CRNN intended for low-latency streaming inference.
- The streaming function here simulates real-time by iterating a WAV buffer with hop-sized steps.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio

# Suppress noisy third-party warnings that do not impact functionality
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec.*",
)

try:
    import pretty_midi  # type: ignore
except Exception as e:  # pragma: no cover
    pretty_midi = None


# ---------------------------
# Hyperparameters (defaults)
# ---------------------------
SR = 16000
N_FFT = 512
HOP = 160  # 10ms at 16kHz
N_MELS = 64
FMIN = 20.0
FMAX = None  # defaults to nyquist when None

N_KEYS = 88
MIDI_LOW = 21  # A0
MIDI_HIGH = 108  # C8 inclusive


# ---------------------------
# Feature extraction
# ---------------------------
class MelSpecProcessor(nn.Module):
    def __init__(
        self,
        sample_rate: int = SR,
        n_fft: int = N_FFT,
        hop: int = HOP,
        n_mels: int = N_MELS,
        f_min: float = FMIN,
        f_max: Optional[float] = FMAX,
    ):
        super().__init__()
        self.resample = None  # filled if input sr != sample_rate
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.amp_to_db = None  # optional; we use log1p below for stability
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop

    def set_input_sr(self, input_sr: int):
        if input_sr != self.sample_rate:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=input_sr, new_freq=self.sample_rate
            )
        else:
            self.resample = None

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, input_sr: int) -> torch.Tensor:
        """wav: [1, T] mono float32, returns [T_frames, n_mels] (time-major)."""
        if self.resample is None or getattr(self, "_resample_sr", None) != input_sr:
            self.set_input_sr(input_sr)
            self._resample_sr = input_sr
        if self.resample is not None:
            wav = self.resample(wav)
        # Compute mel spectrogram
        mel = self.melspec(wav)  # torchaudio returns shape:
                                 # - [n_mels, T] for 1D wav
                                 # - [C, n_mels, T] for 2D wav [C, T]
        if mel.dim() == 3 and mel.size(0) == 1:
            mel = mel.squeeze(0)  # -> [n_mels, T]
        if mel.dim() != 2:
            raise RuntimeError(f"Unexpected mel shape {tuple(mel.shape)}")
        mel = torch.log1p(mel)
        mel = mel.transpose(0, 1).contiguous()  # [T_frames, n_mels]
        # Normalize per-example for stability
        mean = mel.mean()
        std = mel.std().clamp_min(1e-5)
        mel = (mel - mean) / std
        return mel


# ---------------------------
# Model definition
# ---------------------------
class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: Tuple[int, int] = (3, 3),
        s: Tuple[int, int] = (1, 1),
        p: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class OnsetCRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden: int = 192,
        classes: int = N_KEYS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = SeparableConv2d(1, 32, (3, 3), (1, 2), (1, 1))
        self.conv2 = SeparableConv2d(32, 64, (3, 3), (1, 2), (1, 1))
        self.conv3 = SeparableConv2d(64, 96, (3, 3), (1, 2), (1, 1))
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=96, hidden_size=hidden, num_layers=1, batch_first=True
        )
        self.head = nn.Linear(hidden, classes)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, F] -> logits [B, T, 88]."""
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = x.mean(dim=-1).transpose(1, 2)  # [B, C, T, F'] -> [B, T, C]
        y, h = self.gru(x, h)  # [B, T, H]
        logits = self.head(y)
        return logits, h


# ---------------------------
# MIDI to onset labels
# ---------------------------
def midi_onset_frames(
    midi_path: str, total_frames: int, sr: int = SR, hop: int = HOP, widen: int = 1
) -> np.ndarray:
    """Return [T, 88] binary onset labels for a MIDI file.

    Args:
      midi_path: path to MIDI
      total_frames: total number of spectrogram frames in the paired audio
      sr: sample rate used to compute frames
      hop: hop length in samples
      widen: mark +-widen frames around an onset as positive
    """
    if pretty_midi is None:
        raise RuntimeError(
            "pretty_midi not installed. Install with: pip install pretty_midi"
        )
    y = np.zeros((total_frames, N_KEYS), dtype=np.float32)
    pm = pretty_midi.PrettyMIDI(midi_path)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pitch = note.pitch
            if pitch < MIDI_LOW or pitch > MIDI_HIGH:
                continue
            idx = pitch - MIDI_LOW
            t = note.start
            f = int(round(t * sr / hop))
            if 0 <= f < total_frames:
                lo = max(0, f - widen)
                hi = min(total_frames, f + widen + 1)
                y[lo:hi, idx] = 1.0
    return y


# ---------------------------
# Dataset
# ---------------------------
def find_maestro_csv(root: str) -> str:
    # Find the MAESTRO metadata CSV in the root directory
    for fn in os.listdir(root):
        if fn.endswith(".csv") and "maestro" in fn.lower():
            return os.path.join(root, fn)
    # Also check one-level-deeper folder
    for sub in os.listdir(root):
        p = os.path.join(root, sub)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.endswith(".csv") and "maestro" in fn.lower():
                    return os.path.join(p, fn)
    raise FileNotFoundError("Could not locate MAESTRO CSV under --data_root")


@dataclass
class MaestroItem:
    audio_path: str
    midi_path: str
    split: str


class MaestroOnsetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        processor: MelSpecProcessor,
        segment_frames: int = 512,
        augment: bool = False,
        limit_items: Optional[int] = None,
        preproc_root: Optional[str] = None,
        use_preproc: bool = False,
    ):
        super().__init__()
        self.root = root
        self.processor = processor
        self.segment_frames = segment_frames
        self.preproc_root = preproc_root
        self.use_preproc = use_preproc
        self.augment = augment and not use_preproc
        csv_path = find_maestro_csv(root)
        items: List[MaestroItem] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row.get("split", "").strip().lower()
                if s != split:
                    continue
                audio_rel = (
                    row.get("audio_filename")
                    or row.get("audio_path")
                    or row.get("audio")
                )
                midi_rel = (
                    row.get("midi_filename") or row.get("midi_path") or row.get("midi")
                )
                if not audio_rel or not midi_rel:
                    continue
                audio_path = os.path.join(os.path.dirname(csv_path), audio_rel)
                midi_path = os.path.join(os.path.dirname(csv_path), midi_rel)
                items.append(
                    MaestroItem(audio_path=audio_path, midi_path=midi_path, split=s)
                )
        # If using preprocessed data, keep only items with existing files
        if use_preproc and preproc_root is not None:
            filtered: List[MaestroItem] = []
            for it in items:
                if preproc_item_exists(it, preproc_root):
                    filtered.append(it)
            items = filtered
        # Optionally limit items for faster iterations/smoke tests
        if limit_items is not None and limit_items > 0:
            items = items[: int(limit_items)]
        if not items:
            raise RuntimeError(
                f"No items found for split={split}. Check your --data_root"
            )
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav, sr

    def _maybe_augment(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if not self.augment:
            return wav
        # Simple gain and noise augmentations
        if random.random() < 0.5:
            gain_db = random.uniform(-6.0, 6.0)
            wav = wav * (10.0 ** (gain_db / 20.0))
        if random.random() < 0.3:
            noise = torch.randn_like(wav) * 1e-4
            wav = wav + noise
        # Small time shift (<20ms)
        if random.random() < 0.3:
            max_shift = int(0.02 * sr)
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                wav = torch.cat(
                    [wav[:, shift:], torch.zeros_like(wav[:, :shift])], dim=1
                )
            elif shift < 0:
                s = -shift
                wav = torch.cat([torch.zeros_like(wav[:, :s]), wav[:, :-s]], dim=1)
        return wav

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        # Load from preprocessed storage if configured
        if self.use_preproc and self.preproc_root is not None:
            mel_path, labels_path, npz_path = preproc_paths(self.preproc_root, item)
            if mel_path and labels_path and os.path.exists(mel_path) and os.path.exists(labels_path):
                # Fast path: memory-map npy and slice before tensor conversion
                mel_mm = np.load(mel_path, mmap_mode="r")
                labels_mm = np.load(labels_path, mmap_mode="r")
                T = int(mel_mm.shape[0])
                # Choose segment indices (defer actual slicing below)
                seg_start = 0
                if T >= self.segment_frames:
                    seg_start = random.randint(0, T - self.segment_frames)
                    seg_end = seg_start + self.segment_frames
                    mel = torch.from_numpy(np.array(mel_mm[seg_start:seg_end], dtype=np.float32))
                    y_full = np.array(labels_mm[seg_start:seg_end], dtype=np.float32)
                else:
                    mel_arr = np.array(mel_mm, dtype=np.float32)
                    y_arr = np.array(labels_mm, dtype=np.float32)
                    pad = self.segment_frames - T
                    mel = torch.from_numpy(np.pad(mel_arr, ((0, pad), (0, 0)), mode="constant"))
                    y_full = np.pad(y_arr, ((0, pad), (0, 0)), mode="constant")
            elif npz_path and os.path.exists(npz_path):
                data = np.load(npz_path)
                mel = torch.from_numpy(data["mel"]).to(torch.float32)
                y_full = data["labels"].astype(np.float32)
                T = mel.shape[0]
            else:
                raise RuntimeError(
                    f"Preprocessed files not found for item. Looked for NPY and NPZ under {self.preproc_root}"
                )
        else:
            wav, sr = self._load_audio(item.audio_path)
            wav = self._maybe_augment(wav, sr)
            with torch.no_grad():
                mel = self.processor(wav, sr)  # [T, n_mels]
            if mel.dim() != 2:
                raise RuntimeError(f"Expected 2D mel, got {tuple(mel.shape)}")
            T = mel.shape[0]
            y_full = midi_onset_frames(
                item.midi_path,
                total_frames=T,
                sr=self.processor.sample_rate,
                hop=self.processor.hop,
                widen=1,
            )  # np.ndarray [T, 88]

        if self.use_preproc and self.preproc_root is not None and (isinstance(y_full, np.ndarray)):
            # y_full already sliced/padded in the fast path
            mel_seg = mel.contiguous()
            y_seg = torch.from_numpy(y_full).to(torch.float32)
        else:
            if T >= self.segment_frames:
                start = random.randint(0, T - self.segment_frames)
                end = start + self.segment_frames
                mel_seg = mel[start:end].contiguous()
                y_seg = torch.tensor(y_full[start:end], dtype=torch.float32)
            else:
                pad = self.segment_frames - T
                mel_seg = F.pad(mel, (0, 0, 0, pad)).contiguous()  # [segment_frames, n_mels]
                y_t = torch.tensor(y_full, dtype=torch.float32)  # [T, 88]
                y_seg = F.pad(y_t, (0, 0, 0, pad))  # [segment_frames, 88]

        # ensure standalone, contiguous copies
        mel_seg = mel_seg.to(torch.float32).contiguous().clone()
        y_seg = y_seg.to(torch.float32).contiguous().clone()
        return mel_seg, y_seg


# ---------------------------
# Training / Evaluation
# ---------------------------
def frame_metrics(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Simple framewise precision/recall/F1.
    pred/target: [B, T, 88] (logits for pred).
    """
    with torch.no_grad():
        probs = torch.sigmoid(pred)
        yhat = (probs >= threshold).to(target.dtype)
        y = target
        tp = (yhat * y).sum().item()
        fp = (yhat * (1 - y)).sum().item()
        fn = ((1 - yhat) * y).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        return prec, rec, f1


def train_one_epoch(
    model: OnsetCRNN,
    loader: DataLoader,
    optimizer,
    device,
    scaler,
    pos_weight: float = 5.0,
    log_interval: int = 50,
    max_batches: Optional[int] = None,
    profile_batches: Optional[int] = None,
    profile_out: Optional[str] = None,
    epoch_idx: Optional[int] = None,
) -> Tuple[float, float]:
    import cProfile, pstats

    model.train()
    total_loss = 0.0
    total_f1 = 0.0
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    prof = None
    for it, (x, y) in enumerate(loader):
        # Start profiler on first batch if requested
        if prof is None and profile_batches and profile_batches > 0:
            prof = cProfile.Profile()
            prof.enable()
        x = x.to(device, non_blocking=True)  # [B, T, F]
        y = y.to(device, non_blocking=True)  # [B, T, 88]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda"):
            logits, _ = model(x)
            loss = bce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        _, _, f1 = frame_metrics(logits.detach(), y.detach())
        total_f1 += f1
        # Stop profiler after N batches
        if prof is not None and profile_batches and (it + 1) >= profile_batches:
            prof.disable()
            out_path = profile_out or "./profile_train.pstats"
            if epoch_idx is not None:
                root, ext = os.path.splitext(out_path)
                out_path = f"{root}_epoch{epoch_idx}{ext or '.pstats'}"
            with open(out_path, "wb") as f:
                ps = pstats.Stats(prof, stream=f)
                ps.dump_stats(out_path)
            prof = None
        if (it + 1) % log_interval == 0:
            print(
                f"  iter {it + 1}/{len(loader)} | loss {total_loss / (it + 1):.4f} | f1 {total_f1 / (it + 1):.3f}"
            )
        if max_batches is not None and (it + 1) >= max_batches:
            break
    denom = min(len(loader), max_batches) if max_batches else len(loader)
    return total_loss / max(1, denom), total_f1 / max(1, denom)


def preproc_path_for(item: MaestroItem, preproc_root: str) -> str:
    """Compute the NPZ path for a given item under preproc_root.
    Layout: <preproc_root>/<split>/<audio_basename>.npz
    """
    out_dir = os.path.join(preproc_root, item.split)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(item.audio_path))[0]
    return os.path.join(out_dir, f"{base}.npz")


def preproc_base_dir(preproc_root: str, item: MaestroItem) -> str:
    out_dir = os.path.join(preproc_root, item.split)
    base = os.path.splitext(os.path.basename(item.audio_path))[0]
    return os.path.join(out_dir, base)


def preproc_paths(preproc_root: str, item: MaestroItem) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (mel_npy, labels_npy, npz_fallback) paths for an item.
    Prefer NPY files in a subdir; fall back to single NPZ.
    """
    base_dir = preproc_base_dir(preproc_root, item)
    mel_npy = os.path.join(base_dir, "mel.npy")
    labels_npy = os.path.join(base_dir, "labels.npy")
    npz_path = preproc_path_for(item, preproc_root)
    return mel_npy, labels_npy, npz_path


def preproc_item_exists(item: MaestroItem, preproc_root: str) -> bool:
    mel_npy, labels_npy, npz_path = preproc_paths(preproc_root, item)
    return (os.path.exists(mel_npy) and os.path.exists(labels_npy)) or os.path.exists(npz_path)


def preprocess_split(
    data_root: str,
    split: str,
    preproc_root: str,
    processor: MelSpecProcessor,
    overwrite: bool = False,
    limit_items: Optional[int] = None,
    fmt: str = "npy",
):
    """Preprocess a MAESTRO split into mel+label NPZ files."""
    csv_path = find_maestro_csv(data_root)
    items: List[MaestroItem] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("split", "").strip().lower()
            if s != split:
                continue
            audio_rel = (
                row.get("audio_filename") or row.get("audio_path") or row.get("audio")
            )
            midi_rel = (
                row.get("midi_filename") or row.get("midi_path") or row.get("midi")
            )
            if not audio_rel or not midi_rel:
                continue
            audio_path = os.path.join(os.path.dirname(csv_path), audio_rel)
            midi_path = os.path.join(os.path.dirname(csv_path), midi_rel)
            items.append(MaestroItem(audio_path=audio_path, midi_path=midi_path, split=s))
    if limit_items is not None and limit_items > 0:
        items = items[: int(limit_items)]
    print(f"Preprocessing {len(items)} items for split '{split}' -> {preproc_root} ({fmt})")
    os.makedirs(os.path.join(preproc_root, split), exist_ok=True)

    for i, item in enumerate(items, 1):
        out_npz = preproc_path_for(item, preproc_root)
        base_dir = preproc_base_dir(preproc_root, item)
        mel_npy = os.path.join(base_dir, "mel.npy")
        labels_npy = os.path.join(base_dir, "labels.npy")
        if fmt == "npy":
            out_exists = os.path.exists(mel_npy) and os.path.exists(labels_npy)
        elif fmt == "npz":
            out_exists = os.path.exists(out_npz)
        else:  # npz_compressed
            out_exists = os.path.exists(out_npz)
        if (not overwrite) and out_exists:
            if i % 10 == 0:
                marker = mel_npy if fmt == "npy" else out_npz
                print(f"  [{i}/{len(items)}] exists, skipping: {marker}")
            continue
        try:
            wav, sr = torchaudio.load(item.audio_path)
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)
            with torch.no_grad():
                mel = processor(wav, sr)  # [T, F]
            T = mel.shape[0]
            labels = midi_onset_frames(
                item.midi_path, total_frames=T, sr=processor.sample_rate, hop=processor.hop, widen=1
            )
            if fmt == "npy":
                os.makedirs(base_dir, exist_ok=True)
                np.save(mel_npy, mel.cpu().numpy().astype(np.float32), allow_pickle=False)
                np.save(labels_npy, labels.astype(np.float32), allow_pickle=False)
                if i % 10 == 0 or i == len(items):
                    print(f"  [{i}/{len(items)}] wrote {mel_npy} & labels.npy")
            elif fmt == "npz":
                np.savez(out_npz, mel=mel.cpu().numpy().astype(np.float32), labels=labels.astype(np.float32))
                if i % 10 == 0 or i == len(items):
                    print(f"  [{i}/{len(items)}] wrote {out_npz}")
            else:  # npz_compressed
                np.savez_compressed(out_npz, mel=mel.cpu().numpy().astype(np.float32), labels=labels.astype(np.float32))
                if i % 10 == 0 or i == len(items):
                    print(f"  [{i}/{len(items)}] wrote {out_npz}")
        except Exception as e:
            print(f"  [{i}/{len(items)}] ERROR {item.audio_path}: {e}")


@torch.no_grad()
def evaluate(
    model: OnsetCRNN, loader: DataLoader, device, threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    tot_p = tot_r = tot_f1 = 0.0
    for it, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = bce(logits, y)
        total_loss += loss.item()
        p, r, f1 = frame_metrics(logits, y, threshold=threshold)
        tot_p += p
        tot_r += r
        tot_f1 += f1
        # Respect a max batches limit via attribute injected by caller
        max_batches = getattr(loader, "_max_batches", None)
        if max_batches is not None and (it + 1) >= max_batches:
            break
    n = max(1, min(len(loader), getattr(loader, "_max_batches", len(loader))))
    return total_loss / n, tot_p / n, tot_r / n, tot_f1 / n


# ---------------------------
# Streaming inference (simulated from WAV)
# ---------------------------
@dataclass
class StreamState:
    mel_buffer: torch.Tensor  # [W, F]
    hidden: Optional[torch.Tensor]
    sr: int


def stream_from_wav(
    wav_path: str,
    model_path: str,
    device: Optional[str] = None,
    threshold_on: float = 0.55,
    threshold_off: float = 0.40,
    window_frames: int = 64,
) -> List[Tuple[float, int]]:
    """Simulate streaming inference from a WAV file. Returns list of (time_sec, midi_pitch)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Load model
    model = OnsetCRNN()
    model.load_state_dict(torch.load(model_path, map_location=device_t))
    model.to(device_t).eval()

    # Prepare processor and audio
    processor = MelSpecProcessor(sample_rate=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)
    wav, sr = torchaudio.load(wav_path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)

    with torch.no_grad():
        mel_full = processor(wav, sr)  # [T, F]

    # Ring buffer over mel frames
    buffer = torch.zeros((0, mel_full.shape[1]), dtype=mel_full.dtype, device=device_t)
    hidden: Optional[torch.Tensor] = None
    events: List[Tuple[float, int]] = []
    # Hysteresis state per pitch
    active = np.zeros(N_KEYS, dtype=bool)

    step = 1  # one new frame per hop
    for t_idx in range(0, mel_full.shape[0], step):
        new_frame = mel_full[t_idx : t_idx + 1].to(device_t)
        buffer = torch.cat([buffer, new_frame], dim=0)
        if buffer.shape[0] > window_frames:
            buffer = buffer[-window_frames:]
        if buffer.shape[0] < max(8, window_frames // 4):
            continue  # wait until we have minimal context

        x = buffer.unsqueeze(0)  # [1, T, F]
        logits, hidden = model(x, hidden)
        probs = torch.sigmoid(logits[:, -1])  # last frame probs, [1, 88]
        probs = probs.squeeze(0).cpu().numpy()

        for k in range(N_KEYS):
            p = probs[k]
            if not active[k] and p >= threshold_on:
                active[k] = True
                # Emit onset event
                time_sec = (t_idx * HOP) / SR
                midi_pitch = MIDI_LOW + k
                events.append((time_sec, midi_pitch))
            elif active[k] and p <= threshold_off:
                active[k] = False

    # Print a short summary
    print(f"Detected {len(events)} onsets.")
    if events:
        print("First 10 events:")
        for e in events[:10]:
            print(f"  t={e[0]:.3f}s pitch={e[1]}")
    return events


# ---------------------------
# Main (train or stream)
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Piano onset CRNN: train or stream")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to MAESTRO root (contains CSV).",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--segment_frames", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--pos_weight", type=float, default=5.0)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out", type=str, default="onset_crnn.pt")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit_train_items",
        type=int,
        default=None,
        help="Limit number of training items (for quick tests).",
    )
    p.add_argument(
        "--limit_val_items",
        type=int,
        default=None,
        help="Limit number of validation items (for quick tests).",
    )
    p.add_argument(
        "--val_batches",
        type=int,
        default=50,
        help="Num validation batches per epoch (approx).",
    )
    p.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Max training batches per epoch (for quick tests).",
    )
    p.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Max validation batches per epoch (overrides --val_batches).",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    # DataLoader performance
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Prefetch batches per worker (used when num_workers>0).",
    )
    p.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (when num_workers>0).",
    )
    # Preprocessing
    p.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess dataset to mel+label NPZ files and exit.",
    )
    p.add_argument(
        "--preproc_root",
        type=str,
        default=None,
        help="Directory to save/load preprocessed NPZ files (default: <data_root>/preproc).",
    )
    p.add_argument(
        "--preprocess_splits",
        type=str,
        default="train,validation",
        help="Comma-separated splits to preprocess (train,validation,test).",
    )
    p.add_argument(
        "--preprocess_overwrite",
        action="store_true",
        help="Overwrite existing NPZ files if present.",
    )
    p.add_argument(
        "--preprocess_limit_per_split",
        type=int,
        default=None,
        help="Optionally limit items per split during preprocessing.",
    )
    p.add_argument(
        "--preprocess_format",
        type=str,
        default="npy",
        choices=["npy", "npz", "npz_compressed"],
        help="Storage format for preprocessed features (npy=fast, default).",
    )
    # Profiling
    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile during training.",
    )
    p.add_argument(
        "--profile_out",
        type=str,
        default="./profile_train.pstats",
        help="Path to write cProfile stats (appends epoch number).",
    )
    p.add_argument(
        "--profile_batches",
        type=int,
        default=200,
        help="Number of training batches to profile per epoch (if --profile).",
    )
    # Streaming
    p.add_argument(
        "--stream_wav",
        type=str,
        default=None,
        help="WAV path to run streaming simulation.",
    )
    p.add_argument(
        "--model", type=str, default=None, help="Model checkpoint for streaming."
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Streaming-only mode
    if args.stream_wav:
        if not args.model or not os.path.exists(args.model):
            raise SystemExit(
                "--model must point to a trained .pt checkpoint for streaming"
            )
        stream_from_wav(args.stream_wav, args.model)
        return

    # Determine preproc root
    preproc_root = args.preproc_root or (
        os.path.join(args.data_root, "preproc") if args.data_root else None
    )

    # Preprocessing-only mode
    if args.preprocess:
        if not args.data_root:
            raise SystemExit("--data_root is required for --preprocess")
        if preproc_root is None:
            raise SystemExit("Could not determine --preproc_root")
        processor = MelSpecProcessor(sample_rate=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)
        splits = [s.strip().lower() for s in args.preprocess_splits.split(",") if s.strip()]
        for s in splits:
            preprocess_split(
                args.data_root,
                split=s,
                preproc_root=preproc_root,
                processor=processor,
                overwrite=args.preprocess_overwrite,
                limit_items=args.preprocess_limit_per_split,
                fmt=args.preprocess_format,
            )
        return

    # Training mode
    if not args.data_root:
        raise SystemExit("--data_root is required for training (path to MAESTRO root)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = MelSpecProcessor(sample_rate=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)
    # Auto-enable preproc usage if the directory exists and contains files
    use_preproc = False
    if preproc_root and os.path.isdir(preproc_root):
        # Heuristic: check for any npz under preproc_root
        for root_dir, _, files in os.walk(preproc_root):
            if any(f.endswith('.npz') for f in files):
                use_preproc = True
                break

    train_ds = MaestroOnsetDataset(
        args.data_root,
        split="train",
        processor=processor,
        segment_frames=args.segment_frames,
        augment=True,
        limit_items=args.limit_train_items,
        preproc_root=preproc_root,
        use_preproc=use_preproc,
    )

    val_ds = MaestroOnsetDataset(
        args.data_root,
        split="validation",
        processor=processor,
        segment_frames=args.segment_frames,
        augment=False,
        limit_items=args.limit_val_items,
        preproc_root=preproc_root,
        use_preproc=use_preproc,
    )

    dl_common = dict(
        pin_memory=True,
    )
    # Only supply prefetch_factor/persistent_workers when workers are used
    if args.num_workers > 0:
        dl_common.update(
            prefetch_factor=max(2, int(args.prefetch_factor)),
            persistent_workers=bool(args.persistent_workers),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        **dl_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        **dl_common,
    )

    model = OnsetCRNN(
        n_mels=N_MELS, hidden=args.hidden, classes=N_KEYS, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Prefer the new torch.amp GradScaler API; fall back for older PyTorch
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(
                "cuda", enabled=(device.type == "cuda")
            )
        except TypeError:
            # Older signature without device type positional arg
            scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    else:  # pragma: no cover - legacy fallback
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    # Enable cuDNN benchmark for faster convs on fixed input shapes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            pos_weight=args.pos_weight,
            max_batches=args.max_train_batches,
            profile_batches=(args.profile_batches if args.profile else None),
            profile_out=(args.profile_out if args.profile else None),
            epoch_idx=epoch,
        )
        # Inject optional max batches into val loader for evaluate()
        if args.max_val_batches is not None:
            setattr(val_loader, "_max_batches", args.max_val_batches)
        else:
            setattr(val_loader, "_max_batches", args.val_batches)
        val_loss, val_p, val_r, val_f1 = evaluate(
            model, val_loader, device, threshold=args.threshold
        )
        scheduler.step()
        print(
            f"  train: loss={train_loss:.4f} f1={train_f1:.3f} | val: loss={val_loss:.4f} f1={val_f1:.3f} p={val_p:.3f} r={val_r:.3f}"
        )

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.out)
            print(f"  Saved best model to {args.out} (f1={best_val_f1:.3f})")

    print("Done.")


if __name__ == "__main__":
    main()
