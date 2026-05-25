import csv
import os
from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

SR = 16000
N_FFT = 512
HOP = 160  # 10 ms at 16 kHz
N_MELS = 64
FMIN = 20.0
FMAX = None

N_KEYS = 88
MIDI_LOW = 21
MIDI_HIGH = 108

# Global log1p-mel statistics computed over 50 training pieces (562M values).
# Used for fixed normalization in the streaming path so that silence doesn't
# produce extreme values from per-buffer z-normalization.
MEL_GLOBAL_MEAN = 0.2746
MEL_GLOBAL_STD = 0.7824


@dataclass(frozen=True)
class MaestroItem:
    audio_path: str
    midi_path: str
    split: str


def find_maestro_csv(root: str) -> str:
    for fn in os.listdir(root):
        if fn.endswith(".csv") and "maestro" in fn.lower():
            return os.path.join(root, fn)
    for sub in os.listdir(root):
        p = os.path.join(root, sub)
        if not os.path.isdir(p):
            continue
        for fn in os.listdir(p):
            if fn.endswith(".csv") and "maestro" in fn.lower():
                return os.path.join(p, fn)
    raise FileNotFoundError(f"Could not locate MAESTRO CSV under: {root}")


def load_split_items(
    data_root: str,
    split: str,
    limit_items: Optional[int] = None,
) -> List[MaestroItem]:
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
            items.append(
                MaestroItem(audio_path=audio_path, midi_path=midi_path, split=s)
            )
    if limit_items is not None and limit_items > 0:
        items = items[: int(limit_items)]
    if not items:
        raise RuntimeError(f"No items found for split={split}")
    return items


def preproc_base_dir(preproc_root: str, item: MaestroItem) -> str:
    out_dir = os.path.join(preproc_root, item.split)
    base = os.path.splitext(os.path.basename(item.audio_path))[0]
    return os.path.join(out_dir, base)


def preproc_paths(preproc_root: str, item: MaestroItem) -> Tuple[str, str]:
    base_dir = preproc_base_dir(preproc_root, item)
    return os.path.join(base_dir, "mel.npy"), os.path.join(base_dir, "labels.npy")


def preproc_item_exists(item: MaestroItem, preproc_root: str) -> bool:
    mel_path, labels_path = preproc_paths(preproc_root, item)
    return os.path.exists(mel_path) and os.path.exists(labels_path)


def missing_preproc_items(
    items: Iterable[MaestroItem], preproc_root: str
) -> List[MaestroItem]:
    missing: List[MaestroItem] = []
    for item in items:
        if not preproc_item_exists(item, preproc_root):
            missing.append(item)
    return missing


def midi_onset_frames(
    midi_path: str,
    total_frames: int,
    sr: int = SR,
    hop: int = HOP,
    widen: int = 1,
) -> np.ndarray:
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
            f = int(round(note.start * sr / hop))
            if 0 <= f < total_frames:
                lo = max(0, f - widen)
                hi = min(total_frames, f + widen + 1)
                y[lo:hi, idx] = 1.0
    return y


class PreprocessedMaestroDataset(Dataset):
    """MAESTRO dataset that requires precomputed mel/label NPY files."""

    def __init__(
        self,
        data_root: str,
        split: str,
        preproc_root: str,
        segment_frames: int = 512,
        limit_items: Optional[int] = None,
        preproc_cache: str = "mmap",
        random_item_sampling: bool = False,
        virtual_size: Optional[int] = None,
    ):
        super().__init__()
        if preproc_cache not in {"mmap", "ram"}:
            raise ValueError(f"Unsupported preproc_cache={preproc_cache}")

        self.segment_frames = segment_frames
        self.preproc_root = preproc_root
        self.preproc_cache = preproc_cache
        self.random_item_sampling = random_item_sampling
        self.virtual_size = (
            int(virtual_size)
            if virtual_size is not None and int(virtual_size) > 0
            else None
        )

        self.items: List[MaestroItem] = load_split_items(
            data_root=data_root,
            split=split,
            limit_items=limit_items,
        )
        self._preproc_paths: List[Tuple[str, str]] = []
        missing: List[Tuple[str, str]] = []
        for item in self.items:
            mel_path, labels_path = preproc_paths(preproc_root, item)
            self._preproc_paths.append((mel_path, labels_path))
            if not (os.path.exists(mel_path) and os.path.exists(labels_path)):
                missing.append((mel_path, labels_path))
        if missing:
            example = missing[0]
            raise FileNotFoundError(
                f"Missing preprocessed files for {len(missing)} items under {preproc_root}. "
                f"Example missing pair: {example[0]} and {example[1]}"
            )

        self.preproc_ram_bytes = 0
        self._preproc_ram: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if self.preproc_cache == "ram":
            self._preload_preproc_ram()

    def __len__(self) -> int:
        return self.virtual_size if self.virtual_size is not None else len(self.items)

    def _preload_preproc_ram(self) -> None:
        cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        total_bytes = 0
        for mel_path, labels_path in self._preproc_paths:
            mel_arr = np.asarray(
                np.load(mel_path, allow_pickle=False), dtype=np.float32
            )
            labels_arr = np.asarray(
                np.load(labels_path, allow_pickle=False), dtype=np.float32
            )
            total_bytes += int(mel_arr.nbytes + labels_arr.nbytes)
            cache.append((torch.from_numpy(mel_arr), torch.from_numpy(labels_arr)))
        self._preproc_ram = cache
        self.preproc_ram_bytes = total_bytes

    def _sample_item_index(self, idx: int) -> int:
        if self.random_item_sampling:
            return random.randrange(len(self.items))
        return idx % len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item_idx = self._sample_item_index(idx)

        if self.preproc_cache == "ram":
            if self._preproc_ram is None:
                raise RuntimeError("RAM cache requested but not initialized")
            mel_src, labels_src = self._preproc_ram[item_idx]
            total_frames = int(mel_src.shape[0])
            if total_frames >= self.segment_frames:
                start = random.randint(0, total_frames - self.segment_frames)
                end = start + self.segment_frames
                mel_seg = mel_src[start:end].contiguous()
                y_seg = labels_src[start:end].contiguous()
            else:
                pad = self.segment_frames - total_frames
                mel_seg = F.pad(mel_src, (0, 0, 0, pad)).contiguous()
                y_seg = F.pad(labels_src, (0, 0, 0, pad)).contiguous()
        else:
            mel_path, labels_path = self._preproc_paths[item_idx]
            mel_mm = np.load(mel_path, mmap_mode="r")
            labels_mm = np.load(labels_path, mmap_mode="r")
            total_frames = int(mel_mm.shape[0])
            if total_frames >= self.segment_frames:
                start = random.randint(0, total_frames - self.segment_frames)
                end = start + self.segment_frames
                mel_seg = torch.from_numpy(
                    np.array(mel_mm[start:end], dtype=np.float32)
                )
                y_seg = torch.from_numpy(
                    np.array(labels_mm[start:end], dtype=np.float32)
                )
            else:
                mel_arr = np.array(mel_mm, dtype=np.float32)
                y_arr = np.array(labels_mm, dtype=np.float32)
                pad = self.segment_frames - total_frames
                mel_seg = torch.from_numpy(
                    np.pad(mel_arr, ((0, pad), (0, 0)), mode="constant")
                )
                y_seg = torch.from_numpy(
                    np.pad(y_arr, ((0, pad), (0, 0)), mode="constant")
                )

        return (
            mel_seg.to(torch.float32).contiguous().clone(),
            y_seg.to(torch.float32).contiguous().clone(),
        )


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
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop
        self.resample: Optional[torchaudio.transforms.Resample] = None
        self._resample_sr: Optional[int] = None
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

    def _set_input_sr(self, input_sr: int) -> None:
        if input_sr != self.sample_rate:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=input_sr,
                new_freq=self.sample_rate,
            )
        else:
            self.resample = None

    @torch.no_grad()
    def forward(
        self,
        wav: torch.Tensor,
        input_sr: int,
        normalize: str = "per_piece",
    ) -> torch.Tensor:
        """wav: [1, T] mono float32 -> [T_frames, n_mels].

        normalize: "per_piece" (default, training) | "fixed" (streaming) | "none".
        """
        if self._resample_sr != input_sr:
            self._set_input_sr(input_sr)
            self._resample_sr = input_sr
        if self.resample is not None:
            wav = self.resample(wav)
        mel = self.melspec(wav)
        if mel.dim() == 3 and mel.size(0) == 1:
            mel = mel.squeeze(0)
        if mel.dim() != 2:
            raise RuntimeError(f"Unexpected mel shape: {tuple(mel.shape)}")
        mel = torch.log1p(mel)
        mel = mel.transpose(0, 1).contiguous()
        if normalize == "per_piece":
            mean = mel.mean()
            std = mel.std().clamp_min(1e-5)
            mel = (mel - mean) / std
        elif normalize == "fixed":
            mel = (mel - MEL_GLOBAL_MEAN) / MEL_GLOBAL_STD
        elif normalize == "streaming":
            mean = mel.mean()
            std = mel.std().clamp_min(MEL_GLOBAL_STD * 0.5)
            mel = (mel - mean) / std
        return mel


def load_mono_wav(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav, sr
