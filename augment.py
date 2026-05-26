"""Mel-domain data augmentation for training robustness.

All transforms operate on [T, N_MELS] float32 tensors and leave labels unchanged.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch


def add_gaussian_noise(
    mel: torch.Tensor,
    snr_db_range: Tuple[float, float] = (-5.0, 20.0),
) -> torch.Tensor:
    rms = float(mel.pow(2).mean().sqrt())
    if rms < 1e-8:
        return mel
    snr_db = random.uniform(*snr_db_range)
    noise_std = rms / (10.0 ** (snr_db / 20.0))
    return mel + torch.randn_like(mel) * noise_std


def random_eq(
    mel: torch.Tensor,
    max_db: float = 6.0,
    n_control: int = 8,
) -> torch.Tensor:
    n_mels = mel.shape[1]
    ctrl_pts = np.random.uniform(-max_db, max_db, size=n_control).astype(np.float32)
    ctrl_x = np.linspace(0, n_mels - 1, n_control)
    all_x = np.arange(n_mels, dtype=np.float32)
    gains_db = np.interp(all_x, ctrl_x, ctrl_pts).astype(np.float32)
    gains_linear = torch.from_numpy(np.float_power(10.0, gains_db / 20.0).astype(np.float32))
    return mel * gains_linear.unsqueeze(0)


def gain_jitter(
    mel: torch.Tensor,
    range_db: Tuple[float, float] = (-6.0, 6.0),
) -> torch.Tensor:
    gain_db = random.uniform(*range_db)
    return mel * (10.0 ** (gain_db / 20.0))


def dc_offset(
    mel: torch.Tensor,
    max_offset: float = 0.3,
) -> torch.Tensor:
    return mel + random.uniform(-max_offset, max_offset)


def time_masking(
    mel: torch.Tensor,
    max_frames: int = 20,
) -> torch.Tensor:
    T = mel.shape[0]
    if T < 2:
        return mel
    mask_len = random.randint(1, min(max_frames, T - 1))
    start = random.randint(0, T - mask_len)
    mel = mel.clone()
    mel[start : start + mask_len] = 0.0
    return mel


def freq_masking(
    mel: torch.Tensor,
    max_bins: int = 8,
) -> torch.Tensor:
    n_mels = mel.shape[1]
    if n_mels < 2:
        return mel
    mask_len = random.randint(1, min(max_bins, n_mels - 1))
    start = random.randint(0, n_mels - mask_len)
    mel = mel.clone()
    mel[:, start : start + mask_len] = 0.0
    return mel


class MelAugment:
    """Composable mel-domain augmentation applied per training sample."""

    def __init__(
        self,
        noise_p: float = 0.5,
        noise_snr_range: Tuple[float, float] = (-5.0, 20.0),
        eq_p: float = 0.7,
        eq_max_db: float = 6.0,
        gain_p: float = 0.5,
        gain_range_db: Tuple[float, float] = (-6.0, 6.0),
        dc_p: float = 0.3,
        dc_max_offset: float = 0.3,
        time_mask_p: float = 0.2,
        time_mask_max_frames: int = 20,
        freq_mask_p: float = 0.2,
        freq_mask_max_bins: int = 8,
    ):
        self.noise_p = noise_p
        self.noise_snr_range = noise_snr_range
        self.eq_p = eq_p
        self.eq_max_db = eq_max_db
        self.gain_p = gain_p
        self.gain_range_db = gain_range_db
        self.dc_p = dc_p
        self.dc_max_offset = dc_max_offset
        self.time_mask_p = time_mask_p
        self.time_mask_max_frames = time_mask_max_frames
        self.freq_mask_p = freq_mask_p
        self.freq_mask_max_bins = freq_mask_max_bins

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if random.random() < self.eq_p:
            mel = random_eq(mel, max_db=self.eq_max_db)
        if random.random() < self.noise_p:
            mel = add_gaussian_noise(mel, snr_db_range=self.noise_snr_range)
        if random.random() < self.gain_p:
            mel = gain_jitter(mel, range_db=self.gain_range_db)
        if random.random() < self.dc_p:
            mel = dc_offset(mel, max_offset=self.dc_max_offset)
        if random.random() < self.time_mask_p:
            mel = time_masking(mel, max_frames=self.time_mask_max_frames)
        if random.random() < self.freq_mask_p:
            mel = freq_masking(mel, max_bins=self.freq_mask_max_bins)
        return mel
