#!/usr/bin/env python3
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch

from maestro import HOP, MIDI_LOW, N_FFT, N_KEYS, N_MELS, SR, MelSpecProcessor, load_mono_wav
from model import OnsetCRNN


def stream_from_wav(
    wav_path: str,
    model_path: str,
    device: Optional[str] = None,
    threshold_on: float = 0.55,
    threshold_off: float = 0.40,
    window_frames: int = 64,
) -> List[Tuple[float, int]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model = OnsetCRNN()
    model.load_state_dict(torch.load(model_path, map_location=device_t))
    model.to(device_t).eval()

    processor = MelSpecProcessor(sample_rate=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)
    wav, sr = load_mono_wav(wav_path)
    with torch.no_grad():
        mel_full = processor(wav, sr)

    buffer = torch.zeros((0, mel_full.shape[1]), dtype=mel_full.dtype, device=device_t)
    hidden: Optional[torch.Tensor] = None
    events: List[Tuple[float, int]] = []
    active = np.zeros(N_KEYS, dtype=bool)

    for t_idx in range(0, mel_full.shape[0]):
        new_frame = mel_full[t_idx : t_idx + 1].to(device_t)
        buffer = torch.cat([buffer, new_frame], dim=0)
        if buffer.shape[0] > window_frames:
            buffer = buffer[-window_frames:]
        if buffer.shape[0] < max(8, window_frames // 4):
            continue

        x = buffer.unsqueeze(0)
        with torch.no_grad():
            logits, hidden = model(x, hidden)
            probs = torch.sigmoid(logits[:, -1]).squeeze(0).detach().cpu().numpy()

        for key_idx in range(N_KEYS):
            p = probs[key_idx]
            if not active[key_idx] and p >= threshold_on:
                active[key_idx] = True
                events.append(((t_idx * HOP) / SR, MIDI_LOW + key_idx))
            elif active[key_idx] and p <= threshold_off:
                active[key_idx] = False

    return events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run streaming simulation from a WAV file")
    p.add_argument("--wav", required=True, help="Input WAV path")
    p.add_argument("--model", required=True, help="Model checkpoint path")
    p.add_argument("--device", default=None, help="cuda or cpu")
    p.add_argument("--threshold_on", type=float, default=0.55)
    p.add_argument("--threshold_off", type=float, default=0.40)
    p.add_argument("--window_frames", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = stream_from_wav(
        wav_path=args.wav,
        model_path=args.model,
        device=args.device,
        threshold_on=args.threshold_on,
        threshold_off=args.threshold_off,
        window_frames=args.window_frames,
    )
    print(f"Detected {len(events)} onsets.")
    if events:
        print("First 10 events:")
        for t_sec, pitch in events[:10]:
            print(f"  t={t_sec:.3f}s pitch={pitch}")


if __name__ == "__main__":
    main()
