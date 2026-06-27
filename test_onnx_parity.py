"""Parity test: the numpy+ONNX on-device reference vs the PyTorch streaming path.

Run: uv run python test_onnx_parity.py --wav <path>.wav

Checks, on real audio:
  A) numpy log-mel == torchaudio MelSpecProcessor log-mel       (per-frame max abs diff)
  B) ONNX streaming events == PyTorch StreamingOnsetDetector    (same mel; must be EXACT)
  C) full numpy+ONNX pipeline == PyTorch StreamingOnsetDetector (end-to-end; must be EXACT)
  D) ONNX streaming == whole-piece offline inference            (must be EXACT -> IoU 1.0)

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchaudio

from maestro import MelSpecProcessor, load_mono_wav
from model import OnsetCRNN
from stream import StreamingOnsetDetector
from decode import peak_pick
from onnx_runtime import compute_logmel, OnnxStreamingDetector


def live_pytorch_events(model, mel, threshold, refractory, chunk, feed=7):
    """StreamingOnsetDetector fed in small increments. No flush() — a live mic stream
    never flushes, and the ONNX path doesn't emit a final partial chunk either."""
    det = StreamingOnsetDetector(model, chunk_frames=chunk, threshold=threshold,
                                 refractory_frames=refractory)
    ev = []
    for i in range(0, mel.shape[0], feed):
        ev.extend(det.process(mel[i:i + feed]))
    return set(ev)


def onnx_events(cfg, asset_dir, mel, threshold, refractory, feed=7):
    det = OnnxStreamingDetector(cfg, asset_dir=asset_dir, threshold=threshold,
                                refractory_frames=refractory)
    ev = []
    for i in range(0, mel.shape[0], feed):
        ev.extend(det.process(mel[i:i + feed]))
    return set(ev)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", required=True)
    p.add_argument("--model", default="output/mel229_fixed_aug_2x.pt")
    p.add_argument("--config", default="output/mel229_fixed_aug_2x_chunk16.json")
    p.add_argument("--melfb", default="output/mel229_fixed_aug_2x_melfb.npy")
    p.add_argument("--start", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--mel_tol", type=float, default=1e-3)
    args = p.parse_args()

    cfg = json.load(open(args.config))
    asset_dir = os.path.dirname(args.config)
    pre = cfg["preproc"]
    fb = np.load(args.melfb)
    sr_t = pre["sample_rate"]
    chunk = cfg["io"]["chunk_frames"]
    thr = cfg["decode"]["default_threshold"]
    ref = cfg["decode"]["refractory_frames"]

    wav, sr = load_mono_wav(args.wav)
    if sr != sr_t:
        wav = torchaudio.functional.resample(wav, sr, sr_t)
    total = wav.shape[-1] / sr_t
    s = int(round(min(args.start, max(0.0, total - args.duration)) * sr_t))
    e = int(round(min(total, args.start + args.duration) * sr_t))
    wav = wav[:, s:e]
    audio = wav.squeeze(0).numpy().astype(np.float32)
    print(f"[parity] wav={os.path.basename(args.wav)} {audio.shape[0]/sr_t:.1f}s @ {sr_t} Hz")

    model = OnsetCRNN.from_checkpoint(args.model, map_location="cpu").eval()
    mel_mean, mel_std = model.get_mel_stats()

    proc = MelSpecProcessor(sample_rate=sr_t, n_fft=pre["n_fft"], hop=pre["hop"],
                            n_mels=pre["n_mels"], repr_type=pre.get("repr_type", "mel"),
                            mel_mean=mel_mean, mel_std=mel_std)
    with torch.no_grad():
        mel_ta = proc(wav, sr_t, normalize="fixed").numpy().astype(np.float32)
    mel_np = compute_logmel(audio, fb, n_fft=pre["n_fft"], hop=pre["hop"],
                            mel_mean=mel_mean, mel_std=mel_std)
    T = min(mel_ta.shape[0], mel_np.shape[0])
    mel_ta, mel_np = mel_ta[:T], mel_np[:T]

    ok = True

    # A) mel reproduction
    md = float(np.abs(mel_ta - mel_np).max())
    a = md < args.mel_tol
    ok &= a
    print(f"[A] numpy-mel vs torchaudio-mel: max|diff|={md:.2e} tol={args.mel_tol:.0e} -> {'PASS' if a else 'FAIL'}")

    ev_live = live_pytorch_events(model, mel_ta, thr, ref, chunk)

    # B) ONNX vs PyTorch streaming, same mel -> EXACT
    ev_b = onnx_events(cfg, asset_dir, mel_ta, thr, ref)
    dis_b = ev_live ^ ev_b
    b = len(dis_b) == 0
    ok &= b
    print(f"[B] ONNX vs PyTorch-stream (same mel): pt={len(ev_live)} onnx={len(ev_b)} "
          f"disagreements={len(dis_b)} -> {'PASS' if b else 'FAIL'}")
    if dis_b:
        print(f"     {sorted(dis_b)[:10]}")

    # C) full numpy+ONNX (app pipeline) vs PyTorch streaming -> EXACT
    ev_c = onnx_events(cfg, asset_dir, mel_np, thr, ref)
    dis_c = ev_live ^ ev_c
    c = len(dis_c) == 0
    ok &= c
    print(f"[C] full numpy+ONNX vs PyTorch-stream: onnx={len(ev_c)} disagreements={len(dis_c)} "
          f"-> {'PASS' if c else 'FAIL'}")
    if dis_c:
        print(f"     {sorted(dis_c)[:10]}")

    # D) ONNX streaming vs whole-piece offline inference -> EXACT (gold standard).
    # Exclude the final partial chunk (last chunk+ctx frames): a live stream can't have
    # decided it yet without an explicit flush, which an infinite mic stream never does.
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(mel_ta)[None])
        probs = torch.sigmoid(logits).squeeze(0).numpy()
    ev_off = set((int(f), int(k)) for f, k in peak_pick(probs, threshold=thr, refractory_frames=ref))
    cutoff = T - (chunk + cfg["io"]["context_frames"])
    ev_b_d = {(f, k) for (f, k) in ev_b if f < cutoff}
    ev_off_d = {(f, k) for (f, k) in ev_off if f < cutoff}
    dis_d = ev_b_d ^ ev_off_d
    d = len(dis_d) == 0
    ok &= d
    iou_d = len(ev_b_d & ev_off_d) / max(1, len(ev_b_d | ev_off_d))
    print(f"[D] ONNX-stream vs whole-piece offline (excl. final partial chunk): "
          f"offline={len(ev_off_d)} IoU={iou_d:.4f} disagreements={len(dis_d)} -> {'PASS' if d else 'FAIL'}")
    if dis_d:
        print(f"     {sorted(dis_d)[:10]}")

    print(f"\n[parity] {'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
