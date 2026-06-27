"""Export an OnsetCRNN checkpoint to ONNX for on-device (mobile) streaming inference.

The exported graph is the *streaming core*: it takes one fixed-size mel window plus
the carried GRU hidden state, and returns onset probabilities for the chunk plus the
updated GRU state. Mel extraction and peak-picking are done app-side (see MOBILE.md
and onnx_runtime.py for the exact, test-backed reference).

Streaming contract (steady state), for chunk=16 / context=3:
    inputs:
        mel_window : float32 [1, 22, n_mels]   # 3 left-ctx + 16 chunk + 3 right-ctx frames
        h_in       : float32 [1, 1, hidden]    # GRU state (zeros for the first chunk)
    outputs:
        probs      : float32 [1, 16, 88]       # sigmoid(onset logits) for the 16 chunk frames
        h_out      : float32 [1, 1, hidden]    # GRU state to feed into the next chunk

The convs run over the full 22-frame window (so the 16 middle frames see real
neighbours), then the 3 context frames are trimmed off each side before the GRU,
so the GRU state advances by exactly `chunk` frames per call — bit-equivalent to
whole-piece inference.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from model import OnsetCRNN
from stream import CONV_TIME_CONTEXT


class StreamingExportWrapper(nn.Module):
    """Fixed-shape streaming step: (mel_window, h_in) -> (probs, h_out).

    left_trim / right_trim are baked in as Python ints at export time, so the
    graph is specialised to one (chunk, context) configuration with static shapes.
    """

    def __init__(self, model: OnsetCRNN, left_trim: int, right_trim: int):
        super().__init__()
        if model.gru is None:
            raise ValueError("Export wrapper requires a GRU model (temporal_mode='gru').")
        self.model = model
        self.left_trim = int(left_trim)
        self.right_trim = int(right_trim)

    def forward(self, mel_window: torch.Tensor, h_in: torch.Tensor):
        # mel_window: [1, W, n_mels]
        x = self.model._conv_stack(mel_window)  # [1, W, feat]
        T = x.shape[1]
        x = x[:, self.left_trim : T - self.right_trim]  # [1, chunk, feat]
        y, h_out = self.model.gru(x, h_in)
        logits = self.model.head(y)
        probs = torch.sigmoid(logits)
        return probs, h_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="output/mel229_fixed_aug_2x.pt",
                   help="Checkpoint to export.")
    p.add_argument("--out", default=None,
                   help="Output .onnx path. Default: <model_stem>_chunk<N>.onnx next to the model.")
    p.add_argument("--chunk_frames", type=int, default=16)
    p.add_argument("--context_frames", type=int, default=CONV_TIME_CONTEXT)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--config_json", default=None,
                   help="Also write a deployment config JSON sidecar (default: <out>.json).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.model):
        raise SystemExit(f"model not found: {args.model}")

    model = OnsetCRNN.from_checkpoint(args.model, map_location="cpu").eval()
    if model.freq_pool != "flatten":
        print(f"[warn] freq_pool={model.freq_pool!r} (not 'flatten'); exporting anyway.")

    chunk = args.chunk_frames
    ctx = args.context_frames
    window = chunk + 2 * ctx
    n_mels = int(model.n_mels)
    hidden = int(model.hidden)

    stem = os.path.splitext(os.path.basename(args.model))[0]
    out = args.out or os.path.join(os.path.dirname(args.model), f"{stem}_chunk{chunk}.onnx")
    first_out = os.path.splitext(out)[0] + "_first.onnx"
    first_window = chunk + ctx  # 19 frames: no left context at stream start

    def export_one(left_trim, right_trim, w, path):
        wrap = StreamingExportWrapper(model, left_trim=left_trim, right_trim=right_trim).eval()
        mel_window = torch.randn(1, w, n_mels, dtype=torch.float32)
        h_in = torch.zeros(1, 1, hidden, dtype=torch.float32)
        with torch.no_grad():
            probs, h_out = wrap(mel_window, h_in)
        assert tuple(probs.shape) == (1, chunk, model.classes), probs.shape
        assert tuple(h_out.shape) == (1, 1, hidden), h_out.shape
        torch.onnx.export(
            wrap, (mel_window, h_in), path,
            input_names=["mel_window", "h_in"],
            output_names=["probs", "h_out"],
            opset_version=args.opset, dynamo=False, do_constant_folding=True,
        )
        print(f"[export] wrote {path}  (window={w}, left_trim={left_trim}, right_trim={right_trim})")

    # Steady-state graph (used for every chunk after the first): 22-frame real window.
    export_one(ctx, ctx, window, out)
    # First-chunk graph (used once at stream start): 19-frame window, no left context.
    # Reproduces StreamingOnsetDetector's first chunk -> bit-exact with whole-piece inference.
    export_one(0, ctx, first_window, first_out)

    # Deployment config sidecar — everything the app needs to reproduce I/O.
    mel_mean, mel_std = model.get_mel_stats()
    pre = model.get_preproc_config()
    cfg = {
        "checkpoint": os.path.basename(args.model),
        "onnx": os.path.basename(out),
        "io": {
            "chunk_frames": chunk,
            "context_frames": ctx,
            "n_mels": n_mels,
            "hidden": hidden,
            "classes": int(model.classes),
            "h_shape": [1, 1, hidden],
            "probs_shape": [1, chunk, int(model.classes)],
            # Steady-state graph: used for every chunk after the first.
            "steady": {
                "onnx": os.path.basename(out),
                "mel_window_frames": window,
                "mel_window_shape": [1, window, n_mels],
                "left_trim": ctx, "right_trim": ctx,
            },
            # First-chunk graph: used exactly once at stream start (zero h_in).
            "first": {
                "onnx": os.path.basename(first_out),
                "mel_window_frames": first_window,
                "mel_window_shape": [1, first_window, n_mels],
                "left_trim": 0, "right_trim": ctx,
            },
            # Output onset frame index == true global mel-frame index (first chunk
            # covers mel frames 0..chunk-1; reproduces whole-piece inference exactly).
        },
        "preproc": {
            "sample_rate": pre.get("sample_rate", 16000),
            "n_fft": pre.get("n_fft", 2048),
            "hop": pre.get("hop", 160),
            "win_length": pre.get("n_fft", 2048),
            "n_mels": n_mels,
            "f_min": 20.0,
            "f_max": float(pre.get("sample_rate", 16000)) / 2.0,
            "power": 2.0,
            "center": True,
            "pad_mode": "reflect",
            "log": "log1p",          # natural log of (1 + mel_power)
            "normalize": "fixed",
            "mel_mean": mel_mean,
            "mel_std": mel_std,
            "mel_filterbank_npy": f"{stem}_melfb.npy",
        },
        "decode": {
            "default_threshold": 0.65,
            "refractory_frames": 5,
            "midi_low": 21,
            "rule": "is_peak[t] = p[t]>=thr AND p[t]>p[t-1] AND p[t]>=p[t+1]; per-pitch refractory",
            "lookahead_frames": 1,
        },
        "latency_ms": (chunk + ctx) * (pre.get("hop", 160) * 1000 // pre.get("sample_rate", 16000)),
    }
    cfg_path = args.config_json or (os.path.splitext(out)[0] + ".json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[export] wrote {cfg_path}")

    # Dump the EXACT mel filterbank the model was trained with, pulled straight
    # from a MelSpecProcessor built with the checkpoint's preproc config (so the
    # app's mel extraction is bit-for-bit reproducible — no param guessing).
    from maestro import MelSpecProcessor
    proc = MelSpecProcessor(
        sample_rate=pre.get("sample_rate", 16000),
        n_fft=pre.get("n_fft", 2048),
        hop=pre.get("hop", 160),
        n_mels=n_mels,
        repr_type=pre.get("repr_type", "mel"),
    )
    fb = proc.melspec.mel_scale.fb  # [n_freqs, n_mels]
    fb_path = os.path.join(os.path.dirname(out), f"{stem}_melfb.npy")
    np.save(fb_path, fb.numpy().astype(np.float32))
    print(f"[export] wrote {fb_path}  (filterbank {tuple(fb.shape)})")


if __name__ == "__main__":
    main()
