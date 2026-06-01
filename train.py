#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import random
import time
from typing import Optional, Tuple

import cProfile
import pstats
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from maestro import N_KEYS, N_MELS, PreprocessedMaestroDataset
from model import OnsetCRNN, maybe_compile_model, unwrap_compiled_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def frame_counts(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(pred)
    yhat = (probs >= threshold).to(target.dtype)
    y = target
    tp = (yhat * y).sum()
    fp = (yhat * (1 - y)).sum()
    fn = ((1 - yhat) * y).sum()
    return tp, fp, fn


def build_adamw(model: nn.Module, args, device: torch.device) -> torch.optim.Optimizer:
    adamw_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if args.fused_adamw:
        if device.type != "cuda":
            raise RuntimeError("--fused_adamw requires CUDA")
        if "fused" not in inspect.signature(torch.optim.AdamW).parameters:
            raise RuntimeError("This PyTorch build does not support fused AdamW")
        adamw_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    if args.fused_adamw and not bool(optimizer.defaults.get("fused", False)):
        raise RuntimeError(
            "Fused AdamW was requested but optimizer did not enable fused kernels"
        )
    if bool(optimizer.defaults.get("fused", False)):
        print("[info] AdamW fused kernels enabled")
    return optimizer


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    pos_weight: float = 5.0,
    log_interval: int = 50,
    max_batches: Optional[int] = None,
    profile_batches: Optional[int] = None,
    profile_out: Optional[str] = None,
    epoch_idx: Optional[int] = None,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tp = torch.zeros((), device=device, dtype=torch.float32)
    total_fp = torch.zeros((), device=device, dtype=torch.float32)
    total_fn = torch.zeros((), device=device, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    target_batches = min(len(loader), max_batches) if max_batches else len(loader)
    iter_loader = loader
    pbar = None
    if show_progress:
        pbar = tqdm(
            loader,
            total=target_batches,
            desc=progress_desc or "train",
            dynamic_ncols=True,
            leave=True,
        )
        iter_loader = pbar

    prof = None
    for it, (x, y) in enumerate(iter_loader):
        if prof is None and profile_batches and profile_batches > 0:
            prof = cProfile.Profile()
            prof.enable()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=device.type == "cuda", device_type="cuda"):
            logits, _ = model(x)
            loss = bce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()
        tp, fp, fn = frame_counts(logits.detach(), y.detach())
        total_tp += tp
        total_fp += fp
        total_fn += fn

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

        if (it + 1) % log_interval == 0 and not show_progress:
            print(f"  iter {it + 1}/{len(loader)}")
        if max_batches is not None and (it + 1) >= max_batches:
            break

    if pbar is not None:
        pbar.close()

    denom = max(1, target_batches)
    avg_loss = (total_loss / denom).item()
    tp_f = total_tp.item()
    fp_f = total_fp.item()
    fn_f = total_fn.item()
    prec = tp_f / (tp_f + fp_f + 1e-8)
    rec = tp_f / (tp_f + fn_f + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return avg_loss, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tp = torch.zeros((), device=device, dtype=torch.float32)
    total_fp = torch.zeros((), device=device, dtype=torch.float32)
    total_fn = torch.zeros((), device=device, dtype=torch.float32)

    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss = bce(logits, y)
        total_loss += loss
        tp, fp, fn = frame_counts(logits, y, threshold=threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        max_batches = getattr(loader, "_max_batches", None)
        if max_batches is not None and (it + 1) >= max_batches:
            break

    n = max(1, min(len(loader), getattr(loader, "_max_batches", len(loader))))
    avg_loss = (total_loss / n).item()
    tp_f = total_tp.item()
    fp_f = total_fp.item()
    fn_f = total_fn.item()
    p = tp_f / (tp_f + fp_f + 1e-8)
    r = tp_f / (tp_f + fn_f + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return avg_loss, p, r, f1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train onset model from preprocessed MAESTRO NPY data"
    )
    p.add_argument(
        "--data_root", required=True, help="Path to MAESTRO root (contains CSV)"
    )
    p.add_argument(
        "--preproc_root",
        required=True,
        help="Root containing preprocessed mel.npy/labels.npy files",
    )
    p.add_argument("--preproc_cache", default="mmap", choices=["mmap", "ram"])

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--segment_frames", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--pos_weight", type=float, default=5.0)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument(
        "--conv_channels",
        type=str,
        default="32,64,96",
        help='Comma-separated 3 ints for SeparableConv2d widths, e.g. "16,32,48".',
    )
    p.add_argument(
        "--temporal_mode",
        type=str,
        default="gru",
        choices=["gru", "none"],
        help="Temporal layer between conv stack and head: 'gru' (default) or 'none' (drop recurrence).",
    )
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freq_pool", choices=["mean", "flatten"], default="mean")
    p.add_argument("--mel_mean", type=float, default=None,
                   help="Fixed-norm mean used at preprocess. Stored on the checkpoint so inference "
                        "(predict/stream/live) can configure MelSpecProcessor to match.")
    p.add_argument("--mel_std", type=float, default=None,
                   help="Fixed-norm std used at preprocess. See --mel_mean.")
    p.add_argument("--preproc_n_fft", type=int, default=None,
                   help="STFT window used at preprocess. Stored on the checkpoint so inference "
                        "auto-configures (mel-229 needs 2048).")
    p.add_argument("--preproc_hop", type=int, default=None, help="Hop used at preprocess.")
    p.add_argument("--preproc_sample_rate", type=int, default=None, help="Sample rate used at preprocess.")
    p.add_argument("--preproc_repr", choices=["mel", "logfreq"], default=None,
                   help="Representation used at preprocess.")
    p.add_argument("--out", type=str, default="onset_crnn.pt")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--limit_train_items", type=int, default=None)
    p.add_argument("--limit_val_items", type=int, default=None)
    p.add_argument("--train_samples_per_epoch", type=int, default=None)
    p.add_argument("--random_item_sampling", action="store_true")

    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--tqdm", action="store_true")

    p.add_argument(
        "--torch_compile",
        type=str,
        default="none",
        choices=[
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )
    p.add_argument("--torch_compile_backend", type=str, default="inductor")
    p.add_argument("--torch_compile_dynamic", action="store_true")
    p.add_argument("--torch_compile_fullgraph", action="store_true")

    p.add_argument("--fused_adamw", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument(
        "--matmul_precision",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
    )
    p.add_argument("--fuse_eval", action="store_true")

    p.add_argument("--augment", action="store_true", help="Enable mel-domain augmentation on training set.")
    p.add_argument("--aug_noise_p", type=float, default=0.5)
    p.add_argument("--aug_noise_snr_lo", type=float, default=-5.0)
    p.add_argument("--aug_noise_snr_hi", type=float, default=20.0)
    p.add_argument("--aug_eq_p", type=float, default=0.7)
    p.add_argument("--aug_eq_max_db", type=float, default=6.0)
    p.add_argument("--aug_gain_p", type=float, default=0.5)
    p.add_argument("--aug_gain_range_db", type=float, default=6.0)
    p.add_argument("--aug_dc_p", type=float, default=0.3)
    p.add_argument("--aug_dc_max_offset", type=float, default=0.3)
    p.add_argument("--aug_time_mask_p", type=float, default=0.2)
    p.add_argument("--aug_time_mask_max_frames", type=int, default=20)
    p.add_argument("--aug_freq_mask_p", type=float, default=0.2)
    p.add_argument("--aug_freq_mask_max_bins", type=int, default=8)

    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile_out", type=str, default="./profile_train.pstats")
    p.add_argument("--profile_batches", type=int, default=200)
    p.add_argument("--stats_json", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_virtual_size = (
        int(args.train_samples_per_epoch)
        if args.train_samples_per_epoch is not None
        and int(args.train_samples_per_epoch) > 0
        else None
    )
    train_random_item_sampling = bool(
        args.random_item_sampling or train_virtual_size is not None
    )

    mel_transform = None
    if args.augment:
        from augment import MelAugment
        mel_transform = MelAugment(
            noise_p=args.aug_noise_p,
            noise_snr_range=(args.aug_noise_snr_lo, args.aug_noise_snr_hi),
            eq_p=args.aug_eq_p,
            eq_max_db=args.aug_eq_max_db,
            gain_p=args.aug_gain_p,
            gain_range_db=(-args.aug_gain_range_db, args.aug_gain_range_db),
            dc_p=args.aug_dc_p,
            dc_max_offset=args.aug_dc_max_offset,
            time_mask_p=args.aug_time_mask_p,
            time_mask_max_frames=args.aug_time_mask_max_frames,
            freq_mask_p=args.aug_freq_mask_p,
            freq_mask_max_bins=args.aug_freq_mask_max_bins,
        )
        print(f"[info] mel augmentation enabled")

    train_ds = PreprocessedMaestroDataset(
        data_root=args.data_root,
        split="train",
        preproc_root=args.preproc_root,
        segment_frames=args.segment_frames,
        limit_items=args.limit_train_items,
        preproc_cache=args.preproc_cache,
        random_item_sampling=train_random_item_sampling,
        virtual_size=train_virtual_size,
        mel_transform=mel_transform,
    )
    val_ds = PreprocessedMaestroDataset(
        data_root=args.data_root,
        split="validation",
        preproc_root=args.preproc_root,
        segment_frames=args.segment_frames,
        limit_items=args.limit_val_items,
        preproc_cache=args.preproc_cache,
    )

    print(
        f"[info] preprocessed dataset root={args.preproc_root} cache={args.preproc_cache}"
    )
    if args.preproc_cache == "ram":
        train_gib = train_ds.preproc_ram_bytes / (1024**3)
        val_gib = val_ds.preproc_ram_bytes / (1024**3)
        print(
            f"[info] preloaded preproc into RAM: train={train_gib:.2f} GiB "
            f"val={val_gib:.2f} GiB total={train_gib + val_gib:.2f} GiB"
        )
    if train_random_item_sampling:
        print(
            f"[info] train random-item sampling enabled: dataset_items={len(train_ds.items)} "
            f"virtual_samples_per_epoch={len(train_ds)}"
        )

    dl_common = {"pin_memory": True}
    if args.num_workers > 0:
        dl_common.update(
            prefetch_factor=max(2, int(args.prefetch_factor)),
            persistent_workers=bool(args.persistent_workers),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(not train_random_item_sampling),
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

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    conv_channels = tuple(int(x) for x in args.conv_channels.split(","))
    if len(conv_channels) != 3:
        raise SystemExit(
            f"--conv_channels must be 3 comma-separated ints, got {args.conv_channels!r}"
        )
    sample_mel, _ = train_ds[0]
    detected_n_mels = int(sample_mel.shape[-1])
    print(f"[info] detected n_mels={detected_n_mels} freq_pool={args.freq_pool}")
    base_model = OnsetCRNN(
        n_mels=detected_n_mels,
        hidden=args.hidden,
        classes=N_KEYS,
        dropout=args.dropout,
        conv_channels=conv_channels,
        temporal_mode=args.temporal_mode,
        freq_pool=args.freq_pool,
        mel_mean=args.mel_mean,
        mel_std=args.mel_std,
        n_fft=args.preproc_n_fft,
        hop=args.preproc_hop,
        sample_rate=args.preproc_sample_rate,
        repr_type=args.preproc_repr,
    ).to(device)
    model = maybe_compile_model(base_model, args, device)

    optimizer = build_adamw(base_model, args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    run_start = time.perf_counter()
    best_val_f1 = 0.0
    epoch_stats = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        print(f"Epoch {epoch}/{args.epochs}")

        train_start = time.perf_counter()
        train_loss, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            pos_weight=args.pos_weight,
            max_batches=args.max_train_batches,
            profile_batches=(args.profile_batches if args.profile else None),
            profile_out=(args.profile_out if args.profile else None),
            epoch_idx=epoch,
            show_progress=bool(args.tqdm),
            progress_desc=f"train {epoch}/{args.epochs}",
        )
        train_seconds = time.perf_counter() - train_start

        setattr(
            val_loader,
            "_max_batches",
            (
                args.max_val_batches
                if args.max_val_batches is not None
                else args.val_batches
            ),
        )
        val_start = time.perf_counter()
        val_loss, val_p, val_r, val_f1 = evaluate(
            model, val_loader, device, threshold=args.threshold
        )
        val_seconds = time.perf_counter() - val_start
        scheduler.step()

        train_batches = (
            min(len(train_loader), args.max_train_batches)
            if args.max_train_batches is not None
            else len(train_loader)
        )
        val_batches_limit = (
            args.max_val_batches
            if args.max_val_batches is not None
            else args.val_batches
        )
        val_batches = min(len(val_loader), val_batches_limit)

        train_samples = train_batches * args.batch_size
        train_frames = train_samples * args.segment_frames
        train_samples_per_sec = train_samples / max(train_seconds, 1e-8)
        train_frames_per_sec = train_frames / max(train_seconds, 1e-8)
        epoch_seconds = time.perf_counter() - epoch_start

        epoch_stats.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_f1": float(train_f1),
                "val_loss": float(val_loss),
                "val_precision": float(val_p),
                "val_recall": float(val_r),
                "val_f1": float(val_f1),
                "train_seconds": float(train_seconds),
                "val_seconds": float(val_seconds),
                "epoch_seconds": float(epoch_seconds),
                "train_batches": int(train_batches),
                "val_batches": int(val_batches),
                "train_samples": int(train_samples),
                "train_frames": int(train_frames),
                "train_samples_per_sec": float(train_samples_per_sec),
                "train_frames_per_sec": float(train_frames_per_sec),
            }
        )

        print(
            f"  train: loss={train_loss:.4f} f1={train_f1:.3f} "
            f"| val: loss={val_loss:.4f} f1={val_f1:.3f} p={val_p:.3f} r={val_r:.3f} "
            f"| t={epoch_seconds:.2f}s fps={train_frames_per_sec:.0f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(unwrap_compiled_model(model).state_dict(), args.out)
            print(f"  Saved best model to {args.out} (f1={best_val_f1:.3f})")

    total_seconds = time.perf_counter() - run_start

    if args.fuse_eval and os.path.exists(args.out):
        fused_model = OnsetCRNN.from_checkpoint(args.out, map_location="cpu")
        fused_model.eval().fuse_for_eval()
        torch.save(fused_model.state_dict(), args.out)
        print(f"Exported fused eval checkpoint: {args.out}")

    if args.stats_json:
        stats = {
            "run": {
                "total_seconds": float(total_seconds),
                "device": str(device),
                "data_root": args.data_root,
                "preproc_root": args.preproc_root,
                "preproc_cache": args.preproc_cache,
                "train_preproc_ram_gib": float(train_ds.preproc_ram_bytes / (1024**3)),
                "val_preproc_ram_gib": float(val_ds.preproc_ram_bytes / (1024**3)),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "segment_frames": int(args.segment_frames),
                "train_samples_per_epoch": int(args.train_samples_per_epoch)
                if args.train_samples_per_epoch
                else None,
                "random_item_sampling": bool(train_random_item_sampling),
                "num_workers": int(args.num_workers),
                "prefetch_factor": int(args.prefetch_factor),
                "persistent_workers": bool(args.persistent_workers),
                "torch_compile": args.torch_compile,
                "torch_compile_backend": args.torch_compile_backend,
                "torch_compile_dynamic": bool(args.torch_compile_dynamic),
                "torch_compile_fullgraph": bool(args.torch_compile_fullgraph),
                "fused_adamw": bool(args.fused_adamw),
                "allow_tf32": bool(args.allow_tf32),
                "matmul_precision": args.matmul_precision,
                "fuse_eval": bool(args.fuse_eval),
                "max_train_batches": int(args.max_train_batches)
                if args.max_train_batches is not None
                else None,
                "max_val_batches": int(args.max_val_batches)
                if args.max_val_batches is not None
                else None,
                "best_val_f1": float(best_val_f1),
            },
            "epochs": epoch_stats,
        }
        stats_dir = os.path.dirname(args.stats_json)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"Run time: {total_seconds:.2f}s")
    print("Done.")


if __name__ == "__main__":
    main()
