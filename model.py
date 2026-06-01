from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from maestro import N_KEYS, N_MELS


DEFAULT_CONV_CHANNELS: Tuple[int, int, int] = (32, 64, 96)


def _freq_after_convs(n_mels: int, n_blocks: int = 3) -> int:
    """Frequency dim after n_blocks of (k=3, s=2, p=1) convs."""
    f = n_mels
    for _ in range(n_blocks):
        f = (f - 1) // 2 + 1
    return f


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
            in_ch,
            in_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))

    def fuse_for_eval(self) -> "SeparableConv2d":
        if self.training:
            raise RuntimeError("fuse_for_eval() requires eval() mode")
        if isinstance(self.bn, nn.Identity):
            return self
        self.pw = fuse_conv_bn_eval(self.pw, self.bn)
        self.bn = nn.Identity()
        return self


class OnsetCRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden: int = 192,
        classes: int = N_KEYS,
        dropout: float = 0.1,
        conv_channels: Tuple[int, int, int] = DEFAULT_CONV_CHANNELS,
        temporal_mode: str = "gru",
        freq_pool: str = "mean",
        mel_mean: Optional[float] = None,
        mel_std: Optional[float] = None,
        n_fft: Optional[int] = None,
        hop: Optional[int] = None,
        sample_rate: Optional[int] = None,
        repr_type: Optional[str] = None,
    ):
        super().__init__()
        if temporal_mode not in ("gru", "none"):
            raise ValueError(f"temporal_mode must be 'gru' or 'none', got {temporal_mode!r}")
        if freq_pool not in ("mean", "flatten"):
            raise ValueError(f"freq_pool must be 'mean' or 'flatten', got {freq_pool!r}")
        c1, c2, c3 = conv_channels
        self.conv_channels = (c1, c2, c3)
        self.temporal_mode = temporal_mode
        self.freq_pool = freq_pool
        self.n_mels = n_mels
        self.classes = classes
        self.conv1 = SeparableConv2d(1, c1, (3, 3), (1, 2), (1, 1))
        self.conv2 = SeparableConv2d(c1, c2, (3, 3), (1, 2), (1, 1))
        self.conv3 = SeparableConv2d(c2, c3, (3, 3), (1, 2), (1, 1))
        self.dropout = nn.Dropout(dropout)
        if freq_pool == "flatten":
            self.freq_dim = _freq_after_convs(n_mels)
            feat_dim = c3 * self.freq_dim
        else:  # "mean"
            self.freq_dim = 1
            feat_dim = c3
        if temporal_mode == "gru":
            self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden, num_layers=1, batch_first=True)
            head_in = hidden
            self.hidden = hidden
        else:  # "none"
            self.gru = None
            head_in = feat_dim
            self.hidden = 0
        self.head = nn.Linear(head_in, classes)
        # Persisted metadata so from_checkpoint can reconstruct the architecture.
        self.register_buffer("_meta_freq_pool", torch.tensor(0 if freq_pool == "mean" else 1))
        self.register_buffer("_meta_n_mels", torch.tensor(int(n_mels)))
        # NaN sentinels = "no per-rep stats stored; inference should use MelSpecProcessor defaults".
        self.register_buffer(
            "_meta_mel_mean",
            torch.tensor(float("nan") if mel_mean is None else float(mel_mean)),
        )
        self.register_buffer(
            "_meta_mel_std",
            torch.tensor(float("nan") if mel_std is None else float(mel_std)),
        )
        # Preprocessing config (0 sentinel = "unset; inference falls back to maestro.py defaults").
        # repr_type encoded as int: 0=mel, 1=logfreq, -1=unset.
        self.register_buffer("_meta_n_fft",       torch.tensor(0 if n_fft is None else int(n_fft)))
        self.register_buffer("_meta_hop",         torch.tensor(0 if hop is None else int(hop)))
        self.register_buffer("_meta_sample_rate", torch.tensor(0 if sample_rate is None else int(sample_rate)))
        _repr_map = {"mel": 0, "logfreq": 1}
        self.register_buffer(
            "_meta_repr_type",
            torch.tensor(-1 if repr_type is None else _repr_map[repr_type]),
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        map_location=None,
        dropout: float = 0.0,
    ) -> "OnsetCRNN":
        """Instantiate a model whose shape matches the saved state_dict, then load."""
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state and not any(
            k.startswith(("conv1.", "conv2.", "conv3.", "gru.", "head."))
            for k in state.keys()
        ):
            state = state["state_dict"]
        c1 = int(state["conv1.pw.weight"].shape[0])
        c2 = int(state["conv2.pw.weight"].shape[0])
        c3 = int(state["conv3.pw.weight"].shape[0])
        classes = int(state["head.weight"].shape[0])
        if "gru.weight_ih_l0" in state:
            hidden = int(state["gru.weight_ih_l0"].shape[0]) // 3
            temporal_mode = "gru"
        else:
            hidden = 0
            temporal_mode = "none"
        # Recover freq_pool + n_mels from persisted metadata (legacy = mean/64).
        if "_meta_freq_pool" in state:
            freq_pool = "mean" if int(state["_meta_freq_pool"]) == 0 else "flatten"
            n_mels = int(state["_meta_n_mels"])
        else:
            freq_pool = "mean"
            n_mels = N_MELS
        mel_mean = mel_std = None
        if "_meta_mel_mean" in state:
            v = float(state["_meta_mel_mean"])
            if v == v:  # not NaN
                mel_mean = v
        if "_meta_mel_std" in state:
            v = float(state["_meta_mel_std"])
            if v == v:
                mel_std = v
        n_fft = int(state["_meta_n_fft"]) if "_meta_n_fft" in state else 0
        hop = int(state["_meta_hop"]) if "_meta_hop" in state else 0
        sample_rate = int(state["_meta_sample_rate"]) if "_meta_sample_rate" in state else 0
        repr_code = int(state["_meta_repr_type"]) if "_meta_repr_type" in state else -1
        repr_type = {0: "mel", 1: "logfreq"}.get(repr_code)
        model = cls(
            n_mels=n_mels,
            hidden=hidden,
            classes=classes,
            dropout=dropout,
            conv_channels=(c1, c2, c3),
            temporal_mode=temporal_mode,
            freq_pool=freq_pool,
            mel_mean=mel_mean,
            mel_std=mel_std,
            n_fft=n_fft if n_fft > 0 else None,
            hop=hop if hop > 0 else None,
            sample_rate=sample_rate if sample_rate > 0 else None,
            repr_type=repr_type,
        )
        model.load_state_dict(state, strict=False)
        return model

    def get_mel_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns (mel_mean, mel_std) stored in the checkpoint, or (None, None) if unset."""
        mean = float(self._meta_mel_mean)
        std = float(self._meta_mel_std)
        return (mean if mean == mean else None, std if std == std else None)

    def get_preproc_config(self) -> dict:
        """Returns {n_fft, hop, sample_rate, repr_type} from the checkpoint, or {} if unset."""
        cfg = {}
        nf = int(self._meta_n_fft)
        if nf > 0:
            cfg["n_fft"] = nf
        h = int(self._meta_hop)
        if h > 0:
            cfg["hop"] = h
        sr = int(self._meta_sample_rate)
        if sr > 0:
            cfg["sample_rate"] = sr
        rc = int(self._meta_repr_type)
        if rc >= 0:
            cfg["repr_type"] = {0: "mel", 1: "logfreq"}[rc]
        return cfg

    def _conv_stack(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)  # [B, c3, T, F']
        if self.freq_pool == "flatten":
            B, C, T, Fp = x.shape
            return x.permute(0, 2, 1, 3).reshape(B, T, C * Fp).contiguous()  # [B, T, c3*F']
        return x.mean(dim=-1).transpose(1, 2).contiguous()  # [B, T, c3]

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self._conv_stack(x)
        if self.gru is not None:
            y, h = self.gru(x, h)
        else:
            y = x
        logits = self.head(y)
        return logits, h

    def forward_with_context(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        left_trim: int = 0,
        right_trim: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run convs on the full input, trim conv output, then run temporal layer on the trim.

        Useful for streaming: pass [left_context, chunk, right_context] in `x` so
        convs see correct neighbours, then trim those context frames so the
        temporal state advances by exactly `chunk` frames. Avoids conv-padding
        artifacts at chunk boundaries.
        """
        x = self._conv_stack(x)
        T = x.shape[1]
        end_idx = T - right_trim if right_trim > 0 else T
        x = x[:, left_trim:end_idx]
        if self.gru is not None:
            y, h = self.gru(x, h)
        else:
            y = x
        logits = self.head(y)
        return logits, h

    def fuse_for_eval(self) -> "OnsetCRNN":
        if self.training:
            raise RuntimeError("fuse_for_eval() requires eval() mode")
        self.conv1.fuse_for_eval()
        self.conv2.fuse_for_eval()
        self.conv3.fuse_for_eval()
        return self


def maybe_compile_model(model: nn.Module, args, device: torch.device) -> nn.Module:
    if args.torch_compile == "none":
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    compile_kwargs = {
        "backend": args.torch_compile_backend,
        "dynamic": bool(args.torch_compile_dynamic),
        "fullgraph": bool(args.torch_compile_fullgraph),
    }
    if args.torch_compile != "default":
        compile_kwargs["mode"] = args.torch_compile
    compiled = torch.compile(model, **compile_kwargs)
    print(
        "[info] torch.compile enabled: "
        f"mode={args.torch_compile} backend={args.torch_compile_backend} "
        f"dynamic={int(args.torch_compile_dynamic)} fullgraph={int(args.torch_compile_fullgraph)}"
    )
    if device.type != "cuda":
        print("[warn] torch.compile requested on non-CUDA device")
    return compiled


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


