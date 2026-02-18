from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from maestro import N_KEYS, N_MELS


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
    ):
        super().__init__()
        del n_mels  # kept for API compatibility
        self.conv1 = SeparableConv2d(1, 32, (3, 3), (1, 2), (1, 1))
        self.conv2 = SeparableConv2d(32, 64, (3, 3), (1, 2), (1, 1))
        self.conv3 = SeparableConv2d(64, 96, (3, 3), (1, 2), (1, 1))
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=96, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, classes)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = x.mean(dim=-1).transpose(1, 2)
        y, h = self.gru(x, h)
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


