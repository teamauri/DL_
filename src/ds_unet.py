#!/usr/bin/env python3
"""DS-UNet -- depthwise-separable U-Net for bookseg (4-channel output).

Target: RKNN int8 on-NPU inference (see bookseg_pipeline.py, stage 3).

Locked design decisions
-----------------------
1. **Quantisation friendliness** -- ReLU (not SiLU), nearest-neighbour
   upsampling at a CONSTANT scale (`scale_factor=2`, never `size=`), so the
   exported graph contains no Shape/Slice-driven Resize; normal convs, no
   transposed conv, BatchNorm kept in the graph (RKNN folds BN into the
   preceding conv), plain concat skips, no attention / no dynamic shapes.

2. **4 independent sigmoid channels** -- paper / text / hand / gutter. These
   OVERLAP (text sits on paper, a hand can cover paper), so this is a
   multi-label problem, not mutually-exclusive classes:
       * train with BCEWithLogitsLoss on the raw logits;
       * apply sigmoid only at inference;
       * NEVER softmax across the 4 channels.

3. **The model returns RAW LOGITS.** sigmoid lives outside the model so the
   exported ONNX / RKNN graph stays quantisation friendly. Use `probs()` for
   inference and `bce_loss()` for training.

4. `width` is the base channel count. width-32 is the bookseg variant; the
   same file builds width-16/24/48/64.

Output is at the input resolution, so any size divisible by 32 works
(320x320 is a good NPU default; 640x640 also builds).

Usage
-----
    python src/ds_unet.py                       # build width-32, self-check
    python src/ds_unet.py --width 16            # other variants
    python src/ds_unet.py --export ds_unet32.onnx
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4
CHANNEL_NAMES = ("paper", "text", "hand", "gutter")


def autopad(k: int) -> int:
    """'same' padding for an odd kernel."""
    return k // 2


class DSConv(nn.Module):
    """Depthwise separable convolution: DW kxk -> BN -> ReLU -> PW 1x1 -> BN -> ReLU."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        # ReLU, not SiLU: keeps activations non-negative and bounded, which is
        # what RKNN int8 quantisation wants. BN above is folded into the conv.
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class DSBlock(nn.Module):
    """Two DSConv stacked; residual add when the shape is unchanged."""

    def __init__(self, c1: int, c2: int, s: int = 1):
        super().__init__()
        self.cv1 = DSConv(c1, c2, 3, s)
        self.cv2 = DSConv(c2, c2, 3, 1)
        self.add = c1 == c2 and s == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class Up(nn.Module):
    """Nearest x2 upsample -> concat skip -> DSBlock (never a transposed conv)."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.block = DSBlock(c1, c2, s=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # scale_factor=2, NOT size=skip.shape[-2:]: the latter exports as a
        # Shape -> Slice -> Concat -> Resize subgraph, which the RKNN ONNX
        # parser handles badly. A constant scale maps to a plain upsample.
        # Requires an input size divisible by 32 so the skips line up exactly.
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.block(torch.cat([x, skip], dim=1))


class DSUNet(nn.Module):
    """4-level depthwise-separable U-Net. Returns `num_classes` RAW LOGITS."""

    def __init__(self, width: int = 32, num_classes: int = NUM_CLASSES, in_ch: int = 3):
        super().__init__()
        w = width
        self.width = width
        self.num_classes = num_classes

        # encoder: /2 -> /32
        self.stem = DSBlock(in_ch, w, s=2)          # /2
        self.down1 = DSBlock(w, 2 * w, s=2)         # /4
        self.down2 = DSBlock(2 * w, 4 * w, s=2)     # /8
        self.down3 = DSBlock(4 * w, 8 * w, s=2)     # /16
        self.down4 = DSBlock(8 * w, 8 * w, s=2)     # /32

        # decoder: /32 -> /2, each step upsamples and concats its skip
        self.up1 = Up(8 * w + 8 * w, 4 * w)
        self.up2 = Up(4 * w + 4 * w, 2 * w)
        self.up3 = Up(2 * w + 2 * w, w)
        self.up4 = Up(w + w, w)

        # head: back to input resolution, then 1x1 -> the 4 channels, NO sigmoid
        self.head = nn.Conv2d(w, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        b = self.down4(s3)

        x = self.up1(b, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        # constant x2 back to input resolution, same reason as in Up.
        # NOTE: the input size must be divisible by 32 (e.g. 320, 640) so every
        # skip connection and this final x2 line up exactly.
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.head(x)  # raw logits -- sigmoid is applied by the caller


# --- helpers the training / inference code should use ------------------------


def build(width: int = 32, num_classes: int = NUM_CLASSES) -> DSUNet:
    """Convenience constructor. width-32 = the bookseg variant."""
    return DSUNet(width=width, num_classes=num_classes)


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Training loss: BCEWithLogits on the raw logits (4 independent channels).

    `targets` must have the same shape as `logits` (B, 4, H, W), values 0/1.
    Do not softmax the channels -- they overlap on purpose.
    """
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def probs(logits: torch.Tensor) -> torch.Tensor:
    """Inference: per-channel sigmoid -> (B, 4, H, W) in [0, 1]."""
    return torch.sigmoid(logits)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_onnx(model: DSUNet, path: str, size: int = 320, opset: int = 12) -> None:
    """Export raw-logit ONNX for the RKNN int8 toolchain (sigmoid stays outside)."""
    model = model.eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, size, size),
        path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
        dynamic_axes=None,  # static shape: NPU-friendly
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32, help="base channel count")
    ap.add_argument("--size", type=int, default=320, help="square input size for the check")
    ap.add_argument("--export", default="", help="export raw-logit ONNX to this path")
    args = ap.parse_args()

    model = build(args.width).eval()
    n = n_params(model)
    x = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        logits = model(x)
        p = probs(logits)

    print("=" * 62)
    print(f"DS-UNet width-{args.width}  ->  {n/1e6:.3f} M params")
    print(f"bookseg budget: <= 1.5 M  ({(n/1.5e6)*100:.1f}% used)  |  baseline 0.24 M")
    print(f"input   : 1x3x{args.size}x{args.size}")
    print(f"logits  : {tuple(logits.shape)}  (raw, no sigmoid in the graph)")
    print(f"probs   : [{p.min():.3f}, {p.max():.3f}]  channels = {', '.join(CHANNEL_NAMES)}")
    print("=" * 62)

    if args.export:
        export_onnx(model, args.export, size=args.size)
        print("exported raw-logit ONNX ->", args.export)


if __name__ == "__main__":
    main()
