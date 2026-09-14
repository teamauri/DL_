"""Off-the-shelf semantic-segmentation demo (torchvision DeepLabV3).

Purpose: let you SEE what a real semantic segmentation model does and WHAT
CLASSES it has -- then judge how far that is from the 4 channels bookseg needs
(paper / text / hand / gutter).

This is NOT DS-UNet. DS-UNet is bookseg's own architecture and has no public
weights; no public model outputs {paper, text, hand, gutter}.

Usage:
    python seg_demo.py --source data_9.14/books_pile.jpg
    python seg_demo.py --source img.jpg --model deeplabv3_resnet50
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large, deeplabv3_resnet50,
    DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights,
)

# VOC-style palette for the 21 COCO/VOC classes
PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def build(model_name):
    if model_name == "deeplabv3_mobilenet_v3_large":
        w = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        m = deeplabv3_mobilenet_v3_large(weights=w)
    else:
        w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        m = deeplabv3_resnet50(weights=w)
    return m.eval(), w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data_9.14/books_pile.jpg")
    p.add_argument("--model", default="deeplabv3_mobilenet_v3_large",
                   choices=["deeplabv3_mobilenet_v3_large", "deeplabv3_resnet50"])
    p.add_argument("--out", default="runs/seg")
    args = p.parse_args()

    model, weights = build(args.model)
    cats = weights.meta["categories"]
    n_param = sum(x.numel() for x in model.parameters())

    print("=" * 62)
    print(f"模型: {args.model}")
    print(f"参数量: {n_param/1e6:.2f}M   (bookseg 预算上限 1.5M, 基线 0.24M)")
    print(f"类别数: {len(cats)}")
    print("=" * 62)
    print("\n全部类别:")
    for i, c in enumerate(cats):
        print(f"  [{i:2d}] {c}")
    print("\n有 'book' 类吗?", "book" in cats)

    img_bgr = cv2.imread(args.source)
    if img_bgr is None:
        raise FileNotFoundError(args.source)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        inp = weights.transforms()(torch.from_numpy(img_rgb).permute(2, 0, 1))
        out = model(inp.unsqueeze(0))["out"][0]
        prob = out.softmax(0)
        conf, pred = prob.max(0)
        pred = pred.cpu().numpy()
        conf = conf.cpu().numpy()

    pred_full = cv2.resize(pred.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    print("\n本图各像素占比 (只列 >0.1%):")
    total = pred_full.size
    for i in np.bincount(pred_full.flatten(), minlength=len(cats)).argsort()[::-1]:
        share = (pred_full == i).sum() / total
        if share > 0.001:
            print(f"  {cats[i]:14s} {share*100:6.2f}%")

    color = PALETTE[pred_full % len(PALETTE)]
    overlay = cv2.addWeighted(img_bgr, 0.45, color, 0.55, 0)
    mask_only = color

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem
    cv2.imwrite(str(out_dir / f"{stem}_seg_overlay.jpg"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_seg_mask.jpg"), mask_only)
    print(f"\n输出 -> {out_dir}/{stem}_seg_overlay.jpg")
    print(f"        {out_dir}/{stem}_seg_mask.jpg")


if __name__ == "__main__":
    main()
