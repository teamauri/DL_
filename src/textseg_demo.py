"""Off-the-shelf text-region segmentation demo.

Model: LaMOP/Yolo-Seg-TextRegion-TextLine-Typed (YOLOv8-seg, CC0-1.0)
  2 classes: TextRegion / TextLine
  https://huggingface.co/LaMOP/Yolo-Seg-TextRegion-TextLine-Typed

Why this one: it is the closest public analogue of bookseg's `text_region`
channel -- a document-domain, text-specific segmenter. Ultralytics loads it
directly, so no new dependencies.

The key output is the MERGED binary text mask (union of all text instances),
which is what bookseg's text_region channel would emit.

Usage:
    python textseg_demo.py --source data_9.14/textseg_sample.jpg
    python textseg_demo.py --source data_9.14/books_pile.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "model/textseg-SegN.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data_9.14/textseg_sample.jpg")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="runs/textseg")
    args = p.parse_args()

    model = YOLO(args.model)
    names = model.names
    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(args.source)
    h, w = img.shape[:2]

    print("=" * 58)
    print(f"模型: {Path(args.model).name}")
    print(f"类别({len(names)}): " + ", ".join(f"[{i}] {n}" for i, n in names.items()))
    print("=" * 58)

    r = model.predict(args.source, conf=args.conf, verbose=False)[0]

    if r.masks is None:
        print("没检测到任何文字区域")
        return

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    # per-class instance counts
    from collections import Counter
    cnt = Counter(names[c] for c in classes)
    print(f"\n检测到实例: " + ", ".join(f"{k}×{v}" for k, v in cnt.items()))

    # 关键输出：所有文字实例合并成一张二值 mask
    # (这就是 bookseg `text_region` 通道的输出形态)
    merged = np.zeros((h, w), np.uint8)
    for m in masks:
        mm = m.astype(np.float32)
        if mm.shape[:2] != (h, w):
            mm = cv2.resize(mm, (w, h))
        merged = np.maximum(merged, (mm > 0.5).astype(np.uint8))

    cover = merged.sum() / merged.size
    print(f"\n>>> 合并后 text_region mask 覆盖率: {cover*100:.2f}%")

    # 可视化 1: 实例彩色叠加
    vis = img.copy()
    colors = {0: (0, 255, 0), 1: (255, 128, 0)}   # TextRegion 绿 / TextLine 蓝
    for m, c in zip(masks, classes):
        mm = m.astype(np.float32)
        if mm.shape[:2] != (h, w):
            mm = cv2.resize(mm, (w, h))
        sel = mm > 0.5
        col = np.array(colors.get(c, (0, 0, 255)), dtype=np.uint8)
        vis[sel] = (vis[sel] * 0.4 + col * 0.6).astype(np.uint8)

    # 可视化 2: 纯二值 mask (text_region 通道的样子)
    mask_vis = np.zeros_like(img)
    mask_vis[merged > 0] = (255, 255, 255)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem
    cv2.imwrite(str(out_dir / f"{stem}_instances.jpg"), vis)
    cv2.imwrite(str(out_dir / f"{stem}_textmask.jpg"), mask_vis)
    print(f"\n输出 -> {out_dir}/{stem}_instances.jpg   (实例: 绿=TextRegion 蓝=TextLine)")
    print(f"        {out_dir}/{stem}_textmask.jpg   (合并二值 mask)")


if __name__ == "__main__":
    main()
