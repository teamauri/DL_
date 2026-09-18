#!/usr/bin/env python3
"""Demo: 3.pt 检测字符块 -> 页面指纹 -> 以第一张为基准判断是否翻页。

流程:
  1. 用 3.pt 检测每张图里的 "text" 字符块 (整张图当作 ROI)。
  2. 每个字符块提取三个特征: 相对位置 / 面积 / 长宽比, 组成页面指纹。
  3. 第一张作为基准页, 后续每张与基准页算代价。
  4. 代价超过阈值 -> 判定为 "翻页了", 否则 "同一页"。
  5. 在图上画出字符块框, 并用 putText 写上结果。
"""
import argparse
import glob
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ultralytics"))

import cv2
import numpy as np
from ultralytics import YOLO


MODEL = "/Users/marking/Downloads/3.pt"
IMAGES = "/Users/marking/Downloads/textbite-dataset/images"
OUT = "/Users/marking/Dec_book/DL_/runs/page_turn_demo"


def detect_blocks(model, img_path, conf=0.25):
    """Run 3.pt and return text blocks as (cx, cy, w, h, x1, y1, x2, y2)."""
    r = model.predict(img_path, imgsz=640, conf=conf, device="cpu", verbose=False)[0]
    H, W = r.orig_shape[:2]
    blocks = []
    for b in r.boxes:
        if int(b.cls[0]) != 1:  # keep only "text" class
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        blocks.append((cx, cy, w, h, x1, y1, x2, y2))
    return blocks


def descriptor(blocks):
    """Page fingerprint: sorted (cx, cy, area, aspect_ratio)."""
    feats = []
    for cx, cy, w, h, *_ in blocks:
        area = w * h
        ar = w / max(h, 1e-6)
        feats.append((cx, cy, area, ar))
    feats.sort(key=lambda x: (round(x[1], 3), round(x[0], 3)))
    return feats


def page_cost(a, b, w_pos=0.5, w_area=0.25, w_ar=0.25):
    """Cost between two pages."""
    da, db = descriptor(a), descriptor(b)
    if not da and not db:
        return 0.0
    n_pen = abs(len(da) - len(db)) / max(len(da), len(db), 1)
    total, matched, used = 0.0, 0, set()
    for fa in da:
        best_j, best_d = None, None
        for j, fb in enumerate(db):
            if j in used:
                continue
            d = float(np.hypot(fa[0] - fb[0], fa[1] - fb[1]))
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            fb = db[best_j]
            total += w_pos * best_d + w_area * abs(fa[2] - fb[2]) + w_ar * abs(fa[3] - fb[3])
            matched += 1
    return total / max(matched, 1) + n_pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    model = YOLO(MODEL)
    print("classes:", model.names)

    imgs = sorted(glob.glob(os.path.join(IMAGES, "*")))
    rng = random.Random(args.seed)
    imgs = rng.sample(imgs, args.n)

    os.makedirs(OUT, exist_ok=True)

    detected = [(p, detect_blocks(model, p, args.conf)) for p in imgs]
    baseline = detected[0][1]
    print(f"\nbaseline: {os.path.basename(imgs[0])}  #blocks={len(baseline)}\n")

    for i, (p, blocks) in enumerate(detected):
        cost = 0.0 if i == 0 else page_cost(baseline, blocks)
        turned = i > 0 and cost > args.threshold
        label = "基准页" if i == 0 else ("翻页了" if turned else "同一页")

        img = cv2.imread(p)
        for cx, cy, w, h, x1, y1, x2, y2 in blocks:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        color = (0, 0, 255) if turned else (0, 200, 0)
        cv2.putText(img, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)
        out = os.path.join(OUT, f"{i:02d}.jpg")
        cv2.imwrite(out, img)

        print(f"{i}: {os.path.basename(p):32s} #blocks={len(blocks):3d} cost={cost:.3f} -> {label}")

    print(f"\nannotated images saved to: {OUT}")


if __name__ == "__main__":
    main()
