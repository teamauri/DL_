"""Phone-occlusion demo: two models + an occlusion verdict.

Runs TWO off-the-shelf models and judges whether the phone is covering text:

  Model A  model/textseg-SegN.pt      -> TextRegion / TextLine  (text mask)
  Model B  model/yolov8n-seg.pt       -> COCO 'cell phone'      (phone mask)

This mirrors bookseg's `hand ∩ text` occlusion logic, with phone as the occluder.

WHY THE JUDGEMENT IS NOT A PLAIN IoU
  Segmentation only sees VISIBLE pixels. Text under the phone is not detected,
  so phone_mask ∩ text_mask is near-empty exactly when occlusion happened.
  (Same amodal limit as the hand case.) So we judge by three indirect signals:

    1. 接触   phone dilated ∩ text   -- phone touches text at all
    2. 包围   how many of the 4 sides of the phone bbox have text around them
    3. 截断   text lines whose endpoint stops at the phone boundary
    4. 覆盖   phone area inside a TextRegion block -- estimated hidden text

  (Plus the box-level IoU, which is what the very first request asked for.)

Usage:
    python phone_occlusion_demo.py --source data_9.14/phone_on_books.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PHONE_CLASS = 67
DILATE_PX = 25          # 'touching' tolerance
SIDE_MARGIN_PX = 90     # ring width used for the 4-side surround test
IOU_THRESH = 0.5


def iou_box(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def masks_to_binary(res, h, w, wanted=None, name_map=None):
    """Return (merged_uint8_mask, [instances]) resized to (w, h)."""
    if res.masks is None or len(res.masks) == 0:
        return np.zeros((h, w), np.uint8), []
    merged = np.zeros((h, w), np.uint8)
    instances = []
    all_boxes = res.boxes.xyxy.cpu().numpy()
    all_cls = res.boxes.cls.cpu().numpy().astype(int)
    all_conf = res.boxes.conf.cpu().numpy()
    for i, m in enumerate(res.masks.data.cpu().numpy()):
        cls = int(all_cls[i])
        if wanted is not None and cls not in wanted:
            continue
        conf = float(all_conf[i])
        mm = m.astype(np.float32)
        if mm.shape[:2] != (h, w):
            mm = cv2.resize(mm, (w, h))
        binm = (mm > 0.5).astype(np.uint8)
        merged = np.maximum(merged, binm)
        instances.append({"mask": binm, "cls": cls, "conf": conf,
                          "box": [int(v) for v in all_boxes[i].tolist()]})
    return merged, instances


def sides_with_text(text, box, margin):
    h, w = text.shape
    x1, y1, x2, y2 = box
    ys, xs = np.where(text > 0)
    if len(xs) == 0:
        return [], 0
    tx1, ty1, tx2, ty2 = xs.min(), ys.min(), xs.max(), ys.max()
    sides = []
    if ty1 < y1 and ty2 >= y1 - margin and not (tx2 < x1 or tx1 > x2):
        sides.append("上")
    if ty2 > y2 and ty1 <= y2 + margin and not (tx2 < x1 or tx1 > x2):
        sides.append("下")
    if tx1 < x1 and tx2 >= x1 - margin and not (ty2 < y1 or ty1 > y2):
        sides.append("左")
    if tx2 > x2 and tx1 <= x2 + margin and not (ty2 < y1 or ty1 > y2):
        sides.append("右")
    return sides, len(sides)


def lines_cut_by(instances, phone_box, margin):
    """Text lines whose bbox edge comes within `margin` of the phone box."""
    px1, py1, px2, py2 = phone_box
    cut = []
    for ins in instances:
        x1, y1, x2, y2 = ins["box"]
        near_x = (x2 >= px1 - margin) and (x1 <= px2 + margin)
        near_y = (y2 >= py1 - margin) and (y1 <= py2 + margin)
        if near_x and near_y:
            # horizontally adjacent: line stops at the phone's left/right edge
            if (x2 < px1 and px1 - x2 <= margin) or (x1 > px2 and x1 - px2 <= margin):
                cut.append(ins)
            # vertically overlapping and horizontally inside -> spans behind it
            elif x1 >= px1 - margin and x2 <= px2 + margin:
                cut.append(ins)
    return cut


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data_9.14/phone_on_books.jpg")
    p.add_argument("--text-model", default="model/textseg-SegN.pt")
    p.add_argument("--obj-model", default="model/yolov8n-seg.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="runs/phone_occ")
    args = p.parse_args()

    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(args.source)
    h, w = img.shape[:2]

    text_model = YOLO(args.text_model)
    obj_model = YOLO(args.obj_model)

    # --- Model A: text -------------------------------------------------------
    rt = text_model.predict(args.source, conf=args.conf, verbose=False)[0]
    text_mask, text_insts = masks_to_binary(rt, h, w)
    region_mask, region_insts = masks_to_binary(rt, h, w, wanted={0})

    # --- Model B: phone (+ book, for the box-IoU the first version wanted) ---
    ro = obj_model.predict(args.source, conf=0.10, verbose=False)[0]
    phone_mask, phone_insts = masks_to_binary(ro, h, w, wanted={PHONE_CLASS})
    _, book_insts = masks_to_binary(ro, h, w, wanted={73})

    print("=" * 62)
    print(f"图: {args.source}  ({w}x{h})")
    print(f"模型A 文字: {len(text_insts)} 个实例 (mask 覆盖 {text_mask.sum()/text_mask.size*100:.2f}%)")
    print(f"模型B 手机: {len(phone_insts)} 个实例"
          + (f" conf={max(i['conf'] for i in phone_insts):.2f}" if phone_insts else ""))
    print("=" * 62)

    if not phone_insts:
        print("\n未检测到手机 -> 无遮挡判定")
        return
    if text_mask.sum() == 0:
        print("\n未检测到文字 -> 无遮挡判定")
        return

    ph = max(phone_insts, key=lambda i: i["conf"])
    pbox = ph["box"]

    # 判据 1: 接触
    k = np.ones((DILATE_PX * 2 + 1, DILATE_PX * 2 + 1), np.uint8)
    ph_dil = cv2.dilate(ph["mask"], k)
    touch = int((ph_dil & text_mask).sum())
    touching = touch > 0

    # 判据 2: 包围边数
    sides, n_side = sides_with_text(text_mask, pbox, SIDE_MARGIN_PX)

    # 判据 3: 截断的文字行
    cut = lines_cut_by(text_insts, pbox, SIDE_MARGIN_PX)

    # 判据 4: 落在 TextRegion 块内的手机面积 (估计被盖住的文字)
    covered = int((ph["mask"] & region_mask).sum())
    phone_area = int(ph["mask"].sum())
    cover_ratio = covered / phone_area if phone_area else 0.0

    # 附: 框级 IoU (最初需求)
    book_iou = max((iou_box(pbox, b["box"]) for b in book_insts), default=0.0)

    print(f"\n手机框: {pbox}   书框数: {len(book_insts)}   框级IoU(手机-书): {book_iou:.3f}"
          + ("  (>0.5)" if book_iou > IOU_THRESH else ""))

    print(f"\n判据1 接触   : {'是' if touching else '否'}  (膨胀{DILATE_PX}px 后与文字重叠 {touch} px)")
    print(f"判据2 包围   : {n_side}/4 边有文字  {sides}")
    print(f"判据3 截断行 : {len(cut)} 行文字终止于手機边界")
    print(f"判据4 块内覆盖: 手机 {cover_ratio*100:.1f}% 面积落在 TextRegion 块内 "
          f"({covered}/{phone_area} px)")

    # --- 综合判定 ---
    print("\n" + "=" * 62)
    if not touching:
        verdict = "无遮挡 (手机与文字不相邻)"
    elif n_side >= 3 or len(cut) >= 3 or cover_ratio > 0.6:
        verdict = "遮挡 (手机压在文字上)"
    elif n_side >= 2 or len(cut) >= 1 or cover_ratio > 0.3:
        verdict = "疑似遮挡 (部分文字被盖)"
    else:
        verdict = "边缘接触 (紧邻但未明显盖字)"
    print(f"判定: {verdict}")
    print("=" * 62)

    # --- 可视化 ---
    vis = img.copy()
    tm = text_mask > 0
    pm = ph["mask"] > 0
    vis[tm] = (vis[tm] * 0.45 + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
    vis[pm] = (vis[pm] * 0.45 + np.array([0, 0, 255]) * 0.55).astype(np.uint8)
    adj = (ph_dil > 0) & tm & ~pm
    vis[adj] = (vis[adj] * 0.3 + np.array([0, 255, 255]) * 0.7).astype(np.uint8)

    cv2.rectangle(vis, (pbox[0], pbox[1]), (pbox[2], pbox[3]), (0, 0, 255), 4)
    for ins in cut[:20]:
        x1, y1, x2, y2 = ins["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).stem
    cv2.imwrite(str(out_dir / f"{stem}_occ.jpg"), vis)
    print(f"\n可视化 -> {out_dir}/{stem}_occ.jpg")
    print("  绿=文字  红=手机  黄=手机紧邻的文字  橙框=被截断的文字行")


if __name__ == "__main__":
    main()
