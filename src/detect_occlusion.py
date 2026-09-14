"""YOLOv8 daily-necessity occlusion detection (config-driven).

Reads a YAML config that switches between image / video / camera modes.
Detects daily-necessity objects, computes pairwise IoU, and flags the
configured target class (default: book) as "遮挡" when overlap > threshold.

Usage:
    python detect_occlusion.py                                # config/occlusion.yaml
    python detect_occlusion.py --config config/occlusion.yaml
    python detect_occlusion.py --mode video --source input.mp4
    python detect_occlusion.py --mode camera --source 0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# BGR colors
COLOR_NORMAL = (255, 255, 0)   # cyan   - ordinary daily object
COLOR_TARGET = (0, 255, 0)     # green  - target class (e.g. book), not occluded
COLOR_OCCLUDED = (0, 0, 255)   # red    - target class, occluded

_CJK_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def find_cjk_font():
    """Return the first CJK-capable font path available on this system."""
    for path in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iou(a, b):
    """Intersection-over-union of two boxes given as [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_label(draw, xy, text, color_bgr, font):
    """Draw CJK text with a dark background strip via PIL."""
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                   fill=(30, 30, 30))
    draw.text((x, y), text, font=font, fill=tuple(int(c) for c in reversed(color_bgr)))


def annotate(img, boxes, confs, classes, names, cfg, daily_ids):
    """Draw boxes + labels and flag occluded target objects.

    Returns (img, occluded_count, events). `events` lists human-readable
    descriptions of each occlusion found.
    """
    target_names = set(cfg["occlusion"].get("target", []))
    iou_thr = float(cfg["occlusion"].get("iou_threshold", 0.5))

    n = len(boxes)
    occluded = [False] * n
    events = []

    # Flag a target class when it overlaps any other detection beyond threshold.
    for i in range(n):
        name = names[classes[i]]
        if classes[i] not in daily_ids or name not in target_names:
            continue
        best = -1.0
        occluder = -1
        for j in range(n):
            if i == j:
                continue
            s = iou(boxes[i], boxes[j])
            if s > best:
                best = s
                occluder = j
        if occluder >= 0 and best > iou_thr:
            occluded[i] = True
            events.append(f"遮挡: {name} <- {names[classes[occluder]]} (IoU={best:.2f})")

    # Draw rectangles (OpenCV), then text (PIL for CJK, OpenCV fallback).
    for i in range(n):
        if classes[i] not in daily_ids:
            continue
        x1, y1, x2, y2 = boxes[i].astype(int)
        name = names[classes[i]]
        if occluded[i]:
            color = COLOR_OCCLUDED
        elif name in target_names:
            color = COLOR_TARGET
        else:
            color = COLOR_NORMAL
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    font_path = find_cjk_font()
    if font_path:
        font = ImageFont.truetype(font_path, 24)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for i in range(n):
            if classes[i] not in daily_ids:
                continue
            x1, y1, x2, y2 = boxes[i].astype(int)
            name = names[classes[i]]
            color = COLOR_OCCLUDED if occluded[i] else (
                COLOR_TARGET if name in target_names else COLOR_NORMAL)
            label = f"{name} {confs[i]:.2f}"
            if occluded[i]:
                label += " 遮挡"
            draw_label(draw, (x1, max(0, y1 - 26)), label, color, font)
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        for i in range(n):
            if classes[i] not in daily_ids:
                continue
            x1, y1, x2, y2 = boxes[i].astype(int)
            name = names[classes[i]]
            color = COLOR_OCCLUDED if occluded[i] else (
                COLOR_TARGET if name in target_names else COLOR_NORMAL)
            label = f"{name} {confs[i]:.2f}" + (" OCCLUDED" if occluded[i] else "")
            cv2.putText(img, label, (x1, max(10, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, sum(occluded), events


def predict(model, cfg, image):
    return model.predict(
        image,
        conf=float(cfg["model"].get("conf", 0.25)),
        imgsz=int(cfg["model"].get("imgsz", 640)),
        verbose=False,
    )[0]


def process_image(model, cfg, daily_ids):
    src = cfg["source"]
    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {src}")

    result = predict(model, cfg, img)
    names = model.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    annotated, count, events = annotate(img, boxes, confs, classes, names, cfg, daily_ids)

    out_dir = Path(cfg.get("output_dir", "runs/occlusion"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(src).stem + "_occluded.jpg")
    cv2.imwrite(str(out_path), annotated)

    print(f"检测到 {count} 个遮挡目标")
    for e in events:
        print("  " + e)
    print(f"结果已保存 -> {out_path}")

    if cfg.get("show", True):
        cv2.imshow("occlusion", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_stream(model, cfg, daily_ids, cap, out_path):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path is not None:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    names = model.names
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = predict(model, cfg, frame)
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        annotated, count, events = annotate(frame, boxes, confs, classes, names, cfg, daily_ids)
        if count:
            print(f"frame {frame_idx}: {count} 个遮挡  " + " | ".join(events))

        if writer is not None:
            writer.write(annotated)
        if cfg.get("show", True):
            cv2.imshow("occlusion", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if out_path is not None:
        print(f"结果已保存 -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 daily-necessity occlusion detection")
    parser.add_argument("--config", default="config/occlusion.yaml")
    parser.add_argument("--mode", choices=["image", "video", "camera"], help="override config mode")
    parser.add_argument("--source", help="override config source")
    parser.add_argument("--conf", type=float, help="override confidence threshold")
    parser.add_argument("--iou", type=float, help="override occlusion iou_threshold")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=None,
                        help="show/hide result window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    if args.source:
        cfg["source"] = args.source
    if args.conf is not None:
        cfg["model"]["conf"] = args.conf
    if args.iou is not None:
        cfg["occlusion"]["iou_threshold"] = args.iou
    if args.show is not None:
        cfg["show"] = args.show

    daily_ids = set(int(v) for v in cfg["classes"].values())
    model = YOLO(cfg["model"]["path"])

    mode = cfg["mode"]
    if mode == "image":
        process_image(model, cfg, daily_ids)
    elif mode in ("video", "camera"):
        src = cfg["source"]
        cap = cv2.VideoCapture(int(src) if mode == "camera" else src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {src}")
        out_path = None
        if mode == "video":
            out_dir = Path(cfg.get("output_dir", "runs/occlusion"))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (Path(str(src)).stem + "_occluded.mp4")
        process_stream(model, cfg, daily_ids, cap, out_path)
    else:
        raise ValueError(f"Unknown mode: {mode} (choose image / video / camera)")


if __name__ == "__main__":
    main()
