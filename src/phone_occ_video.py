"""Video phone-occlusion demo with OpenCV visualization + a reject/release state machine.

Two models per frame:
  A  model/textseg-SegN.pt   -> text mask
  B  model/yolov8n-seg.pt    -> 'cell phone' mask

Judgement is the 4-signal one from phone_occlusion_demo.py (NOT a plain mask IoU:
text under the phone is invisible, so intersection is near-empty exactly when
occlusion happens).

On top of the per-frame verdict this adds the board-side state machine:
  CLEAR     --occlusion starts-->  OCCLUDED
  OCCLUDED  --clears---------->    CLEAR + 放行 (page may be sent)
  OCCLUDED  --exceeds timeout-->   降级放行  (宁送勿漏: never lose a page)

Usage:
    python phone_occ_video.py --source data_9.14/phone_occlusion_test.mp4
    python phone_occ_video.py --source 0                    # webcam
    python phone_occ_video.py --source in.mp4 --no-show --out runs/phone_occ/out.mp4
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from phone_occlusion_demo import (
    DILATE_PX, SIDE_MARGIN_PX, PHONE_CLASS,
    masks_to_binary, sides_with_text, lines_cut_by, iou_box,
)

CJK_FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

C_CLEAR = (80, 220, 80)
C_SUSPECT = (0, 200, 255)
C_OCC = (60, 60, 255)
C_TEXT = (0, 255, 0)
C_PHONE = (0, 0, 255)
C_ADJ = (0, 255, 255)
C_CUT = (0, 165, 255)


def find_font():
    for p in CJK_FONTS:
        if Path(p).exists():
            return p
    return None


class CJK:
    """Cached OpenCV<->PIL bridge so Chinese HUD text renders."""

    def __init__(self, size=22):
        self.ok = False
        fp = find_font()
        if fp:
            self.font = ImageFont.truetype(fp, size)
            self.ok = True

    def put(self, img, text, xy, color_bgr, size=None):
        if not self.ok:
            cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
            return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = self.font if size is None else ImageFont.truetype(self.font.path, size)
        d.text(xy, text, font=f, fill=tuple(int(c) for c in reversed(color_bgr)))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def judge(text_mask, text_insts, phone_mask, region_mask, pbox):
    """The 4-signal occlusion judgement. Returns a dict."""
    k = np.ones((DILATE_PX * 2 + 1, DILATE_PX * 2 + 1), np.uint8)
    ph_dil = cv2.dilate(phone_mask, k)
    touch = int((ph_dil & text_mask).sum())
    sides, n_side = sides_with_text(text_mask, pbox, SIDE_MARGIN_PX)
    cut = lines_cut_by(text_insts, pbox, SIDE_MARGIN_PX)
    covered = int((phone_mask & region_mask).sum())
    ph_area = int(phone_mask.sum())
    cover_ratio = covered / ph_area if ph_area else 0.0

    if touch == 0:
        level, label = 0, "无遮挡"
    elif n_side >= 3 or len(cut) >= 3 or cover_ratio > 0.6:
        level, label = 2, "遮挡"
    elif n_side >= 2 or len(cut) >= 1 or cover_ratio > 0.3:
        level, label = 1, "疑似遮挡"
    else:
        level, label = 0, "边缘接触"

    return {"level": level, "label": label, "touch": touch, "sides": sides,
            "n_side": n_side, "cut": cut, "cover_ratio": cover_ratio, "dil": ph_dil}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data_9.14/phone_occlusion_test.mp4")
    p.add_argument("--text-model", default="model/textseg-SegN.pt")
    p.add_argument("--obj-model", default="model/yolov8n-seg.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--phone-conf", type=float, default=0.15)
    p.add_argument("--timeout", type=float, default=3.0,
                   help="遮挡持续超过该秒数则降级放行 (宁送勿漏)")
    p.add_argument("--out", default=None, help="输出视频路径 (默认 runs/phone_occ/<stem>_out.mp4)")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true", help="打印状态机跳转")
    args = p.parse_args()

    src = args.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {src}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = args.out
    if out_path is None:
        d = Path("runs/phone_occ"); d.mkdir(parents=True, exist_ok=True)
        stem = "camera" if src.isdigit() else Path(src).stem
        out_path = str(d / f"{stem}_out.mp4")
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))

    text_model = YOLO(args.text_model)
    obj_model = YOLO(args.obj_model)
    cjk = CJK(22)

    # --- state machine ---
    # Timing is FRAME-based (frames / fps_in), not wall-clock: offline video and
    # the board both advance one "display frame" per processed frame, so
    # wall-clock would make the timeout fire early whenever processing is slower
    # than realtime (observed: 3s wall == 11 video frames at 3.4fps).
    # Debounce tolerates single-frame detection dropouts (observed at the
    # slide-in/out edges), which would otherwise flip the state every few frames.
    HIT_TOL = 2     # consecutive occluded frames before entering OCCLUDED
    MISS_TOL = 3    # consecutive clear frames before releasing

    state = "CLEAR"
    occ_frames = 0
    miss_frames = 0
    occ_start_idx = None
    released = None          # (frame_idx, reason)
    stats = {"occluded": 0, "suspect": 0, "frames": 0, "phone": 0}
    fps_hist = deque(maxlen=15)

    idx = 0
    t_prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        rt = text_model.predict(frame, conf=args.conf, verbose=False)[0]
        text_mask, text_insts = masks_to_binary(rt, H, W)
        region_mask, _ = masks_to_binary(rt, H, W, wanted={0})

        ro = obj_model.predict(frame, conf=args.phone_conf, verbose=False)[0]
        _, phone_insts = masks_to_binary(ro, H, W, wanted={PHONE_CLASS})
        _, book_insts = masks_to_binary(ro, H, W, wanted={73})

        vis = frame.copy()
        j = None
        pbox = None

        if phone_insts:
            stats["phone"] += 1
            ph = max(phone_insts, key=lambda i: i["conf"])
            pbox = ph["box"]
            j = judge(text_mask, text_insts, ph["mask"], region_mask, pbox)

            tm = text_mask > 0
            pm = ph["mask"] > 0
            vis[tm] = (vis[tm] * 0.5 + np.array(C_TEXT) * 0.5).astype(np.uint8)
            vis[pm] = (vis[pm] * 0.45 + np.array(C_PHONE) * 0.55).astype(np.uint8)
            adj = (j["dil"] > 0) & tm & ~pm
            vis[adj] = (vis[adj] * 0.3 + np.array(C_ADJ) * 0.7).astype(np.uint8)
            for ins in j["cut"][:20]:
                x1, y1, x2, y2 = ins["box"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), C_CUT, 2)
            cv2.rectangle(vis, (pbox[0], pbox[1]), (pbox[2], pbox[3]), C_PHONE, 3)

            if j["level"] > 0:
                stats["occluded" if j["level"] == 2 else "suspect"] += 1

            # --- state machine ---
            if j["level"] > 0:
                occ_frames += 1
                miss_frames = 0
                if state == "CLEAR" and occ_frames >= HIT_TOL:
                    state = "OCCLUDED"
                    occ_start_idx = idx - occ_frames + 1
                    if args.verbose:
                        print(f"  f{idx:4d} CLEAR -> OCCLUDED   (level={j['level']} 判据 "
                              f"包围{j['n_side']}/4 截断{len(j['cut'])} 覆盖{j['cover_ratio']*100:.0f}%)")
                elif state == "OCCLUDED" and occ_start_idx is not None:
                    dur_s = (idx - occ_start_idx) / fps_in
                    if dur_s > args.timeout:
                        state = "CLEAR"
                        released = (idx, "超时降级放行")
                        if args.verbose:
                            print(f"  f{idx:4d} OCCLUDED -> CLEAR  超时降级 (视频时长 {dur_s:.1f}s > {args.timeout}s)")
                        occ_start_idx = None; occ_frames = 0; miss_frames = 0
            else:
                miss_frames += 1
                occ_frames = 0
                if state == "OCCLUDED" and miss_frames >= MISS_TOL:
                    state = "CLEAR"
                    released = (idx, "遮挡解除，放行")
                    if args.verbose:
                        print(f"  f{idx:4d} OCCLUDED -> CLEAR  遮挡解除 (连续{miss_frames}帧无遮挡)")
                    occ_start_idx = None
        else:
            miss_frames += 1
            occ_frames = 0
            if state == "OCCLUDED" and miss_frames >= MISS_TOL:
                state = "CLEAR"
                released = (idx, "手机离开，放行")
                if args.verbose:
                    print(f"  f{idx:4d} OCCLUDED -> CLEAR  手机连续{miss_frames}帧未检出")
                occ_start_idx = None

        stats["frames"] += 1
        now = time.time()
        fps_hist.append(1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        fps = sum(fps_hist) / len(fps_hist)

        # ---------- HUD ----------
        vis = cjk.put(vis, f"帧 {idx}   {fps:.1f} FPS", (12, 8), (255, 255, 255), 22)

        if j is None:
            banner, col = "无手机", C_CLEAR
        elif j["level"] == 2:
            banner, col = "遮挡：手机压在文字上", C_OCC
        elif j["level"] == 1:
            banner, col = "疑似遮挡：部分文字被盖", C_SUSPECT
        else:
            banner, col = "无遮挡", C_CLEAR

        cv2.rectangle(vis, (0, 44), (W, 94), (25, 25, 25), -1)
        vis = cjk.put(vis, banner, (12, 54), col, 30)

        panel = [
            f"文字实例 {len(text_insts)}   手机 {len(phone_insts)}",
            f"判据1 接触   {'是' if (j and j['touch'] > 0) else '否'}"
            + (f"  ({j['touch']} px)" if j else ""),
            f"判据2 包围   {j['n_side'] if j else 0}/4 {''.join(j['sides']) if j else ''}",
            f"判据3 截断行 {len(j['cut']) if j else 0}",
            f"判据4 覆盖   {j['cover_ratio']*100:.1f}%" if j else "判据4 覆盖   --",
        ]
        cv2.rectangle(vis, (0, 102), (330, 102 + 30 * len(panel) + 12), (25, 25, 25), -1)
        for i, line in enumerate(panel):
            vis = cjk.put(vis, line, (10, 108 + 30 * i), (220, 220, 220), 20)

        state_col = C_OCC if state == "OCCLUDED" else C_CLEAR
        occ_dur = ((idx - occ_start_idx) / fps_in) if (state == "OCCLUDED" and occ_start_idx is not None) else 0.0
        cv2.rectangle(vis, (0, H - 46), (W, H), (25, 25, 25), -1)
        vis = cjk.put(vis, f"状态 {state}" + (f"  遮挡持续 {occ_dur:.1f}s / 超时 {args.timeout:.0f}s"
                                             if state == "OCCLUDED" else ""),
                      (12, H - 40), state_col, 24)
        if released and idx - released[0] < int(fps_in * 1.5):
            vis = cjk.put(vis, f">> 放行：{released[1]}", (12, H - 76), (255, 255, 255), 22)

        writer.write(vis)
        if args.show:
            cv2.imshow("phone-occlusion", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    n = max(stats["frames"], 1)
    print(f"处理 {stats['frames']} 帧  |  检出手机 {stats['phone']} 帧 "
          f"({stats['phone']/n*100:.0f}%)")
    print(f"  遮挡帧 {stats['occluded']} ({stats['occluded']/n*100:.0f}%)  "
          f"疑似 {stats['suspect']} ({stats['suspect']/n*100:.0f}%)")
    print(f"输出视频 -> {out_path}")


if __name__ == "__main__":
    main()
