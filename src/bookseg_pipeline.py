"""bookseg board-pipeline demo (macOS stand-in for the RV1106 board).

Runs the 6 board stages end-to-end so the pipeline can be exercised on a Mac
before the real 4-channel RKNN model exists:

  1. 取两路图   640 small stream for every decision; full-res original only on trigger
  2. 判翻动     翻页状态机 -- 停稳 + "内容变了"(PageTurnDetector); NOT a frame diff
  3. 分割(NPU)  pluggable Segmenter -- demo uses yolov8n-seg "book" mask as the
                paper-channel stand-in (TODO: real DS-UNet 4ch RKNN int8)
  4. 粗筛闸门   reject no-paper / not-settled / 同一页 (text-at-edge stubbed)
  5. 定位+裁剪  paper mask bbox x1.3 -> crop from full-res original -> JPEG
  6. 触发+发送  beep (afplay) + "send" = write crop to runs/bookseg/

Usage:
    python bookseg_pipeline.py --source data_9.14/books_pile.jpg
    python bookseg_pipeline.py --source some.mp4
    python bookseg_pipeline.py --source 0            # webcam
    python bookseg_pipeline.py --source x.jpg --segmenter stub --mute
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

SMALL_WIDTH = 640
DIFF_GRID = 128
STABLE_THRESH = 2.0          # mean |diff| on the 128x128 grid below which the scene is "still"
SETTLE_FRAMES = 3            # consecutive still frames before the scene counts as settled
CROP_EXPAND = 1.3
MIN_PAPER_AREA = 0.005       # paper mask must cover >= this fraction of the frame
SAME_HAMMING = 12            # pHash 汉明 <= 此值 -> 同一页
TURN_HAMMING = 16            # pHash 汉明 >= 此值 -> 新的一页 (中间地带算"疑似")
# 阈值是实测的, 不是拍的。同一页的噪声地板(含头动): 合成视频最大 8, 纯静止 0-2;
# 翻页距离: 34 (带纸面 mask) / 22 (整帧) / 20 (单页测试图)。12/16 落在 [8,20] 这个
# 空隙里, 两边各留 4 bit。SAME 这条更要紧 -- 判成"同一页"就是漏页, 不可逆;
# 而翻页被低估只会落进"疑似"带, 疑似同样放行, 不会丢页。

BOOK_CLASS = 73              # COCO "book"


# --- stage 3: segmenter -----------------------------------------------------
class YoloPaperSegmenter:
    """Demo stand-in: yolov8n-seg 'book' mask as the 'paper' channel.

    TODO(board): replace with the real DS-UNet 4-channel model
    (paper/text/hand/gutter) exported to RKNN int8. The other three channels
    are None here, so text_at_edge / hand-over-text gates are no-ops in the demo.
    """

    def __init__(self, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO("yolov8n-seg.pt")
        self.conf = conf

    def predict(self, small):
        h, w = small.shape[:2]
        paper = np.zeros((h, w), np.uint8)
        r = self.model.predict(small, conf=self.conf, verbose=False)[0]
        if r.masks is not None and r.boxes is not None:
            for seg, cls in zip(r.masks.data, r.boxes.cls):
                if int(cls) != BOOK_CLASS:
                    continue
                m = seg.cpu().numpy().astype(np.float32)
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h))
                paper = np.maximum(paper, (m > 0.5).astype(np.uint8))
        return {"paper": paper, "text": None, "hand": None, "gutter": None}


class StubSegmenter:
    """Full-frame 'paper' -- only for exercising pipeline logic w/o the model."""

    def predict(self, small):
        h, w = small.shape[:2]
        return {"paper": np.ones((h, w), np.uint8), "text": None, "hand": None, "gutter": None}


class ThresholdPaperSegmenter:
    """Bright-region paper mask -- stand-in for the DS-UNet paper channel.

    For synthetic scenes (bright page on a dark desk). No weights, so it is fast
    enough to run the pipeline at full frame rate and it isolates the page-turn
    logic from YOLO's unreliable 'book' detection.
    """

    def __init__(self, thresh=150, min_area=0.02):
        self.thresh = thresh
        self.min_area = min_area

    def predict(self, small):
        h, w = small.shape[:2]
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(g, self.thresh, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        paper = np.zeros((h, w), np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))   # largest blob
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area * h * w:
                paper[lab == i] = 1
        return {"paper": paper, "text": None, "hand": None, "gutter": None}


# --- stage 2: 书本翻动了吗 ---------------------------------------------------
class PageTurnDetector:
    """翻页判定: a state machine, NOT a frame diff.

    The camera is head-mounted, so leaning in, head drift and lighting changes
    all produce large frame diffs with NO page turn. Three properties separate a
    real turn from camera motion:

      1. A turn is an EVENT, not a level. Content must change AND then stop.
         Moving alone is not a turn; still-and-unchanged is idle.
      2. The change is measured in the PAPER's frame of reference. The signature
         (page_signature) is taken on the paper-mask bbox crop, so the book
         sliding across the sensor normalises away instead of firing.
      3. Two thresholds, not one: hamming <= SAME_HAMMING -> 同一页, >= TURN_HAMMING
         -> 新的一页, and the band between them is 疑似. The band is real -- on
         dense text pages two genuinely different paragraphs can still hash
         close together, which is exactly why the cloud OCR dedup stays gate two.

    States: IDLE -> MOVING -> (diff falls back) -> settles after settle_frames
    still frames, at which point the caller must call resolve() with a signature.
    """

    def __init__(self, grid=DIFF_GRID, motion_thresh=STABLE_THRESH,
                 settle_frames=SETTLE_FRAMES,
                 same_hamming=SAME_HAMMING, turn_hamming=TURN_HAMMING):
        self.grid = grid
        self.motion_thresh = motion_thresh
        self.settle_frames = settle_frames
        self.same_hamming = same_hamming
        self.turn_hamming = turn_hamming
        self.prev = None
        self.state = "IDLE"
        self.still = 0
        self.last_sig = None
        self.want_sig = True     # 开机就要一张基线, 不然第一页永远没得比

    def update(self, frame):
        """Cheap per-frame motion update. `settled` fires ONCE, on the frame the
        scene becomes still again -- that is the only moment a turn can be
        confirmed, and the only moment the caller needs to segment.

        `want_sig` is what makes that safe at startup: it is set on the first
        frame and again after every motion, and cleared once a settle has been
        emitted. Without it the very first page never gets a baseline signature
        (the scene starts still and never passes through MOVING), so the first
        signature the detector ever takes is the page AFTER the first turn --
        and that turn is silently lost.
        """
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.grid, self.grid)).astype(np.float32)
        if self.prev is None:
            self.prev = g
            return {"state": "IDLE", "settled": False, "diff": 0.0}
        diff = float(np.abs(g - self.prev).mean())
        self.prev = g

        settled = False
        if diff >= self.motion_thresh:
            self.state, self.still, self.want_sig = "MOVING", 0, True
        else:
            self.still += 1
            if self.want_sig and self.still >= self.settle_frames:
                self.state, settled, self.want_sig = "IDLE", True, False
            else:
                self.state = "IDLE"
        return {"state": self.state, "settled": settled, "diff": diff}

    def resolve(self, signature):
        """Call on the frame update() reported settled=True.

        signature=None means 纸面看不见: no verdict, but the baseline is dropped
        so the next visible page counts as a turn rather than matching a stale one.
        """
        if signature is None:
            self.last_sig = None
            return {"result": "NO_PAPER", "hamming": None, "turn": False}
        if self.last_sig is None:
            self.last_sig = signature
            return {"result": "FIRST", "hamming": None, "turn": False}

        d = hamming(signature, self.last_sig)
        self.last_sig = signature
        if d >= self.turn_hamming:
            return {"result": "TURNED", "hamming": d, "turn": True}
        if d <= self.same_hamming:
            return {"result": "SAME", "hamming": d, "turn": False}
        # 疑似: 宁送勿漏 -- a false send is deduped in the cloud, a missed page is
        # gone forever, so the ambiguous band ships.
        return {"result": "SUSPECT", "hamming": d, "turn": True}


# --- helpers ----------------------------------------------------------------
def to_small(frame):
    scale = SMALL_WIDTH / frame.shape[1]
    return cv2.resize(frame, (SMALL_WIDTH, int(frame.shape[0] * scale)))


def paper_bbox(paper):
    ys, xs = np.where(paper > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(box, factor, w, h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = (x2 - x1) / 2 * factor, (y2 - y1) / 2 * factor
    x1 = max(0, int(cx - nw)); y1 = max(0, int(cy - nh))
    x2 = min(w, int(cx + nw)); y2 = min(h, int(cy + nh))
    return x1, y1, x2, y2


def phash(gray, size=32, keep=8):
    """DCT low-frequency signature (63 bit).

    Preferred over dHash for text pages. dHash only compares adjacent-pixel
    gradients, and a page of running text is a regular line rhythm -- two
    genuinely different paragraphs hash nearly alike. Measured on a synthetic
    page turn: dHash 8x8  -> 11/64 bits differ (below any usable threshold),
                          pHash    -> 34/63 bits differ, and 0 on head motion.
    The DCT low frequencies catch the macro layout (where the ink is) rather
    than the line texture.

    Upgrading later: compute this on the text-mask crop instead of grayscale and
    it also becomes immune to exposure changes -- grayscale pHash still reacts
    to the camera going in and out of shadow.
    """
    r = cv2.resize(gray, (size, size)).astype(np.float32)
    d = cv2.dct(r)[:keep, :keep].flatten()[1:]   # 丢掉 DC, 否则它一个分量压过全部
    return (d > np.median(d)).astype(np.uint8)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def page_signature(small, box):
    """pHash of the paper-mask bbox crop.

    Taken on the crop rather than the raw frame so the signature is invariant to
    WHERE the book sits in frame -- that is what keeps head motion from reading
    as a page turn.
    """
    x1, y1 = max(0, box[0]), max(0, box[1])
    x2, y2 = min(small.shape[1], box[2]), min(small.shape[0], box[3])
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return phash(cv2.cvtColor(small[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY))


def beep(mute):
    if mute:
        return
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
    except FileNotFoundError:
        print("    [BEEP]")


# --- stage 4: coarse gate ---------------------------------------------------
def coarse_gate(paper, text, stable, is_dup):
    """Return (pass, reason). Biased toward SEND -- only reject what is certain."""
    if paper is None or paper.sum() / paper.size < MIN_PAPER_AREA:
        return False, "no_paper"
    if not stable:
        return False, "not_stable"
    if is_dup:
        return False, "duplicate"
    # text_at_edge needs the text channel (absent in the demo stand-in).
    # TODO(board): text mask touching frame border -> reject "text_at_edge".
    return True, "ok"


# --- image mode -------------------------------------------------------------
def run_image(seg, args):
    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    small = to_small(frame)
    sx = frame.shape[1] / small.shape[1]
    sy = frame.shape[0] / small.shape[0]

    print("[1] 取两路图: small", small.shape[:2], "<- original", frame.shape[:2])
    print("[2] 判停稳:   单帧视为已停稳 (视频模式才做帧差)")

    masks = seg.predict(small)
    paper = masks["paper"]
    print("[3] 分割:     paper mask 覆盖 %.2f%%" % (100 * paper.sum() / paper.size))

    box = paper_bbox(paper)
    passed, reason = coarse_gate(paper, masks.get("text"), stable=True, is_dup=False)
    print("[4] 粗筛闸门:", "PASS" if passed else f"REJECT ({reason})")
    if not passed:
        return

    x1, y1, x2, y2 = expand_bbox(box, args.expand, small.shape[1], small.shape[0])
    ox1, oy1, ox2, oy2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    crop = frame[oy1:oy2, ox1:ox2]
    print(f"[5] 定位+裁剪: paper bbox 小图[{box[0]},{box[1]},{box[2]},{box[3]}] "
          f"-> 裁剪框(外扩{args.expand}) 原图[{ox1},{oy1},{ox2},{oy2}] crop {crop.shape[:2]}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_jpg = out_dir / f"page_{ts}.jpg"
    cv2.imwrite(str(out_jpg), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("[6] 触发+发送: BEEP + 发送 ->", out_jpg)
    beep(args.mute)

    vis = small.copy()
    cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, "paper", (box[0], max(12, box[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "crop", (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f"page_{ts}_viz.jpg"), vis)
    print("    (可视化小图 ->", out_dir / f"page_{ts}_viz.jpg", ")")


# --- video / camera mode ----------------------------------------------------
def draw_viz(vis, idx, ev, vz, turns, sent, expand_box=None):
    """HUD for the page-turn logic. ASCII only -- cv2.putText cannot draw CJK and
    bookseg_pipeline deliberately has no PIL dependency (phone_occ_video owns that)."""
    H, W = vis.shape[:2]
    if expand_box is not None:
        x1, y1, x2, y2 = expand_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "paper", (x1, max(16, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(vis, (0, 0), (W, 44), (25, 25, 25), -1)
    col = (0, 165, 255) if ev["state"] == "MOVING" else (120, 220, 120)
    cv2.putText(vis, f"frame {idx}  state {ev['state']}  diff {ev['diff']:.2f}"
                     f"   turns {turns}  sent {sent}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    r = vz.get("r")
    if r:
        txt = f"{r['result']}  hamming {r['hamming']}" if r["hamming"] is not None \
            else f"{r['result']}  (baseline)"
        vcol = {"TURNED": (60, 60, 255), "SUSPECT": (0, 200, 255),
                "SAME": (120, 220, 120), "FIRST": (200, 200, 200),
                "NO_PAPER": (160, 160, 160)}.get(r["result"], (200, 200, 200))
        cv2.rectangle(vis, (0, 48), (W, 92), (25, 25, 25), -1)
        cv2.putText(vis, txt, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, vcol, 2)

    # 签名条: 每一位一格, 红 = 与上一张签名不同。翻页时整条会大片变红。
    a, b = vz.get("sig_before"), vz.get("sig_after")
    if a is not None and b is not None:
        n = len(b)
        cw = max(4, (W - 24) // n)
        y0 = H - 34
        for i in range(n):
            c = (60, 60, 240) if int(a[i]) != int(b[i]) else (60, 120, 60)
            cv2.rectangle(vis, (12 + i * cw, y0), (12 + i * cw + cw - 2, y0 + 22), c, -1)
        cv2.putText(vis, "phash bits  red = flipped", (12, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if vz.get("flash", 0) > 0:
        cv2.putText(vis, ">> SEND", (W - 210, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 3)
        vz["flash"] -= 1


def run_video(seg, args):
    src = args.source
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    turn = PageTurnDetector()
    sent = 0
    turns = 0
    frame_idx = 0

    writer, viz = None, {"r": None, "sig_before": None, "sig_after": None, "flash": 0}
    if args.viz:
        writer = cv2.VideoWriter(str(out_dir / f"{Path(src).stem}_viz.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.get(cv2.CAP_PROP_FPS) or 25,
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        vis, expand_box = frame.copy(), None

        small = to_small(frame)                    # [1] 取两路图
        ev = turn.update(small)                    # [2] 判翻动 (每帧, 很便宜)
        handled = False

        if ev["settled"]:
            # 只有"停稳"这一刻才需要分割 -- 一页一次, 不是每帧都跑 (旧版每帧都跑)
            masks = seg.predict(small)             # [3] 分割
            paper = masks["paper"]
            box = paper_bbox(paper)
            if box is None or paper.sum() / paper.size < MIN_PAPER_AREA:
                viz["r"] = turn.resolve(None)
                print(f"    frame {frame_idx}: 停稳但没有纸面 -> 不判定")
                handled = True
            else:
                sx = frame.shape[1] / small.shape[1]
                sy = frame.shape[0] / small.shape[0]
                sb = turn.last_sig
                sig = page_signature(small, box)
                r = turn.resolve(sig)              # [4] 粗筛闸门: 翻页了吗
                viz.update(r=r, sig_before=sb, sig_after=turn.last_sig)

                bx1, by1, bx2, by2 = expand_bbox(box, args.expand,
                                                 small.shape[1], small.shape[0])
                expand_box = (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy))

                if not r["turn"]:
                    print(f"    frame {frame_idx}: 停稳 {r['result']:8s} "
                          f"hamming={r['hamming']} -> 未翻页, 不发送")
                else:
                    if r["result"] == "SUSPECT":
                        print(f"    frame {frame_idx}: 疑似翻动 hamming={r['hamming']} "
                              f"-> 宁送勿漏, 仍然发送")
                    turns += 1
                    crop = frame[expand_box[1]:expand_box[3], expand_box[0]:expand_box[2]]
                    if crop.size:                  # [5] 定位+裁剪 (从原图裁)
                        sent += 1
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        out_jpg = out_dir / f"page_{sent}_{ts}.jpg"
                        cv2.imwrite(str(out_jpg), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        print(f"[触发] frame {frame_idx}: 翻页 #{turns} "
                              f"hamming={r['hamming']} -> 发送 {out_jpg.name}")
                        beep(args.mute)
                        viz["flash"] = 25
                handled = True

        if not handled and args.verbose and frame_idx % 25 == 0:
            print(f"    frame {frame_idx}: {ev['state']:8s} diff={ev['diff']:.2f}")

        if writer:
            draw_viz(vis, frame_idx, ev, viz, turns, sent, expand_box)
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()
        print(f"可视化 -> {out_dir / f'{Path(src).stem}_viz.mp4'}")
    print(f"done: 翻页 {turns} 次, 共发送 {sent} 页")


def main():
    p = argparse.ArgumentParser(description="bookseg board-pipeline demo (macOS)")
    p.add_argument("--source", default="data_9.14/books_pile.jpg")
    p.add_argument("--segmenter", choices=["yolo", "stub", "thresh"], default="yolo")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--expand", type=float, default=CROP_EXPAND)
    p.add_argument("--out", default="runs/bookseg")
    p.add_argument("--mute", action="store_true", help="disable the beep sound")
    p.add_argument("--verbose", action="store_true", help="print every state change")
    p.add_argument("--viz", action="store_true", help="write an annotated HUD video")
    args = p.parse_args()

    seg = {"yolo": YoloPaperSegmenter, "stub": StubSegmenter,
           "thresh": ThresholdPaperSegmenter}[args.segmenter]
    seg = seg(args.conf) if args.segmenter == "yolo" else seg()

    src = args.source
    if src.isdigit():
        run_video(seg, args)
    elif Path(src).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        run_image(seg, args)
    else:
        run_video(seg, args)


if __name__ == "__main__":
    main()
