"""Regression tests for the page-turn logic (run: python src/test.py).

No pytest, no models, no video files -- the detector takes plain frames and the
signature takes plain grayscale, so the whole thing runs in milliseconds.

The first test is the important one. The startup-baseline bug failed SILENTLY:
the scene starts still and never passes through MOVING, so `settled` never
fired, the first signature the detector ever took was the page AFTER the first
turn, and every turn was reported as 0. Nothing errored. Only a test that
asserts on the count catches that.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bookseg_pipeline import (                     # noqa: E402
    PageTurnDetector, SAME_HAMMING, TURN_HAMMING, hamming, page_signature,
)

W, H = 480, 360
BOX = (120, 60, 360, 300)          # the page, excluding the desk border
FAILED = []


def frame(kind="A", dx=0, dy=0):
    """A bright page on a dark desk, with text-like line blocks."""
    img = np.full((H, W, 3), 50, np.uint8)
    img[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 240
    rng = np.random.default_rng(1 if kind == "A" else 2)
    y = BOX[1] + 20
    while y < BOX[3] - 20:
        w = int(60 + rng.random() * 130)
        img[y:y + 8, BOX[0] + 20:BOX[0] + 20 + w] = 90
        y += 18
    if kind == "A":                # 给 A 页加一个 A 独有的图块, 两页必须明显不同
        img[BOX[1] + 20:BOX[1] + 80, BOX[0] + 20:BOX[0] + 150] = 90
    if dx or dy:
        img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(50, 50, 50))
    return img


def settled_event(det, img, tries=8):
    """Feed duplicate frames until update() reports a settle; return that event."""
    for _ in range(tries):
        ev = det.update(img)
        if ev["settled"]:
            return ev
    return None


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


# --- 1. 开机基线: 静帧也要吐 settled, 否则第一页永远没得比 ------------------
def test_startup_baseline():
    det = PageTurnDetector()
    ev = settled_event(det, frame("A"))
    check("开机静帧就给出基线 settle", ev is not None)
    if ev is None:
        return None
    r = det.resolve(page_signature(frame("A"), BOX))
    check("第一张签名判为 FIRST (不是翻页)", r["result"] == "FIRST" and not r["turn"])
    return det


# --- 2. 翻页: 运动 -> 停稳 -> 内容变了 -> TURNED ---------------------------
def test_turn_detected(det):
    for _ in range(3):
        det.update(frame("A", dx=40))                   # 运动
    ev = settled_event(det, frame("B"))
    check("翻页后能停稳", ev is not None)
    if ev is None:
        return
    r = det.resolve(page_signature(frame("B"), BOX))
    check("翻页判为 TURNED", r["result"] == "TURNED" and r["turn"],
          f"hamming={r['hamming']}")


# --- 3. 头动: 同样大的运动, 内容没变 -> 不能判翻页 --------------------------
def test_head_motion_is_not_a_turn(det):
    for _ in range(3):
        det.update(frame("B", dx=40))                   # 和翻页一样大的位移
    ev = settled_event(det, frame("B"))
    check("头动后能停稳", ev is not None)
    if ev is None:
        return
    r = det.resolve(page_signature(frame("B"), BOX))
    check("头动判为 SAME, 不发送", r["result"] == "SAME" and not r["turn"],
          f"hamming={r['hamming']}")


# --- 4. 疑似带: 中间地带按宁送勿漏放行 --------------------------------------
def test_suspect_band_ships(det):
    for _ in range(3):
        det.update(frame("B", dx=40))
    settled_event(det, frame("B"))
    mid = page_signature(frame("B"), BOX).copy()
    n = (SAME_HAMMING + TURN_HAMMING) // 2              # 落在两条线中间
    mid[:n] ^= 1
    r = det.resolve(mid)
    check(f"疑似带 (hamming={n}) 判为 SUSPECT 且放行",
          r["result"] == "SUSPECT" and r["turn"])


# --- 5. 签名本身的鉴别力: 这条会把 dHash 打回去 ------------------------------
def test_signature_discriminates():
    sa, sb = page_signature(frame("A"), BOX), page_signature(frame("B"), BOX)
    d = hamming(sa, sb)
    check(f"两页签名距离 {d} >= TURN_HAMMING({TURN_HAMMING})", d >= TURN_HAMMING,
          "(dHash 8x8 在同一对图上实测只有 10, 会失败 -- 这就是换 pHash 的原因)")


if __name__ == "__main__":
    print("翻页逻辑回归测试")
    det = test_startup_baseline()
    if det is not None:
        test_turn_detected(det)
        test_head_motion_is_not_a_turn(det)
        test_suspect_band_ships(det)
    test_signature_discriminates()
    print(f"\n{'全部通过' if not FAILED else str(len(FAILED)) + ' 项失败: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
