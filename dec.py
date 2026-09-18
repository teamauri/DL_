#!/usr/bin/env python3
"""视频检测脚本：支持摄像头或视频文件。

用法:
  # 摄像头
  python dec.py --source 0 --weights 4.pt --device mps

  # 视频文件
  python dec.py --source /path/to/video.mp4 --weights 4.pt

  # 视频文件并保存结果
  python dec.py --source /path/to/video.mp4 --save --output out.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ultralytics"))

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS = "/Users/marking/Downloads/4.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default="0", help="camera index (0/1/...) or video file path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="cpu / mps / 0")
    parser.add_argument("--save", action="store_true", help="save the annotated video")
    parser.add_argument("--output", default="output.mp4", help="output video path")
    args = parser.parse_args()

    model = YOLO(args.weights)
    print("classes:", model.names)

    # source: 数字当作摄像头序号，否则当作文件路径
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"按 q 退出。save={args.save}  output={args.output if args.save else 'N/A'}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("视频结束或读取失败，退出。")
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        annotated = results[0].plot()  # 自动画 bbox + 类别 + 置信度

        cv2.imshow("YOLO detection", annotated)
        if writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
