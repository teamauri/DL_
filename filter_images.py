#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片数据筛选工具

功能：
  - 选择一个图片文件夹，逐张显示图片并读取同名标签文件（默认 .txt）
  - A / 左方向键：上一张
  - D / 右方向键：下一张
  - 空格：删除当前图片和它的标签
  - U：撤销上一次删除
  - R：重新扫描当前文件夹
  - Q / Esc：退出

默认“删除”是把图片和标签移动到文件夹里的 _deleted/ 目录，方便反悔。
如需真正永久删除，加上 --hard 参数。

用法示例：
  python filter_images.py
  python filter_images.py --folder /path/to/images
  python filter_images.py --folder /path/to/images --label-dir /path/to/labels --label-ext txt
  python filter_images.py --folder /path/to/images --hard
"""

import argparse
import re
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    print("缺少 Pillow，请先安装：pip install Pillow")
    sys.exit(1)


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tif", ".tiff", ".webp",
}

MAX_DISPLAY_W = 1000
MAX_DISPLAY_H = 680


def natural_key(text: str):
    """让文件名按数字大小自然排序，例如 2 < 10。"""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ".txt"
    return ext if ext.startswith(".") else "." + ext


def scan_images(folder: Path):
    images = []
    if not folder.exists() or not folder.is_dir():
        return images
    for child in folder.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            images.append(child)
    images.sort(key=lambda p: natural_key(p.name))
    return images


def find_label(image_path: Path, label_ext: str, label_dir: Path | None):
    """查找与图片同名的标签文件。"""
    stem = image_path.stem
    candidates = []

    if label_dir is not None:
        candidates.append(label_dir / (stem + label_ext))

    candidates.append(image_path.with_name(stem + label_ext))
    candidates.append(image_path.parent / "labels" / (stem + label_ext))
    candidates.append(image_path.parent / "label" / (stem + label_ext))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def read_label(label_path: Path | None) -> str:
    if label_path is None:
        return "（没有找到标签文件）"
    try:
        text = label_path.read_text(encoding="utf-8", errors="replace").strip()
        return text if text else "（标签文件为空）"
    except OSError as exc:
        return f"（读取标签失败：{exc}）"


class ImageFilterApp:
    def __init__(self, root: tk.Tk, folder: Path, label_ext: str,
                 label_dir: Path | None, hard: bool):
        self.root = root
        self.folder = folder
        self.label_ext = label_ext
        self.label_dir = label_dir
        self.hard = hard

        self.images: list[Path] = []
        self.index = 0
        self.undo_stack: list[tuple[Path, Path | None, Path, Path | None]] = []
        self.current_tk_image = None

        self.deleted_dir = folder / "_deleted"

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        self.root.title("图片数据筛选工具")
        self.root.bind_all("<Key>", self.on_key)

        self.canvas = tk.Canvas(
            self.root,
            width=MAX_DISPLAY_W,
            height=MAX_DISPLAY_H,
            bg="#222222",
        )
        self.canvas.pack(padx=8, pady=(8, 4))

        self.status_var = tk.StringVar(value="正在扫描...")
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=8, pady=(0, 2))

        self.label_text = tk.Text(self.root, height=7, wrap="word", state="disabled")
        self.label_text.pack(fill="both", expand=False, padx=8, pady=(0, 4))

        button_bar = tk.Frame(self.root)
        button_bar.pack(fill="x", padx=8, pady=(0, 8))

        tk.Button(button_bar, text="上一张 (A/←)", command=self.prev).pack(side="left")
        tk.Button(button_bar, text="下一张 (D/→)", command=self.next).pack(side="left", padx=(6, 0))
        tk.Button(button_bar, text="删除 (空格)", command=self.delete_current).pack(side="left", padx=(6, 0))
        tk.Button(button_bar, text="撤销 (U)", command=self.undo).pack(side="left", padx=(6, 0))
        tk.Button(button_bar, text="退出 (Q/Esc)", command=self.root.destroy).pack(side="right")

    def refresh(self):
        self.images = scan_images(self.folder)
        self.index = 0
        self.show_current()

    def show_current(self):
        if not self.images:
            self.current_tk_image = None
            self.canvas.delete("all")
            self.canvas.create_text(
                MAX_DISPLAY_W // 2,
                MAX_DISPLAY_H // 2,
                text="文件夹里没有图片",
                fill="#ffffff",
                font=("Arial", 18),
            )
            self.status_var.set("没有可筛选的图片")
            self._set_label("")
            return

        self.index = max(0, min(self.index, len(self.images) - 1))
        image_path = self.images[self.index]
        label_path = find_label(image_path, self.label_ext, self.label_dir)

        self._show_image(image_path)
        self._set_label(read_label(label_path))
        self.status_var.set(
            f"{self.index + 1} / {len(self.images)}  |  {image_path.name}"
        )

    def _show_image(self, image_path: Path):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((MAX_DISPLAY_W, MAX_DISPLAY_H), Image.LANCZOS)
                self.current_tk_image = ImageTk.PhotoImage(img)
        except OSError as exc:
            self.current_tk_image = None
            self.canvas.delete("all")
            self.canvas.create_text(
                MAX_DISPLAY_W // 2,
                MAX_DISPLAY_H // 2,
                text=f"无法读取图片：{exc}",
                fill="#ff8888",
                font=("Arial", 14),
            )
            return

        self.canvas.delete("all")
        self.canvas.config(
            width=self.current_tk_image.width(),
            height=self.current_tk_image.height(),
        )
        self.canvas.create_image(
            self.current_tk_image.width() // 2,
            self.current_tk_image.height() // 2,
            image=self.current_tk_image,
        )

    def _set_label(self, text: str):
        self.label_text.config(state="normal")
        self.label_text.delete("1.0", "end")
        self.label_text.insert("1.0", text)
        self.label_text.config(state="disabled")

    def prev(self):
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self.show_current()

    def next(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.show_current()

    def _unique_pair_dest(self, image_path: Path, label_path: Path | None):
        """返回 _deleted 中不冲突的目标路径，保持图片和标签同名。"""
        self.deleted_dir.mkdir(exist_ok=True)
        suffix = ""
        counter = 1
        while True:
            image_dest = self.deleted_dir / f"{image_path.stem}{suffix}{image_path.suffix}"
            label_dest = None
            if label_path is not None:
                label_dest = self.deleted_dir / f"{image_path.stem}{suffix}{label_path.suffix}"

            image_ok = not image_dest.exists()
            label_ok = label_dest is None or not label_dest.exists()
            if image_ok and label_ok:
                return image_dest, label_dest

            suffix = f"_{counter}"
            counter += 1

    def delete_current(self):
        if not self.images:
            return

        image_path = self.images[self.index]
        label_path = find_label(image_path, self.label_ext, self.label_dir)

        if self.hard:
            try:
                image_path.unlink()
            except OSError as exc:
                messagebox.showerror("删除失败", f"无法删除图片：\n{exc}")
                return
            if label_path is not None:
                try:
                    label_path.unlink()
                except OSError:
                    pass
            self.images.pop(self.index)
        else:
            image_dest, label_dest = self._unique_pair_dest(image_path, label_path)
            try:
                shutil.move(str(image_path), str(image_dest))
            except OSError as exc:
                messagebox.showerror("移动失败", f"无法移动图片：\n{exc}")
                return
            if label_path is not None and label_dest is not None:
                try:
                    shutil.move(str(label_path), str(label_dest))
                except OSError as exc:
                    messagebox.showwarning(
                        "标签移动失败",
                        f"图片已移动，但标签移动失败：\n{exc}",
                    )
            self.undo_stack.append((image_path, label_path, image_dest, label_dest))
            self.images.pop(self.index)

        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)
        self.show_current()

    def undo(self):
        if not self.undo_stack:
            messagebox.showinfo("提示", "没有可撤销的删除")
            return

        image_orig, label_orig, image_moved, label_moved = self.undo_stack.pop()
        try:
            if image_moved.exists():
                shutil.move(str(image_moved), str(image_orig))
            if label_moved is not None and label_moved.exists() and label_orig is not None:
                shutil.move(str(label_moved), str(label_orig))
        except OSError as exc:
            messagebox.showerror("撤销失败", str(exc))
            return

        self.refresh()
        for idx, image_path in enumerate(self.images):
            if image_path == image_orig:
                self.index = idx
                break
        self.show_current()

    def on_key(self, event: tk.Event):
        key = event.keysym.lower()
        if key in ("left", "a"):
            self.prev()
        elif key in ("right", "d"):
            self.next()
        elif key == "space":
            self.delete_current()
        elif key == "u":
            self.undo()
        elif key == "r":
            self.refresh()
        elif key in ("q", "escape"):
            self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="图片数据筛选工具")
    parser.add_argument("--folder", help="图片文件夹路径；不填则弹出选择框")
    parser.add_argument("--label-dir", help="标签文件夹路径；默认尝试同名目录、labels/ 和 label/")
    parser.add_argument("--label-ext", default="txt", help="标签文件扩展名，默认 txt")
    parser.add_argument("--hard", action="store_true", help="永久删除，而不是移动到 _deleted/")
    return parser.parse_args()


def main():
    args = parse_args()
    label_ext = normalize_ext(args.label_ext)

    root = tk.Tk()
    root.withdraw()

    if args.folder:
        folder = Path(args.folder).expanduser()
    else:
        selected = filedialog.askdirectory(title="选择要筛选的图片文件夹")
        if not selected:
            print("未选择文件夹，已退出。")
            root.destroy()
            return
        folder = Path(selected)

    if not folder.exists() or not folder.is_dir():
        messagebox.showerror("路径错误", f"文件夹不存在：\n{folder}")
        root.destroy()
        return

    label_dir = Path(args.label_dir).expanduser() if args.label_dir else None
    if label_dir is not None and (not label_dir.exists() or not label_dir.is_dir()):
        messagebox.showerror("路径错误", f"标签文件夹不存在：\n{label_dir}")
        root.destroy()
        return

    root.deiconify()
    ImageFilterApp(root, folder, label_ext, label_dir, args.hard)
    root.mainloop()


if __name__ == "__main__":
    main()
