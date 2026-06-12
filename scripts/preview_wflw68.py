"""Visualize WFLW68 subset mapping on a single sample to diagnose alignment."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.data import _load_wflw_rows, select_landmark_subset


def main() -> None:
    train_dir = Path("data/wflw_extracted/train_data")
    samples = _load_wflw_rows(train_dir)
    print(f"loaded {len(samples)} WFLW samples")
    out = Image.new("RGB", (320 * 4, 320))
    for i, (image_path, coords_98_norm, _pose) in enumerate(samples[:4]):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        # full-image: scale norm → pixel
        coords_98 = coords_98_norm.clone()
        coords_98[:, 0] *= w
        coords_98[:, 1] *= h
        # subset to 68
        coords_68 = select_landmark_subset(coords_98_norm, "wflw68")
        coords_68 = coords_68.clone()
        coords_68[:, 0] *= w
        coords_68[:, 1] *= h
        # draw
        canvas = image.copy().resize((320, 320), Image.BILINEAR)
        sx, sy = 320 / w, 320 / h
        d = ImageDraw.Draw(canvas)
        for j, (x, y) in enumerate(coords_68.tolist()):
            xs, ys = x * sx, y * sy
            color = (255, 0, 0) if j < 17 else (0, 255, 0) if j < 27 else (0, 0, 255) if j < 36 else (255, 255, 0) if j < 48 else (255, 0, 255)
            d.ellipse((xs - 2, ys - 2, xs + 2, ys + 2), fill=color)
            d.text((xs + 3, ys), str(j), fill=(255, 255, 255))
        out.paste(canvas, (i * 320, 0))
    out_path = Path("runs/wflw68_preview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
