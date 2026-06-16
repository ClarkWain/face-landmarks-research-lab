"""Visualize generated pseudo labels on CelebA images for quick sanity check."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="data/celeba_pseudo_smoke.npz")
    parser.add_argument("--image-dir", default="data/celeba_ssl_20k/img_align_celeba")
    parser.add_argument("--output", default="runs/celeba_pseudo_preview.png")
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()

    payload = np.load(args.npz, allow_pickle=False)
    names = payload["names"]
    landmarks = payload["landmarks"]
    crops = payload["crops"]
    image_size = int(payload["image_size"])

    grid_cols = args.num
    tile = 256
    canvas = Image.new("RGB", (tile * grid_cols, tile))
    for i in range(min(args.num, len(names))):
        path = Path(args.image_dir) / str(names[i])
        image = Image.open(path).convert("RGB")
        left, top, side = crops[i]
        cropped = image.crop((left, top, left + side, top + side)).resize((tile, tile), Image.BILINEAR)
        draw = ImageDraw.Draw(cropped)
        for x_norm, y_norm in landmarks[i]:
            x = float(x_norm) * tile
            y = float(y_norm) * tile
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 0, 0))
        canvas.paste(cropped, (i * tile, 0))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"saved preview to {args.output}")


if __name__ == "__main__":
    main()
