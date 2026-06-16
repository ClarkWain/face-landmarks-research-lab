"""Generate a hero image for face106/README: 4 LaPa test samples with 106-point predictions.

Usage:
    py -3.12 scripts/make_hero.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarklab.model import build_model

ROOT = Path(__file__).resolve().parent.parent
LAPA_TEST = ROOT.parent / "data" / "LaPa" / "test"
CKPT = ROOT / "runs" / "lapa_hrnet_w18_awing_mixed_e80" / "best.pt"
OUT = ROOT / "docs" / "images" / "hero.png"

# 106-point semantic groups (LaPa convention)
GROUPS = {
    "contour": (range(0, 33), (200, 200, 200)),
    "right_brow": (range(33, 42), (66, 165, 245)),
    "left_brow": (range(42, 51), (66, 165, 245)),
    "nose": (range(51, 66), (76, 175, 80)),
    "right_eye": (range(66, 80), (244, 67, 54)),
    "left_eye": (range(80, 94), (244, 67, 54)),
    "mouth": (range(94, 106), (255, 193, 7)),
}

PICKS = [
    "10009865324_0",
    "10012551673_5",
    "10014368575_1",
    "262826415_0",
]


def resolve_pick(name: str) -> Path | None:
    p = LAPA_TEST / "images" / f"{name}.jpg"
    if p.exists():
        return p
    cands = list((LAPA_TEST / "images").glob(f"{name}*.jpg"))
    return cands[0] if cands else None


def crop_face(image: Image.Image, bbox: tuple[float, float, float, float],
              image_size: int = 256, crop_scale: float = 1.35):
    w, h = image.size
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * crop_scale
    left = max(0, cx - side / 2)
    top = max(0, cy - side / 2)
    right = min(w, left + side)
    bottom = min(h, top + side)
    crop_w = right - left
    crop_h = bottom - top
    cropped = image.crop((int(left), int(top), int(right), int(bottom))).resize(
        (image_size, image_size), Image.BILINEAR
    )
    return cropped, (left, top, crop_w, crop_h)


def landmarks_to_bbox(lms: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1 = lms[:, 0].min(), lms[:, 1].min()
    x2, y2 = lms[:, 0].max(), lms[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


def draw_panel(image: Image.Image, lms: np.ndarray, title: str) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    r = max(2, int(min(img.size) / 200))
    for name, (idx_range, color) in GROUPS.items():
        for i in idx_range:
            x, y = lms[i]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    # Title bar
    bar_h = max(28, int(img.size[1] * 0.06))
    panel = Image.new("RGB", (img.size[0], img.size[1] + bar_h), (24, 24, 28))
    panel.paste(img, (0, bar_h))
    pdraw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(bar_h * 0.55)))
    except OSError:
        font = ImageFont.load_default()
    pdraw.text((10, max(2, (bar_h - font.size) // 2)), title, fill=(220, 220, 220), font=font)
    return panel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["model"])

    panels = []
    for pick in PICKS:
        img_path = resolve_pick(pick)
        if img_path is None:
            print(f"[skip] {pick} not found")
            continue
        # Use GT landmarks file to derive a bbox, fallback to full image
        lm_txt = LAPA_TEST / "landmarks" / f"{img_path.stem}.txt"
        image = Image.open(img_path).convert("RGB")
        if lm_txt.exists():
            arr = np.loadtxt(lm_txt, skiprows=1)  # first line is count "106"
            arr = arr.reshape(-1, 2)
            bbox = landmarks_to_bbox(arr)
        else:
            w, h = image.size
            bbox = (w * 0.15, h * 0.1, w * 0.85, h * 0.95)

        cropped, (cx, cy, cw, ch) = crop_face(image, bbox, image_size=256, crop_scale=1.35)
        tensor = TF.to_tensor(cropped)
        tensor = TF.normalize(tensor, mean=[0.5] * 3, std=[0.5] * 3)
        with torch.no_grad():
            out = model(tensor.unsqueeze(0).to(device))
        lms_norm = out.cpu().numpy()[0]
        lms_px = lms_norm.copy()
        lms_px[:, 0] = lms_norm[:, 0] * cw + cx
        lms_px[:, 1] = lms_norm[:, 1] * ch + cy

        # Crop panel area for display
        x1, y1, x2, y2 = bbox
        side = max(x2 - x1, y2 - y1) * 1.35
        cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
        l = max(0, cx0 - side / 2)
        t = max(0, cy0 - side / 2)
        r = min(image.size[0], l + side)
        b = min(image.size[1], t + side)
        crop_panel = image.crop((int(l), int(t), int(r), int(b))).resize((400, 400), Image.LANCZOS)
        # Adjust landmarks to the panel coords
        scale = 400 / (r - l)
        lms_panel = lms_px.copy()
        lms_panel[:, 0] = (lms_px[:, 0] - l) * scale
        lms_panel[:, 1] = (lms_px[:, 1] - t) * scale

        panel = draw_panel(crop_panel, lms_panel, title=img_path.stem)
        panels.append(panel)
        print(f"[ok] {img_path.name} -> 106 landmarks")

    if not panels:
        raise SystemExit("No panels produced")

    # Concat horizontally
    h = panels[0].size[1]
    total_w = sum(p.size[0] for p in panels) + 8 * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, h), (24, 24, 28))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.size[0] + 8

    # Add caption strip at the bottom
    cap_h = 40
    final = Image.new("RGB", (canvas.size[0], canvas.size[1] + cap_h), (16, 16, 20))
    final.paste(canvas, (0, 0))
    cdraw = ImageDraw.Draw(final)
    try:
        cf = ImageFont.truetype("arial.ttf", size=18)
    except OSError:
        cf = ImageFont.load_default()
    caption = ("face106 HRNet W18 + AWing + Mixed Data  |  LaPa test  |  "
               "contour=gray  brow=blue  nose=green  eye=red  mouth=yellow")
    cdraw.text((12, canvas.size[1] + (cap_h - 18) // 2), caption,
               fill=(180, 180, 180), font=cf)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUT, quality=92, optimize=True)
    print(f"\nSaved hero: {OUT}  size={final.size}  bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
