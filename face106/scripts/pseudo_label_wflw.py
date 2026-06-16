"""Generate 106-point pseudo labels on WFLW images using a trained HRNet checkpoint.

Usage:
    cd face106
    py -3.12 scripts/pseudo_label_wflw.py --config configs/lapa_hrnet_w18_heatmap_ft.yaml --run runs/lapa_hrnet_w18_heatmap_ft --output data/wflw_pseudo_106 --batch-size 16
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarklab.core import load_config
from landmarklab.model import build_model


def load_wflw_image_paths(split_dir: Path) -> list[Path]:
    """Get all WFLW image paths from labels.csv without loading coords."""
    paths = []
    with (split_dir / "labels.csv").open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 206:
                continue
            # last column is the image path (absolute or relative)
            img_name = Path(row[-1]).name
            img_path = split_dir / "imgs" / img_name
            if img_path.exists():
                paths.append(img_path)
    return paths


def crop_and_resize_face(
    image: Image.Image,
    *,
    crop_scale: float = 1.35,
    image_size: int = 256,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Crop face region using simple center crop with scale, then resize."""
    w, h = image.size
    # Use center of image as face center (WFLW images are already face-cropped)
    cx, cy = w / 2, h / 2
    crop_size = max(w, h) * crop_scale
    left = max(0.0, cx - crop_size / 2)
    top = max(0.0, cy - crop_size / 2)
    right = min(float(w), left + crop_size)
    bottom = min(float(h), top + crop_size)
    crop_w = max(2.0, right - left)
    crop_h = max(2.0, bottom - top)

    image = TF.resized_crop(
        image,
        top=int(round(top)),
        left=int(round(left)),
        height=max(2, int(round(crop_h))),
        width=max(2, int(round(crop_w))),
        size=[image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return image, (left, top, crop_w, crop_h)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 106-point pseudo labels on WFLW images")
    parser.add_argument("--config", required=True, help="Config YAML of the trained 106-point model")
    parser.add_argument("--run", required=True, help="Run directory containing best.pt")
    parser.add_argument("--output", required=True, help="Output directory for pseudo labels")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = Path(args.run)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_size = args.image_size

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    imgs_out = output_dir / "imgs"
    imgs_out.mkdir(exist_ok=True)

    # Process both train and test splits
    wflw_root = Path(checkpoint["config"]["data"].get("root", "../data")) / "wflw_extracted"
    for split_name in ["train_data", "test_data"]:
        split_dir = wflw_root / split_name
        if not split_dir.exists():
            print(f"Skipping {split_name}: not found at {split_dir}")
            continue

        print(f"\n=== Processing {split_name} ===")
        image_paths = load_wflw_image_paths(split_dir)
        print(f"Found {len(image_paths)} images")

        pseudo_rows = []
        batch_images = []
        batch_paths = []

        for img_path in tqdm(image_paths, desc=split_name):
            image = Image.open(img_path).convert("RGB")
            w, h = image.size

            # Crop + resize
            cropped, (crop_left, crop_top, crop_w, crop_h) = crop_and_resize_face(
                image, crop_scale=args.crop_scale, image_size=input_size
            )
            tensor = TF.to_tensor(cropped)
            tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            batch_images.append(tensor)
            batch_paths.append(img_path)

            if len(batch_images) >= args.batch_size:
                batch = torch.stack(batch_images).to(device)
                preds = model(batch).cpu()  # (B, 106, 2) normalized [0,1]
                for i, pred in enumerate(preds):
                    # Convert normalized coords back to pixel coords
                    pixels = pred.numpy()
                    pixels[:, 0] = pixels[:, 0] * w
                    pixels[:, 1] = pixels[:, 1] * h
                    # Copy image to output
                    src = batch_paths[i]
                    dst = imgs_out / src.name
                    if not dst.exists():
                        Image.open(src).convert("RGB").save(dst)
                    pseudo_rows.append([str(dst.resolve())] + pixels.flatten().tolist())
                batch_images = []
                batch_paths = []

        # Process remaining
        if batch_images:
            batch = torch.stack(batch_images).to(device)
            preds = model(batch).cpu()
            for i, pred in enumerate(preds):
                pixels = pred.numpy()
                pixels[:, 0] = pixels[:, 0] * w
                pixels[:, 1] = pixels[:, 1] * h
                src = batch_paths[i]
                dst = imgs_out / src.name
                if not dst.exists():
                    Image.open(src).convert("RGB").save(dst)
                pseudo_rows.append([str(dst.resolve())] + pixels.flatten().tolist())

        # Write CSV: image_path, 106*2 coords (pixel), source
        csv_path = output_dir / f"{split_name}.csv"
        print(f"Writing {len(pseudo_rows)} pseudo labels to {csv_path}")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in pseudo_rows:
                writer.writerow(row)

    print(f"\nDone! Pseudo labels saved to {output_dir}")


if __name__ == "__main__":
    main()
