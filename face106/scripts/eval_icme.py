"""Evaluate a trained 106-point model on ICME 2019 Test_data1.

Usage:
    cd face106
    py -3.12 scripts/eval_icme.py --run runs/lapa_hrnet_w18_awing_mixed_e80 --image-size 384
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarklab.core import load_config
from landmarklab.model import build_model


def load_icme_test_data(test_dir: Path) -> list[dict]:
    """Load ICME Test_data1: picture + landmark + rect."""
    samples = []
    pic_dir = test_dir / "picture"
    lmk_dir = test_dir / "landmark"
    rect_dir = test_dir / "rect"
    
    for pic_path in sorted(pic_dir.glob("*.jpg")):
        stem = pic_path.stem
        lmk_path = lmk_dir / f"{stem}.jpg.txt"
        rect_path = rect_dir / f"{stem}.jpg.rect"
        
        if not lmk_path.exists() or not rect_path.exists():
            continue
        
        # Load landmark: first line is count (106), rest are x y
        with lmk_path.open("r") as f:
            lines = [l.strip() for l in f if l.strip()]
        count = int(lines[0])
        coords = []
        for line in lines[1:count+1]:
            x, y = line.split()
            coords.append([float(x), float(y)])
        gt = torch.tensor(coords, dtype=torch.float32)  # (106, 2) pixel coords
        
        # Load rect: x1 y1 x2 y2
        with rect_path.open("r") as f:
            parts = f.read().strip().split()
        rect = [float(v) for v in parts[:4]]
        
        samples.append({
            "image_path": pic_path,
            "gt": gt,
            "rect": rect,
        })
    
    return samples


def compute_nme(pred: np.ndarray, gt: np.ndarray, eye_groups: list[list[int]]) -> float:
    """Compute NME normalized by inter-ocular distance."""
    left_eye_center = gt[eye_groups[0]].mean(axis=0)
    right_eye_center = gt[eye_groups[1]].mean(axis=0)
    interocular = np.linalg.norm(left_eye_center - right_eye_center)
    if interocular < 1e-6:
        return 999.0
    errors = np.linalg.norm(pred - gt, axis=1)
    return float(errors.mean() / interocular)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate on ICME 2019 Test_data1")
    parser.add_argument("--run", required=True, help="Run directory with best.pt")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    run_dir = Path(args.run)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    input_size = args.image_size
    
    # Load test data
    test_dir = Path(args.run).parent.parent / "data" / "Test_data1"
    if not test_dir.exists():
        test_dir = Path("../data/Test_data1")
    samples = load_icme_test_data(test_dir)
    print(f"Loaded {len(samples)} test samples")
    
    # ICME uses same eye groups as LaPa (indices 66-79 left eye, 80-93 right eye)
    eye_groups = [[66,67,68,69,70,71,72,73,74,75,76,77,78,79],
                  [80,81,82,83,84,85,86,87,88,89,90,91,92,93]]
    
    nme_list = []
    acc_005_list = []
    acc_008_list = []
    acc_010_list = []
    
    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        gt = sample["gt"].numpy()  # (106, 2)
        rect = sample["rect"]  # x1, y1, x2, y2
        x1, y1, x2, y2 = rect
        
        # Crop using rect with some margin
        w, h = image.size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = (x2 - x1) * args.crop_scale
        bh = (y2 - y1) * args.crop_scale
        crop_size = max(bw, bh)
        
        crop_left = max(0, cx - crop_size / 2)
        crop_top = max(0, cy - crop_size / 2)
        crop_right = min(w, crop_left + crop_size)
        crop_bottom = min(h, crop_top + crop_size)
        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        
        # Crop and resize
        cropped = TF.resized_crop(
            image,
            top=int(round(crop_top)),
            left=int(round(crop_left)),
            height=max(2, int(round(crop_h))),
            width=max(2, int(round(crop_w))),
            size=[input_size, input_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        
        tensor = TF.to_tensor(cropped)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        tensor = tensor.unsqueeze(0).to(device)
        
        # Inference
        pred_norm = model(tensor).cpu().squeeze(0).numpy()  # (106, 2) in [0,1]
        
        # Convert normalized coords back to original image pixel coords
        pred_px = pred_norm.copy()
        pred_px[:, 0] = pred_norm[:, 0] * crop_w + crop_left
        pred_px[:, 1] = pred_norm[:, 1] * crop_h + crop_top
        
        # Compute per-point errors
        errors = np.linalg.norm(pred_px - gt, axis=1)
        
        # NME
        left_eye = gt[eye_groups[0]].mean(axis=0)
        right_eye = gt[eye_groups[1]].mean(axis=0)
        interocular = np.linalg.norm(left_eye - right_eye)
        if interocular < 1e-6:
            continue
        nme = errors.mean() / interocular
        nme_list.append(nme)
        
        # Per-point accuracy
        acc_005_list.append((errors / interocular < 0.05).mean())
        acc_008_list.append((errors / interocular < 0.08).mean())
        acc_010_list.append((errors / interocular < 0.10).mean())
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(samples)}: running NME={np.mean(nme_list):.4f}")
    
    nme_arr = np.array(nme_list)
    acc5_arr = np.array(acc_005_list)
    acc8_arr = np.array(acc_008_list)
    acc10_arr = np.array(acc_010_list)
    
    # FR@0.08
    fr_008 = (nme_arr > 0.08).mean() * 100
    
    print(f"\n{'='*50}")
    print(f"ICME 2019 Test_data1 Results ({len(nme_list)} images)")
    print(f"{'='*50}")
    print(f"  NME:         {nme_arr.mean():.4f}")
    print(f"  acc@0.05:    {acc5_arr.mean()*100:.2f}%")
    print(f"  acc@0.08:    {acc8_arr.mean()*100:.2f}%")
    print(f"  acc@0.10:    {acc10_arr.mean()*100:.2f}%")
    print(f"  FR@0.08:     {fr_008:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
