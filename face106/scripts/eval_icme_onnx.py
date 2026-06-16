"""Evaluate INT8 ONNX model on ICME 2019 Test_data1.

Usage:
    cd face106
    py -3.12 scripts/eval_icme_onnx.py --onnx runs/lapa_hrnet_w18_awing_mixed_e80/model_int8.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_icme_test_data(test_dir: Path) -> list[dict]:
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
        with lmk_path.open("r") as f:
            lines = [l.strip() for l in f if l.strip()]
        coords = []
        for line in lines[1:107]:
            x, y = line.split()
            coords.append([float(x), float(y)])
        gt = np.array(coords, dtype=np.float32)
        with rect_path.open("r") as f:
            parts = f.read().strip().split()
        rect = [float(v) for v in parts[:4]]
        samples.append({"image_path": pic_path, "gt": gt, "rect": rect})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=[args.provider])
    input_name = sess.get_inputs()[0].name
    print(f"Loaded ONNX: {args.onnx}")

    test_dir = Path("../data/Test_data1")
    samples = load_icme_test_data(test_dir)
    print(f"Loaded {len(samples)} test samples")

    eye_groups = [
        list(range(66, 80)),
        list(range(80, 94)),
    ]

    nme_list = []
    acc5_list = []
    acc8_list = []
    acc10_list = []
    for i, s in enumerate(samples):
        image = Image.open(s["image_path"]).convert("RGB")
        gt = s["gt"]
        x1, y1, x2, y2 = s["rect"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * args.crop_scale
        w, h = image.size
        left = max(0.0, cx - side / 2)
        top = max(0.0, cy - side / 2)
        right = min(float(w), left + side)
        bottom = min(float(h), top + side)
        crop_w = max(2.0, right - left)
        crop_h = max(2.0, bottom - top)

        cropped = image.crop((int(round(left)), int(round(top)),
                              int(round(left + crop_w)), int(round(top + crop_h))))
        cropped = cropped.resize((args.image_size, args.image_size), Image.BILINEAR)
        arr = np.asarray(cropped, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr.transpose(2, 0, 1)[None].astype(np.float32)

        out = sess.run(None, {input_name: arr})[0][0]
        pred_px = out.copy()
        pred_px[:, 0] = out[:, 0] * crop_w + left
        pred_px[:, 1] = out[:, 1] * crop_h + top

        errors = np.linalg.norm(pred_px - gt, axis=1)
        left_eye = gt[eye_groups[0]].mean(axis=0)
        right_eye = gt[eye_groups[1]].mean(axis=0)
        interocular = np.linalg.norm(left_eye - right_eye)
        if interocular < 1e-6:
            continue
        nme = errors.mean() / interocular
        nme_list.append(nme)
        acc5_list.append((errors / interocular < 0.05).mean())
        acc8_list.append((errors / interocular < 0.08).mean())
        acc10_list.append((errors / interocular < 0.10).mean())
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(samples)}: running NME={np.mean(nme_list):.4f}")

    nme_arr = np.array(nme_list)
    fr8 = (nme_arr > 0.08).mean() * 100
    print("\n" + "=" * 50)
    print(f"ICME 2019 Test_data1 INT8 Results ({len(nme_list)} images)")
    print("=" * 50)
    print(f"  NME:         {nme_arr.mean():.4f}")
    print(f"  acc@0.05:    {np.mean(acc5_list)*100:.2f}%")
    print(f"  acc@0.08:    {np.mean(acc8_list)*100:.2f}%")
    print(f"  acc@0.10:    {np.mean(acc10_list)*100:.2f}%")
    print(f"  FR@0.08:     {fr8:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
