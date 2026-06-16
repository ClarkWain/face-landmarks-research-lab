"""Generate a compact hero image for README — three example faces with our INT8 vs FAN4 INT8 vs GT overlaid on each face."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.data import _collect_samples_from_folder, _load_pts
from landmarklab.model import build_model


def _crop(landmarks, image, size=320, scale=1.30):
    width, height = image.size
    mn = landmarks.min(dim=0).values
    mx = landmarks.max(dim=0).values
    cx, cy = ((mn + mx) * 0.5).tolist()
    side = max((mx - mn).tolist()) * scale
    left = max(0.0, cx - side * 0.5)
    top = max(0.0, cy - side * 0.5)
    right = min(float(width - 1), left + side)
    bottom = min(float(height - 1), top + side)
    cw = max(2.0, right - left)
    ch = max(2.0, bottom - top)
    cropped = TF.resized_crop(
        image, top=int(round(top)), left=int(round(left)),
        height=max(2, int(round(ch))), width=max(2, int(round(cw))),
        size=[size, size], interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    lmk = landmarks.clone()
    lmk[:, 0] = (lmk[:, 0] - left) / cw
    lmk[:, 1] = (lmk[:, 1] - top) / ch
    return cropped, lmk


def _draw(image, points, color, r=3):
    out = image.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for x, y in points.tolist():
        cx, cy = x * w, y * h
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(255, 255, 255))
    return out


@torch.no_grad()
def main() -> None:
    indices = [350, 200, 410]  # hand-picked diverse poses
    int8_path = "runs/300w_lmnet_w26_100k_finetune/model_int8.onnx"
    teacher_path = "runs/300w_fan4_finetune/best.pt"
    dataset_root = Path("data/300w_extracted/300w_extracted/300W")

    samples = _collect_samples_from_folder(dataset_root / "01_Indoor")
    samples += _collect_samples_from_folder(dataset_root / "02_Outdoor")

    student = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher_model = build_model(teacher_payload["config"])
    teacher_model.load_state_dict(teacher_payload["model"])
    teacher_model.eval()
    teacher_size = int(teacher_payload["config"]["data"]["image_size"])
    student_size = 224

    tile_h = 360
    grid = Image.new("RGB", (tile_h * 3, tile_h + 36), (245, 245, 245))
    draw_top = ImageDraw.Draw(grid)
    titles = ["Ground Truth (green)", "FAN4 INT8 (red)", "Our INT8 LMNet (blue)"]
    legend = "GT (green) - FAN4 INT8 24.79MB (red) - Ours INT8 15.48MB (blue)"

    for col, idx in enumerate(indices):
        image_path, pts_path = samples[idx]
        image = Image.open(image_path).convert("RGB")
        landmarks = _load_pts(pts_path)
        student_crop, gt_lmk = _crop(landmarks, image, size=student_size)
        teacher_crop, _ = _crop(landmarks, image, size=teacher_size)

        s_tensor = TF.to_tensor(student_crop)
        s_tensor = TF.normalize(s_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        student_pred = student.run(None, {"images": s_tensor.unsqueeze(0).numpy().astype(np.float32)})[0][0]

        t_tensor = TF.to_tensor(teacher_crop)
        t_tensor = TF.normalize(t_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        teacher_pred, _, _ = teacher_model.forward_train(t_tensor.unsqueeze(0))
        teacher_pred = teacher_pred[0].cpu()

        # overlay all three on the same student-resolution crop
        canvas = student_crop.resize((tile_h, tile_h), Image.BILINEAR)
        canvas = _draw(canvas, gt_lmk, (40, 200, 40))
        canvas = _draw(canvas, torch.from_numpy(student_pred), (60, 90, 240))
        canvas = _draw(canvas, teacher_pred, (220, 60, 60))
        grid.paste(canvas, (col * tile_h, 36))

    draw_top.text((10, 10), legend, fill=(20, 20, 20))
    out = Path("docs/images/hero.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
