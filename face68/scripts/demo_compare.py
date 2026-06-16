"""Compare predictions of our INT8 LMNet vs FAN4 teacher vs ground truth on 300W test images.

Outputs a side-by-side visualization grid to `runs/demo_compare.png`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.data import _collect_samples_from_folder, _load_pts
from landmarklab.model import build_model


def _crop_landmarks(landmarks: torch.Tensor, image: Image.Image, crop_scale: float = 1.30, image_size: int = 224) -> tuple[Image.Image, torch.Tensor, tuple[float, float, float, float]]:
    width, height = image.size
    min_xy = landmarks.min(dim=0).values
    max_xy = landmarks.max(dim=0).values
    box_w, box_h = (max_xy - min_xy).tolist()
    cx, cy = ((min_xy + max_xy) * 0.5).tolist()
    crop_size = max(box_w, box_h) * crop_scale
    left = max(0.0, cx - crop_size * 0.5)
    top = max(0.0, cy - crop_size * 0.5)
    right = min(float(width - 1), left + crop_size)
    bottom = min(float(height - 1), top + crop_size)
    crop_w = max(2.0, right - left)
    crop_h = max(2.0, bottom - top)
    cropped = TF.resized_crop(
        image,
        top=int(round(top)), left=int(round(left)),
        height=max(2, int(round(crop_h))), width=max(2, int(round(crop_w))),
        size=[image_size, image_size],
        interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    landmarks_in_crop = landmarks.clone()
    landmarks_in_crop[:, 0] = (landmarks_in_crop[:, 0] - left) / crop_w
    landmarks_in_crop[:, 1] = (landmarks_in_crop[:, 1] - top) / crop_h
    return cropped, landmarks_in_crop, (left, top, crop_w, crop_h)


def _draw_landmarks(image: Image.Image, landmarks: torch.Tensor, color: tuple[int, int, int], radius: int = 2) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    for x, y in landmarks.tolist():
        cx, cy = x * width, y * height
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    return out


def _label_image(image: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + 26), (240, 240, 240))
    out.paste(image, (0, 26))
    draw = ImageDraw.Draw(out)
    draw.text((6, 4), text, fill=(20, 20, 20))
    return out


def _compute_nme(pred: torch.Tensor, gt: torch.Tensor, left_eye: list[int], right_eye: list[int]) -> float:
    le = gt[left_eye].mean(dim=0)
    re = gt[right_eye].mean(dim=0)
    inter = float(torch.linalg.norm(le - re))
    if inter < 1e-6:
        return float("nan")
    err = torch.linalg.norm(pred - gt, dim=-1).mean().item()
    return err / inter


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8-onnx", default="runs/300w_lmnet_w26_100k_finetune/model_int8.onnx")
    parser.add_argument("--teacher-checkpoint", default="runs/300w_fan4_finetune/best.pt")
    parser.add_argument("--dataset-root", default="data/300w_extracted")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="runs/demo_compare.png")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if (dataset_root / "01_Indoor").exists():
        samples = _collect_samples_from_folder(dataset_root / "01_Indoor")
        samples += _collect_samples_from_folder(dataset_root / "02_Outdoor")
    else:
        raise SystemExit(f"Cannot find 300W under {dataset_root}")

    rng = np.random.default_rng(7)
    indices = rng.choice(len(samples), size=args.num_samples, replace=False)

    int8_session = ort.InferenceSession(args.int8_onnx, providers=["CPUExecutionProvider"])

    teacher_payload = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher = build_model(teacher_payload["config"])
    teacher.load_state_dict(teacher_payload["model"])
    teacher.eval()
    teacher_image_size = int(teacher_payload["config"]["data"]["image_size"])

    left_eye = [36, 37, 38, 39, 40, 41]
    right_eye = [42, 43, 44, 45, 46, 47]

    tile_size = args.image_size
    panels_per_row = 3  # GT, Teacher, Student
    grid = Image.new("RGB", (tile_size * panels_per_row, (tile_size + 26) * args.num_samples), (255, 255, 255))

    for row, idx in enumerate(indices):
        image_path, pts_path = samples[int(idx)]
        image = Image.open(image_path).convert("RGB")
        landmarks = _load_pts(pts_path)

        # crop + resize for student (image_size=224)
        student_crop, gt_in_student_crop, _ = _crop_landmarks(landmarks, image, image_size=tile_size)
        student_tensor = TF.to_tensor(student_crop)
        student_tensor = TF.normalize(student_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        student_pred = int8_session.run(None, {"images": student_tensor.unsqueeze(0).numpy().astype(np.float32)})[0][0]
        student_pred_t = torch.from_numpy(student_pred)
        nme_student = _compute_nme(student_pred_t, gt_in_student_crop, left_eye, right_eye)

        # teacher (image_size=256)
        teacher_crop, gt_in_teacher_crop, _ = _crop_landmarks(landmarks, image, image_size=teacher_image_size)
        teacher_tensor = TF.to_tensor(teacher_crop)
        teacher_tensor = TF.normalize(teacher_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        teacher_pred, _, _ = teacher.forward_train(teacher_tensor.unsqueeze(0)) if hasattr(teacher, "forward_train") else (teacher(teacher_tensor.unsqueeze(0)), None, None)
        teacher_pred = teacher_pred[0].cpu()
        nme_teacher = _compute_nme(teacher_pred, gt_in_teacher_crop, left_eye, right_eye)

        gt_panel = _label_image(_draw_landmarks(student_crop, gt_in_student_crop, (0, 200, 0)), "Ground Truth")
        teacher_panel = _label_image(_draw_landmarks(teacher_crop.resize((tile_size, tile_size), Image.BILINEAR), teacher_pred, (255, 0, 0)),
                                     f"FAN4 Teacher (NME {nme_teacher:.4f})")
        student_panel = _label_image(_draw_landmarks(student_crop, student_pred_t, (0, 80, 255)),
                                     f"Our INT8 LMNet (NME {nme_student:.4f})")

        grid.paste(gt_panel, (0, row * (tile_size + 26)))
        grid.paste(teacher_panel, (tile_size, row * (tile_size + 26)))
        grid.paste(student_panel, (tile_size * 2, row * (tile_size + 26)))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"saved comparison to {args.output}")


if __name__ == "__main__":
    main()
