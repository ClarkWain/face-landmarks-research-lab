"""Generate 68-landmark pseudo labels on CelebA using fine-tuned FAN4 teacher."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.model import build_model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/300w_fan4_finetune/best.pt")
    parser.add_argument("--image-dir", default="data/celeba_ssl_20k/img_align_celeba")
    parser.add_argument("--output", default="data/celeba_pseudo_landmarks.npz")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = payload["config"]
    model = build_model(model_config)
    model.load_state_dict(payload["model"])
    model = model.to(args.device).eval()

    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob("*.jpg"))[: args.max_samples]
    print(f"found {len(image_paths)} images under {image_dir}")

    landmarks_buffer: list[np.ndarray] = []
    paths_buffer: list[str] = []
    crop_buffer: list[np.ndarray] = []  # (left, top, side) of the center crop in original image

    def preprocess(path: Path) -> tuple[torch.Tensor, np.ndarray]:
        image = Image.open(path).convert("RGB")
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        cropped = image.crop((left, top, left + side, top + side))
        cropped = cropped.resize((args.image_size, args.image_size), Image.BILINEAR)
        tensor = TF.to_tensor(cropped)
        # match training pipeline: normalize to [-1, 1]
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return tensor, np.array([left, top, side], dtype=np.float32)

    iterator = iter(image_paths)
    pbar = tqdm(total=len(image_paths))
    while True:
        batch_tensors: list[torch.Tensor] = []
        batch_crops: list[np.ndarray] = []
        batch_paths: list[Path] = []
        for path in iterator:
            try:
                tensor, crop = preprocess(path)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {path}: {exc}")
                pbar.update(1)
                continue
            batch_tensors.append(tensor)
            batch_crops.append(crop)
            batch_paths.append(path)
            if len(batch_tensors) >= args.batch_size:
                break
        if not batch_tensors:
            break

        batch = torch.stack(batch_tensors).to(args.device)
        prediction, _, _ = model.forward_train(batch) if hasattr(model, "forward_train") else (model(batch), None, None)
        landmarks = prediction.cpu().numpy()  # (B, 68, 2) in [0, 1] within the 256 crop

        for path, crop, lmk in zip(batch_paths, batch_crops, landmarks):
            paths_buffer.append(path.name)
            landmarks_buffer.append(lmk.astype(np.float32))
            crop_buffer.append(crop)
        pbar.update(len(batch_tensors))

    pbar.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        names=np.array(paths_buffer),
        landmarks=np.stack(landmarks_buffer),
        crops=np.stack(crop_buffer),
        image_size=args.image_size,
    )
    print(f"saved {len(paths_buffer)} pseudo labels to {output_path}")


if __name__ == "__main__":
    main()
