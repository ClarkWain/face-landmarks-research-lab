from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.data import _collect_samples_from_folder, _prepare_300w, _load_pts


def normalize_box(landmarks, image_width: int, image_height: int):
    min_xy = landmarks.min(dim=0).values
    max_xy = landmarks.max(dim=0).values
    center = (min_xy + max_xy) * 0.5
    size = (max_xy - min_xy) * 1.10
    x_center = float(center[0] / image_width)
    y_center = float(center[1] / image_height)
    width = float(size[0] / image_width)
    height = float(size[1] / image_height)
    return x_center, y_center, width, height


def write_label(label_path: Path, landmarks, image_width: int, image_height: int) -> None:
    x_center, y_center, width, height = normalize_box(landmarks, image_width, image_height)
    values = [0, x_center, y_center, width, height]
    for point in landmarks:
        values.extend([float(point[0] / image_width), float(point[1] / image_height), 2])
    label_path.write_text(" ".join(str(value) for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a 32-image 300W YOLO pose overfit dataset")
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default="data/yolo300w32")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    dataset_root = _prepare_300w(args.root, download=False, archive_url="")
    if (dataset_root / "01_Indoor").exists() and (dataset_root / "02_Outdoor").exists():
        samples = _collect_samples_from_folder(dataset_root / "01_Indoor")
        samples.extend(_collect_samples_from_folder(dataset_root / "02_Outdoor"))
    else:
        raise RuntimeError("Expected 300W mirror with 01_Indoor/02_Outdoor folders")

    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.count]

    out_root = Path(args.out)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_path, pts_path in samples:
        landmarks = _load_pts(pts_path)
        destination_name = image_path.name
        for split in ("train", "val"):
            target_image = out_root / "images" / split / destination_name
            target_label = out_root / "labels" / split / f"{image_path.stem}.txt"
            shutil.copy2(image_path, target_image)
            from PIL import Image
            image = Image.open(image_path)
            write_label(target_label, landmarks, image.width, image.height)

    yaml_path = out_root / "300w32.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [68, 3]",
                f"flip_idx: [{', '.join(str(index) for index in range(68))}]",
                "names:",
                "  0: face",
            ]
        ),
        encoding="utf-8",
    )
    print(f"prepared {len(samples)} images at {out_root}")


if __name__ == "__main__":
    main()
