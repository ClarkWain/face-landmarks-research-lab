from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))

from landmarklab.core import compute_metrics


def read_label(path: Path) -> torch.Tensor:
    values = path.read_text(encoding="utf-8").split()
    coords = [float(value) for value in values[5:]]
    points = []
    for index in range(0, len(coords), 3):
        points.append([coords[index], coords[index + 1]])
    return torch.tensor(points, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO pose predictions with NME")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="data/yolo300w32")
    args = parser.parse_args()

    model = YOLO(args.weights)
    image_dir = Path(args.dataset) / "images" / "val"
    label_dir = Path(args.dataset) / "labels" / "val"
    predictions = []
    targets = []
    for image_path in sorted(image_dir.glob("*.*")):
        result = model.predict(source=str(image_path), verbose=False, device=0)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            prediction = torch.zeros((68, 2), dtype=torch.float32)
        else:
            prediction = result.keypoints.xyn[0].cpu().to(torch.float32)
        target = read_label(label_dir / f"{image_path.stem}.txt")
        predictions.append(prediction.unsqueeze(0))
        targets.append(target.unsqueeze(0))

    metrics = compute_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        eye_groups=[[36, 37, 38, 39, 40, 41], [42, 43, 44, 45, 46, 47]],
    )
    print(metrics)


if __name__ == "__main__":
    main()