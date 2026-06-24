from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default="yolo_face_baseline")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_yaml = root / "face-det" / "configs" / "face_yolo_dataset.yaml"

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(root / "face-det" / "runs"),
        name=args.name,
        pretrained=True,
        degrees=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        cache=False,
        device=0,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()