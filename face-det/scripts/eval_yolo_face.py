from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="face-det/configs/face_yolo_dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_yaml = root / args.data
    weights = root / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=False,
        plots=False,
        verbose=True,
    )

    summary = {
        "weights": str(weights),
        "split": args.split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "mp": float(metrics.box.mp),
        "mr": float(metrics.box.mr),
    }
    print(summary)

    if args.out:
        out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()