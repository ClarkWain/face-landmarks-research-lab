# face-det — Face Detection Research

> Goal: derive face boxes from the existing landmark datasets, train a strong face detector baseline, and benchmark it cross-domain on the 2,000-image `Test_data1` set.

[中文（默认）](README.md) | English

## Benchmark definition

- **Training sources**
  - LaPa: 18,168 train + 2,000 val + 2,000 test
  - JD-landmark: 26,386 images
  - WFLW_augmented: 74,950 train + 2,500 test
- **Total bbox-able images**: **126,004**
- **External benchmark**: `../data/Test_data1/`
  - `picture/`: 2,000 images
  - `rect/`: 2,000 bbox annotations in `x1 y1 x2 y2` format

## Metrics

- Train/validation: standard YOLO detection metrics (`mAP50-95`, `mAP50`, precision, recall)
- External benchmark (`Test_data1`):
  - `AP50`
  - `AP50-95`
  - `Recall@0.5`

## Design choices

1. **Bounding boxes**
   - LaPa / JD / WFLW have landmarks but not face boxes; we derive a face bbox from the landmark min/max box expanded by a fixed margin.
2. **Detector family**
   - Start with a strong single-class Ultralytics YOLO baseline.
3. **SOTA track**
   - First push the cross-domain `Test_data1` benchmark as high as possible on the self-built dataset, then decide whether to move to a RetinaFace / SCRFD family.

## Quick start

```powershell
# 1. Build the YOLO dataset (fast baseline uses LaPa only)
py -3.12 face-det/scripts/build_face_detection_dataset.py --profile lapa_only

# 2. Smoke train
py -3.12 face-det/scripts/train_yolo_face.py --epochs 1 --imgsz 640 --batch 16 --model yolov8n.pt --name smoke

# 3. Full training
py -3.12 face-det/scripts/train_yolo_face.py --epochs 30 --imgsz 640 --batch 32 --model yolov8n.pt --name yolo_face_baseline
```

## Current result

- Main run: `yolo_face_lapa_s` (`yolov8s.pt`, imgsz=640, batch=32)
- Training data: LaPa train 18,168 images (face boxes derived from 106-point landmarks)
- Validation set: LaPa val 2,000 images
- External benchmark: `Test_data1` 2,000 images + official `.rect` boxes

### Best numbers so far

**LaPa val (in-domain validation)**
- epoch 17: `Precision=0.997`, `Recall=0.997`, `mAP50=0.9948`, `mAP50-95=0.9638`

**Test_data1 (cross-domain benchmark)**
- `AP50 = 0.995`
- `AP50-95 = 0.5817`
- `Precision = 0.99997`
- `Recall = 1.00000`

Interpretation: on `Test_data1`, **AP50 is already near saturation**, meaning the detector is already a SOTA-level strong baseline for "does it find the face". Further gains would mainly come from improving the stricter `AP50-95` metric.

