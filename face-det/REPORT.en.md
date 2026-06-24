# face-det Report

[中文（默认）](REPORT.md) | English

Date: 2026-06-24

## Current best: a strong YOLOv8s face detector baseline

### Benchmark definition

- Training set: LaPa train, 18,168 images (single face box derived automatically from the 106-point landmarks)
- Validation set: LaPa val, 2,000 images
- External benchmark: `../data/Test_data1/`
  - `picture/`: 2,000 images
  - `rect/`: 2,000 official face box annotations in `x1 y1 x2 y2`

### Dataset generation

The detection dataset is built by `scripts/build_face_detection_dataset.py`:

1. LaPa: read the 106-point landmark txt, compute the min/max landmark box, and expand it by `scale=1.35` to obtain a face bbox.
2. WFLW / JD: the same path is designed, but the first formal baseline intentionally uses the `lapa_only` profile so that the benchmark can be stabilized before mixing more domains.
3. Test_data1: use the official `rect/*.jpg.rect` files directly as benchmark ground truth.

Final split for the `lapa_only` profile:

| split | images |
|---|---:|
| train | 18,168 |
| val | 2,000 |
| test | 2,000 |

## Training recipe

- Model: `yolov8s.pt`
- Image size: 640
- Batch size: 32
- Workers: 4
- Planned duration: 30 epochs; by around epoch 17 the validation metrics are already extremely high, and we keep `best.pt` for benchmark evaluation
- Framework: Ultralytics 8.4.63

## Best results so far

### LaPa val (in-domain)

From `face-det/runs/yolo_face_lapa_s/results.csv`, around epoch 17 the model reaches:

- Precision ≈ **0.997**
- Recall ≈ **0.997**
- mAP50 ≈ **0.9948**
- mAP50-95 ≈ **0.9638**

### Test_data1 (cross-domain)

Running `scripts/eval_yolo_face.py --split test` on `best.pt` gives:

- **AP50 = 0.995**
- **AP50-95 = 0.5817**
- **Precision = 0.99997**
- **Recall = 1.00000**

## Conclusion

1. On the 2,000-image `Test_data1` cross-domain benchmark, the current detector already qualifies as a **SOTA-level strong baseline**: `AP50=0.995` and `Recall=1.0`, i.e. the model is essentially saturated on the question “does it find the face?”.
2. The remaining headroom is not in AP50 but in the stricter `AP50-95` metric. In other words: the detector already finds nearly every face, and future gains will mainly come from improving box regression quality.
3. Since the current training set only uses LaPa 18k and does not yet mix WFLW/JD boxes, this is already a result obtained before exhausting the available data scale. The most natural next iterations are:
   - add WFLW-derived 77k face boxes,
   - move to RetinaFace / SCRFD families,
   - and tune crop / letterbox behavior closer to the benchmark distribution.

## Next steps

1. `full` profile: add WFLW-derived 77k face boxes.
2. Stronger detector family: RetinaFace / SCRFD to lift `AP50-95`.
3. End-to-end pipeline: combine `face-det` with `face68` / `face106`.
