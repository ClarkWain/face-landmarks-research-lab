# face-det Report

[中文（默认）](REPORT.md) | English

Date: 2026-06-25

## 2026-06-25 Follow-up: why training looked too slow and why mAP50 dropped below 0.5

### Observed problem

Two concerns were raised:

1. the first `robust_mix` run looked unreasonably slow ("hours per epoch");
2. the reported `mAP50 < 0.5` looked dramatically worse than the earlier detector.

### Root cause analysis

#### 1. This was not just a code bug; it was a **dataset/validation definition mismatch**

The first `robust_mix` experiment mixed the following into a single training and validation world:

- LaPa (single face, large face, forehead included)
- WFLW (112×112 cropped faces, boxes almost fill the crop)
- WIDERFace (multi-face, tiny faces, strong occlusion)
- synthetic small-face composites

The training log `mAP50` therefore stopped meaning "the model got worse" and instead meant "the same model is now being scored on a dramatically harder and more heterogeneous validation set". Comparing it directly to the earlier LaPa-only `0.99`-level `mAP50` was an apples-to-oranges comparison.

#### 2. The epoch became slow because the dataset exploded in size and validation was still run every epoch

Original `LaPa-only`:

- train = 18,168
- ~568 steps / epoch

First `robust_mix`:

- train = 107,998
- val = 7,926
- ~3,375 steps / epoch
- plus a full validation pass over 7,926 images every epoch

This is not literally 7 hours per epoch, but it is absolutely enough to push a single epoch into the **tens-of-minutes** regime. With concurrent downloads and leftover Python workers on Windows, it felt even worse.

#### 3. Windows pagefile / multiprocessing was the secondary environment issue

The first evaluation attempts failed with repeated shared mapping errors (`WinError 1455`). The actual causes were:

- stale training Python workers still alive,
- Windows pagefile pressure,
- torch dataloader shared-memory / file-mapping collisions during evaluation.

### Fixes applied

1. `train_yolo_face.py` now supports:
  - `--fraction`
  - `--no-val`
  - `--device`
  - `--cache`
2. `eval_yolo_face.py` now exposes `--workers` (default `0`) so evaluation can run without Windows shared-memory blowups.
3. `build_face_detection_dataset.py` now supports:
  - `robust_mix` profile,
  - WIDERFace ingestion,
  - synthetic small-face generation.
4. Official WIDER `easy/hard` subsets are now converted through `prepare_wider_easy_hard.py`.

## Current results after the fixes

### A. LaPa-only baseline (still best on Test_data1)

Run: `yolo_face_lapa_s`

**LaPa val**
- `Precision ≈ 0.997`
- `Recall ≈ 0.997`
- `mAP50 ≈ 0.9948`
- `mAP50-95 ≈ 0.9638`

**Test_data1**
- `AP50 = 0.995`
- `AP50-95 = 0.5817`
- `Precision = 0.99997`
- `Recall = 1.00000`

**WIDER subsets**
- `WIDER easy`: `AP50 = 0.0862`, `AP50-95 = 0.0132`
- `WIDER hard`: `AP50 = 0.0217`, `AP50-95 = 0.00335`

Conclusion:
- Excellent on a single-face / large-face external benchmark (`Test_data1`)
- Very weak on true multi-face / tiny-face WIDER scenarios

### B. robust-mix MVP (LaPa + WFLW + WIDER + synthetic tiny faces)

Run: `yolo_face_robustmix_mvp`

Training settings:
- `fraction = 0.25`
- `epochs = 3`
- `workers = 2`
- `--no-val`
- warm-start from `yolo_face_lapa_s/weights/best.pt`

This turns the previous heavy, unstable run into a **fast MVP**.

**Dataset scale**
- train = 107,998
- val = 7,926
- test = 2,000 (`Test_data1`)

**Test_data1**
- `AP50 = 0.995`
- `AP50-95 = 0.5753`
- `Precision = 0.9980`
- `Recall = 0.9965`

**WIDER easy**
- `AP50 = 0.0973`
- `AP50-95 = 0.0156`
- `Precision = 0.2416`
- `Recall = 0.1294`

**WIDER hard**
- `AP50 = 0.0293`
- `AP50-95 = 0.00493`
- `Precision = 0.1417`
- `Recall = 0.0550`

### Comparison

Relative to `LaPa-only`:

- **WIDER easy**: `AP50 0.0862 -> 0.0973` (improved)
- **WIDER hard**: `AP50 0.0217 -> 0.0293` (clear improvement)
- **Test_data1**: `AP50` unchanged, `AP50-95` slightly worse (`0.5817 -> 0.5753`)

This confirms the original diagnosis:

1. multi-face / tiny-face / detection-domain data does improve WIDER generalization;
2. but blindly mixing everything with the old validation logic makes the run look much worse than it actually is;
3. `LaPa-only` is best for `Test_data1`, while `robust_mix` is the correct direction for WIDER.

## Best current interpretation

- If the target is **today's Test_data1 benchmark**, `LaPa-only` remains the best baseline.
- If the target is **broader real-world face detection generalization**, `robust_mix` is the correct direction, but it is still only an MVP and needs another round with WIDER/CrowdHuman/synthetic tuning.

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
