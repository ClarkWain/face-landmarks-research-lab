# NewFaceDetect Report (English)

[中文（默认）](REPORT.md) | English

Date: 2026-06-12

## Current best: self-designed LMNet w=2.6 + 100 k pseudo-labels + fine-tune, INT8 NME 5.94%

`300w_lmnet_w26_100k_finetune` first pre-trains on 100 k CelebA pseudo-labels + 300 W ground truth (Stage A), then fine-tunes Stage A's `best.pt` with a cosine LR (3e-4 → 1e-6) and `real_ratio=0.7` (Stage B). Stage B epoch 1 directly reaches valid 0.0514 / test 0.0587, and the next 12 epochs plateau and we stop.

| Metric | FP32 (epoch 1, best) | INT8 |
|---|---|---|
| valid NME | **0.0514** | – |
| test NME | **0.0587** | **0.0594** |
| valid_acc@0.08 | 83.01% | – |
| test_acc@0.08 | 77.30% | 76.87% |
| test_acc@0.10 | 85.24% | 85.02% |
| test_acc@0.05 | 54.98% | 53.54% |
| Disk size | 58.99 MB | **15.48 MB** ✓ |

**Key comparisons:**

- vs. INT8 FAN4 teacher (24.79 MB / NME 0.0961): we are **38% smaller and 38% more accurate**.
- vs. LMNet historical baseline (NME 0.298): **5.0× improvement**.
- vs. strict 5% target: FP32 valid is only 0.14 points above; INT8 test is 0.94 points above.
- vs. 10% fallback target: we are about 4.06 points below.

## Run history

| Stage | run | valid NME | test NME | INT8 NME | INT8 size |
|---|---|---|---|---|---|
| 1 | `300w_lmnet_pseudo_balanced` (50 ep, w=2.25, 20 k pseudo) | 0.0817 | 0.0766 | 0.0772 | 11.61 MB |
| 2 | `300w_lmnet_pseudo_finetune` (40 ep, w=2.25, ssl) | 0.0595 | 0.0667 | 0.0676 | 11.61 MB |
| 3 | `300w_lmnet_w26` (60 ep, w=2.6, 20 k pseudo) | 0.0800 | 0.0750 | – | 14.75 MB |
| 4 | `300w_lmnet_w26_finetune` (40 ep, w=2.6, ssl) | 0.0564 | 0.0639 | 0.0647 | 15.48 MB |
| 5 | `300w_lmnet_w26_celeba100k` (60 ep, w=2.6, 100 k pseudo) | 0.0744 | 0.0717 | – | 14.75 MB |
| 6 | **`300w_lmnet_w26_100k_finetune` (40 ep, w=2.6, ssl)** | **0.0514** | **0.0587** | **0.0594** | **15.48 MB** |

## Tech stack at a glance

1. **Data expansion**: the FAN4 teacher predicts 68-point pseudo-labels on 100 k CelebA face crops → `data/celeba_pseudo_100k.npz` (300 W has only 432 real training images, 230× expansion).
2. **WeightedRandomSampler**: per batch, real and pseudo are sampled by `real_ratio` (0.5 in Stage A, 0.7 in Stage B), preventing pseudo-data from dominating.
3. **Two-stage training**: Stage A uses big data + strong augmentation + OneCycle to teach the backbone a face representation; Stage B loads `best.pt` + small-LR cosine + higher `real_ratio` + gentle augmentation to refit the 300 W distribution.
4. **EMA decay 0.99**: keeps up at 13–200 steps/epoch (the default 0.999 cannot follow training on small data).
5. **`width_mult` 2.25 → 2.6**: parameters 11.5 M → 15.5 M, leaves room within INT8 size budget while improving expressiveness.
6. **`mean_shape` initialization + global FC head**: converges faster and is more stable than a PIP heatmap head on a small training set.

## Negative experiments (this round)

### `300w_lmnet_pip_pseudo` (PIP head)
50-epoch training plateaus at epoch 22 on valid_nme=0.235 / test_nme=0.215. The PIP head quantizes coordinates to roughly 1/28 of the inter-ocular distance on the 28 × 28 stage-2 grid; without `mean_shape` initialization the cls cross-entropy starts at ~6.6 (random), and even after the cls drops to 2.3 the sub-pixel offset hardly improves. Global FC head is decisively better at this scale.

### `300w_lmnet_wflw_pseudo` (add WFLW augmented ground truth)
50-epoch training ends at valid_nme=0.1037 / test_nme=0.0915 — *worse* than `300w_lmnet_pseudo_balanced`. Causes: (i) WFLW augmented is provided as 112 × 112 face crops (resolution too low); (ii) the WFLW68 subset has sub-pixel semantic offsets relative to 300 W ground truth at non-corner mouth/eye points; (iii) WFLW takes 60% of every batch, drifting the model away from the 300W test distribution.

## Earlier milestones

### Transfer learning hits NME < 10%

`300w_fan4_finetune`: with 2DFAN4 pretrained weights as initialization for `FANHeatmapNet(num_modules=4)`, fine-tuned on the full 300 W training set for 15 epochs (lr=2e-4 → 0 cosine, batch=8, raw MSE heatmap loss, EMA, no AMP).

- Test set `NME = 0.0802` (8.02%), `Acc@0.05 = 47.14%`, `Acc@0.08 = 70.34%`.
- Validation set `NME = 0.0789` (best epoch 15).
- 23.82 M parameters, FP32 ~90.87 MB.

### `300w_fan4_finetune` INT8 quantization
ONNX QDQ + per-channel + MinMax calibration.

- INT8 disk size **24.79 MB** (slightly above the 20 MB ceiling).
- INT8 test `NME = 0.0961` (9.61%) — still below the 10% fallback.
- INT8 `Acc@0.05 = 38.28%`, `Acc@0.08 = 61.51%`, `Acc@0.10 = 72.25%`.

### Critical fixes for FAN4 reproduction

1. The inference path of `FANHeatmapNet._heatmaps_to_coordinates` must use argmax + neighbour-difference sub-pixel offsets, not soft-argmax — otherwise pretrained weights deliver NME 0.91 instead of 0.115.
2. `forward_train` re-normalizes to `[0, 1]` (our pipeline outputs `[-1, 1]`).
3. The training-time loss is `mse_raw` (raw heatmap vs. Gaussian target), not `mse(sigmoid(heatmap), gaussian)` — the pretrained heatmap peaks at ≈0.96 and a sigmoid would saturate at 1.0 and destroy the calibration.

### Bare 300W training caps at NME ~0.30

Many earlier scratch LMNet experiments on 300W only converge in the 0.296–0.310 band regardless of head/loss/scheduler choices. A 32-image overfit experiment did succeed (NME 0.0568) but a 320-image overfit failed (NME 0.336), confirming that the bottleneck was data, not architecture.

## Pre-summary historical notes

- The full training/eval/quantization/visualization pipeline is functional.
- WFLW augmented data is structured as `train_data/imgs + train_data/labels.csv` and `test_data/...`, ready for use.
- On Windows we found `num_workers=8` triggers `shm.dll` / pagefile failures; `num_workers=0` is the stable setting.
- The earlier 98-point production model is around 7.13 M parameters (theoretical INT8 ~6.79 MB), already inside the size budget.
- Up to this round, the strict `Acc@0.08 > 98` target had not been met; only the 5% / 10% NME variant survived.
