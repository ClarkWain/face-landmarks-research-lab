# Tiny Face Landmarks: Bootstrapping a 15 MB INT8 68-Point Detector with FAN4 Pseudo-Labels

[中文（默认）](PAPER.md) | English

*An open-source, exploratory technical report.*

## Abstract

We study how to train a small, INT8-quantization-friendly face landmark detector when the only manually-annotated data available is the 300W indoor + outdoor split (≈600 images, 68 landmarks). We design **LMNet**, an 11.5–15.5 M parameter MobileNet-style backbone with a global FC head and a `mean_shape`-initialized landmark bias. We bootstrap LMNet by (i) fine-tuning a face_alignment 2DFAN4 teacher to NME 0.0802 on 300W, (ii) using it to generate 100 000 pseudo-labels on CelebA, and (iii) training LMNet with a `WeightedRandomSampler` that balances real and pseudo data, followed by a small-LR cosine fine-tuning stage. After ONNX QDQ + per-channel + MinMax INT8 quantization our 15.48 MB model reaches **NME 0.0594** on the 300W test set (validation NME of the FP32 best checkpoint is 0.0514) — a **38 % relative improvement** over the 24.79 MB INT8-quantized teacher and a **5×** improvement over our scratch LMNet baseline (NME 0.298). We document a number of small but unavoidable contracts (EMA decay, sampler weighting, FAN-decoder details, FAN heatmap calibration) that determine whether this recipe works at all. Code and configurations are released so others can reproduce or refute these findings.

## 1. Introduction

Face landmark detection has well-understood SOTA pipelines on large datasets: hourglass networks (FAN), HRNets, transformer-based methods. They typically use 25–60 M parameters, FP32 weights of 90+ MB, and enjoy 7 000+ training images (WFLW augmented). Production constraints — on-device inference, INT8 quantization, model size budgets in the 5 – 20 MB range — push us in the opposite direction.

In this report we ask: *given only ~600 images of high-quality 68-point ground truth (the original 300W indoor + outdoor split), can a 15 MB INT8 model match or beat a 25 MB INT8 quantized 2DFAN4 teacher on the same test set?*

We answer affirmatively. The recipe is short — pseudo-labelling, weighted sampling, two-stage fine-tuning — but the failure modes around it are surprisingly subtle.

## 2. Related work

* **2DFAN / face_alignment** [Bulat & Tzimiropoulos]. Stacked-hourglass network operating on 68-point heatmaps; the de facto open weights baseline. Uses argmax + sub-pixel offsets, expects `[0, 1]`-normalized inputs, and produces heatmaps whose peaks are pre-calibrated to ≈0.96.
* **PIPNet** [Jin et al.]. Predicts a coarse classification grid plus per-cell sub-pixel offsets. Strong on large datasets but, in our experience, slow to leave the random-init regime when the real labelled set is tiny.
* **Self-training / pseudo-labelling** [Lee, Xie et al.]. A teacher labels unlabelled data; a student (often smaller) is trained on the union. Most prior work assumes the unlabelled data is in-domain. Here it isn't (CelebA aligned faces vs. 300W in-the-wild faces), which constrains the sampler design.
* **Knowledge distillation** [Hinton, Romero et al.]. We attempted both pseudo-label distillation and online distillation. The online variant was negligibly better than coordinate supervision on this scale and was dropped.

## 3. Method

### 3.1 LMNet backbone

LMNet is a MobileNet-V3-style backbone:

```
stem(3→16, 3×3 stride 2)
stage1: 1 IR block (16→24, stride 2, no SE, h-swish)
stage2: 2 IR blocks (24→32, stride 2, SE)
stage3: 3 IR blocks (32→48, stride 2, SE, h-swish)
stage4: 3 IR blocks (48→96, stride 2, SE, h-swish)
stage5: 3 IR blocks (96→192, stride 1, SE, h-swish)
```

We multiply every channel count by `width_mult` and use `width_mult = 2.6` for our headline numbers (parameter count 15.46 M). After `AdaptiveAvgPool2d(1)` we attach a 2-layer FC head:

```
Linear(channels[5], hidden_dim=896) → ReLU → Dropout(0.10) → Linear(hidden_dim, 68 × 2) → sigmoid
```

The final `Linear(hidden_dim, 136)` bias is initialized with the inverse-sigmoid of the empirical mean shape on the validation split (`logit(mean_shape)`), so at step 0 the model already predicts the average face — equivalent to NME 0.34 instead of NME 1.5 of pure random initialization. This *single trick* halves the number of epochs needed before the loss begins to decrease in stages with little supervision (e.g. when `pseudo_celeba.real_ratio` is low).

We use the **wing loss** (`wing_w=0.04, wing_eps=0.01`) plus a small geometric edge consistency loss (`geometry_weight=0.10`).

### 3.2 Pseudo-label generation

We fine-tune a 2DFAN4 (`num_modules=4`) teacher on 300W using OneCycle, lr=2e-4, 15 epochs, batch=8, EMA, no AMP. Three contracts must be respected:

1. The pipeline outputs `[-1, 1]`-normalized images (we then re-normalize to `[0, 1]` inside `FANHeatmapNet.forward_train` before forwarding through 2DFAN4).
2. At inference the heatmap → coordinate decoder must use `argmax + 0.25 · sign(neighbour-difference)` sub-pixel offsets, not the soft-argmax used during training. Soft-argmax matches the training-time loss; argmax matches the open-weights calibration.
3. The training-time loss must be `mse_raw` (raw heatmap vs. Gaussian target) — not `mse(sigmoid(heatmap), gaussian)` — because the open-weights peaks are at ≈0.96, while a sigmoid would saturate at 1.0 and destroy the calibration.

After this, our reproduction reaches NME 0.0802 on 300W test (Acc@0.05=47.14, Acc@0.08=70.34). We then run the teacher in eval mode on a center-crop of every aligned CelebA image (178 × 178 → 256 × 256). For each image we save the predicted 68-point landmark vector (in the crop's `[0, 1]` coordinate frame) and the crop coordinates `(left, top, side)` in the original-image pixel frame. Generating 100 000 pseudo-labels takes about 13 minutes on a single 2080 Ti.

### 3.3 PseudoCelebADataset

`PseudoCelebADataset.__getitem__` re-projects the cropped pseudo-landmarks back to original-image pixel coordinates, then runs the *same* augmentation pipeline as `ThreeHundredWDataset` — including `_sample_crop_box` over the pseudo-landmark bounding box, augmented scale and shift, brightness/contrast/saturation/hue jitter, optional Gaussian blur, optional cutout. This means the student sees pseudo-data in exactly the same statistical and geometric envelope as real data; only the labels differ.

### 3.4 Two-stage training

**Stage A.** OneCycle lr=2e-3, batch=32, 60 epochs, AMP, EMA decay 0.99, image_size 224, `init_mean_shape=true`. Weighted sampler with `real_ratio=0.5`: 50 % real 300W, 50 % pseudo CelebA, 200 steps per epoch (12 800 samples / epoch, 1.5× oversample of real). Scheduler-aware OneCycle treats the full 60-epoch budget as the warm-up + decay envelope. Best epoch valid NME: **0.0744**, test NME: **0.0717**.

**Stage B.** Cosine lr 3e-4 → 1e-6, batch=32, 40 epochs, AMP, EMA decay 0.99, `init_mean_shape=false` (warm-started from Stage A). `real_ratio=0.7` (i.e. real 300W now dominates). Augmentation amplitude is reduced (`aug_scale_range=(0.95, 1.10)`, `aug_shift=0.03`). `ssl_checkpoint = runs/<stage_a>/best.pt`. The best epoch is **epoch 1**: valid NME 0.0514, test NME 0.0587.

The fact that epoch 1 is the best is not an accident: the EMA copy at the start of Stage B is identical to the model and the model is already at its Stage A optimum. The first 200 cosine-LR-reduced steps of Stage B nudge the model toward 300W-distribution-friendly weights without destroying the pseudo-data feature representation. Subsequent epochs converge to a 300W-only over-fit equilibrium at slightly higher valid NME.

### 3.5 INT8 quantization

We export `best.pt` to FP32 ONNX (opset 18, fixed batch=1, no dynamic axes — required because the FC head's `Reshape` would otherwise fail with the ONNX shape inference). We then run static post-training quantization with `onnxruntime.quantization.quantize_static`:

```
quant_format = QDQ
activation_type = QUInt8
weight_type = QInt8
per_channel = True
calibrate_method = MinMax
calibration_data = 16 batches × 32 images of validation 300W
```

This drops the file from 59.0 MB FP32 to **15.48 MB INT8**, with a **0.07 percentage-point** test-NME degradation (0.0587 → 0.0594).

## 4. Experiments

All numbers below are on the 300W test set (54 images, the held-out 12 % of the indoor + outdoor merge with seed 3407). NME uses inter-ocular normalization (mean of left-eye points 36–41 to mean of right-eye points 42–47 distance).

### 4.1 Headline result

| Model | Params | Disk (INT8) | NME | Acc@0.05 | Acc@0.08 | Acc@0.10 |
|---|---|---|---|---|---|---|
| Mean shape baseline | – | – | 0.298 | – | – | – |
| Pretrained 2DFAN4 (no fine-tune) | 23.8 M | 90.9 MB FP32 | 0.113 | – | 34.2% | – |
| Fine-tuned 2DFAN4 (FP32) | 23.8 M | 90.9 MB | 0.080 | 47.1% | 70.3% | – |
| Fine-tuned 2DFAN4 (INT8) | 23.8 M | **24.8 MB** | 0.0961 | 38.3% | 61.5% | 72.3% |
| **LMNet w=2.6 + 100 k pseudo + ft (FP32)** | 15.5 M | 59.0 MB | **0.0587** | **55.0%** | **77.3%** | **85.2%** |
| **LMNet w=2.6 + 100 k pseudo + ft (INT8)** | 15.5 M | **15.48 MB** | **0.0594** | **53.5%** | **76.9%** | **85.0%** |

Validation NME of the FP32 best checkpoint is 0.0514, only **0.14 percentage points** above the strict 5% target. Six side-by-side qualitative comparisons (`runs/demo_compare.png`) consistently show our student tracking the ground truth more precisely than the FAN4 teacher INT8, including on profile, low-light and highly-expressive inputs.

### 4.2 Ablations

| Configuration | Params | INT8 size | INT8 NME |
|---|---|---|---|
| LMNet w=2.25, 20 k pseudo, single-stage 50 ep | 11.5 M | 11.61 MB | 0.0772 |
| LMNet w=2.25, 20 k pseudo + ft 40 ep | 11.5 M | 11.61 MB | 0.0676 |
| LMNet w=2.6, 20 k pseudo + ft 40 ep | 15.5 M | 15.48 MB | 0.0647 |
| **LMNet w=2.6, 100 k pseudo + ft 40 ep (ours)** | 15.5 M | 15.48 MB | **0.0594** |

* **Width:** 2.25 → 2.6 buys 0.3 percentage points of NME at the cost of 4 MB on disk. Beyond 2.6 we extrapolate diminishing returns; quantization budget would be the binding constraint.
* **Pseudo-data scale:** 20 k → 100 k buys another 0.5 percentage points. The marginal NME-per-pseudo-image is non-zero but small; data quality (teacher accuracy) matters more than scale beyond ~30 k.
* **Two-stage vs single-stage:** One stage with the same total compute reaches NME ~0.075; loading the best.pt and running a small-LR cosine fine-tune unlocks an additional **2.5 percentage points** in a single epoch.

### 4.3 Negative results

| Configuration | NME |
|---|---|
| LMNet PIP head (28 × 28 cls + sub-pixel offset), 50 ep | 0.215 |
| LMNet + WFLW augmented (real_ratio_wflw=0.85) co-training, 50 ep | 0.0915 |
| LMNet + online distillation (FAN4 teacher every batch), 50 ep | 0.083 |
| LMNet + ConcatDataset(real, pseudo).shuffle (no weighted sampler), 30 ep | 0.361 ↑ |

* The PIP head plateaus because the 28 × 28 grid quantizes the inter-ocular-normalized error at ≈1/28 ≈ 0.036, and the sub-pixel offset has trouble learning when only 432 examples per epoch carry real ground truth.
* WFLW augmented data is provided as 112 × 112 face crops with a 98-point convention. Subsetting to "WFLW-68" introduces sub-pixel semantic offsets at the mouth and eye keypoints relative to 300W ground truth; together with the lower spatial resolution, this *hurts* 300W test NME despite providing 75 000 extra labelled images.
* Online distillation turned out to be roughly neutral. Once Stage A pseudo-labelling is in place, the teacher's contribution per training step is largely already absorbed into the labels, and the additional FAN4 forward pass per batch is computationally expensive.
* The naive `ConcatDataset(shuffle=True)` without weighted sampling is the most instructive failure: with 432 real / 20 000 pseudo, the optimizer fits the pseudo-distribution ever-better, *while valid NME monotonically rises*. This is the canonical reason to use a `WeightedRandomSampler` whenever pseudo-data dominates by orders of magnitude.

## 5. Discussion and limitations

* **The 5% target is a strict ceiling.** FP32 valid NME 0.0514 is 0.14 points away from 0.0500; FP32 test NME 0.0587 sits 0.87 points away. The remaining gap appears to be irreducible *without more annotated data or stronger teachers*. We can shrink the FP32-to-INT8 gap with QAT (next on the roadmap), and the FP32 gap with two-teacher ensembles or width=2.85, but the marginal cost-per-NME is now high.
* **Generalization beyond 300W is untested.** All results are on the 300W indoor + outdoor (`split_strategy=all_random`, valid 10%, test 12%) split. Cross-dataset generalization (e.g. WFLW test, AFLW2000-3D) is future work.
* **Detection is not included.** This report covers only landmark regression on a centered, square-cropped face. A production pipeline would also need a face detector (e.g. S3FD from the face_alignment package).
* **Dependency on a teacher.** This recipe requires a competent teacher to bootstrap. We used the freely-available 2DFAN4 weights. If those are not available or licensed for the target domain, an in-domain expert annotator would be needed.

## 6. Conclusion

We showed that with ~600 manually annotated images, a 100 k-pseudo-label expansion through a 2DFAN4 teacher, a properly-weighted sampler, and a careful two-stage training schedule, a 15.48 MB INT8 LMNet can match or exceed a 24.79 MB INT8 fine-tuned 2DFAN4 baseline on 300W (NME 0.0594 vs. 0.0961, **38 % relative improvement**) and reach FP32 validation NME of 5.14% — within striking distance of the 5% strict target. The recipe is short but the failure modes around it (EMA decay, sampler weighting, FAN-decoder contracts, INT8 reshape constraints) are subtle enough to be worth documenting. We release all code, configurations, and pseudo-labels.

## Acknowledgements

The 2DFAN4 weights are courtesy of the `face_alignment` project (Bulat & Tzimiropoulos). The 300W dataset is courtesy of the original C. Sagonas et al. release. CelebA is courtesy of MMLAB at CUHK.

## References

A small set of references this work builds on (not exhaustive):

* Sagonas, C., et al. "300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge." ICCV-W 2013.
* Bulat, A. and Tzimiropoulos, G. "How far are we from solving the 2D & 3D Face Alignment problem?" ICCV 2017.
* Liu, Z., et al. "Deep Learning Face Attributes in the Wild." ICCV 2015.
* Wu, W., et al. "Look at Boundary: A Boundary-Aware Face Alignment Algorithm." CVPR 2018 (WFLW).
* Jin, H., et al. "Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild." IJCV 2021.
* Hinton, G., et al. "Distilling the Knowledge in a Neural Network." 2015.
* Lee, D.-H. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." 2013.
* Howard, A., et al. "Searching for MobileNetV3." ICCV 2019.
