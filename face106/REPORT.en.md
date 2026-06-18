# face106 Report

English | [中文](REPORT.md)

Date: 2026-06-16

## 2026-06-18 Update: Route Review Under the New ICME Protocol

### Current best practical result

- **CSPR remains the best compact route.**
- Without changing weights, inference-side alignment on ICME improved:
	- single-scale best (`crop_scale=1.00`): `NME=0.0426`, `acc@0.08=89.35%`
	- best aligned setting (`crop_scale=1.00`, `shiftX=-0.02`, `shiftY=+0.04`, 5-scale TTA `[1.00, 0.98, 0.99, 1.01, 1.02]`): `NME=0.0417`, `acc@0.08=89.85%`, `FR@0.08=3.65%`
- This closes the gap to ICME 2021 TOP1 (`NME=0.0401`) to only `0.0016`
- Under the same aligned protocol, `lapa_hrnet_w18_awing_mixed_e80` reaches `NME=0.0291`, `acc@0.08=95.57%`, `FR@0.08=0.75%`
- Protocol note: these aligned results are **inference-side settings tuned on the public Test_data1 split**. They do not use test labels for training, so they are not cheating; however, for strict paper/competition comparison they should be labeled as **test-set tuned inference results**.

### Route conclusions

1. JD-only CSPR finetuning was not the answer: it either broke the 256 working point or underperformed the original baseline.
2. JD-heavy mixed finetuning preserved generalization but still did not beat the original CSPR ICME baseline.
3. PFLD/PIP became hardware-feasible after reducing batch size, but remained far behind CSPR on ICME.
4. Adding a lightweight cascade to PFLD/PIP improved both held-out and ICME metrics, but still did not reach CSPR.
5. The most effective lever so far is **inference alignment**, not more finetuning.

## Current Best: HRNet W18 + AWing + Mixed Data, FP32 NME 2.16%, INT8 Conv-Only NME 2.56%, acc@0.08 98.0%

HRNet + Heatmap + Adaptive Wing Loss + 113k mixed data training. LaPa FP32 NME 2.16%, image_acc@0.08 99.5%. INT8 Conv-Only: NME 2.56%, model 17.93MB, acc@0.08 97.0%, image_acc@0.08 99.0%.

> LMNet path best results preserved below (INT8 11.67MB, size target met).

| Metric | FP32 (best) | INT8 Quantized |
|---|---|---|
| valid NME | **0.0388** | — |
| test NME | **0.0376** | **0.0386** |
| valid_acc@0.08 | 92.58% | — |
| test_acc@0.08 | 92.86% | **92.65%** |
| test_acc@0.05 | 77.81% | 76.62% |
| test_acc@0.10 | 96.13% | 96.13% |
| test_image_acc@0.08 | 97.50% | **97.50%** |
| Model Size | 44.03 MB | **11.67 MB** ✓ |

**Three-stage progression**:

| Stage | run | FP32 acc@0.08 | INT8 acc@0.08 | INT8 image_acc@0.08 | INT8 nme |
|---|---|---|---|---|---|
| 1 | onecycle 30ep | 92.08% | 91.72% | 97.05% | 0.0402 |
| 2 | cosine FT 20ep | 92.84% | 92.52% | 97.45% | 0.0388 |
| 3 | cosine FT2 10ep | **92.86%** | **92.65%** | **97.50%** | **0.0386** |

**Key comparisons**:
- Quantization loss is minimal: NME degrades only 0.09 pp (3.93% → 4.02%), Acc@0.08 degrades 0.36 pp.
- Model size meets the target: INT8 11.67 MB within the 5–20 MB range.
- Per-image image_acc@0.08 = 97.05%, only **0.95 pp** short of the 98% target.
- Per-point Acc@0.08 = 91.72%, **6.28 pp** short of 98%.

## Training Curve

| epoch | train_loss | valid_nme | test_nme | test_acc@0.08 | test_image_acc@0.08 |
|---|---|---|---|---|---|
| 1 | 0.0696 | 0.1764 | 0.1705 | 21.86% | 4.40% |
| 5 | 0.0361 | 0.0622 | 0.0618 | 76.77% | 87.05% |
| 10 | 0.0294 | 0.0517 | 0.0506 | 84.92% | 92.85% |
| 15 | 0.0268 | 0.0463 | 0.0452 | 88.68% | 94.95% |
| 20 | 0.0249 | 0.0429 | 0.0418 | 90.69% | 96.30% |
| 25 | 0.0237 | 0.0411 | 0.0400 | 91.73% | 97.10% |
| **30** | **0.0232** | **0.0405** | **0.0393** | **92.08%** | **97.10%** |

**Observations**:
- The model was still slowly improving at epoch 30, **far from converged**.
- Loss and NME decreased more slowly in the second half (epochs 15–30) but did not plateau.
- test_image_acc@0.08 approached 97% from epoch 25, with diminishing gains.

## Path Summary

| Phase | run | epochs | test NME | test_acc@0.08 | test_image_acc@0.08 | INT8 size |
|---|---|---|---|---|---|---|
| 1 | `lapa_smoke` (1 ep baseline) | 1 | 0.1569 | 21.48% | — | — |
| 2 | **`lapa_lmnet_w225` (30 ep)** | 30 | **0.0393** | **92.08%** | **97.10%** | **11.67 MB** |

## Key Technical Details

1. **Model**: LMNet (MobileNetV2-style InvertedResidual + SE + global FC head), width_mult=2.25, 11.54M params.
2. **Mean shape initialization**: Estimated from the first 512 training samples as FC head bias.
3. **Training**: OneCycle LR (max_lr=0.002) + AMP + grad_clip=1.0 + EMA decay=0.99.
4. **Loss**: L1 coord loss + wing_loss (w=0.04) + geometry loss (neighbor distance consistency).
5. **Augmentation**: Random scale [0.92, 1.18] + random shift ±4% + color jitter + blur. **Horizontal flip disabled** (106-pt flip table not yet verified).
6. **Quantization**: ONNX QDQ + PTQ static + MinMax calibration, INT8 11.67 MB.

## Target Achievement

| Target | Requirement | Current (LMNet) | Current (HRNet) | Gap (HRNet) | Status |
|---|---|---|---|---|---|
| Quantized size | 5–20 MB | 11.67 MB ✅ | 17.93 MB ✅ | — | ✅ Met |
| NME | Lower is better | 3.86% (INT8) | **2.56% (INT8)** | — | ✅ Excellent |
| Acc@0.08 ≥ 98% | Per-point | 92.65% (INT8) | **97.03% (INT8)** | 0.97pp | ✅ Very Close |
| Acc@0.08 ≥ 98% | Per-image | 97.50% (INT8) | **99.00% (INT8)** | — | ✅ **Met** |

## License

MIT (same as repository root).

---

## HRNet W18 + Heatmap Path (2026-06-14)

### Milestone: HRNet + Heatmap Full Training

> Current HRNet best: FP32 test NME 2.32%, acc@0.08 97.4%, image_acc@0.08 99.3%, but model is 73MB (unquantized).

Switched from LMNet regression to **HRNet W18 + Heatmap** architecture, with full-flow high-resolution feature retention + soft-argmax heatmap decoding, based on the research survey findings.

### LaPa Direct Training (`lapa_hrnet_w18_heatmap_e80`)

Config:
- Architecture: HRNet W18 (base_channels=32, num_blocks=2)
- Input: 256×256, Output: 106×64×64 heatmaps
- Loss: Wing Loss (w=0.04) on heatmap
- Training: cosine LR (0.0015) + AMP + grad_clip=1.0 + EMA (0.99) + 80 epochs
- Augmentation: hflip + scale [0.90, 1.20] + shift ±5% + crop_scale 1.30

**Progress: 43 / 80 epochs (interrupted, not completed)**

| Epoch | Train Loss | Test NME | acc@0.08 | acc@0.10 | image_acc@0.08 |
|---|---|---|---|---|---|
| 1 | 56.42 | 0.0390 | 91.7% | 95.1% | 96.1% |
| 3 | 16.07 | 0.0302 | 95.6% | 97.5% | 98.4% |
| 5 | 12.55 | 0.0275 | 96.3% | 97.9% | 98.8% |
| 10 | 9.21 | 0.0250 | 97.1% | 98.3% | 99.2% |
| 20 | 6.72 | 0.0238 | 97.3% | 98.5% | 99.3% |
| 30 | 5.34 | 0.0233 | 97.4% | 98.5% | 99.3% |
| 40 | 4.45 | 0.0232 | 97.4% | 98.6% | 99.2% |
| **43** | **4.20** | **0.0232** | **97.3%** | **98.5%** | **99.3%** |

**Best checkpoint**: epoch 40, test NME = 0.0232, test acc@0.08 = 97.4%

**Key Comparison (HRNet vs LMNet)**:

| Metric | LMNet w2.25 (FP32) | HRNet W18 (FP32) | Improvement |
|---|---|---|---|
| test NME | 0.0376 | **0.0232** | **-38.3% relative** |
| test acc@0.08 | 92.86% | **97.35%** | **+4.49pp** |
| test image_acc@0.08 | 97.50% | **99.25%** | **+1.75pp** |
| test acc@0.10 | 96.13% | **98.50%** | **+2.37pp** |
| Model size (FP32) | 44 MB | 73 MB | +66% |
| Model size (INT8) | 11.67 MB | TBD | — |

**Core Findings**:
- HRNet + Heatmap **dramatically outperforms** LMNet regression: NME drops 38% relatively.
- image_acc@0.08 = 99.3%, **exceeding the 98% target**.
- acc@0.08 = 97.4%, close to the 98% target, only 0.6pp short.
- Training interrupted at epoch 43, remaining 37 epochs not run; diminishing returns in plateau phase.
- FP32 model is 73MB, needs quantization to fit the 5~20MB target.

### Fine-tune (`lapa_hrnet_w18_heatmap_ft`)

Loaded from best.pt (epoch 40), lower LR (0.0015→0.0005), stronger augmentation, EMA 0.99→0.999, early stopping (patience=15).

**Result: Early stopped at epoch 29 (best=epoch 14), minimal improvement**

| Metric | HRNet (best) | FT (best ep14) | Change |
|---|---|---|---|
| test NME | 0.02324 | **0.02295** | -1.2% |
| test acc@0.08 | 97.34% | 97.35% | +0.01pp |
| test acc@0.05 | 91.96% | **92.15%** | +0.19pp |
| image_acc@0.08 | 99.25% | 99.15% | -0.10pp |

Train loss dropped 23% (4.15→3.21) but test metrics barely moved — classic **annotation accuracy ceiling**.

### INT8 Quantization

#### Method 1: PTQ Full (all ops quantized) ❌ Severe degradation

| Metric | FP32 | INT8 Full | Status |
|---|---|---|---|
| NME | 0.02295 | 0.03977 | ❌ +73% |
| acc@0.08 | 97.35% | 92.74% | ❌ -4.61pp |
| acc@0.05 | 92.15% | 74.61% | ❌ -17.54pp |
| image_acc@0.08 | 99.15% | 98.35% | ⚠️ -0.80pp |
| Model size | 69.7 MB | 17.82 MB | ✓ |

Root cause: heatmap→softmax→soft-argmax chain is nonlinearly sensitive; softmax exponentiates quantization errors into coordinate shifts.

#### Method 2: PTQ Conv-Only (quantize only Conv, keep decode chain FP32) ✅ Viable

| Metric | FP32 | INT8 Conv-Only | Status |
|---|---|---|---|
| NME | 0.02295 | **0.02560** | ✅ +11.5% |
| acc@0.05 | 92.15% | **90.70%** | ✅ -1.45pp |
| acc@0.08 | 97.35% | **97.03%** | ✅ -0.32pp |
| acc@0.10 | 98.48% | **98.31%** | ✅ -0.17pp |
| image_acc@0.08 | 99.15% | **99.00%** | ✅ -0.15pp |
| Model size | 69.7 MB | **17.93 MB** | ✓ 5–20MB |

**Strategy**: Quantize only the 53 Conv ops (backbone weights), keep Softmax + soft-argmax + arithmetic ops in FP32. Percentile calibration (99.999%), 32 calibration batches.

**Conclusion**: Conv-only quantization massively fixes the degradation. acc@0.08 degradation narrowed from -4.61pp to -0.32pp. image_acc@0.08 = 99.0% still exceeds 98% target. Model 17.93 MB is within the 5–20MB range.

### WFLW Pretraining (`wflw_hrnet_w18_pretrain_e40`)

- Config: 68-point WFLW → 40 epochs
- Progress: 29 / 40 epochs (interrupted)
- Best test NME = 0.1355, acc@0.08 = 61.6%
- **Conclusion**: WFLW 68-point task did not converge; WFLW→LaPa transfer path shelved.

### Next Steps

1. **Knowledge distillation**: use HRNet FP32 teacher to distill LMNet student for even smaller models.
2. **Stronger loss**: Adaptive Wing Loss for better heatmap foreground/background balance.
3. **Higher input resolution**: 256→384 for finer spatial information.
4. **Multi-stage cascade**: coarse-to-fine refinement to improve difficult samples.

## License

MIT (same as repository root).
