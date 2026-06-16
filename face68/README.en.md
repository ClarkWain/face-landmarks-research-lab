# NewFaceDetect

> An exploratory open-source project for tiny face landmark detection.
> A fully from-scratch 11.5–15.5 M parameter "LMNet" reaches **INT8 NME 5.94%** on 300W (15.48 MB on disk) — beating a 24.79 MB FAN4 INT8 teacher by ~38% relative.

[中文（默认）](README.md) | English

![hero](docs/images/hero.png)

*Three examples overlaid: ground truth (green) vs FAN4 INT8 teacher (red) vs our INT8 LMNet (blue). Blue points sit on top of the green ground truth; red points clearly drift.*

## Headline numbers (300W test set)

| Model | Params | Disk (INT8) | NME | Acc@0.05 | Acc@0.08 | Acc@0.10 |
|---|---|---|---|---|---|---|
| Mean shape baseline | – | – | 0.298 | – | – | – |
| Pre-trained face_alignment 2DFAN4 (no fine-tune) | 23.8 M | 90.9 MB (FP32) | 0.113 | – | 34.2% | – |
| Fine-tuned 2DFAN4 (our reproduction) | 23.8 M | 90.9 MB (FP32) | 0.080 | 47.1% | 70.3% | – |
| Fine-tuned 2DFAN4, INT8 quantized | 23.8 M | **24.8 MB** | 0.0961 | 38.3% | 61.5% | 72.3% |
| **LMNet w=2.6 + 100k pseudo + fine-tune (FP32)** | 15.5 M | 59.0 MB | **0.0587** | **55.0%** | **77.3%** | **85.2%** |
| **LMNet w=2.6 + 100k pseudo + fine-tune (INT8)** | 15.5 M | **15.48 MB** | **0.0594** | **53.5%** | **76.9%** | **85.0%** |

(Validation NME of the FP32 best checkpoint is **0.0514**, only 0.14 percentage points above the strict 5% target.)

The full ablation/run history is in [REPORT.md](REPORT.md).

## Why this repo

300W has only ~600 manually annotated images, which is far too small to train a competitive landmark model from scratch. We bootstrap a small custom backbone (LMNet) by:

1. Fine-tuning a 2DFAN4 teacher on 300W to NME 0.0802.
2. Using that teacher to **pseudo-label 100 000 CelebA face crops**.
3. Training LMNet from scratch with a `WeightedRandomSampler` that mixes 50% real 300W and 50% pseudo CelebA per batch (*Stage A*, 60 epochs, OneCycle lr=2e-3).
4. Loading Stage A's best checkpoint and fine-tuning with a smaller LR, gentler augmentation, and a higher real-data ratio (`real_ratio=0.7`) (*Stage B*, cosine lr=3e-4 → 1e-6).
5. Quantizing to INT8 with ONNX QDQ + per-channel + MinMax calibration.

We also tried PIP-style heatmap heads, a WFLW-augmented landmark co-training mix, and an online-distillation loss — all underperformed the recipe above. Details are in [REPORT.md](REPORT.md).

## Repository layout

```
landmarklab/
  data.py          # 300W / WFLW / FaceSynthetics / pseudo-CelebA datasets and loaders
  model.py         # LMNet (MobileNet-style backbone with several heads), FAN heatmap wrapper, ResNet18+deconv
  train.py         # YAML-driven trainer with EMA, AMP, OneCycle / Cosine, optional online distillation
  export_quant.py  # ONNX INT8 (QDQ + per-channel + MinMax) export and evaluation
  core.py          # losses, metrics, IO utilities
configs/           # all experiment recipes (yaml)
scripts/
  pseudo_label_celeba.py  # FAN4 -> CelebA pseudo-label generator
  extract_celeba_extra.py # expand CelebA disk dump to 100k images
  preview_celeba_pseudo.py
  demo_compare.py         # GT vs teacher vs student visualization
data/              # datasets (not committed)
runs/              # training outputs: best.pt, history.csv, ONNX, summary.json
REPORT.md          # exhaustive experiment log
PAPER.md           # short technical paper-style write-up of the recipe and results
```

## Setup

```powershell
py -3.12 -m pip install -r requirements.txt
```

PyTorch 2.9 with CUDA, `face_alignment` 1.5.0 (loaded with `compile=False` to avoid Triton on Windows), `onnxruntime` (CPU is fine for INT8 evaluation), `tqdm`, `pyyaml`, `pillow`. The project has been tested on Python 3.12 + Windows with `num_workers=0` (Windows shared-memory limitations).

## Datasets

Place under `data/`:

- **300W**: original 600 indoor/outdoor images, automatically downloaded if `data.download=true`.
- **CelebA aligned 20 k → 100 k**: extracted JPGs into `data/celeba_ssl_20k/img_align_celeba/`. Use `scripts/extract_celeba_extra.py` to grow from 20 k → 100 k starting from `img_align_celeba.zip`.
- **WFLW augmented** (optional, not in the headline recipe): tar.gz under `data/`.
- **FAN4 teacher**: train it once with `configs/300w_fan4_finetune.yaml`.

## Reproducing the headline result

```powershell
# 1. fine-tune the FAN4 teacher on 300W
python -m landmarklab.train --config configs/300w_fan4_finetune.yaml --note teacher

# 2. generate 100 k pseudo-labels on CelebA
python -m scripts.extract_celeba_extra
python -m scripts.pseudo_label_celeba `
    --max-samples 100000 --batch-size 24 `
    --output data/celeba_pseudo_100k.npz

# 3. Stage A: train LMNet w=2.6 with 100 k pseudo + 300W
python -m landmarklab.train --config configs/300w_lmnet_w26_celeba100k.yaml --note stage_a

# 4. Stage B: fine-tune Stage A's best checkpoint on a real-heavy mix
python -m landmarklab.train --config configs/300w_lmnet_w26_100k_finetune.yaml --note stage_b

# 5. INT8 export + evaluation
$env:PYTHONIOENCODING='utf-8'
python -m landmarklab.export_quant `
    --config configs/300w_lmnet_w26_100k_finetune.yaml `
    --run runs/300w_lmnet_w26_100k_finetune

# 6. side-by-side visual comparison
python -m scripts.demo_compare --dataset-root data/300w_extracted/300w_extracted/300W
```

Each run writes:

- `runs/<run_name>/best.pt` — model state + EMA + config snapshot
- `runs/<run_name>/history.csv` — per-epoch loss / NME / accuracies
- `runs/<run_name>/preview_best.png` — predicted vs ground-truth grid
- `runs/<run_name>/model_fp32.onnx` and `model_int8.onnx`
- `runs/<run_name>/summary.json`, `quant_summary.json`

## Smoke test

```powershell
python -m landmarklab.train --config configs/300w_lmnet_w26_100k_finetune.yaml `
    --override train.epochs=1 train.log_interval_steps=20 `
    run_name=smoke
```

## Lessons learned

The bumpy version is in `REPORT.md`; the short version:

1. **`ema_decay=0.999` is wrong on small datasets**. With 13 steps/epoch on raw 300W, the EMA copy never catches up. Drop to `0.99`.
2. **Pseudo-labelling without weighted sampling fails**. At 432 real / 20 000 pseudo, naive `ConcatDataset(shuffle=True)` makes valid NME *increase* with epochs because the model overfits to pseudo-distribution. `WeightedRandomSampler` with `real_ratio≈0.5` (Stage A) then `0.7` (Stage B) is the fix.
3. **Two-stage beats one-stage**. One-shot Stage A reaches NME ~0.075; loading its best.pt + a small-LR cosine schedule + tighter `real_ratio` jumps to **0.0514 in a single epoch**.
4. **WFLW augmented** at 112 × 112 is too low resolution to mix with 300W; the WFLW-68 subset has sub-pixel semantic offsets at the mouth/eye keypoints that hurt 300W test NME.
5. **PIP-style heads** (28 × 28 cls + sub-pixel offset) plateau on small data; a global FC head with a `mean_shape` bias initialization converges much faster.
6. **face_alignment 2DFAN4 has hidden contracts** — without these three fixes we measured NME 0.91 instead of 0.115:
   - it expects `[0, 1]`-normalized RGB (so `forward_train` re-normalizes from the rest of our `[-1, 1]` pipeline);
   - uses argmax + neighbour sub-pixel decoding (not soft-argmax);
   - its pretrained heatmap peaks at ≈0.96, so plain MSE on `sigmoid(heatmap)` destroys the calibration — use raw MSE on the un-sigmoided heatmap.

## License

MIT (research / experimentation). The individual datasets keep their own licenses.
