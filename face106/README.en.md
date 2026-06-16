# face106 — 106-point Face Landmark Research

> Goal: push a small self-designed model on LaPa (22 000 images, 106 points) to a 2019 top-5 level (NME ≤ 1.5%).

[中文（默认）](README.md) | English

## Current status

- ✅ Training stack bootstrapped (re-uses code from `face68/landmarklab/`, adds `LaPaDataset` and a LaPa landmark loader).
- ✅ LaPa extracted into `../data/LaPa/` (train 18 168 / val 2 000 / test 2 000).
- ✅ Visualization confirms the 106-point grouping (contour 0–32, right brow 33–41, left brow 42–50, nose 51–65, right eye 66–79, left eye 80–93, mouth 94–105).
- ⏳ Training pipeline not yet launched; pending smoke verification, then formal runs.

## Differences from face68

| Item | face68 | face106 |
|---|---|---|
| Training samples | 600 (300W) | 18 168 (LaPa train) |
| Landmark count | 68 | 106 |
| Need pseudo-labels? | Yes (CelebA 100k) | Not yet (data is already 30× larger) |
| Eye groups for NME | (36–41) / (42–47) | (66–79) / (80–93) |
| Horizontal flip | Enabled (FLIP_ORDER_68 known) | Disabled (106-point flip table TBD) |

## Reproduce the initial training

```powershell
# From the repo root
py -3.12 face106/landmarklab/train.py `
    --config face106/configs/lapa_lmnet.yaml `
    --override train.epochs=1 train.log_interval_steps=50 run_name=lapa_smoke
```

The trainer is launched from the repo root so that `../data/LaPa/...` relative paths resolve.

## Known TODO

1. **106-point horizontal-flip table**. Need to extract LaPa's left-right correspondence from the LaPa repo / paper. Until then, `LaPaDataset.enable_hflip=False`.
2. **Optimization protocol**. First iteration directly reuses the face68 OneCycle + LMNet w=2.25 recipe. If NME plateaus we will replay face68 lessons (width_mult, real_ratio, ema_decay, two-stage fine-tune, ...).
3. **2019 SOTA baseline**. The LaPa paper reports NME 1.4–1.5% (normalized by face bbox; not directly comparable to inter-ocular). We will recalibrate the target after our first results.

## License

MIT (same as the monorepo root).
