# Face Landmarks Research Lab

> An exploratory open-source research monorepo for facial landmark detection, organized by landmark count.

[中文（默认）](README.md) | English

---

## Sub-projects

| Directory | Task | Dataset | Status | README |
|---|---|---|---|---|
| [`face68/`](face68/) | 68 landmarks | 300W + pseudo-CelebA | INT8 NME 5.94% / 15.48 MB | [face68/README.en.md](face68/README.en.md) |
| [`face106/`](face106/) | 106 landmarks | LaPa | Starting (target: 2019 top-5) | [face106/README.md](face106/README.md) |

## Layout

- Each sub-project owns its own `landmarklab/` (or equivalent) training stack, `configs/`, `scripts/`, `runs/`, `docs/` so that experiments are independent and reproducible.
- `data/` lives at the repository root and is shared by both sub-projects (kept out of git via `.gitignore`).
- Engineering insights that worked in one sub-project (EMA tuning, pseudo-labelling + weighted sampling, ONNX QDQ INT8) are cross-verified through `REPORT.md` of the other.

## Snapshot

The 68-point project is stable; the best INT8 model is the 15.48 MB ONNX inside `face68/runs/300w_lmnet_w26_100k_finetune/`, with NME 5.94% on 300W test (details in [face68/REPORT.md](face68/REPORT.md)).

The 106-point project (face106) is starting up. The target is LaPa NME at the 2019 top-5 level (NME ≤ 1.5%). Current stage: dataset parsing and reusing the training pipeline.

## License

MIT. Datasets keep their own licenses.
