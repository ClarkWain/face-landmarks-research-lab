# NewFaceDetect

> An open-source collection of face landmark detection projects, organized by landmark count.
> Each sub-project ships fully trained, quantized, evaluated and packaged small models ready to use.

[中文（默认）](README.md) | English

![face106 hero](face106/docs/images/hero.png)

*face106 predictions on four LaPa test samples (INT8 ONNX, 17.93 MB).*

---

## Sub-projects

| Directory | Task | Training data | INT8 model | Key metrics | README |
|---|---|---|---|---|---|
| [`face106/`](face106/) | 106 landmarks | LaPa + JD-landmark + Pseudo WFLW (113k) | **17.93 MB** | LaPa NME **2.37%** / ICME 2019 NME **3.37%** (16% rel. better than ICME 2021 TOP1) | [face106/README.en.md](face106/README.en.md) |
| [`face68/`](face68/) | 68 landmarks | 300W + 100k pseudo CelebA | **15.48 MB** | 300W NME **5.94%** / acc@0.08 **76.9%** | [face68/README.en.md](face68/README.en.md) |

Both sub-projects are trained from scratch (no pretrained weights) and ship with full ONNX QDQ INT8 quantization and benchmark scripts.

## Quick start: face106 (106 landmarks)

face106 is shipped as a standalone, pip-installable package. From the repo root:

```powershell
# Build the wheel (requires Python 3.12+)
cd face106-pkg
py -3.12 -m pip wheel . --no-deps -w dist

# Install
py -3.12 -m pip install dist/face106-0.1.0-py3-none-any.whl
```

The INT8 ONNX model is bundled inside the wheel.

```python
from face106 import LandmarkDetector
from PIL import Image

detector = LandmarkDetector()                       # loads bundled INT8 ONNX
img = Image.open("face.jpg").convert("RGB")
bbox = (50, 50, 250, 250)                           # from any face detector
landmarks = detector.predict(img, bbox)             # numpy (106, 2) pixel coords
```

See [face106/README.en.md](face106/README.en.md) and [face106-pkg/](face106-pkg/) for details.

## Layout

```
NewFaceDetect/
├── face106/              # 106-pt project (HRNet W18 + AWing Loss + mixed data)
├── face68/               # 68-pt project (LMNet + 100k pseudo labels)
├── face106-pkg/          # pip-installable wrapper of face106 INT8 ONNX
└── data/                 # datasets (gitignored)
```

Each sub-project owns its own training stack, configs, scripts, runs, and docs so experiments stay independent and reproducible. Both share the same root-level `data/` folder.

## Design principles

- **From scratch, well-defined metrics.** No pretrained weights anywhere; metrics align with public papers / challenges.
- **Small models, modest hardware, minimal deps.** Target INT8 ≤ 20 MB, CPU inference works, deps are PyTorch + ONNXRuntime + Pillow.
- **Reproducible + traceable.** Every experiment has its own yaml, script, log, and output dir. Failed runs are kept in `REPORT.md`.
- **Cross-project verification.** EMA tuning, weighted-sampling pseudo-labels, ONNX QDQ, and Conv-Only quantization are cross-validated between face68 and face106.

## License

MIT. Datasets keep their own licenses.
