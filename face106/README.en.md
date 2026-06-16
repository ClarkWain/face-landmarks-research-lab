# face106 — 106-point Face Landmark Detection

> A from-scratch lightweight 106-point face landmark model, **17.93 MB** after INT8 quantization, runs in real-time on CPU.
> NME **2.37%** on LaPa test set, NME **3.37%** on ICME 2019 Test\_data1 (FP32 384 inference) — **16% relatively better** than ICME 2021 TOP1 (Meituan, NME 4.01%).

[中文（默认）](README.md) | English

---

## Performance

### LaPa test set (256×256, matches training resolution)

| Model | Params | Disk size | NME | acc@0.05 | acc@0.08 | image\_acc@0.08 |
|---|---|---|---|---|---|---|
| HRNet W18 + AWing + Mixed Data (FP32) | 18.3 M | 69.7 MB | **2.16%** | **93.11%** | **97.98%** | **99.55%** |
| Same, INT8 (Conv-Only) | 18.3 M | **17.93 MB** | **2.37%** | 91.83% | **97.60%** | **99.50%** |

### ICME 2019 Test\_data1 (vs public competition results)

| Model | Inference resolution | NME | acc@0.05 |
|---|---|---|---|
| **Ours (FP32, 384 inference)** | 384 | **3.37%** | **82.49%** |
| ICME 2021 TOP1 (Meituan TuringTest) | – | 4.01% | 79.05% |
| ICME 2021 TOP2 (Tencent) | – | 4.21% | 78.25% |
| ICME 2021 TOP3 (Streamax) | – | 4.15% | 75.93% |
| Ours (FP32, 256 inference) | 256 | 4.03% | 75.13% |
| Ours (INT8, 256 inference) | 256 | 4.31% | 72.80% |

> Already on par with ICME 2021 TOP1 at 256 inference; upsampling to 384 inference beats TOP1 by 16% relative NME.
> Note: this model is **not** subject to the ICME 2021 limits on model size (≤2 MB) and FLOPs (≤100 MFLOPs).

Full training log, ablations and failed experiments live in [PROJECT\_SUMMARY.md](PROJECT_SUMMARY.md) and [REPORT.en.md](REPORT.en.md).

---

## Three ways to use it

### 1. pip package (recommended)

```powershell
# Build the wheel from the repo root
cd ../face106-pkg
py -3.12 -m pip wheel . --no-deps -w dist
py -3.12 -m pip install dist/face106-0.1.0-py3-none-any.whl
```

The INT8 ONNX model (~15 MB) is bundled in the wheel.

```python
from face106 import LandmarkDetector
from PIL import Image

detector = LandmarkDetector()                       # loads bundled INT8 ONNX
img = Image.open("face.jpg").convert("RGB")
bbox = (50, 50, 250, 250)                           # supplied by your face detector
landmarks = detector.predict(img, bbox)             # numpy (106, 2) pixel coords
```

Visualization helper:

```python
from face106 import draw_landmarks
out = draw_landmarks(img, landmarks)
out.save("out.jpg")
```

### 2. Repo demo script

`scripts/demo.py` ships with a built-in OpenCV Haar face detector and supports both single-image and webcam modes:

```powershell
# Single image
py -3.12 scripts/demo.py --image path/to/face.jpg --output out.jpg

# Live webcam
py -3.12 scripts/demo.py --webcam

# INT8 ONNX backend
py -3.12 scripts/demo.py --image face.jpg --onnx runs/lapa_hrnet_w18_awing_mixed_e80/model_int8.onnx
```

### 3. Direct ONNX call

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("runs/lapa_hrnet_w18_awing_mixed_e80/model_int8.onnx")
# Input:  (N, 3, 256, 256) float32, [-1, 1] normalized, BGR→RGB
# Output: (N, 106, 2) normalized coordinates [0, 1]
```

---

## Architecture

```
Input 256×256×3
   ↓
HRNet W18 backbone (base_channels=32, num_blocks=2)
   ↓
heatmap head → (106, 64, 64)
   ↓
Spatial Softmax → soft-argmax decoding
   ↓
Output (106, 2) normalized coordinates
```

- Training: `sigmoid + Adaptive Wing Loss` (ω=14, ε=0.5, α=2.1, θ=0.5).
- Inference: soft-argmax — avoids argmax non-differentiability and quantization distortion.
- INT8 quantization uses **Conv-Only** strategy: only the 53 Conv nodes are quantized; Softmax + soft-argmax decoding chain stays FP32, keeping degradation to NME +9.6%.

See [PROJECT\_SUMMARY.md](PROJECT_SUMMARY.md) for full technical decisions and ablations.

---

## Training data

**113,504 training images total**:

| Dataset | Count | Annotation source |
|---|---|---|
| LaPa train | 18,168 | Official 106-point manual annotation |
| JD-landmark FLL3 | ~20,000 | Official 106-point manual annotation |
| Pseudo WFLW | ~75,000 | 106-point pseudo-labels from HRNet teacher |

Loading and sampling logic lives in [`landmarklab/data.py`](landmarklab/data.py) (`lapa_mixed` dataset type + `WeightedRandomSampler`).

---

## Reproduce training

Final best config: [`configs/lapa_hrnet_w18_awing_mixed.yaml`](configs/lapa_hrnet_w18_awing_mixed.yaml).

```powershell
# 1. Prepare data
#    - LaPa        → ../data/LaPa/{train,val,test}/
#    - JD          → ../data/jd_landmark/FLL3_dataset/
#    - Pseudo WFLW → face106/data/wflw_pseudo_106/train_data.csv

# 2. Train (80 epochs, batch=16; ~few hours on RTX 2080 Ti 22GB)
py -3.12 -m landmarklab.train `
    --config configs/lapa_hrnet_w18_awing_mixed.yaml

# 3. Export INT8 ONNX (Conv-Only + Percentile 99.999% calibration)
py -3.12 -m landmarklab.export_quant `
    --config configs/lapa_hrnet_w18_awing_mixed.yaml `
    --run runs/lapa_hrnet_w18_awing_mixed_e80 `
    --quant-mode conv_only `
    --calibrate-method Percentile

# 4. Evaluate on ICME 2019 Test_data1
py -3.12 scripts/eval_icme.py `
    --checkpoint runs/lapa_hrnet_w18_awing_mixed_e80/best.pt `
    --test-root ../data/Test_data1 `
    --resolution 384

# Evaluate INT8 ONNX
py -3.12 scripts/eval_icme_onnx.py `
    --onnx runs/lapa_hrnet_w18_awing_mixed_e80/model_int8.onnx `
    --test-root ../data/Test_data1 `
    --resolution 256
```

Smoke test (sanity-check the pipeline):

```powershell
py -3.12 -m landmarklab.train `
    --config configs/lapa_hrnet_w18_awing_mixed.yaml `
    --override train.epochs=1 train.log_interval_steps=20 run_name=smoke
```

---

## Layout

```
face106/
├── landmarklab/
│   ├── core.py            # losses (incl. Adaptive Wing Loss), metrics, geometry
│   ├── data.py            # LaPa / JD-landmark / PseudoWFLW datasets and mixed loader
│   ├── model.py           # HRNet W18 + heatmap head
│   ├── train.py           # YAML-driven trainer (EMA / AMP / Cosine LR / early stop)
│   ├── export_quant.py    # ONNX QDQ INT8 export (supports conv_only + percentile)
│   └── ssl_pretrain.py
├── configs/
│   ├── lapa_hrnet_w18_awing_mixed.yaml   # final best config
│   ├── lapa_lmnet.yaml                   # earlier LMNet baseline
│   └── ...                               # configs from comparison runs
├── scripts/
│   ├── demo.py            # single-image / webcam demo
│   ├── eval_icme.py       # ICME 2019 Test_data1 eval (PyTorch)
│   ├── eval_icme_onnx.py  # same, ONNX backend
│   └── preview_lapa.py
├── runs/
│   └── lapa_hrnet_w18_awing_mixed_e80/   # final outputs
│       ├── best.pt        (69.7 MB FP32)
│       ├── model_int8.onnx (17.93 MB)
│       ├── history.csv
│       └── summary.json
├── PROJECT_SUMMARY.md
├── REPORT.md / REPORT.en.md
├── PAPER.md / PAPER.en.md
└── README.md / README.en.md
```

---

## Key takeaways

- **Data quantity is the real bottleneck.** With 18k LaPa only, NME plateaus at 2.30%; adding 20k JD + 75k pseudo WFLW drops it to 2.16%.
- **Adaptive Wing Loss is significantly better than Wing / MSE**, but you must `clamp(d, min=1e-6)` to avoid divergence at `d → 0`.
- **Conv-Only INT8 is critical.** Quantizing the full graph (including Softmax) inflates LaPa NME from 2.30% to 4.0%+; quantizing only the 53 Conv nodes while keeping Softmax + soft-argmax FP32 limits degradation to +9.6%.
- **Train at 256, infer at 384.** The model is more robust to inference upsampling than to training resolution increases — ICME NME drops from 4.03% to 3.37%.

The full version (including failed experiments and untaken roads) is in [PROJECT\_SUMMARY.md](PROJECT_SUMMARY.md) and [REPORT.en.md](REPORT.en.md).

---

## License

MIT. LaPa / JD-landmark / WFLW datasets keep their own licenses.
