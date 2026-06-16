# face106 pip package

A simple, reusable Python package for **106-point facial landmark detection**, based on HRNet + AWing Loss + 113k mixed dataset training.

## Installation

```bash
pip install face106
```

Or install from source:

```bash
git clone https://github.com/yourname/NewFaceDetect.git
cd NewFaceDetect/face106-pkg
pip install -e .
```

## Quick start

```python
from face106 import LandmarkDetector
from PIL import Image

detector = LandmarkDetector()  # auto-downloads INT8 ONNX model
image = Image.open("face.jpg").convert("RGB")

# bbox: (x1, y1, x2, y2) - face bounding box from a detector (Haar / RetinaFace / etc.)
bbox = (50, 50, 250, 250)
landmarks = detector.predict(image, bbox)  # numpy (106, 2) in pixel coords
```

## Performance

| Variant | NME (LaPa) | acc@0.08 | Model size | Latency (CPU, 256×256) |
|---|---|---|---|---|
| FP32 PyTorch | 2.16% | 97.98% | 69.7 MB | ~80 ms |
| **INT8 ONNX** | **2.56%** | **97.03%** | **17.93 MB** | **~40 ms** |

**ICME 2019 Test_data1** (vs ICME 2021 winners):
| Model | NME | acc@0.05 |
|---|---|---|
| **face106 (ours)** | **3.37%** | **82.49%** |
| ICME 2021 TOP1 (Meituan) | 4.01% | 79.05% |

## License

MIT
