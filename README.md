# NewFaceDetect

> 一个开源的人脸关键点检测项目集合，按"关键点数量"组织子项目。
> 提供经过完整训练 / 量化 / 评测 / 打包的小模型，可直接拿来用。

中文（默认） | [English](README.en.md)
![face106 hero](face106/docs/images/hero.png)

*face106 在 LaPa 测试集上的 106 点预测示例（INT8 ONNX，17.93 MB）。*
---

## 子项目

| 目录 | 任务 | 训练数据 | INT8 模型 | 关键指标 | README |
|---|---|---|---|---|---|
| [`face-det/`](face-det/) | 人脸检测 | LaPa landmarks → bbox + Test_data1 benchmark | FP32 | `Test_data1`: **AP50 99.5 / Recall 100% / AP50-95 58.2** | [face-det/README.md](face-det/README.md) |
| [`face106/`](face106/) | 106 点关键点 | LaPa + JD-landmark + 伪标 WFLW（共 113k）| **17.93 MB** | LaPa NME **2.37%** / ICME 2019 NME **3.37%**（超越 ICME 2021 TOP1 16%）| [face106/README.md](face106/README.md) |
| [`face68/`](face68/) | 68 点关键点 | 300W + CelebA 100k 伪标 | **15.48 MB** | 300W NME **5.94%** / acc@0.08 **76.9%** | [face68/README.md](face68/README.md) |

两个子项目同样都从零训练（不使用任何预训练权重），都已完成 ONNX QDQ INT8 量化和端到端的评测。

## 快速使用 face106（106 点）

face106 以可独立安装的 pip 包形式提供。在本仓库根目录：

```powershell
# 构建 wheel（需 Python 3.12+）
cd face106-pkg
py -3.12 -m pip wheel . --no-deps -w dist

# 安装
py -3.12 -m pip install dist/face106-0.1.0-py3-none-any.whl
```

INT8 ONNX 模型已内嵌在 wheel 中。

```python
from face106 import LandmarkDetector
from PIL import Image

detector = LandmarkDetector()                       # 加载内置 INT8 ONNX
img = Image.open("face.jpg").convert("RGB")
bbox = (50, 50, 250, 250)                           # 来自任意人脸检测器
landmarks = detector.predict(img, bbox)             # numpy (106, 2) 像素坐标
```

详情见 [face106/README.md](face106/README.md) 和 [face106-pkg/](face106-pkg/)。

## 仓库布局

```
NewFaceDetect/
├── face-det/             # 人脸检测子项目（YOLO strong baseline + Test_data1 benchmark）
│   ├── configs/
│   ├── scripts/
│   ├── REPORT.md
│   └── README.md
├── face106/              # 106 点子项目（HRNet W18 + AWing Loss + 混合数据）
│   ├── landmarklab/      # 训练、模型、损失、ONNX 导出
│   ├── configs/          # 所有训练 yaml
│   ├── scripts/          # demo.py、ICME 评测、可视化
│   ├── runs/             # 训练产物（best.pt、ONNX、history.csv）
│   ├── PROJECT_SUMMARY.md
│   ├── REPORT.md
│   └── README.md
├── face68/               # 68 点子项目（LMNet + 100k 伪标签）
│   ├── landmarklab/
│   ├── configs/
│   ├── scripts/
│   ├── runs/
│   ├── REPORT.md
│   ├── PAPER.md
│   └── README.md
├── face106-pkg/          # face106 的 pip 包
│   ├── face106/
│   │   ├── detector.py
│   │   └── assets/face106_int8.onnx
│   └── pyproject.toml
└── data/                 # 数据集（不入库，由 .gitignore 屏蔽）
    ├── LaPa/
    ├── 300w_extracted/
    ├── celeba_ssl_20k/
    └── ...
```

每个子项目的训练栈相互独立、可单独跑实验，但共享同一份 `data/` 数据目录。

## 设计理念

- **从零训练，明确口径**：所有结果都来自从零训练，不复用任何预训练权重；评测口径与公开论文 / 比赛对齐。
- **小模型、小硬件、小依赖**：目标 INT8 后 ≤ 20 MB，CPU 推理可用，依赖 PyTorch + ONNXRuntime + Pillow。
- **可复现 + 可追溯**：每个实验有独立的 yaml 配置、训练脚本、log、产物目录。失败实验也保留在 `REPORT.md`。
- **跨子项目交叉验证**：EMA 调参、伪标签 + 加权采样、ONNX QDQ INT8、Conv-Only 量化等技巧在 face68 和 face106 互相印证。

## 许可证

MIT。各数据集保留各自原许可证。
