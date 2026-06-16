# Face Landmarks Research Lab

> 一个探索性的人脸关键点检测开源研究仓库，按"点数"分子项目组织。
> An exploratory open-source research monorepo for facial landmark detection, organized by landmark count.

中文（默认） | [English](README.en.md)

---

## 子项目

| 目录 | 任务 | 数据集 | 当前状态 | README |
|---|---|---|---|---|
| [`face68/`](face68/) | 68 点关键点 | 300W + CelebA 伪标签 | INT8 NME 5.94% / 15.48 MB | [face68/README.md](face68/README.md) |
| [`face106/`](face106/) | 106 点关键点 | LaPa | INT8 NME 3.88% / 11.67 MB | [face106/README.md](face106/README.md) |

## 设计理念

- 每个子项目自带独立的 `landmarklab/`（或类似）训练栈、`configs/`、`scripts/`、`runs/`、`docs/`，以便单独实验和可复现。
- `data/` 在仓库根，两个子项目共享数据集存放位置（用 `.gitignore` 屏蔽）。
- 共有的工程经验（EMA 调参、伪标签 + 加权采样、ONNX QDQ INT8 等）通过两个子项目的 `REPORT.md` 互相验证。

## 当前总览

68 点项目已稳定，最佳 INT8 模型为 `face68/runs/300w_lmnet_w26_100k_finetune/` 中的 15.48 MB ONNX，300W 测试集 NME 5.94%（详见 [face68/REPORT.md](face68/REPORT.md)）。

106 点项目（face106）在 LaPa 数据集上已取得显著成果，最佳 INT8 模型为 `face106/runs/lapa_w225_cosine_ft/` 中的 11.67 MB ONNX，LaPa 测试集 NME 3.88%，image_acc@0.08 97.45%（详见 [face106/REPORT.md](face106/REPORT.md)）。

## 许可证

MIT。各数据集保留原许可证。
