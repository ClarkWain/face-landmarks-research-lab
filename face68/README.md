# NewFaceDetect

> 一个探索性的开源人脸关键点项目。
> 自研 11.5–15.5 M 参数小模型 LMNet 在 300W 测试集上达到 **INT8 NME 5.94%**（磁盘 15.48 MB），相对于 24.79 MB 的 FAN4 INT8 教师模型 **相对提升约 38%**。

中文（默认） | [English](README.en.md)

![hero](docs/images/hero.png)

*三个样例叠加展示：真值（绿）vs FAN4 INT8 教师（红）vs 我们的 INT8 LMNet（蓝）。蓝点几乎完全覆盖绿点，红点显著偏离。*

## 结果一览（300W 测试集）

| 模型 | 参数量 | 磁盘 (INT8) | NME | Acc@0.05 | Acc@0.08 | Acc@0.10 |
|---|---|---|---|---|---|---|
| 平均脸基线 | – | – | 0.298 | – | – | – |
| 预训练 face_alignment 2DFAN4（未微调）| 23.8 M | 90.9 MB FP32 | 0.113 | – | 34.2% | – |
| 微调 2DFAN4（FP32）| 23.8 M | 90.9 MB | 0.080 | 47.1% | 70.3% | – |
| 微调 2DFAN4（INT8）| 23.8 M | **24.79 MB** | 0.0961 | 38.3% | 61.5% | 72.3% |
| **LMNet w=2.6 + 100k 伪标签 + 微调（FP32）** | 15.5 M | 59.0 MB | **0.0587** | **55.0%** | **77.3%** | **85.2%** |
| **LMNet w=2.6 + 100k 伪标签 + 微调（INT8）** | 15.5 M | **15.48 MB** | **0.0594** | **53.5%** | **76.9%** | **85.0%** |

（FP32 最佳 checkpoint 的验证集 NME 为 **0.0514**，距离严格 5% 目标仅差 0.14 个百分点。）

完整的实验日志在 [REPORT.md](REPORT.md)，论文风格的技术写作在 [PAPER.md](PAPER.md)。

## 项目背景

300W 标准训练集只有约 600 张高质量手动标注图片，对从零训练具有竞争力的关键点模型来说远远不够。我们的方案：

1. 在 300W 上把 face_alignment 2DFAN4 教师微调到 NME 0.0802。
2. 用该教师在 **100,000 张 CelebA 对齐人脸** 上生成 68 点伪标签。
3. 用 `WeightedRandomSampler` 让每个 batch 包含 50% 真值 300W + 50% 伪标签 CelebA，从零训练 LMNet（**Stage A**，60 epoch，OneCycle lr=2e-3）。
4. 加载 Stage A 的 best.pt，用更小学习率、更温和增强、更高真值占比（`real_ratio=0.7`）继续微调（**Stage B**，cosine lr=3e-4 → 1e-6）。
5. 用 ONNX QDQ + per-channel + MinMax 校准做 INT8 量化。

我们也试过 PIP 风格 heatmap head、加 WFLW augmented 联合训练、在线蒸馏等方案，全部不如上述配方。失败实验细节见 [REPORT.md](REPORT.md)。

## 仓库结构

```
landmarklab/
  data.py          # 300W / WFLW / FaceSynthetics / 伪标签 CelebA 数据集与加载器
  model.py         # LMNet（MobileNet 风格主干 + 多种 head）、FAN heatmap 包装器、ResNet18+deconv
  train.py         # YAML 驱动的训练器：EMA、AMP、OneCycle/Cosine、可选在线蒸馏
  export_quant.py  # ONNX INT8（QDQ + per-channel + MinMax）导出与评估
  core.py          # 损失、指标、IO 工具
configs/           # 所有实验配置 (yaml)
scripts/
  pseudo_label_celeba.py  # FAN4 -> CelebA 伪标签生成器
  extract_celeba_extra.py # 把 CelebA 解压目录扩展到 100k 张图
  preview_celeba_pseudo.py
  demo_compare.py         # 真值 vs 教师 vs 学生可视化（详细多行版）
  demo_hero.py            # 紧凑横排 hero 图
data/              # 数据集（不入库）
runs/              # 训练产物：best.pt、history.csv、ONNX、summary.json
docs/images/       # 文档图片
REPORT.md          # 详尽实验日志（中文）
REPORT.en.md       # 详尽实验日志（英文）
PAPER.md           # 论文风格技术报告（中文）
PAPER.en.md        # 论文风格技术报告（英文）
README.md          # 本文件（中文）
README.en.md       # 本文件英文版
```

## 安装

```powershell
py -3.12 -m pip install -r requirements.txt
```

依赖：PyTorch 2.9 + CUDA、`face_alignment` 1.5.0（必须用 `compile=False` 避免在 Windows 上加载 Triton）、`onnxruntime`（INT8 推理纯 CPU 即可）、`tqdm`、`pyyaml`、`pillow`。在 Python 3.12 + Windows 下测试通过，由于 Windows 共享内存限制建议 `num_workers=0`。

## 数据集准备

放到 `data/` 下：

- **300W**：原始 600 张室内 + 室外图片，设置 `data.download=true` 可自动下载。
- **CelebA aligned 20k → 100k**：解压到 `data/celeba_ssl_20k/img_align_celeba/`。用 `scripts/extract_celeba_extra.py` 从 `img_align_celeba.zip` 扩展到 100k。
- **WFLW augmented**（可选，本次最终配方未使用）：`tar.gz` 放在 `data/`。
- **FAN4 教师**：用 `configs/300w_fan4_finetune.yaml` 训一次。

## 复现最佳结果

```powershell
# 1. 微调 FAN4 教师
python -m landmarklab.train --config configs/300w_fan4_finetune.yaml --note teacher

# 2. 生成 100k 伪标签
python -m scripts.extract_celeba_extra
python -m scripts.pseudo_label_celeba `
    --max-samples 100000 --batch-size 24 `
    --output data/celeba_pseudo_100k.npz

# 3. Stage A：100k 伪标签 + 300W 训练
python -m landmarklab.train --config configs/300w_lmnet_w26_celeba100k.yaml --note stage_a

# 4. Stage B：加载 Stage A 的 best.pt 继续微调
python -m landmarklab.train --config configs/300w_lmnet_w26_100k_finetune.yaml --note stage_b

# 5. INT8 导出 + 评估
$env:PYTHONIOENCODING='utf-8'
python -m landmarklab.export_quant `
    --config configs/300w_lmnet_w26_100k_finetune.yaml `
    --run runs/300w_lmnet_w26_100k_finetune

# 6. 对比可视化
python -m scripts.demo_compare --dataset-root data/300w_extracted/300w_extracted/300W
python -m scripts.demo_hero
```

每次训练产生：

- `runs/<run_name>/best.pt`：模型 + EMA + 配置快照
- `runs/<run_name>/history.csv`：每个 epoch 的 loss / NME / 准确率
- `runs/<run_name>/preview_best.png`：预测 vs 真值可视化
- `runs/<run_name>/model_fp32.onnx`、`model_int8.onnx`
- `runs/<run_name>/summary.json`、`quant_summary.json`

## 冒烟测试

```powershell
python -m landmarklab.train --config configs/300w_lmnet_w26_100k_finetune.yaml `
    --override train.epochs=1 train.log_interval_steps=20 `
    run_name=smoke
```

## 实验教训

完整版本在 `REPORT.md`，简版：

1. **`ema_decay=0.999` 在小数据集上是错的**：300W 仅 13 step/epoch，EMA 永远跟不上模型。降到 `0.99`。
2. **不带加权采样的伪标签注定失败**：432 真值 / 20000 伪标签直接 `ConcatDataset(shuffle=True)`，valid NME 会随 epoch *上升*，因为模型完全过拟合到伪标签分布。`WeightedRandomSampler` + `real_ratio≈0.5`（Stage A）和 `0.7`（Stage B）才是解。
3. **两阶段优于单阶段**：单阶段 Stage A 能到 NME ~0.075；加载 best.pt + 小学习率 cosine + 提高 real_ratio，单 epoch 即跳到 **0.0514**。
4. **WFLW augmented 是 112×112 已 crop 数据**，分辨率太低；WFLW68 子集与 300W 真值在嘴/眼非端点存在亚像素语义差异，反而会拖累 300W 测试 NME。
5. **PIP head**（28×28 cls + 亚像素 offset）在小数据上 plateau，远不如带 `mean_shape` 偏置初始化的全局 FC head 收敛快。
6. **face_alignment 2DFAN4 有几个隐藏契约** —— 不修复这三点会得到 NME 0.91 而非 0.115：
   - 期望 `[0, 1]` 归一化输入（我们流水线是 `[-1, 1]`，所以 `forward_train` 内部再做反归一化）；
   - 推理时用 argmax + 邻域亚像素偏移解码，不是 soft-argmax；
   - 预训练 heatmap 峰值≈0.96，对其做 `sigmoid` 再 MSE 会破坏校准 —— 必须用 raw MSE。

## 许可证

MIT（科研 / 实验目的）。各数据集保留原许可证。
