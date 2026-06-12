# 微小人脸关键点：用 FAN4 伪标签把 INT8 模型压到 15 MB

中文（默认） | [English](PAPER.en.md)

*一份开源探索性技术报告。*

## 摘要

我们研究在仅有 300W 室内 + 室外切分（约 600 张图片，每张 68 点标注）作为人工标注数据的情况下，如何训练一个对 INT8 量化友好的小型人脸关键点检测器。我们设计了 **LMNet**，一个 11.5–15.5 M 参数的 MobileNet 风格主干，配 global FC 头并用平均脸初始化最后一层 bias。LMNet 的 bootstrap 流程：(i) 微调 face_alignment 2DFAN4 教师到 300W NME 0.0802；(ii) 用它在 100,000 张 CelebA 上生成伪标签；(iii) 用 `WeightedRandomSampler` 平衡真值与伪标签训练 LMNet（Stage A），再用小 LR cosine 微调（Stage B）。经 ONNX QDQ + per-channel + MinMax 校准后的 INT8 模型 (15.48 MB) 在 300W 测试集上达到 **NME 0.0594**（FP32 best 验证集 NME 为 0.0514）—— 相对于 24.79 MB INT8 量化的教师模型 **相对改进 38%**，相对于 LMNet 从零基线（NME 0.298）改进 **5 倍**。我们记录了若干小但不可绕开的契约（EMA decay、采样器加权、FAN 解码细节、FAN heatmap 校准），它们决定了配方是否有效。代码与所有配置一并开源。

## 1. 引言

人脸关键点检测在大规模数据集上的 SOTA pipeline 已经被充分研究：hourglass（FAN）、HRNet、Transformer 系列。它们通常使用 25–60 M 参数、FP32 权重 90+ MB，并享有 7000+ 训练样本（WFLW augmented）。但部署约束 —— 端侧推理、INT8 量化、模型大小预算 5–20 MB —— 把我们推向相反方向。

本报告问：*仅有约 600 张高质量 68 点真值图片（300W 原始室内 + 室外切分），一个 15 MB 的 INT8 模型能否在同一个测试集上匹配甚至超过 25 MB 的 INT8 量化 2DFAN4 教师？*

我们给出肯定的答案。配方很短 —— 伪标签、加权采样、两阶段微调 —— 但围绕它的失败模式相当微妙，值得一记。

## 2. 相关工作

* **2DFAN / face_alignment**（Bulat & Tzimiropoulos）。基于堆叠 hourglass 的 68 点 heatmap 模型；事实上的开源权重基线。推理用 argmax + 亚像素偏移，期望 `[0, 1]` 归一化输入，预训练 heatmap 峰值校准在≈0.96。
* **PIPNet**（Jin 等）。预测一个粗分类网格 + 每格亚像素偏移。在大数据集上很强，但在我们仅有 ~600 真值图片的设定下，跳出随机初始化区域所需的 epoch 数远多于全局 FC head。
* **自训练 / 伪标签**（Lee, Xie 等）。教师标无标签数据，学生（通常更小）在合并集上训练。多数前作假设无标签数据与训练集同分布，本工作不是（CelebA 对齐人脸 vs 300W in-the-wild），从而约束了采样器的设计。
* **知识蒸馏**（Hinton, Romero 等）。我们尝试了伪标签蒸馏与在线蒸馏。在线版本在本规模下相对坐标监督几乎没有改进，最终弃用。

## 3. 方法

### 3.1 LMNet 主干

LMNet 是一个 MobileNet-V3 风格的主干：

```
stem(3→16, 3×3 stride 2)
stage1: 1 个 IR block (16→24, stride 2, 无 SE, h-swish)
stage2: 2 个 IR block (24→32, stride 2, SE)
stage3: 3 个 IR block (32→48, stride 2, SE, h-swish)
stage4: 3 个 IR block (48→96, stride 2, SE, h-swish)
stage5: 3 个 IR block (96→192, stride 1, SE, h-swish)
```

每个 channel 数乘以 `width_mult`，正式配方使用 `width_mult = 2.6`（参数量 15.46 M）。`AdaptiveAvgPool2d(1)` 后接 2 层 FC head：

```
Linear(channels[5], hidden_dim=896) → ReLU → Dropout(0.10) → Linear(hidden_dim, 68 × 2) → sigmoid
```

最后一层 `Linear(hidden_dim, 136)` 的 bias 用经验平均脸的 inverse-sigmoid 初始化（`logit(mean_shape)`），所以第 0 步模型已经预测平均脸 —— 等价于 NME 0.34（而非纯随机初始化的 NME 1.5）。**仅这一招** 就能把 loss 开始下降所需 epoch 数减半，尤其在 `pseudo_celeba.real_ratio` 较低的弱监督阶段。

我们使用 wing loss（`wing_w=0.04, wing_eps=0.01`）+ 一个小的几何边一致性损失（`geometry_weight=0.10`）。

### 3.2 伪标签生成

我们用 OneCycle、lr=2e-4、15 epoch、batch=8、EMA、不开 AMP 把 2DFAN4（`num_modules=4`）教师在 300W 上微调。三个契约必须满足：

1. 流水线输出 `[-1, 1]` 归一化的图像（`FANHeatmapNet.forward_train` 内部再反归一化到 `[0, 1]`）。
2. 推理时 heatmap → 坐标解码必须用 `argmax + 0.25 · sign(邻域差分)` 亚像素偏移，**不能** 用训练时的 soft-argmax。soft-argmax 与训练损失匹配，argmax 与开源权重的校准匹配。
3. 训练损失必须是 `mse_raw`（heatmap 原值 vs 高斯目标），不是 `mse(sigmoid(heatmap), gaussian)`，因为开源权重峰值≈0.96，sigmoid 会饱和到 1.0 破坏校准。

修复后我们的复现达到 300W 测试 NME 0.0802（Acc@0.05=47.14, Acc@0.08=70.34）。我们随即在 CelebA 每张对齐人脸的中心 crop（178 × 178 → 256 × 256）上以 eval 模式运行教师，把每张图的 68 点预测（在 crop 的 `[0, 1]` 坐标系中）和 crop 在原图中的坐标 `(left, top, side)` 一起存盘。100,000 张伪标签在单张 2080 Ti 上约 13 分钟生成完毕。

### 3.3 PseudoCelebADataset

`PseudoCelebADataset.__getitem__` 把 crop 内的伪标签反投影回原图像素坐标，然后跑 *与 `ThreeHundredWDataset` 完全相同* 的增强流水线 —— 包括基于伪标签 bbox 的 `_sample_crop_box`、随机 scale 与 shift、亮度/对比度/饱和度/色调抖动、可选高斯模糊、可选 cutout。这意味着学生看到的伪数据与真值数据具有完全相同的统计与几何包络，区别只在标签。

### 3.4 两阶段训练

**Stage A.** OneCycle lr=2e-3，batch=32，60 epoch，AMP，EMA decay 0.99，image_size 224，`init_mean_shape=true`。加权采样器 `real_ratio=0.5`：50% 真值 300W、50% 伪标签 CelebA，每 epoch 200 步（12,800 sample/epoch，真值 1.5× oversample）。OneCycle 把整个 60 epoch 预算视为热身 + 衰减包络。最佳 epoch valid NME：**0.0744**，test NME：**0.0717**。

**Stage B.** Cosine lr 3e-4 → 1e-6，batch=32，40 epoch，AMP，EMA decay 0.99，`init_mean_shape=false`（从 Stage A 暖启动）。`real_ratio=0.7`（真值现在主导）。增强幅度减小（`aug_scale_range=(0.95, 1.10)`，`aug_shift=0.03`）。`ssl_checkpoint=runs/<stage_a>/best.pt`。最佳 **epoch=1**：valid NME 0.0514，test NME 0.0587。

epoch 1 即最佳并非偶然：Stage B 开始时 EMA copy 与模型相同，模型已经位于 Stage A 最优点。Stage B 前 200 步小 LR 余弦把模型微微推向偏 300W 分布的权重，同时不破坏伪数据的特征表示。后续 epoch 收敛到偏 300W 过拟合的均衡，valid NME 略升。

### 3.5 INT8 量化

把 `best.pt` 导出为 FP32 ONNX（opset 18，固定 batch=1，无 dynamic axes —— 必须如此，否则 FC head 的 `Reshape` 会触发 ONNX shape inference 失败）。然后用 `onnxruntime.quantization.quantize_static` 做静态后训练量化：

```
quant_format = QDQ
activation_type = QUInt8
weight_type = QInt8
per_channel = True
calibrate_method = MinMax
calibration_data = 16 个 batch × 32 张验证集 300W
```

文件大小从 59.0 MB FP32 降到 **15.48 MB INT8**，测试 NME 仅退化 **0.07 个百分点**（0.0587 → 0.0594）。

## 4. 实验

下面所有数字都在 300W 测试集上（54 张图，seed 3407 随机切分的室内 + 室外合并集 12% 测试比例）。NME 使用左右眼中心点（36-41 与 42-47 的均值）距离归一化。

### 4.1 主结果

| 模型 | 参数 | 磁盘 (INT8) | NME | Acc@0.05 | Acc@0.08 | Acc@0.10 |
|---|---|---|---|---|---|---|
| 平均脸基线 | – | – | 0.298 | – | – | – |
| 2DFAN4（未微调）| 23.8 M | 90.9 MB FP32 | 0.113 | – | 34.2% | – |
| 2DFAN4 微调（FP32）| 23.8 M | 90.9 MB | 0.080 | 47.1% | 70.3% | – |
| 2DFAN4 微调（INT8）| 23.8 M | **24.79 MB** | 0.0961 | 38.3% | 61.5% | 72.3% |
| **LMNet w=2.6 + 100k 伪标签 + 微调（FP32）** | 15.5 M | 59.0 MB | **0.0587** | **55.0%** | **77.3%** | **85.2%** |
| **LMNet w=2.6 + 100k 伪标签 + 微调（INT8）** | 15.5 M | **15.48 MB** | **0.0594** | **53.5%** | **76.9%** | **85.0%** |

FP32 best 的验证集 NME 为 0.0514，距严格 5% 目标仅 **0.14 个百分点**。`runs/demo_compare.png` 与 `docs/images/hero.png` 中的可视化对比一致地显示我们的学生比 FAN4 INT8 教师更精准，尤其在侧脸、低光、夸张表情等输入上。

### 4.2 消融

| 配置 | 参数 | INT8 大小 | INT8 NME |
|---|---|---|---|
| LMNet w=2.25, 20k 伪标签, 单阶段 50 ep | 11.5 M | 11.61 MB | 0.0772 |
| LMNet w=2.25, 20k 伪标签 + 微调 40 ep | 11.5 M | 11.61 MB | 0.0676 |
| LMNet w=2.6, 20k 伪标签 + 微调 40 ep | 15.5 M | 15.48 MB | 0.0647 |
| **LMNet w=2.6, 100k 伪标签 + 微调 40 ep（最佳）** | 15.5 M | 15.48 MB | **0.0594** |

* **宽度**：2.25 → 2.6 换来 0.3 个百分点 NME，磁盘多 4 MB。再增大 width 收益迅速衰减；量化预算很快成为瓶颈。
* **伪标签规模**：20k → 100k 又拉低 0.5 个百分点。每张伪标签的边际 NME 下降非零但很小，超过 30k 后教师质量比规模更重要。
* **两阶段 vs 单阶段**：同 compute 下单阶段 NME ~0.075；加载 best.pt + 小 LR cosine + 提升 real_ratio 单 epoch 内突进 **2.5 个百分点**。

### 4.3 失败实验

| 配置 | NME |
|---|---|
| LMNet PIP head（28×28 cls + 亚像素 offset）50 ep | 0.215 |
| LMNet + WFLW augmented 联合训练（real_ratio_wflw=0.85）50 ep | 0.0915 |
| LMNet + 在线蒸馏（每 batch 跑 FAN4 教师）50 ep | 0.083 |
| LMNet + ConcatDataset(real, pseudo).shuffle（不加权）30 ep | 0.361 ↑ |

* PIP head plateau 是因为 28×28 网格把眼距归一化误差量化在≈1/28≈0.036；亚像素偏移在每 epoch 仅 432 张真值的情况下学不动。
* WFLW augmented 是 112×112 face crop。把 98 点子集映射到 "WFLW-68" 时嘴/眼非端点存在亚像素语义偏差；加上分辨率太低，反而 *拉高* 300W 测试 NME，尽管它带来了 75,000 张额外标注。
* 在线蒸馏几近中性。Stage A 伪标签生成完毕后教师在每步的额外贡献已经被吸收进伪标签里，每 batch 多一次 FAN4 forward 计算开销不划算。
* 不加权 `ConcatDataset(shuffle=True)` 是最具教益的失败：432 真值 / 20,000 伪标签下，优化器越来越好地拟合伪分布，而 valid NME 单调上升。这正是只要伪数据数量级压制真值就必须用 `WeightedRandomSampler` 的原因。

## 5. 讨论与限制

* **5% 是个硬天花板**。FP32 valid NME 0.0514 距 0.0500 还有 0.14 个百分点；FP32 test NME 0.0587 距 0.87 个百分点。剩余 gap 看起来在不增加标注数据或更强教师的情况下难以再降。我们可以用 QAT 缩小 FP32 → INT8 的 gap（路线图中的下一步），用双教师 ensemble 或 width=2.85 缩小 FP32 gap，但每个百分点 NME 的边际成本已经很高。
* **300W 之外未验证**。所有结果都在 300W 室内 + 室外（`split_strategy=all_random`，valid 10%、test 12%）切分上。WFLW 测试集、AFLW2000-3D 等跨数据集泛化是后续工作。
* **不含人脸检测**。本报告仅覆盖在已 crop 的方形人脸上做关键点回归。生产 pipeline 还需配人脸检测器（如 face_alignment 自带的 S3FD）。
* **依赖教师**。这套配方需要一个胜任的教师 bootstrap。我们用了开源的 2DFAN4 权重；若目标域无可用 / 可商用预训练教师，需要域内人工标注。

## 6. 结论

我们展示了：在仅有约 600 张人工标注图片的设定下，通过 100k 伪标签经 2DFAN4 教师扩展、合理加权的采样器、严谨的两阶段训练，一个 15.48 MB INT8 LMNet 可以在 300W 上匹配甚至超过 24.79 MB INT8 微调 2DFAN4（NME 0.0594 vs 0.0961，**相对改进 38%**），FP32 验证集 NME 触及 5.14% —— 距严格 5% 目标已是触手可及。配方简短，但围绕它的失败模式（EMA decay、采样器加权、FAN 解码契约、INT8 reshape 约束）足够微妙，值得文档化。所有代码、配置和伪标签均已开源。

## 致谢

2DFAN4 权重来源于 `face_alignment` 项目（Bulat & Tzimiropoulos）。300W 数据集来自 C. Sagonas 等的原始发布。CelebA 来自 CUHK MMLAB。

## 参考文献

简要列出本工作所基于的少量参考文献（不完备）：

* Sagonas, C. 等. "300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge." ICCV-W 2013.
* Bulat, A. 与 Tzimiropoulos, G. "How far are we from solving the 2D & 3D Face Alignment problem?" ICCV 2017.
* Liu, Z. 等. "Deep Learning Face Attributes in the Wild." ICCV 2015.
* Wu, W. 等. "Look at Boundary: A Boundary-Aware Face Alignment Algorithm." CVPR 2018（WFLW）.
* Jin, H. 等. "Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild." IJCV 2021.
* Hinton, G. 等. "Distilling the Knowledge in a Neural Network." 2015.
* Lee, D.-H. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." 2013.
* Howard, A. 等. "Searching for MobileNetV3." ICCV 2019.
