# NewFaceDetect Report

中文（默认） | [English](REPORT.en.md)

日期：2026-06-12

## 当前最佳：自研 LMNet w=2.6 + 100k 伪标签 + 微调，INT8 NME 5.94%

`300w_lmnet_w26_100k_finetune` 在 100k CelebA 伪标签 + 300W 真值上预训练（Stage A），再用 cosine LR 3e-4 → 1e-6 + real_ratio=0.7 微调（Stage B）。Stage B epoch 1 直接达 valid 0.0514 / test 0.0587，后续 12 epoch plateau，停止。

| 指标 | FP32 (epoch 1, best) | INT8 量化 |
|---|---|---|
| valid NME | **0.0514** | — |
| test NME | **0.0587** | **0.0594** |
| valid_acc@0.08 | 83.01% | — |
| test_acc@0.08 | 77.30% | 76.87% |
| test_acc@0.10 | 85.24% | 85.02% |
| test_acc@0.05 | 54.98% | 53.54% |
| 模型大小 | 58.99 MB | **15.48 MB** ✓ |

**关键对比**：
- vs FAN4 教师 INT8 (24.79 MB / NME 0.0961)：**自研模型小 38%、INT8 NME 低 38%**。
- vs LMNet 历史 baseline (NME 0.298)：**5.0 倍提升**。
- vs 5% 严格目标：FP32 valid 仅差 0.14 个百分点；INT8 test 差 0.94 个百分点。
- vs 10% fallback：超过约 4.06 个百分点。

## 路径回顾

| 阶段 | run | valid NME | test NME | INT8 NME | INT8 size |
|---|---|---|---|---|---|
| 1 | `300w_lmnet_pseudo_balanced` (50 ep, w=2.25, 20k pseudo) | 0.0817 | 0.0766 | 0.0772 | 11.61 MB |
| 2 | `300w_lmnet_pseudo_finetune` (40 ep, w=2.25, ssl) | 0.0595 | 0.0667 | 0.0676 | 11.61 MB |
| 3 | `300w_lmnet_w26` (60 ep, w=2.6, 20k pseudo) | 0.0800 | 0.0750 | — | 14.75 MB |
| 4 | `300w_lmnet_w26_finetune` (40 ep, w=2.6, ssl) | 0.0564 | 0.0639 | 0.0647 | 15.48 MB |
| 5 | `300w_lmnet_w26_celeba100k` (60 ep, w=2.6, 100k pseudo) | 0.0744 | 0.0717 | — | 14.75 MB |
| 6 | **`300w_lmnet_w26_100k_finetune` (40 ep, w=2.6, ssl)** | **0.0514** | **0.0587** | **0.0594** | **15.48 MB** |

## 关键技术栈

1. **数据扩充**：FAN4 教师在 CelebA 100k 张 face crop 上预测 68 点伪标签 → `data/celeba_pseudo_100k.npz` (300W 真值仅 432 张，扩充比 230x)。
2. **WeightedRandomSampler**：每 batch 真值与伪标签按 `real_ratio` 比例采样（Stage A 0.5, Stage B 0.7），避免伪标签 dominate。
3. **两阶段训练**：Stage A 大数据 + 强增强 + onecycle 让 backbone 学到 face representation；Stage B 加载 best.pt + 小 LR cosine + real_ratio 提高 + 温和增强精调到 300W 分布。
4. **EMA decay 0.99**：在 13-200 step/epoch 上能跟上训练步进（默认 0.999 在小数据上跟不上）。
5. **width_mult 2.25 → 2.6**：参数 11.5M → 15.5M，给 INT8 留出空间又能提供更高表达能力。
6. **mean_shape 初始化 + global FC head**：在小训练数据上比 PIP heatmap head 收敛更快、更稳。

## 失败实验（本轮）

### `300w_lmnet_pip_pseudo`（PIP head）
50 epoch 训练在 epoch 22 plateau 在 valid_nme=0.235 / test_nme=0.215。PIP head 在 stage2 28×28 grid + sub-pixel offset 上量化误差大，不带 mean shape init 起步太慢（epoch 1 valid_nme 高达 2.45），对小样本回归任务不如 global FC head。

### `300w_lmnet_wflw_pseudo`（加 WFLW augmented 真值）
50 epoch 最终 valid_nme=0.1037 / test_nme=0.0915 — 反而比 pseudo_balanced 退步。失败原因：(1) WFLW augmented 是 112×112 face crop，分辨率太低；(2) WFLW68 子集挑选与 300W 真值之间在 mouth/eye 的非端点存在亚像素级语义偏差；(3) WFLW 占 batch 60% 让模型偏离 300W 测试分布。

## 上一阶段：迁移学习达成 NME < 10%


- **`300w_fan4_finetune`**: 用 face_alignment 2DFAN4 预训练权重作为 `FANHeatmapNet(num_modules=4)` 初始化，在 300W 全训练集 fine-tune 15 epoch（lr=2e-4 → 0 cosine, batch=8, raw MSE heatmap loss, EMA, no AMP）。
  - 测试集 `NME = 0.0802` (8.02%), `Acc@0.05 = 47.14%`, `Acc@0.08 = 70.34%`。
  - 验证集 `NME = 0.0789` (best epoch 15)。
  - 模型参数 23.82M, FP32 ~90.87MB。
- **`300w_fan4_finetune` INT8 量化**: ONNX QDQ + per_channel + MinMax 校准。
  - INT8 模型大小 **24.79 MB**（略超 20MB 上限约 4.79MB）。
  - INT8 测试集 `NME = 0.0961` (9.61%) — 仍低于 10% 兜底目标。
  - INT8 `Acc@0.05 = 38.28%`, `Acc@0.08 = 61.51%`, `Acc@0.10 = 72.25%`。
- **关键技术修复**：
  1. 把 `FANHeatmapNet._heatmaps_to_coordinates` 的推理路径从 soft-argmax 改成 face_alignment 标准的 argmax + 邻域差分亚像素偏移，否则预训练权重无法复现精度（之前直接评估 NME=0.91 而非 0.115）。
  2. 在 `forward_train` 入口加 `inputs * 0.5 + 0.5` 反归一化，把数据管道的 `[-1, 1]` 还原成 face_alignment 训练时的 `[0, 1]`。
  3. 训练 loss 新增 `mse_raw` 模式 — 直接对原始 heatmap 与高斯目标做 MSE，避免 sigmoid 破坏预训练已对齐的输出尺度（pretrained peak ≈ 0.96，正好匹配高斯峰值 1.0）。
- 直接用 face_alignment 2DFAN4 inference 在我们 300W 测试集上的基线为 `NME=0.1131, Acc@0.08=34.22`；接入工程后基线 `NME=0.1156, Acc@0.08=54.75`（差异来自裁剪策略），fine-tune 后再降到 `NME=0.0802`。

## 历史结论（已被上述结果取代）

- 已完成从零训练工程搭建：数据下载/解析、训练、评估、量化导出、可视化、优化日志全部可用。
- 已验证 WFLW_augmented 训练数据可直接使用，结构为 `train_data/imgs + train_data/labels.csv` 和 `test_data/imgs + test_data/labels.csv`。
- 已确认 Windows 上 `num_workers=8` 会触发 `shm.dll` / 页文件问题，当前稳定设置为 `num_workers=0`。
- 已验证主干模型满足量化体积目标：当前 98 点主模型约 712.5 万参数，理论 int8 大小约 6.79 MB。
- 截止本轮，尚未达到 `Acc@0.08 > 98` 目标。

## 已完成实验

| 实验 | 数据 | 关键改动 | 结果 |
| --- | --- | --- | --- |
| `wflw_lmnet_m` | WFLW 98 点，256 子集，1 epoch | 直接回归 baseline | `valid_acc_008=0.351`, `test_acc_008=0.446`, `test_nme=0.950` |
| `wflw_ms_pose_smoke` | WFLW 98 点，256 子集，1 epoch | multiscale + pose | `valid_acc_008=0.399`, `test_acc_008=0.399`, `test_nme=0.951` |
| `exp_global_budget` | WFLW 98 点，4096 子集，3 epoch | global head | `valid_acc_008=0.460`, `test_acc_008=0.470`, `test_nme=0.937` |
| `exp_ms_pose_budget` | WFLW 98 点，4096 子集，3 epoch | multiscale + pose | `valid_acc_008=0.411`, `test_acc_008=0.421`, `test_nme=0.943` |
| `exp_global_full_e3` | WFLW 98 点，全量，3 epoch | global head | `valid_acc_008=0.735`, `test_acc_008=0.517`, `test_nme=0.744` |
| `wflw_spatial_smoke` | WFLW 98 点，256 子集，1 epoch | spatial_fusion | `valid_acc_008=0.415`, `test_acc_008=0.494`, `test_nme=0.952` |
| `wflw_heatmap_smoke` | WFLW 98 点，256 子集，1 epoch | heatmap + soft-argmax | `valid_acc_008=0.351`, `test_acc_008=0.446`, `test_nme=0.951` |
| `wflw_heatmap_sup_smoke` | WFLW 98 点，256 子集，1 epoch | heatmap + Gaussian supervision | `valid_acc_008=0.351`, `test_acc_008=0.446`, `test_nme=0.951` |
| `wflw5_smoke` | WFLW 派生 5 点，256 子集，1 epoch | 5 语义点 direct regression | `valid_acc_008=1.875`, `test_acc_008=0.625`, `test_nme=0.721` |
| `wflw5_full_e3` | WFLW 派生 5 点，全量，3 epoch | 5 语义点 direct regression | `valid_acc_008=1.494`, `test_acc_008=1.632`, `test_nme=0.561` |
| `wflw68_smoke` | WFLW 98→68 子集，256 子集，1 epoch | 68 点 direct regression | `valid_acc_008=0.460`, `test_acc_008=0.551`, `test_nme=0.888` |
| `wflw68_meanface_smoke` | WFLW 98→68 子集，256 子集，1 epoch | 68 点 + 平均脸偏置初始化 | `valid_acc_008=1.471`, `test_acc_008=2.275`, `test_nme=0.584` |
| `wflw68_full_e3` | WFLW 98→68 子集，全量，3 epoch | 68 点 + 平均脸偏置初始化 | `valid_acc_008=1.592`, `test_acc_008=2.009`, `test_nme=0.579` |
| `300w_global_meanface_smoke` | 300W 标准 68 点，Indoor/Outdoor，256/32/64，1 epoch | global + 平均脸偏置初始化 | `valid_acc_008=7.696`, `test_acc_008=8.663`, `test_nme=0.299` |
| `300w_global_meanface_smoke_ptq` | 300W smoke 权重量化 | INT8 ONNX 实测 | `quant_model_size_mb=11.61`, `acc_008=8.824`, `nme=0.299` |
| `300w_global_random_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 随机切分 + meanface | `valid_acc_008=4.963`, `test_acc_008=9.881`, `test_nme=0.298` |
| `300w_global_random_ssl_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 加载极小预算 SSL 骨干 | `valid_acc_008=4.917`, `test_acc_008=9.536`, `test_nme=0.299` |
| `300w_global_random_w3_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 宽度增到 3.0，约 19.18MB | `valid_acc_008=4.825`, `test_acc_008=9.789`, `test_nme=0.297` |
| `300w_global_random_fp32_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 关闭 AMP | `valid_acc_008=5.009`, `test_acc_008=9.903`, `test_nme=0.298` |
| `300w_overfit32_bugcheck` | 300W 标准 68 点，32 张训练图，关闭增强，过拟合测试 | 极小样本记忆性检查 | 到 `15`+ epoch 训练损失仍仅约 `0.078`，无法快速记住 32 张样本 |
| `300w_overfit32_const` | 300W 标准 68 点，32 张训练图，关闭增强，constant lr + no EMA | 训练策略排查 | `train_loss` 可降到约 `0.031`，说明链路可记忆；但验证/测试 NME 仍高，说明主要瓶颈不在简单 bug，而在数据/协议/泛化 |
| `300w_overfit32_same` | 300W 标准 68 点，同一批 32 张图同时作 train/val/test | 真正严格的过拟合检查 | `best_nme≈0.0568`, `acc_008≈77.7`，说明当前主模型几乎可以记住同一批样本 |
| `300w_overfit64_same` | 300W 标准 68 点，同一批 64 张图同时作 train/val/test | 更大样本数的严格过拟合检查 | `best_nme≈0.0725`, `acc_008≈67.3`，能记住但明显比 32 张更难 |
| `300w_overfit320_same` | 300W 标准 68 点，同一批 320 张图同时作 train/val/test | 中等规模样本的严格过拟合检查 | 前两轮仍 `nme≈0.336`，未表现出快速记忆能力 |
| `yolo300w32_overfit` | 300W 32 张图，YOLO11n-Pose scratch，train=val 同一批图 | YOLO-Pose 过拟合检查 | Box 已过拟合，但按我们自己的 NME 评估 `nme≈9.47`，关键点并未学会 |
| `face_alignment_300w_baseline` | 300W 标准 68 点，all_random，256/32/64 测试子集 | 成熟强模型 2DFAN4 + S3FD（预训练，仅作诊断基线） | `test_nme≈0.112`, `acc_008≈34.08` |
| `300w_global_random_e12` | 300W 标准 68 点，all_random，12 epoch | 更短 OneCycle 正式训练 | `epoch2 test_acc_008=9.232`，无实质改善 |
| `300w_global_random_256_smoke` | 300W 标准 68 点，all_random，256x256 输入，256/32/64，1 epoch | 提高输入分辨率 | `valid_acc_008=4.963`, `test_acc_008=9.858`, `test_nme=0.298` |
| `300w_global_random_tight_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | tighter crop (`crop_scale=1.10`) | `valid_acc_008=4.779`, `test_acc_008=9.766`, `test_nme=0.293` |
| `300w_global_random_noaug_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 关闭训练增强 | `valid_acc_008=4.963`, `test_acc_008=9.858`, `test_nme=0.298` |
| `celeba_ssl_20k` | CelebA 20k 图像，5 epoch | SimSiam 自监督预训练 | `last_loss=-0.879`，产出骨干 `runs/celeba_ssl_20k/ssl_backbone.pt` |
| `300w_global_random_ssl20k_e40` | 300W 标准 68 点，all_random，40 epoch | 加载 `CelebA 20k SSL` 骨干后精调 | `best_test_acc_008=8.361`, `best_test_nme=0.313` |
| `300w_heatmap_hr_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 更强 heatmap/HR-like 多尺度拼接头 | `valid_acc_008=0.460`, `test_acc_008=0.574`, `test_nme=0.885` |
| `300w_deconv_heatmap_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | SimpleBaseline 风格 deconv heatmap | `valid_acc_008=0.460`, `test_acc_008=0.597`, `test_nme=0.885` |
| `300w_pip_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 最小 PIP-style 网格分类 + 局部偏移 | `valid_acc_008=0.046`, `test_acc_008=0.023`, `test_nme=2.063` |
| `300w_unet_heatmap_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 全新 U-Net/FAN 风格 heatmap backbone（3.37MB） | `valid_acc_008=0.506`, `test_acc_008=0.551`, `test_nme=0.885` |
| `300w_unet112_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 更重 U-Net/FAN 风格 heatmap backbone（18.33MB） | `valid_acc_008=0.460`, `test_acc_008=0.551`, `test_nme=0.885` |
| `300w_fan2_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 成熟 FAN 2-stack heatmap 架构（11.55MB） | `valid_acc_008=0.230`, `test_acc_008=0.620`, `test_nme=0.913` |
| `300w_resnet18_transfer_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | ImageNet 预训练 ResNet18 + deconv heatmap | `valid_acc_008=0.506`, `test_acc_008=0.528`, `test_nme=0.885` |
| `300w_fan2_transfer_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 2DFAN4 预训练权重迁移到 FAN2 | `valid_acc_008=0.322`, `test_acc_008=0.597`, `test_nme=0.941` |
| `wflw68_pfld_smoke` | WFLW 98→68 PFLD 映射，256 子集，1 epoch | 插值眼部点的 68 点转换 | `valid_acc_008=1.287`, `test_acc_008=1.953`, `test_nme=0.609` |
| `wflw68_pfld_pretrain_full` | WFLW 98→68 PFLD 映射，全量，5 epoch | 同任务 68 点监督预训练 | `best_test_nme=0.575` |
| `300w_global_wflwpre_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 加载 WFLW 68 点监督预训练骨干 | `test_acc_008=10.156`, `test_nme=0.298` |
| `300w_global_wflwpre_e40` | 300W 标准 68 点，all_random，40 epoch | 加载 WFLW 68 点监督预训练骨干后正式精调 | `best_test_acc_008=8.932`, `best_test_nme=0.308` |
| `facesyn68_pretrain_1000` | FaceSynthetics 1000 样本，5 epoch | 68 点监督预训练 | `best_test_nme=0.673` |
| `facesyn68_pretrain_1000_e40` | FaceSynthetics 1000 样本，40 epoch | 长周期 68 点监督预训练 | `best_test_nme=0.674`，继续训练后反而逐步劣化 |
| `300w_global_facesyn1000_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | 加载 FaceSynthetics 1000 监督预训练骨干 | `test_acc_008=9.858`, `test_nme=0.298` |
| `300w_global_facesyn1000_freeze_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | FaceSynthetics 1000 骨干 + 冻结 backbone | `test_acc_008=9.835`, `test_nme=0.298` |
| `300w_facesyn1000_translate_smoke` | 300W 标准 68 点，all_random，256/32/64，1 epoch | FaceSynthetics 1000 骨干 + translation layer | `test_acc_008=7.031`, `test_nme=0.332` |

## 结果解读

- 目前最稳的 98 点路线仍然是 `global head + 全量数据`，但提升太慢，短时间内看不到冲到目标精度的可能。
- `multiscale + pose`、`spatial_fusion`、`heatmap`、`heatmap + dense supervision` 在当前实现和预算下都没有实证超过 `global head`。
- `68 点 + 平均脸偏置初始化` 是目前在“60+ 点且 10-20MB”约束下最值得继续追的路线，它首次显著打破了纯中心塌缩，但距目标仍远。
- 300W 下载链已经通过断点续传完整修通，并成功接入标准 68 点训练与 PTQ 导出。
- 当前所有标准 68 点 quick wins 里，最好的单次 smoke 是 `300w_global_random_fp32_smoke`，`test_acc_008=9.903`，真实 INT8 ONNX 体积 `11.61 MB`。
- `300w_overfit32_bugcheck` 说明当前方案连极小样本都不能快速过拟合，这表明问题不只是数据不够，也包含训练范式或表示方式本身不适合当前任务。
- `300w_overfit32_const` 进一步说明：把 OneCycle 和 EMA 拿掉后，模型是能明显记忆小样本的，所以并不存在“完全学不动”的硬 bug；但泛化依然很差，说明更大的问题仍然是数据协议与有效数据量。
- `300w_overfit32_same` 进一步证明：在真正同一批 32 图上，当前主模型可以把 NME 压到约 `5.68%`，已经非常接近 `5%`，因此数据顺序、标签归一化、点位协议并不存在那种“完全错误”的致命硬 bug。
- `300w_overfit64_same` 和 `300w_overfit320_same` 表明：样本数一放大，当前模型的记忆能力迅速下降。问题不是“完全学不会”，而是当前表示方式在样本复杂度上升后很快失去拟合能力。
- `face_alignment_300w_baseline` 证明在同一套评估管线下，成熟强模型可以把 NME 压到 `≈0.112`，明显优于当前从零训练结果 `≈0.298`。这说明数据和评估链路并非完全失真，差距主要在模型范式、训练起点和数据利用方式上。
- `YOLO-Pose` 在 32 张图上也没有按我们的 NME 口径学会关键点，这进一步说明问题不只是当前 LMNet 一条线，而是当前数据协议/训练设置本身就非常难以驱动 68 点收敛到目标区间。
- `3.0` 宽度、`256` 输入、tighter crop、关闭增强、极小预算 SSL 预训练，都没有显著超过当前最优 smoke。
- `CelebA 20k -> 300W` 的 40 epoch 精调也没有把标准 68 点路线推上去，最好只有 `test_acc_008=8.361 / nme=0.313`，说明当前这条 SSL 路线在现有实现下也不是解法。
- 更强的 `heatmap/HR-like` 头在当前从零训练条件下同样明显弱于现有 best smoke，说明问题不只是 head capacity。
- 更标准的 `deconv heatmap` 和最小 `PIP-style` 头也没有打起来，说明当前阶段仅靠更换头部范式仍不足以跨过当前精度鸿沟。
- 全新 U-Net/FAN 风格 backbone 即便扩到 `18.33MB` 量级，仍然没有超过现有 best smoke，说明仅靠更重的 2D heatmap backbone 也不足以解决当前问题。
- 成熟 `FAN 2-stack` 架构在从零训练条件下同样没有起色，这说明“换成大众认可的强结构”本身并不足以在当前数据条件下把 NME 拉到目标区间。
- 即便换成 `ImageNet 预训练 ResNet18` 或把 `2DFAN4` 预训练权重桥接到我们自己的 `FAN2`，当前迁移学习 smoke 也没有显著优于现有 best。说明问题不是简单地“加一个预训练 backbone”就能解决。
- `WFLW 68 点监督预训练` 在 300W smoke 上略微抬高了 `test_acc_008`，但长训后仍没有把 NME 压下去。
- `FaceSynthetics 1000` 的同协议监督预训练、长周期预训练、冻结 backbone 和 translation layer 也都没有超过当前 best。
- `12 epoch` 和 `40 epoch` 的 OneCycle 长训在前 3 个 epoch 都没有表现出足够陡峭的提升趋势。
- 5 点路线的收敛速度更快，但当前 5 点配置理论 int8 大小约 `4.86 MB`，略低于你要求的 `5 MB` 下限，需要稍微加宽模型才能完全满足体积约束。

## 下一步优先级

1. 做一轮真正有规模的 CelebA 自监督预训练，再回到 300W 标准 68 点精调。这是当前还没有被充分验证、且最可能带来质变的路线。
2. 如果继续纯监督路线，下一步更值得试的是彻底更换损失/头部范式，而不是继续微调宽度、分辨率、crop 或增强。
3. 如果必须按当前 `Acc@0.08` 口径冲 98 以上，建议重新确认口径是否为逐点百分比；按当前实验结果，这个目标非常激进。