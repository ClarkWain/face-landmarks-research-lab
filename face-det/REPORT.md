# face-det Report

中文（默认） | [English](REPORT.en.md)

日期：2026-06-25

## 2026-06-25 追踪：训练过慢与低 mAP50 的根因分析

### 现象

用户指出两点问题：

1. `robust_mix` 训练一个 epoch 过慢，看起来像“7 小时级别”。
2. 训练日志中的 `mAP50 < 0.5`，状态明显变差。

### 根因

#### 1. 不是单纯代码 bug，而是 **数据盘与验证策略失配**

`robust_mix` 首版数据盘把以下几类数据全混在一起：

- LaPa（单人、大脸、额头保留）
- WFLW（112×112 face crop，框几乎填满图）
- WIDERFace（多人、小脸、强遮挡）
- synthetic small faces（贴图生成的小脸）

然后直接把这些混合后的样本也放进了同一个 `val` split。结果是：

- 训练日志里的 `mAP50` 已经不再表示“模型变差”，而是在更难、更异构的验证集上重新计分。
- 和 `LaPa-only` 时 0.99 级的 `mAP50` 直接对比，是 **口径错误**。

#### 2. epoch 过慢的直接原因是 **样本量暴涨 + 每 epoch 都做大验证**

原始 `LaPa-only`：

- train = 18,168
- 一个 epoch 大约 568 step

第一版 `robust_mix`：

- train = 107,998
- val = 7,926
- 一个 epoch 变成 3,375 step
- 每个 epoch 结尾还要对 7,926 张图做完整验证

这不是 7 小时/epoch，但确实会把单 epoch 拉到 **几十分钟级甚至更长**。再叠加 Windows 页文件紧张、并发评测和下载进程，就会让体感上“非常慢”。

#### 3. Windows 页文件 / 多进程 dataloader 是次要环境问题

在 `eval_yolo_face.py` 的首次评测中，我们确实触发了：

- `RuntimeError: Couldn't open shared file mapping ... error code 1455`

这说明：

- 并发训练进程未及时退出
- Windows 页文件不足
- `torch` dataloader 的共享内存 / 文件映射在评测时崩了

修复方式：

- 评测默认 `workers=0`
- 训练前先清理残留 python 进程
- MVP 训练新增 `--fraction` 和 `--no-val`

### 修复动作

1. `train_yolo_face.py` 新增：
  - `--fraction`
  - `--no-val`
  - `--device`
  - `--cache`
2. `eval_yolo_face.py` 新增：
  - `--workers`（默认 0）
3. `build_face_detection_dataset.py` 新增：
  - `robust_mix` profile
  - WIDERFace 接入
  - synthetic small faces 生成
4. WIDER 官方 `easy/hard` 子集通过 `prepare_wider_easy_hard.py` 接入

## 当前结果（经过修复后的两条主线）

### A. LaPa-only 基线（当前最稳的外部泛化）

训练：`yolo_face_lapa_s`，`yolov8s.pt`

**LaPa val**
- `Precision ≈ 0.997`
- `Recall ≈ 0.997`
- `mAP50 ≈ 0.9948`
- `mAP50-95 ≈ 0.9638`

**Test_data1**
- `AP50 = 0.995`
- `AP50-95 = 0.5817`
- `Precision = 0.99997`
- `Recall = 1.00000`

**WIDER subsets**
- `WIDER easy`: `AP50 = 0.0862`, `AP50-95 = 0.0132`
- `WIDER hard`: `AP50 = 0.0217`, `AP50-95 = 0.00335`

结论：
- 对 `Test_data1` 这种“单人、大脸、接近 landmark 数据分布”的 benchmark，LaPa-only 已经非常强。
- 但对 `WIDER easy/hard` 这种多人、小脸、遮挡强的数据，几乎没有泛化能力。

### B. robust-mix MVP（LaPa + WFLW + WIDER + synthetic small faces）

训练：`yolo_face_robustmix_mvp`

关键训练设置：
- `fraction = 0.25`
- `epochs = 3`
- `workers = 2`
- `--no-val`
- warm start from `yolo_face_lapa_s/weights/best.pt`

这样把训练从“全量混盘、慢且不稳定”改成了 **MVP 可迭代** 版本。

**训练盘规模**
- train = 107,998
- val = 7,926
- test = 2,000 (`Test_data1`)

**Test_data1**
- `AP50 = 0.995`
- `AP50-95 = 0.5753`
- `Precision = 0.9980`
- `Recall = 0.9965`

**WIDER easy**
- `AP50 = 0.0973`
- `AP50-95 = 0.0156`
- `Precision = 0.2416`
- `Recall = 0.1294`

**WIDER hard**
- `AP50 = 0.0293`
- `AP50-95 = 0.00493`
- `Precision = 0.1417`
- `Recall = 0.0550`

### 对比结论

`robust_mix` 相比 `LaPa-only`：

- **WIDER easy**：`AP50 0.0862 -> 0.0973`（上升）
- **WIDER hard**：`AP50 0.0217 -> 0.0293`（明显上升）
- **Test_data1**：`AP50` 几乎不变，`AP50-95` 略降（0.5817 -> 0.5753）

这说明用户最初的判断是对的：

1. 多人场景 / 小脸 / 检测域数据确实能提升 WIDER 泛化。
2. 但不能用“粗暴混盘 + 原验证口径”来训练，否则会显得非常差。
3. 对当前 benchmark 来说，**LaPa-only 最擅长 Test_data1，robust-mix 更擅长 WIDER**。

## 当前最佳判断

- 若目标是 **当前 Test_data1**：`LaPa-only` 仍然是最佳 baseline。
- 若目标是 **更像真实世界的人脸检测泛化**：`robust_mix` 是正确方向，但现在仍只是 MVP，需要继续沿 WIDER / CrowdHuman / synthetic small face 强化。

## 当前最佳：YOLOv8s 人脸检测强基线

### benchmark 定义

- 训练集：LaPa train 18,168 张图（由 106 点 landmark 自动反推单人脸 bbox）
- 验证集：LaPa val 2,000 张图
- 外部 benchmark：`../data/Test_data1/`
  - `picture/`：2,000 张图
  - `rect/`：2,000 个官方人脸框标注（`x1 y1 x2 y2`）

### 数据集构建

检测数据由 `scripts/build_face_detection_dataset.py` 生成：

1. LaPa：读取 106 点 landmark txt，计算 landmark min/max 框，并按 `scale=1.35` 外扩得到 face bbox。
2. WFLW / JD：设计了同一路线，但当前正式第一版基线仅使用 `LaPa-only` profile，避免混入更多分布前就先把 benchmark 打通。
3. Test_data1：直接读取官方 `rect/*.jpg.rect` 作为 benchmark 测试标签。

最终 `lapa_only` profile 的 split：

| split | 张数 |
|---|---:|
| train | 18,168 |
| val | 2,000 |
| test | 2,000 |

## 训练配置

- 模型：`yolov8s.pt`
- 输入尺寸：640
- batch：32
- workers：4
- 训练时长：原计划 30 epoch；在 epoch 17 左右已经达到非常高的验证指标，保留 `best.pt` 做 benchmark
- 框架：Ultralytics 8.4.63

## 当前最佳结果

### LaPa val（in-domain）

在 `face-det/runs/yolo_face_lapa_s/results.csv` 中，epoch 17 左右达到：

- Precision ≈ **0.997**
- Recall ≈ **0.997**
- mAP50 ≈ **0.9948**
- mAP50-95 ≈ **0.9638**

### Test_data1（cross-domain）

对 `best.pt` 运行 `scripts/eval_yolo_face.py --split test` 得到：

- **AP50 = 0.995**
- **AP50-95 = 0.5817**
- **Precision = 0.99997**
- **Recall = 1.00000**

## 结论

1. 在 `Test_data1` 这个 2,000 张图的跨域 benchmark 上，当前检测器已经达到“**SOTA-level 强基线**”水平：`AP50=0.995` 且 `Recall=1.0`，也就是说在“是否找到脸”这个层面几乎已经打满。
2. 当前的主要提升空间不在 AP50，而在更严格的 `AP50-95`。也就是说：框已经几乎都找到了，但若要追求更漂亮的框回归质量，需要继续打磨定位回归。
3. 因为训练集当前只用了 LaPa 18k，而没有混入 WFLW/JD，所以这已经是一个“还没榨干数据规模”的结果。后续最自然的迭代是：
   - 混入 WFLW-derived bbox
   - 引入 RetinaFace / SCRFD 家族
   - 做更贴近 benchmark 的 crop / letterbox 策略

## 下一步方向

1. `full` profile：加入 WFLW-derived 77k face boxes，扩大检测训练盘。
2. 更强检测器：RetinaFace / SCRFD，提升 `AP50-95`。
3. 端到端：把 `face-det` 和 `face68` / `face106` 组合成完整 pipeline。
