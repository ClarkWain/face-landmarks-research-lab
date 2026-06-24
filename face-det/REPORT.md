# face-det Report

中文（默认） | [English](REPORT.en.md)

日期：2026-06-24

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
