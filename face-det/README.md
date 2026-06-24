# face-det — 人脸检测研究

> 目标：基于现有 landmark 数据反推人脸框，训练一个强人脸检测基线，并在 `Test_data1` 2000 张图片上做跨域 benchmark。

中文（默认） | [English](README.en.md)

## 当前 benchmark 定义

- **训练集来源**：
  - LaPa: 18,168 train + 2,000 val + 2,000 test
  - JD-landmark: 26,386 images
  - WFLW_augmented: 74,950 train + 2,500 test
- **总计可导出 bbox 图像**：**126,004** 张
- **评测集**：`../data/Test_data1/`
  - `picture/`：2,000 张图片
  - `rect/`：2,000 个框标注，格式为 `x1 y1 x2 y2`

## 指标

- 训练/验证：YOLO 默认 detection metrics（`mAP50-95`, `mAP50`, precision, recall）
- 外部 benchmark：在 `Test_data1` 上计算：
  - `AP50`
  - `AP50-95`
  - `Recall@0.5`

## 设计选择

1. **检测框来源**：
   - LaPa / JD / WFLW 没有现成 bbox，但有人脸关键点；我们以 landmark 的 min/max 框为基础，外扩一定 margin 得到 face bbox。
2. **训练模型**：
   - 先用 Ultralytics YOLO 检测器做强基线（单类：face）。
3. **SOTA 路线定义**：
   - 先在自建 benchmark 上把跨域 `Test_data1` 指标推高，再决定是否切到 RetinaFace / SCRFD 路线。

## 快速开始

```powershell
# 1. 生成 YOLO 数据集（快速基线只用 LaPa）
py -3.12 face-det/scripts/build_face_detection_dataset.py --profile lapa_only

# 2. smoke train
py -3.12 face-det/scripts/train_yolo_face.py --epochs 1 --imgsz 640 --batch 16 --model yolov8n.pt --name smoke

# 3. 正式训练
py -3.12 face-det/scripts/train_yolo_face.py --epochs 30 --imgsz 640 --batch 32 --model yolov8n.pt --name yolo_face_baseline
```

## 当前结果

- 训练主线：`yolo_face_lapa_s`（`yolov8s.pt`, imgsz=640, batch=32）
- 训练数据：LaPa train 18,168 张（由 106 点 landmark 自动反推 bbox）
- 验证集：LaPa val 2,000 张
- 外部 benchmark：`Test_data1` 2,000 张 + 官方 `.rect` 框

### 当前最佳结果

**LaPa val（训练内验证）**
- epoch 17: `Precision=0.997`, `Recall=0.997`, `mAP50=0.9948`, `mAP50-95=0.9638`

**Test_data1（跨域 benchmark）**
- `AP50 = 0.995`
- `AP50-95 = 0.5817`
- `Precision = 0.99997`
- `Recall = 1.00000`

解释：在 `Test_data1` 上，**AP50 已经接近饱和**，说明这个检测器在“找得到脸”这件事上已经达到 SOTA-level 强基线；后续若要继续提升，主要空间会体现在更严格的 `AP50-95` 上。
