# face106 — 106 点人脸关键点研究

> 目标：在 LaPa（22 000 张图，106 点）上把自研小模型推到 2019 年 top-5 水平（NME ≤ 1.5%）。

中文（默认） | [README.en.md](README.en.md)

## 当前状态

- ✅ 已搭好训练栈（复用 face68/landmarklab/ 的代码，加 LaPaDataset、LaPa landmark loader）。
- ✅ LaPa 数据已解压到 `../data/LaPa/`（train 18168 / val 2000 / test 2000）。
- ✅ 可视化验证 LaPa 的 106 点分组（face contour 0-32、右眉 33-41、左眉 42-50、鼻 51-65、右眼 66-79、左眼 80-93、嘴 94-105）。
- ⏳ 训练 pipeline 未启动，等待跑 smoke 验证然后开始正式实验。

## 与 face68 的差异

| 项 | face68 | face106 |
|---|---|---|
| 训练样本量 | 600（300W）| 18 168（LaPa train）|
| 关键点数 | 68 | 106 |
| 是否需要伪标签 | 是（CelebA 100k）| 暂时不需要（数据已经多 30 倍）|
| eye groups for NME | (36-41) / (42-47) | (66-79) / (80-93) |
| 水平翻转 | 可（FLIP_ORDER_68 已知）| 暂禁用（106 点翻转表待整理）|

## 复现初始训练

```powershell
# 在仓库根
py -3.12 face106/landmarklab/train.py `
    --config face106/configs/lapa_lmnet.yaml `
    --override train.epochs=1 train.log_interval_steps=50 run_name=lapa_smoke
```

注意：训练入口要从仓库根运行，这样 `../data/LaPa/...` 相对路径才能解析。

## 已知未做的事

1. **106 点水平翻转表**。LaPa 106 点的左右对应顺序需要从 LaPa repo 或论文整理。当前 `LaPaDataset.enable_hflip=False`。
2. **优化协议**。第一版直接复用 face68 的 OneCycle + LMNet w=2.25。如果 NME 卡住会按 face68 经验调（width_mult、real_ratio、ema_decay、两阶段 fine-tune 等）。
3. **2019 SOTA 对照**。LaPa 论文报告的 baseline NME 1.4-1.5%（用 face bbox 归一化，与 inter-ocular 不直接可比）。等初版结果出来再校准目标。

## 许可证

MIT（与仓库根一致）。
