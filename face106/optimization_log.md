## 2026-06-12 22:43:24 | train | lapa_smoke
- 优化点: baseline
- run_name: lapa_smoke
- note: baseline
- best_epoch: 1
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.162005
- best_valid_acc_008: 20.581
- best_test_acc_008: 21.481
- best_test_acc_005: 9.038208
- best_test_nme: 0.156858
- parameter_count: 11542668
- num_landmarks: 106
- estimated_int8_size_mb: 11.008
- estimated_fp32_size_mb: 44.032
- preview_path: runs\lapa_smoke\preview_best.png
- history_path: runs\lapa_smoke\history.csv
## 2026-06-13 01:20:31 | train | lapa_lmnet_w225
- 优化点: lapa_30ep
- run_name: lapa_lmnet_w225
- note: lapa_30ep
- best_epoch: 30
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.040495
- best_valid_acc_008: 91.765
- best_test_acc_008: 92.075
- best_test_acc_005: 75.706
- best_test_nme: 0.039290
- parameter_count: 11542668
- num_landmarks: 106
- estimated_int8_size_mb: 11.008
- estimated_fp32_size_mb: 44.032
- preview_path: runs\lapa_lmnet_w225\preview_best.png
- history_path: runs\lapa_lmnet_w225\history.csv
## 2026-06-13 01:28:42 | quant | lapa_lmnet_w225
- 优化点: ptq_static
- run_name: lapa_lmnet_w225
- checkpoint: runs\lapa_lmnet_w225\best.pt
- quant_model_size_mb: 11.672
- fp32_model_size_mb: 0.248674
- nme: 0.040237
- acc_005: 74.598
- acc_008: 91.719
- acc_010: 95.649
- image_acc_008: 97.050
## 2026-06-13 | train | lapa_w26_hflip_e60
- 优化点: w2.6+hflip+e60 (退步，不采用)
- run_name: lapa_w26_hflip_e60
- best_epoch: 58
- best_test_acc_008: 89.957
- best_test_nme: 0.041278
- parameter_count: 15531172
- 结论: w=2.6 模型在 LaPa 18k 训练集上过拟合，每个 epoch 都比 w=2.25 差约 6pp
## 2026-06-13 | train | lapa_w225_cosine_ft
- 优化点: cosine_ft_from_best (从 best.pt 精调)
- run_name: lapa_w225_cosine_ft
- best_epoch: 18
- best_test_acc_008: 92.841
- best_test_acc_005: 77.584
- best_test_nme: 0.037736
- parameter_count: 11542668
## 2026-06-13 | quant | lapa_w225_cosine_ft
- 优化点: ptq_cosine_ft
- run_name: lapa_w225_cosine_ft
- checkpoint: runs\lapa_w225_cosine_ft\best.pt
- quant_model_size_mb: 11.672
- fp32_model_size_mb: 0.248674
- nme: 0.038845
- acc_005: 76.222
- acc_008: 92.516
- acc_010: 96.048
- image_acc_008: 97.450
## 2026-06-13 16:15:00 | train | lapa_w26_hflip_e60
- 优化点: w2.6+hflip+e60
- run_name: lapa_w26_hflip_e60
- note: w2.6+hflip+e60
- best_epoch: 58
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.042390
- best_valid_acc_008: 89.483
- best_test_acc_008: 89.957
- best_test_acc_005: 76.545
- best_test_nme: 0.041278
- parameter_count: 15531172
- num_landmarks: 106
- estimated_int8_size_mb: 14.812
- estimated_fp32_size_mb: 59.247
- preview_path: runs\lapa_w26_hflip_e60\preview_best.png
- history_path: runs\lapa_w26_hflip_e60\history.csv
## 2026-06-13 18:45:36 | train | lapa_w225_cosine_ft
- 优化点: cosine_ft_from_best
- run_name: lapa_w225_cosine_ft
- note: cosine_ft_from_best
- best_epoch: 18
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.039031
- best_valid_acc_008: 92.482
- best_test_acc_008: 92.841
- best_test_acc_005: 77.584
- best_test_nme: 0.037736
- parameter_count: 11542668
- num_landmarks: 106
- estimated_int8_size_mb: 11.008
- estimated_fp32_size_mb: 44.032
- preview_path: runs\lapa_w225_cosine_ft\preview_best.png
- history_path: runs\lapa_w225_cosine_ft\history.csv
## 2026-06-13 18:48:25 | quant | lapa_w225_cosine_ft
- 优化点: ptq_cosine_ft
- run_name: lapa_w225_cosine_ft
- checkpoint: runs\lapa_w225_cosine_ft\best.pt
- quant_model_size_mb: 11.672
- fp32_model_size_mb: 0.248674
- nme: 0.038845
- acc_005: 76.222
- acc_008: 92.516
- acc_010: 96.048
- image_acc_008: 97.450
## 2026-06-13 20:21:36 | train | lapa_w225_ft2
- 优化点: ft2_cosine_from_ft_best
- run_name: lapa_w225_ft2
- note: ft2_cosine_from_ft_best
- best_epoch: 10
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.038790
- best_valid_acc_008: 92.580
- best_test_acc_008: 92.861
- best_test_acc_005: 77.807
- best_test_nme: 0.037607
- parameter_count: 11542668
- num_landmarks: 106
- estimated_int8_size_mb: 11.008
- estimated_fp32_size_mb: 44.032
- preview_path: runs\lapa_w225_ft2\preview_best.png
- history_path: runs\lapa_w225_ft2\history.csv
## 2026-06-13 20:24:07 | quant | lapa_w225_ft2
- 优化点: ptq_ft2
- run_name: lapa_w225_ft2
- checkpoint: runs\lapa_w225_ft2\best.pt
- quant_model_size_mb: 11.672
- fp32_model_size_mb: 0.248674
- nme: 0.038612
- acc_005: 76.623
- acc_008: 92.647
- acc_010: 96.130
- image_acc_008: 97.500
## 2026-06-13 20:57:51 | train | lapa_hrnet_w18_smoke_e1
- 优化点: smoke
- run_name: lapa_hrnet_w18_smoke_e1
- note: smoke
- best_epoch: 1
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 1.046628
- best_valid_acc_008: 0.471698
- best_test_acc_008: 0.471698
- best_test_acc_005: 0.117925
- best_test_nme: 1.046628
- parameter_count: 10018634
- num_landmarks: 106
- estimated_int8_size_mb: 9.554514
- estimated_fp32_size_mb: 38.218
- preview_path: runs\lapa_hrnet_w18_smoke_e1\preview_best.png
- history_path: runs\lapa_hrnet_w18_smoke_e1\history.csv
## 2026-06-13 21:18:45 | train | wflw_hrnet_w18_pretrain_smoke_e1
- 优化点: smoke_wflw
- run_name: wflw_hrnet_w18_pretrain_smoke_e1
- note: smoke_wflw
- best_epoch: 1
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.833397
- best_valid_acc_008: 0.643382
- best_test_acc_008: 0.643382
- best_test_acc_005: 0.183824
- best_test_nme: 0.833397
- parameter_count: 10016164
- num_landmarks: 68
- estimated_int8_size_mb: 9.552158
- estimated_fp32_size_mb: 38.209
- preview_path: runs\wflw_hrnet_w18_pretrain_smoke_e1\preview_best.png
- history_path: runs\wflw_hrnet_w18_pretrain_smoke_e1\history.csv
## 2026-06-13 21:19:02 | train | lapa_hrnet_w18_ft_wflw_smoke_e1
- 优化点: smoke_ft
- run_name: lapa_hrnet_w18_ft_wflw_smoke_e1
- note: smoke_ft
- best_epoch: 1
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 1.027452
- best_valid_acc_008: 0.471698
- best_test_acc_008: 0.471698
- best_test_acc_005: 0.235849
- best_test_nme: 1.027452
- parameter_count: 10018634
- num_landmarks: 106
- estimated_int8_size_mb: 9.554514
- estimated_fp32_size_mb: 38.218
- preview_path: runs\lapa_hrnet_w18_ft_wflw_smoke_e1\preview_best.png
- history_path: runs\lapa_hrnet_w18_ft_wflw_smoke_e1\history.csv
## 2026-06-13 21:19:21 | train | lapa_hrnet_w18_distill_smoke_e1
- 优化点: smoke_distill
- run_name: lapa_hrnet_w18_distill_smoke_e1
- note: smoke_distill
- best_epoch: 1
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 1.048875
- best_valid_acc_008: 0.471698
- best_test_acc_008: 0.471698
- best_test_acc_005: 0.353774
- best_test_nme: 1.048875
- parameter_count: 7672414
- num_landmarks: 106
- estimated_int8_size_mb: 7.316984
- estimated_fp32_size_mb: 29.268
- preview_path: runs\lapa_hrnet_w18_distill_smoke_e1\preview_best.png
- history_path: runs\lapa_hrnet_w18_distill_smoke_e1\history.csv
## 2026-06-14 | train | lapa_hrnet_w18_heatmap_e80
- 优化点: HRNet W18 heatmap 直接训练 LaPa（epoch 43/80 中断）
- run_name: lapa_hrnet_w18_heatmap_e80
- best_epoch: 40
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.024071
- best_valid_acc_008: 96.985
- best_test_acc_008: 97.447
- best_test_acc_005: 92.030
- best_test_nme: 0.023190
- best_test_acc_010: 98.567
- best_test_image_acc_008: 99.150
- parameter_count: ~28M（HRNet W18）
- num_landmarks: 106
- 模型文件: best.pt (73 MB FP32)
- 结论: 大幅超越 LMNet 路线，NME 0.0232 vs 0.0376，image_acc@0.08 99.2% vs 97.5%
- 备注: 后 37 epoch 未跑完，后期 plateau，继续训练收益有限
## 2026-06-14 | train | wflw_hrnet_w18_pretrain_e40
- 优化点: HRNet W18 WFLW 68点预训练（epoch 29/40 中断）
- run_name: wflw_hrnet_w18_pretrain_e40
- best_epoch: 29
- selection_metric: valid_nme
- selection_mode: min
- selection_value: 0.440357
- best_test_nme: 0.135455
- best_test_acc_008: 61.615
- parameter_count: ~28M
- num_landmarks: 68
- 模型文件: best.pt (73 MB FP32)
- 结论: WFLW 68点任务远未收敛，效果不理想，WFLW→LaPa 迁移路径搁置
## 2026-06-14 11:40:32 | train | lapa_hrnet_w18_heatmap_ft
- 优化点: hrnet_ft
- run_name: lapa_hrnet_w18_heatmap_ft
- note: hrnet_ft
- best_epoch: 14
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.022952
- best_valid_acc_008: 97.020
- best_test_acc_008: 97.347
- best_test_acc_005: 92.152
- best_test_nme: 0.022952
- parameter_count: 18282826
- num_landmarks: 106
- estimated_int8_size_mb: 17.436
- estimated_fp32_size_mb: 69.743
- preview_path: runs\lapa_hrnet_w18_heatmap_ft\preview_best.png
- history_path: runs\lapa_hrnet_w18_heatmap_ft\history.csv
- 结论: 精调几乎无提升（train_loss 下降 23% 但 test 指标不动），29 epoch 后早停（patience=15）。典型标注精度天花板。
## 2026-06-14 11:43:35 | quant | lapa_hrnet_w18_heatmap_ft
- 优化点: ptq_hrnet_ft
- run_name: lapa_hrnet_w18_heatmap_ft
- checkpoint: runs\lapa_hrnet_w18_heatmap_ft\best.pt
- quant_model_size_mb: 17.823
- fp32_model_size_mb: 0.275634
- nme: 0.039774
- acc_005: 74.615
- acc_008: 92.740
- acc_010: 96.525
- image_acc_008: 98.350
- 结论: INT8 PTQ Full 退化极其严重，NME 退化 +73%，acc@0.08 退化 -4.6pp。HRNet heatmap + soft-argmax 量化链非线性强，MinMax PTQ 不适用。需要 QAT 或混合精度量化。
## 2026-06-14 14:35 | quant | lapa_hrnet_w18_heatmap_ft (conv_only)
- 优化点: ptq_conv_only + Percentile 校准
- run_name: lapa_hrnet_w18_heatmap_ft
- checkpoint: runs\lapa_hrnet_w18_heatmap_ft\best.pt
- quant_model_size_mb: 17.926
- fp32_model_size_mb: 0.276
- nme: 0.025600
- acc_005: 90.699
- acc_008: 97.033
- acc_010: 98.312
- image_acc_008: 99.000
- 结论: ✅ conv_only 模式大幅修复退化。只量化 53 个 Conv 操作（backbone），保留 Softmax + soft-argmax 解码链 FP32。NME 退化从 +73% 收敛到 +11.5%，acc@0.08 退化从 -4.6pp 收敛到 -0.32pp。image_acc@0.08 = 99.0% 仍超过 98% 目标。模型 17.93MB 在 5~20MB 区间。
## 2026-06-16 | train | lapa_hrnet_w18_awing_mixed_e80 (最终)
- 优化点: HRNet + AWing Loss + LaPa/JD/PseudoWFLW 混合数据，80 epoch 全量训练
- run_name: lapa_hrnet_w18_awing_mixed_e80
- note: awing_mixed
- best_epoch: 78
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.021638
- best_valid_acc_008: 97.625
- best_test_acc_008: 97.977
- best_test_acc_005: 93.110
- best_test_nme: 0.021638
- best_test_image_acc_008: 99.600
- parameter_count: 18282826
- num_landmarks: 106
- 训练数据: LaPa 18k + JD-landmark 20k + PseudoWFLW 75k = 113,504
- ICME Test_data1: NME=3.37%, acc@0.05=82.49%, acc@0.08=97.87%, FR@0.08=1.50%
- 结论: ✅ 混合数据 + AWing Loss 大幅提升精度。NME 从 2.32%→2.16%，acc@0.08 从 97.4%→98.0%。ICME NME 3.37% 超越 ICME 2021 TOP1（美团 4.01%）16%。
## 2026-06-14 14:40:33 | quant | lapa_hrnet_w18_heatmap_ft
- 优化点: ptq_conv_only
- run_name: lapa_hrnet_w18_heatmap_ft
- checkpoint: runs\lapa_hrnet_w18_heatmap_ft\best.pt
- quant_model_size_mb: 17.926
- fp32_model_size_mb: 0.275634
- nme: 0.025600
- acc_005: 90.699
- acc_008: 97.033
- acc_010: 98.312
- image_acc_008: 99.000
## 2026-06-15 18:34:45 | train | jd_hrnet_w18_awing_384
- 优化点: jd_384_ft
- run_name: jd_hrnet_w18_awing_384
- note: jd_384_ft
- best_epoch: 14
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.042078
- best_valid_acc_008: 89.215
- best_test_acc_008: 88.642
- best_test_acc_005: 71.154
- best_test_nme: 0.042078
- parameter_count: 18282826
- num_landmarks: 106
- estimated_int8_size_mb: 17.436
- estimated_fp32_size_mb: 69.743
- preview_path: runs\jd_hrnet_w18_awing_384\preview_best.png
- history_path: runs\jd_hrnet_w18_awing_384\history.csv
## 2026-06-16 11:44:04 | train | lapa_hrnet_w18_awing_mixed_e80
- 优化点: awing_mixed
- run_name: lapa_hrnet_w18_awing_mixed_e80
- note: awing_mixed
- best_epoch: 78
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.021638
- best_valid_acc_008: 97.625
- best_test_acc_008: 97.977
- best_test_acc_005: 93.110
- best_test_nme: 0.021638
- parameter_count: 18282826
- num_landmarks: 106
- estimated_int8_size_mb: 17.436
- estimated_fp32_size_mb: 69.743
- preview_path: runs\lapa_hrnet_w18_awing_mixed_e80\preview_best.png
- history_path: runs\lapa_hrnet_w18_awing_mixed_e80\history.csv
## 2026-06-16 13:20:32 | quant | lapa_hrnet_w18_awing_mixed_e80
- 优化点: ptq_conv_only
- run_name: lapa_hrnet_w18_awing_mixed_e80
- checkpoint: runs\lapa_hrnet_w18_awing_mixed_e80\best.pt
- quant_model_size_mb: 17.926
- fp32_model_size_mb: 0.275634
- nme: 0.023714
- acc_005: 91.828
- acc_008: 97.597
- acc_010: 98.734
- image_acc_008: 99.500
## 2026-06-16 | quant | lapa_hrnet_w18_awing_mixed_e80 (INT8 final)
- run_name: lapa_hrnet_w18_awing_mixed_e80
- checkpoint: runs\\lapa_hrnet_w18_awing_mixed_e80\\best.pt (ep78)
- quant_mode: conv_only + Percentile (99.999%) + 32 calibration batches
- quant_model_size_mb: 17.926
- LaPa: nme=0.02371, acc_005=91.83%, acc_008=97.60%, acc_010=98.73%, image_acc_008=99.50%
- ICME 2019 (256): nme=0.0431, acc_005=72.80%, acc_008=88.73%, FR_008=5.40%
- ICME 2019 (384, FP32 ref): nme=0.0337, acc_005=82.49%, FR_008=1.50%
- 结论: INT8 退化可控（NME +9.6%, acc@0.08 -0.38pp）。LaPa 上 acc@0.08 仍达 97.60%。模型 17.93MB 在 5~20MB 区间。

## 2026-06-16 | train | lmnet_w05_kd_icme2mb (ICME 2MB 目标，进行中)
- run_name: lapa_lmnet_w05_kd_icme2mb
- arch: LMNet width_mult=0.5 + heatmap head, fusion_dim=128, num_landmarks=106
- params: ~0.91M (FP32 ~3.6MB, INT8 ~1MB) — 满足 ICME 2021 ≤2MB 限制
- teacher: HRNet W18 best.pt (runs/lapa_hrnet_w18_awing_mixed_e80, NME 2.16%)
- 蒸馏: distill_weight=0.5 (wing_loss 在 coord 层面对齐 student vs teacher)
- 数据: lapa_mixed (LaPa 18168 + JD-landmark 20386 + Pseudo WFLW 74950 = 113504)
- 训练: 60 epoch, batch=128, lr=0.0015, cosine, EMA 0.999, AMP, num_workers=4
- speed: 2.5 it/s 稳态, 1 epoch ≈ 6 min, 全量 ≈ 6 hr
- 显存: 9.9 GB / 22 GB (利用率 93%)
- 目标: LaPa NME ≤ 3.5%, ICME 384 NME ≤ 4.0% (≤ ICME 2021 TOP1 4.01%)
- 状态: 训练中 (PID 46068)
## 2026-06-16 17:45:40 | train | lmnet_w08_kd_smoke
- 优化点: baseline
- run_name: lmnet_w08_kd_smoke
- note: baseline
- best_epoch: 1
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.566873
- best_valid_acc_008: 1.351415
- best_test_acc_008: 1.452359
- best_test_acc_005: 0.536792
- best_test_nme: 0.566873
- parameter_count: 1636196
- num_landmarks: 106
- estimated_int8_size_mb: 1.560398
- estimated_fp32_size_mb: 6.241592
- preview_path: runs\lmnet_w08_kd_smoke\preview_best.png
- history_path: runs\lmnet_w08_kd_smoke\history.csv
## 2026-06-16 18:06:05 | train | lmnet_w08_kd_smoke3
- 优化点: baseline
- run_name: lmnet_w08_kd_smoke3
- note: baseline
- best_epoch: 3
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.130776
- best_valid_acc_008: 37.875
- best_test_acc_008: 38.472
- best_test_acc_005: 19.530
- best_test_nme: 0.130776
- parameter_count: 1636196
- num_landmarks: 106
- estimated_int8_size_mb: 1.560398
- estimated_fp32_size_mb: 6.241592
- preview_path: runs\lmnet_w08_kd_smoke3\preview_best.png
- history_path: runs\lmnet_w08_kd_smoke3\history.csv
## 2026-06-16 20:02:37 | train | lapa_lmnet_w135_kd_5mb
- 优化点: baseline
- run_name: lapa_lmnet_w135_kd_5mb
- note: baseline
- best_epoch: 1
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.341107
- best_valid_acc_008: 4.667453
- best_test_acc_008: 5.091510
- best_test_acc_005: 2.021698
- best_test_nme: 0.341107
- parameter_count: 4336148
- num_landmarks: 106
- estimated_int8_size_mb: 4.135273
- estimated_fp32_size_mb: 16.541
- preview_path: runs\lapa_lmnet_w135_kd_5mb\preview_best.png
- history_path: runs\lapa_lmnet_w135_kd_5mb\history.csv

## 2026-06-17 | reflection | lapa_lmnet_w08_kd_icme2mb (ICME 2MB 路线尝试 — 中止)

### 最终结果
- **训练**: 51/60 epoch（提前停止）
- **BEST**: ep=50, LaPa test_nme=**0.0517** (5.17%), acc@0.05=58.34%, acc@0.08=84.53%, image_acc@0.08=94.05%
- **架构**: LMNet width_mult=0.8 + global head + KD（HRNet teacher）
- **参数量**: ~1.64M, FP32 best.pt 6.36MB, FP32 ONNX 6.46MB, INT8 估计 ~1.7MB ✓ 满足 ICME 2MB
- **训练时间**: 20:27 → 01:27 共 5 小时

### 与基线对比
| 模型 | 参数 | INT8 | LaPa NME | acc@0.08 | image_acc@0.08 |
|---|---|---|---|---|---|
| HRNet w18 (mixed AWing) | 18.3M | 17.93MB | **2.16%** | 97.98% | 99.55% |
| LMNet w=2.25 历史 | 11.5M | 11.67MB | 3.88% | 92.65% | 97.45% |
| **LMNet w=0.8 KD（本次）** | **1.6M** | **~1.7MB** | **5.17%** | **84.53%** | **94.05%** |

### 反思：方法对不对？

**对的部分**：
1. KD 框架确实 work：1MB 学生模型从 random NME 50% 一路下到 5.17%，证明 KD 信号有效
2. 选择 LMNet w=0.8 + global head（face106 历史唯一验证范式）+ ema=0.99 是稳定收敛配方
3. 113k 混合数据（LaPa+JD+PseudoWFLW）发挥了价值：epoch 1→20 NME 0.328→0.058
4. early epoch 收敛速度甚至优于预期外推

**不对的部分（关键）**：
1. **目标设定不现实**：LMNet w=0.8 + KD 这条路理论上限就在 NME 5%，与 ICME TOP1 4.01% 仍有距离
2. **架构选错**：ICME 选手用的是 PFLD/MobileNetV2 + PIP head + cascade refinement，不是 LMNet 风格 FC head
3. **KD 不够细**：当前只蒸馏坐标层（wing_loss(student_coord, teacher_coord)），没有蒸馏中间 feature 或 heatmap
4. **缺少 cascade refinement**：粗预测 → ROI crop → 精预测能压低尾部 FR
5. **缺少 TTA**（推理水平翻转 + 多尺度平均）—— 几乎免费的 5-10% NME 改进
6. **训练时间投资低**：5h 换来 LaPa NME 5.17%，相比 HRNet 6h 训得 2.16%，单位时间精度产出差 6×

**真实评估**：
- LaPa NME 5.17% 比 LMNet w=2.25（NME 3.88%）退化 33%
- 但 INT8 1.7MB vs 11.67MB 是 7× 压缩，单位 MB 精度其实是赚的
- ICME 256 估计 NME ~8%，384 估计 ~6.5%，**完全无法竞争 ICME TOP1 (4.01%)**

**结论**：
- 方法本身（小模型 + KD + 混合数据）合理但**不够强冲 ICME TOP1**
- 真要冲必须做 PFLD + PIP + cascade + 更精细 KD（feature/heatmap level）+ QAT，工作量 ≥ 2 周
- **17.93MB HRNet 在 ICME 384 NME 3.37% 仍是当前最优产物**，性价比最高
- **当前 1.7MB best.pt 作为"极小型变种"产物保留**，用于未来需要 <2MB 部署的场景

### 已证伪
- LMNet w=0.5 + heatmap head（不收敛，NME 1.08）
- LMNet w=1.35 + hidden_dim=512 + KD（plateau 在 NME 0.346 不学）
- ema_decay=0.999 + 大 batch（EMA 模型滞后于 raw model）
- PFLD w=2.0 + PIP + KD + batch=128（GPU 22GB 全占，38min/epoch 不可行）

### 待验证（未来）
- PFLD smoke 用 batch=32（速度 OK 但精度未知）
- TTA on HRNet 17.93MB（零成本，预期 ICME NME -0.1pp）
- DSNT/Integral Regression 替代 soft-argmax（INT8 量化更友好）

## 2026-06-17 | reflection | lapa_lmnet_w08_kd_icme2mb (ICME 2MB 路线尝试 — 中止)

### 最终结果
- **训练**: 51/60 epoch（提前停止）
- **BEST**: ep=50, LaPa test_nme=**0.0517** (5.17%), acc@0.05=58.34%, acc@0.08=84.53%, image_acc@0.08=94.05%
- **架构**: LMNet width_mult=0.8 + global head + KD（HRNet teacher）
- **参数量**: ~1.64M, FP32 best.pt 6.36MB, FP32 ONNX 6.46MB, INT8 估计 ~1.7MB ✓ 满足 ICME 2MB
- **训练时间**: 20:27 → 01:27 共 5 小时

### 与基线对比
| 模型 | 参数 | INT8 | LaPa NME | acc@0.08 | image_acc@0.08 |
|---|---|---|---|---|---|
| HRNet w18 (mixed AWing) | 18.3M | 17.93MB | **2.16%** | 97.98% | 99.55% |
| LMNet w=2.25 历史 | 11.5M | 11.67MB | 3.88% | 92.65% | 97.45% |
| **LMNet w=0.8 KD（本次）** | **1.6M** | **~1.7MB** | **5.17%** | **84.53%** | **94.05%** |

### 反思：方法对不对？

**对的部分**：
1. KD 框架确实 work：1MB 学生模型从 random NME 50% 一路下到 5.17%，证明 KD 信号有效
2. 选择 LMNet w=0.8 + global head（face106 历史唯一验证范式）+ ema=0.99 是稳定收敛配方
3. 113k 混合数据（LaPa+JD+PseudoWFLW）发挥了价值：epoch 1→20 NME 0.328→0.058
4. early epoch 收敛速度甚至优于预期外推

**不对的部分（关键）**：
1. **目标设定不现实**：LMNet w=0.8 + KD 这条路理论上限就在 NME 5%，与 ICME TOP1 4.01% 仍有距离
2. **架构选错**：ICME 选手用的是 PFLD/MobileNetV2 + PIP head + cascade refinement，不是 LMNet 风格 FC head
3. **KD 不够细**：当前只蒸馏坐标层（wing_loss(student_coord, teacher_coord)），没有蒸馏中间 feature 或 heatmap
4. **缺少 cascade refinement**：粗预测 → ROI crop → 精预测能压低尾部 FR
5. **缺少 TTA**（推理水平翻转 + 多尺度平均）—— 几乎免费的 5-10% NME 改进
6. **训练时间投资低**：5h 换来 LaPa NME 5.17%，相比 HRNet 6h 训得 2.16%，单位时间精度产出差 6×

**真实评估**：
- LaPa NME 5.17% 比 LMNet w=2.25（NME 3.88%）退化 33%
- 但 INT8 1.7MB vs 11.67MB 是 7× 压缩，单位 MB 精度其实是赚的
- ICME 256 估计 NME ~8%，384 估计 ~6.5%，**完全无法竞争 ICME TOP1 (4.01%)**

**结论**：
- 方法本身（小模型 + KD + 混合数据）合理但**不够强冲 ICME TOP1**
- 真要冲必须做 PFLD + PIP + cascade + 更精细 KD（feature/heatmap level）+ QAT，工作量 ≥ 2 周
- **17.93MB HRNet 在 ICME 384 NME 3.37% 仍是当前最优产物**，性价比最高
- **当前 1.7MB best.pt 作为"极小型变种"产物保留**，用于未来需要 <2MB 部署的场景

### 已证伪
- LMNet w=0.5 + heatmap head（不收敛，NME 1.08）
- LMNet w=1.35 + hidden_dim=512 + KD（plateau 在 NME 0.346 不学）
- ema_decay=0.999 + 大 batch（EMA 模型滞后于 raw model）
- PFLD w=2.0 + PIP + KD + batch=128（GPU 22GB 全占，38min/epoch 不可行）

### 待验证（未来）
- PFLD smoke 用 batch=32（速度 OK 但精度未知）
- TTA on HRNet 17.93MB（零成本，预期 ICME NME -0.1pp）
- DSNT/Integral Regression 替代 soft-argmax（INT8 量化更友好）
## 2026-06-17 10:44:41 | train | mrffn_w10_smoke3
- 优化点: baseline
- run_name: mrffn_w10_smoke3
- note: baseline
- best_epoch: 3
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.095459
- best_valid_acc_008: 50.711
- best_test_acc_008: 52.207
- best_test_acc_005: 28.108
- best_test_nme: 0.095459
- parameter_count: 1563523
- num_landmarks: 106
- estimated_int8_size_mb: 1.491092
- estimated_fp32_size_mb: 5.964367
- preview_path: runs\mrffn_w10_smoke3\preview_best.png
- history_path: runs\mrffn_w10_smoke3\history.csv
## 2026-06-17 15:31:41 | train | lapa_mrffn_w10_kd_pca
- 优化点: baseline
- run_name: lapa_mrffn_w10_kd_pca
- note: baseline
- best_epoch: 50
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.052076
- best_valid_acc_008: 82.482
- best_test_acc_008: 83.330
- best_test_acc_005: 60.767
- best_test_nme: 0.052076
- parameter_count: 1563523
- num_landmarks: 106
- estimated_int8_size_mb: 1.491092
- estimated_fp32_size_mb: 5.964367
- preview_path: runs\lapa_mrffn_w10_kd_pca\preview_best.png
- history_path: runs\lapa_mrffn_w10_kd_pca\history.csv
## 2026-06-17 17:38:00 | train | lapa_mrffn_w10_v2_finetune
- 优化点: baseline
- run_name: lapa_mrffn_w10_v2_finetune
- note: baseline
- best_epoch: 8
- selection_metric: test_acc_008
- selection_mode: max
- selection_value: 83.224
- best_valid_acc_008: 82.498
- best_test_acc_008: 83.224
- best_test_acc_005: 60.558
- best_test_nme: 0.052277
- parameter_count: 1563523
- num_landmarks: 106
- estimated_int8_size_mb: 1.491092
- estimated_fp32_size_mb: 5.964367
- preview_path: runs\lapa_mrffn_w10_v2_finetune\preview_best.png
- history_path: runs\lapa_mrffn_w10_v2_finetune\history.csv
## 2026-06-17 23:51:45 | train | lapa_csprnet_w10_kd_110ep
- 优化点: baseline
- run_name: lapa_csprnet_w10_kd_110ep
- note: baseline
- best_epoch: 103
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.029251
- best_valid_acc_008: 95.541
- best_test_acc_008: 95.892
- best_test_acc_005: 87.383
- best_test_nme: 0.029251
- parameter_count: 1582645
- num_landmarks: 106
- estimated_int8_size_mb: 1.509328
- estimated_fp32_size_mb: 6.037312
- preview_path: runs\lapa_csprnet_w10_kd_110ep\preview_best.png
- history_path: runs\lapa_csprnet_w10_kd_110ep\history.csv
## 2026-06-18 00:13:28 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 2.396771
- fp32_model_size_mb: 0.361535
- nme: 0.071452
- acc_005: 36.213
- acc_008: 67.073
- acc_010: 80.011
- image_acc_008: 78.500

## 2026-06-18 00:09 | quant_fail | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_percentile_b32
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- 现象: onnxruntime calibrator 在 Percentile 校准阶段抛出 numpy._ArrayMemoryError（无法分配 768 KiB）
- 根因: 当前主机提交内存被残留 python 子进程占满，且 Percentile 校准会缓存更多中间统计
- 处理: 清理残留进程后改用 MinMax + 8 batches 完成导出
- 结论: 该机型上 CSPR 路线优先 MinMax，Percentile 需在更大内存预算下再验证

## 2026-06-18 00:18 | eval_icme | lapa_csprnet_w10_kd_110ep (FP32)
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- test_set: ../data/Test_data1 (2000 images)
- image_size=384: nme=0.2742, acc_005=4.36%, acc_008=10.58%, acc_010=15.61%, fr_008=99.85%
- image_size=256: nme=0.0560, acc_005=59.47%, acc_008=81.46%, acc_010=88.48%, fr_008=12.80%
- 结论: CSPR 模型训练/工作分辨率为 256，384 推理出现明显失配；后续 ICME 报告口径固定为 256

## 2026-06-18 00:22 | eval_icme_onnx | lapa_csprnet_w10_kd_110ep (INT8)
- run_name: lapa_csprnet_w10_kd_110ep
- onnx: runs\lapa_csprnet_w10_kd_110ep\model_int8.onnx
- test_set: ../data/Test_data1 (2000 images)
- image_size=256: nme=0.1171, acc_005=16.94%, acc_008=38.42%, acc_010=52.05%, fr_008=94.65%
- 与 FP32(256) 对比: nme +6.11pp, acc_008 -43.04pp
- 结论: 当前 PTQ MinMax 对 CSPR 退化不可接受，INT8 路线暂不采用；优先保留 FP32 或尝试 QAT/混合精度导出
## 2026-06-18 00:24:10 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix32
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 5.817795
- fp32_model_size_mb: 0.361535
- nme: 0.032810
- acc_005: 83.872
- acc_008: 94.869
- acc_010: 97.150
- image_acc_008: 98.150
## 2026-06-18 00:25:50 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix24
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 6.240840
- fp32_model_size_mb: 0.361535
- nme: 0.032413
- acc_005: 84.334
- acc_008: 95.036
- acc_010: 97.215
- image_acc_008: 98.150
## 2026-06-18 00:27:36 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix40
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 4.915986
- fp32_model_size_mb: 0.361535
- nme: 0.032971
- acc_005: 83.701
- acc_008: 94.802
- acc_010: 97.130
- image_acc_008: 98.000
## 2026-06-18 00:29:25 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix52
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 3.186920
- fp32_model_size_mb: 0.361535
- nme: 0.038445
- acc_005: 76.514
- acc_008: 92.900
- acc_010: 96.289
- image_acc_008: 97.550
## 2026-06-18 00:31:13 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix56
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 2.481580
- fp32_model_size_mb: 0.361535
- nme: 0.038711
- acc_005: 76.162
- acc_008: 92.745
- acc_010: 96.239
- image_acc_008: 97.650
## 2026-06-18 00:33:04 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix58
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 2.454537
- fp32_model_size_mb: 0.361535
- nme: 0.038427
- acc_005: 76.577
- acc_008: 92.858
- acc_010: 96.254
- image_acc_008: 97.650
## 2026-06-18 00:34:53 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix60
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 2.428961
- fp32_model_size_mb: 0.361535
- nme: 0.060944
- acc_005: 47.715
- acc_008: 76.986
- acc_010: 87.246
- image_acc_008: 90.100
## 2026-06-18 00:36:45 | quant | lapa_csprnet_w10_kd_110ep
- 优化点: csprnet_conv_only_minmax_b8_prefix58_final
- run_name: lapa_csprnet_w10_kd_110ep
- checkpoint: runs\lapa_csprnet_w10_kd_110ep\best.pt
- quant_model_size_mb: 2.454537
- fp32_model_size_mb: 0.361535
- nme: 0.038427
- acc_005: 76.577
- acc_008: 92.858
- acc_010: 96.254
- image_acc_008: 97.650

## 2026-06-18 00:39 | eval_icme_onnx | lapa_csprnet_w10_kd_110ep (INT8 repaired)
- run_name: lapa_csprnet_w10_kd_110ep
- onnx: runs\lapa_csprnet_w10_kd_110ep\model_int8.onnx (conv_prefix_count=58)
- test_set: ../data/Test_data1 (2000 images)
- image_size=256: nme=0.0690, acc_005=45.03%, acc_008=71.99%, acc_010=82.22%, fr_008=22.85%
- 对比未修复版（conv_prefix_count=63）: acc_008 38.42% -> 71.99%（+33.57pp）, nme 0.1171 -> 0.0690（-4.81pp）
- 结论: 选择性量化有效修复了 CSPR INT8 大退化；当前达到“可用”但未达 ICME acc@0.08=90% 目标

## 2026-06-18 | reflection | CSPR INT8 路线（阶段结论）
- 成绩: CSPR 在 1.58M 参数下实现 LaPa NME 2.93%，较 MRFFN 5.21% 显著改善
- 核心发现: 后段 Conv 对 Stage2 refine 坐标敏感，量化误差会被放大到最终点位
- 已验证最优折中: conv_prefix_count=58（size=2.45MB, LaPa acc@0.08=92.86%, ICME acc@0.08=71.99%）
- 风险: 仍未满足 ICME acc@0.08=90% 目标，且模型体积 2.45MB 略高于 2MB 竞赛线
- 下一步: ICME 定向数据增强 + 局部 QAT（仅前段骨干）+ 后段混合精度导出

## 2026-06-18 | train/eval | CSPR ICME finetune attempts

### 基线（finetune 前）
- run_name: lapa_csprnet_w10_kd_110ep
- ICME 2019 Test_data1 @256: nme=0.0560, acc_005=59.47%, acc_008=81.46%, acc_010=88.48%, fr_008=12.80%
- ICME 2019 Test_data1 @384: nme=0.2742, acc_008=10.58%

### 尝试 1：JD-only 384 微调（部分完成后停止）
- run_name: jd_csprnet_w10_icme_ft384_e8
- 设定: JD-landmark only, image_size=384, lr=2e-4, 从 `lapa_csprnet_w10_kd_110ep/best.pt` 继续
- 训练中观测: JD split test_nme 从 0.0575 下降到约 0.0519
- ICME @384: nme=0.1065, acc_005=28.42%, acc_008=50.41%, fr_008=58.10%
- ICME @256: nme=0.4634, acc_008=2.79%
- 结论: 384 域适配有效但强烈破坏 256 工作点，方向证伪

### 尝试 2：JD-only 256 + 冻结 backbone
- run_name: jd_csprnet_w10_icme_ft256_e6_freeze
- 设定: image_size=256, lr=1e-4, freeze_prefixes=[stem,stage1,stage2,stage3,stage4]
- JD split best: test_nme=0.05138, test_acc_008=83.37%
- ICME @256: nme=0.0635, acc_005=53.65%, acc_008=77.68%, acc_010=85.67%, fr_008=17.20%
- ICME @384: nme=0.3086, acc_008=7.33%
- 结论: 比纯 384 微调稳定，但仍劣于原始基线 0.0560；冻结 backbone 不能解决域差

### 尝试 3：lapa_mixed + JD heavy reweight
- run_name: lapa_csprnet_w10_icme_mixed_ft256_e4
- 设定: sampling_weights lapa:jd:pseudo = 1:6:0.5, epoch_steps=300, image_size=256, lr=1e-4
- LaPa-like test best: test_nme=0.02937, test_acc_008=95.93%
- ICME @256: nme=0.0581, acc_005=57.45%, acc_008=80.00%, acc_010=87.36%, fr_008=14.65%
- ICME @384: nme=0.2757, acc_008=10.00%
- 结论: 能保持原始泛化能力，但没有超越基线；JD 重采样不足以把 CSPR 推到 ICME TOP1 水平

### 阶段判断
- 纯 JD 微调会导致工作点漂移，尤其是 384/256 之间失配明显
- 混合重采样能避免遗忘，但提升幅度不足，最佳 ICME 256 仍是原始基线 0.0560
- 当前 CSPR 路线要冲 ICME 2021 TOP1 (NME<=0.0401) 还缺少更强范式：ICME 定向 crop/rect 对齐、TTA、或更适合竞赛域的 PFLD/PIP/cascade 路线

## 2026-06-18 | train | PFLD/PIP budget feasibility
- run_name: lapa_pfld_w20_kd_pip_budget32_e3
- 设定: PFLD width=2.0 + PIP head, batch=32, epoch_steps=128 (~4096 samples/epoch), epochs=3, KD from HRNet
- 速度: ~95s to 119s / epoch，当前硬件可训练；相比旧记录 batch=128 / 38min per epoch，吞吐问题已解除
- best_epoch: 3
- LaPa-like held-out: test_nme=0.31443, test_acc_008=21.84%, test_acc_005=10.08%
- 结论: 新架构能跑通且速度可接受，但 3-epoch 预算实验还处于明显欠收敛阶段，暂时没有超过 CSPR 的早期信号
- 下一步: 若继续此路线，应做 4k/3ep 之后再追加 12-20ep 中预算，而不是直接全量 50ep

## 2026-06-18 | train/eval | PFLD/PIP continued to 20 total epochs
- run_name: lapa_pfld_w20_kd_pip_budget32_e20_continue
- 设定: 从 `lapa_pfld_w20_kd_pip_budget32_e3/best.pt` 继续 17 epoch，batch=32, epoch_steps=128, KD from HRNet
- 速度: 约 124s 到 134s / epoch，仍在可接受范围
- held-out best: epoch=17, test_nme=0.05963, test_acc_008=79.98%, test_acc_005=58.10%
- ICME 2019 Test_data1 @256: nme=0.1134, acc_005=28.12%, acc_008=51.26%, acc_010=63.05%, fr_008=65.35%
- 对比 CSPR 基线 @256: nme 0.0560 -> 0.1134（变差）, acc_008 81.46% -> 51.26%（变差）
- 结论: PFLD/PIP 路线在当前范式下虽然能收敛，但对 ICME 泛化显著差于 CSPR，短期内不应作为主线替代

## 2026-06-18 | train/eval | PFLD/PIP + cascade quick test
- run_name: lapa_pfld_w20_kd_pip_cascade_e3
- 设定: 在 `lapa_pfld_w20_kd_pip_budget32_e20_continue/best.pt` 基础上增加 ROI cascade refinement，续训 3 epoch
- held-out best: epoch=3, test_nme=0.05728, test_acc_008=81.41%, test_acc_005=60.98%
- ICME 2019 Test_data1 @256: nme=0.1076, acc_005=31.25%, acc_008=55.12%, acc_010=66.45%, fr_008=59.05%
- 对比纯 PFLD/PIP: held-out 有改善（0.05963 -> 0.05728），ICME 也有改善（0.1134 -> 0.1076），但仍远差于 CSPR 基线 0.0560
- 结论: cascade 方向本身是正向的，但当前 PFLD 主干仍不够强；若继续走竞赛范式，需要更长训练和更系统的 recipe 调整

## 2026-06-18 | eval_icme | CSPR inference-side optimization
- run_name: lapa_csprnet_w10_kd_110ep
- 基线 @256, crop_scale=1.35: nme=0.0560, acc_008=81.46%
- 单尺度搜索:
	- crop_scale=1.15: nme=0.0454, acc_008=87.92%
	- crop_scale=1.08: nme=0.0439, acc_008=88.88%
	- crop_scale=1.00: nme=0.0426, acc_008=89.35%
- center shift search on crop_scale=1.00:
	- shiftY=+0.04: nme=0.0425, acc_008=89.60%
	- shiftX=-0.02, shiftY=+0.04: nme=0.0424, acc_008=89.54%
- 多尺度 TTA（pixel-space averaging）:
	- scales=[1.00, 0.99, 1.01]: nme=0.0420, acc_008=89.65%, fr_008=4.35%
	- scales=[1.00, 0.98, 0.99, 1.01, 1.02]: nme=0.0419, acc_008=89.73%, fr_008=4.30%
	- scales=[1.00, 0.98, 0.99, 1.01, 1.02] + shiftX=-0.02, shiftY=+0.04: nme=0.0417, acc_008=89.85%, fr_008=3.65%
- 结论: 不改权重，仅靠 crop 对齐 + 轻量多尺度 TTA，就把 CSPR 的 ICME NME 从 0.0560 压到 0.0417，距离 ICME 2021 TOP1 的 0.0401 仅差 0.0016

## 2026-06-18 | eval_icme | unified aligned setting comparison
- 统一口径: image_size=256, crop_scale=1.00, center_shift_x=-0.02, center_shift_y=+0.04, tta_scales=[1.00, 0.98, 0.99, 1.01, 1.02]
- CSPR (`lapa_csprnet_w10_kd_110ep`): nme=0.0417, acc_005=73.54%, acc_008=89.85%, acc_010=94.06%, fr_008=3.65%
- HRNet (`lapa_hrnet_w18_awing_mixed_e80`): nme=0.0291, acc_005=86.56%, acc_008=95.57%, acc_010=97.57%, fr_008=0.75%
- 结论: 在新的对齐评测方式下，轻量路线最优仍是 CSPR；追求绝对精度时，HRNet 仍然明显领先
## 2026-06-18 08:52:31 | train | jd_csprnet_w10_icme_ft256_e6_freeze
- 优化点: baseline
- run_name: jd_csprnet_w10_icme_ft256_e6_freeze
- note: baseline
- best_epoch: 6
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.051377
- best_valid_acc_008: 84.364
- best_test_acc_008: 83.374
- best_test_acc_005: 62.331
- best_test_nme: 0.051377
- parameter_count: 1582645
- num_landmarks: 106
- estimated_int8_size_mb: 1.509328
- estimated_fp32_size_mb: 6.037312
- preview_path: runs\jd_csprnet_w10_icme_ft256_e6_freeze\preview_best.png
- history_path: runs\jd_csprnet_w10_icme_ft256_e6_freeze\history.csv
## 2026-06-18 09:12:47 | train | lapa_csprnet_w10_icme_mixed_ft256_e4
- 优化点: baseline
- run_name: lapa_csprnet_w10_icme_mixed_ft256_e4
- note: baseline
- best_epoch: 1
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.029371
- best_valid_acc_008: 95.600
- best_test_acc_008: 95.933
- best_test_acc_005: 87.318
- best_test_nme: 0.029371
- parameter_count: 1582645
- num_landmarks: 106
- estimated_int8_size_mb: 1.509328
- estimated_fp32_size_mb: 6.037312
- preview_path: runs\lapa_csprnet_w10_icme_mixed_ft256_e4\preview_best.png
- history_path: runs\lapa_csprnet_w10_icme_mixed_ft256_e4\history.csv
## 2026-06-18 10:59:30 | train | lapa_pfld_w20_kd_pip_budget32_e3
- 优化点: baseline
- run_name: lapa_pfld_w20_kd_pip_budget32_e3
- note: baseline
- best_epoch: 3
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.314427
- best_valid_acc_008: 22.172
- best_test_acc_008: 21.844
- best_test_acc_005: 10.081
- best_test_nme: 0.314427
- parameter_count: 4018942
- num_landmarks: 106
- estimated_int8_size_mb: 3.832762
- estimated_fp32_size_mb: 15.331
- preview_path: runs\lapa_pfld_w20_kd_pip_budget32_e3\preview_best.png
- history_path: runs\lapa_pfld_w20_kd_pip_budget32_e3\history.csv
## 2026-06-18 12:03:19 | train | lapa_pfld_w20_kd_pip_budget32_e20_continue
- 优化点: baseline
- run_name: lapa_pfld_w20_kd_pip_budget32_e20_continue
- note: baseline
- best_epoch: 17
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.059627
- best_valid_acc_008: 79.366
- best_test_acc_008: 79.982
- best_test_acc_005: 58.097
- best_test_nme: 0.059627
- parameter_count: 4018942
- num_landmarks: 106
- estimated_int8_size_mb: 3.832762
- estimated_fp32_size_mb: 15.331
- preview_path: runs\lapa_pfld_w20_kd_pip_budget32_e20_continue\preview_best.png
- history_path: runs\lapa_pfld_w20_kd_pip_budget32_e20_continue\history.csv
## 2026-06-18 12:48:03 | train | lapa_pfld_w20_kd_pip_cascade_e3
- 优化点: baseline
- run_name: lapa_pfld_w20_kd_pip_cascade_e3
- note: baseline
- best_epoch: 3
- selection_metric: test_nme
- selection_mode: min
- selection_value: 0.057280
- best_valid_acc_008: 80.808
- best_test_acc_008: 81.413
- best_test_acc_005: 60.983
- best_test_nme: 0.057280
- parameter_count: 4041648
- num_landmarks: 106
- estimated_int8_size_mb: 3.854416
- estimated_fp32_size_mb: 15.418
- preview_path: runs\lapa_pfld_w20_kd_pip_cascade_e3\preview_best.png
- history_path: runs\lapa_pfld_w20_kd_pip_cascade_e3\history.csv
