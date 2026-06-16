# 人脸 Landmark 预测技术研究报告

更新时间：2026-06-14（UTC+8）

## 1. 结论摘要[text]

过去十年，人脸 landmark/face alignment 的主线不是单纯“点回归更准”，而是从稀疏 2D 点定位，演进到鲁棒的 2D/3D 几何估计、密集 face mesh、视频稳定性和端侧实时部署。

核心判断：

- 300W、AFLW、COFW 这类经典静态 2D benchmark 已经基本饱和。2017 年 FAN 系列论文已经明确指出当时模型在现有 2D/3D 数据集上接近饱和；后续改进更多体现在困难子集、失败率、跨域和部署效率，而不是主榜绝对误差的大幅下降。
- 真实应用还没有“到天花板”。遮挡、极端姿态、低分辨率、运动模糊、口罩/眼镜/手遮挡、非真人风格脸、视频抖动、表情细节、3D 一致性、隐私和功耗，仍是工程痛点。
- 模型架构从早期的级联回归和小 CNN，发展到深度级联、hourglass/heatmap、HRNet 高分辨率表示、边界/结构约束、轻量化单阶段网络、局部 patch/Transformer 关系建模、3DMM/mesh 联合估计。
- 如果目标是移动端产品级落地，目前优先选择 MediaPipe Face Landmarker/Face Mesh 这类成熟端侧方案：它把 BlazeFace 检测、Face Mesh V2、blendshape 预测打包成完整 pipeline，输出 478 个 3D landmarks、52 个 blendshape 系数和 face transform matrix，Android/iOS/Web/Python 均有官方支持。
- 如果只需要 68/98/106 个 2D 点，并且需要完全可控、可训练、模型极小，PFLD、PIPNet、Lite-HRNet/轻量 HRNet 派生方案仍然有价值。PFLD 论文报告 2.1MB、骁龙 845 上每张脸 140+ FPS；PIPNet 轻量版报告 CPU 35.7 FPS、GPU 200 FPS。

## 2. 任务定义与常见指标

人脸 landmark 预测通常输入人脸框或图像，输出预定义语义点：

- 稀疏 2D：5 点、21 点、68 点、98 点、106 点。
- 稀疏 3D：每个点含 x/y/z 或由 2D 点和姿态/深度共同估计。
- 密集 3D/mesh：468/478 点、几百到上千点，覆盖脸轮廓、五官、眼睛、虹膜或牙齿等区域。

常见评估指标：

- NME（Normalized Mean Error）：平均 landmark 距离除以归一化尺度，常用眼距、瞳距、脸框尺寸或外眼角距离。不同论文归一化方式不同，不能无脑横向比较。
- FR（Failure Rate）：误差超过阈值的失败比例，常见阈值 0.08 或 0.1。
- AUC（Cumulative Error Distribution）：误差分布曲线面积，越高越好。
- 视频任务还会关注 jitter、时序平滑、延迟和跟踪丢失率。
- 移动端还必须看模型大小、端到端延迟、功耗、NPU/GPU/CPU 兼容性和多脸性能。

## 3. 数据集和 Benchmark 演进

### 3.1 经典 2D 数据集

- 300W：68 点，长期作为 2D face alignment 标准基准，包括 LFPW、HELEN、AFW、iBUG 等来源。
- AFLW：更强调大姿态和野外场景。
- COFW：强调遮挡。
- WFLW：98 点，覆盖姿态、表情、光照、妆容、遮挡、模糊等属性。LAB 论文提出 WFLW，用于统一评估更真实的困难因素。
- 300VW：视频 landmark/tracking benchmark，强调时序稳定性。

### 3.2 更密集与更实用的数据

- LS3D-W：FAN/2D&3D face alignment 论文构建约 23 万张 3D landmark 数据，用于跨数据集 3D 对齐。
- JD-landmark/106 点：2019 年 106 点 challenge 认为传统 68 点不足以描述复杂面部结构，推动更密集稀疏点集。
- MediaPipe Face Mesh：468/478 3D 点，更适合 AR、虚拟头像、滤镜和实时表情驱动。
- Dense landmarks：微软 2022 年 dense landmark 工作展示了用更多 landmark 辅助 3D 重建和 performance capture，论文报告 dense landmark + 3D fitting 在单 CPU 线程可超过 150 FPS。

## 4. Benchmark 与阶段性分数

### 4.1 读 benchmark 时的注意事项

人脸 landmark 的“排行榜”比分类/检测任务更容易误读，主要原因是：

- NME 的归一化尺度不同：300W 常见 inter-ocular/inter-pupil，WFLW 常按外眼角距离，AFLW 又可能按脸框尺寸或其他尺度。
- 输入条件不同：有的用 ground-truth bbox，有的用检测器 bbox；face crop 策略差一点，NME 会明显变化。
- 点数不同：68 点、98 点、106 点、468 点不能直接比较。
- 静态图和视频不等价：单帧 NME 低，不代表视频 jitter 低。
- 公开网站中的 SOTA 页面不总是稳定。此次检索中 Papers with Code 的 SOTA URL 出现重定向到 Hugging Face trending page 的情况，因此本报告不把它作为主来源；优先采用论文摘要、论文表格、官方数据集页面和官方项目页。

下面的表是“可靠 benchmark 锚点 + 代表性阶段排名”，不是全量最新榜单。

### 4.2 WFLW 官方首版 benchmark：2018 年困难 98 点数据集

WFLW 是过去十年最有代表性的 2D face alignment 困难 benchmark 之一。官方页面说明其包含 10,000 张脸，7,500 张训练、2,500 张测试，98 个手工标注 landmarks，并提供 pose、expression、illumination、makeup、occlusion、blur 属性划分。

WFLW 官方/LAB 论文中的首版排名如下，指标为 NME（%）、AUC@0.1 和 FR@0.1，NME/FR 越低越好，AUC 越高越好。

| 排名（2018 首版 fullset） | 方法 | 年份 | 架构阶段 | WFLW Fullset NME ↓ | AUC@0.1 ↑ | FR@0.1 ↓ | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | LAB | 2018 | 边界感知 + landmark | 5.27 | 0.5323 | 7.56 | WFLW 发布论文方法 |
| 2 | DVLN | 2017 | 深度网络 + shape/view constraint | 6.08 | 0.4551 | 10.84 | 2018 前强基线 |
| 3 | CFSS | 2015 | 粗到细 shape search | 9.07 | 0.3661 | 20.56 | 传统级联强基线 |
| 4 | SDM | 2013 | supervised descent | 10.29 | 0.3002 | 29.40 | 传统回归基线 |
| 5 | ESR | 2012 | ensemble regression trees | 11.13 | 0.2774 | 35.24 | 早期强基线 |

WFLW 首版结果的意义：

- 2015-2018 年从传统/级联方法到深度结构约束，fullset NME 从约 11% 降到约 5.3%，失败率从 35% 降到 7.6%。
- pose 子集的改善更关键：传统方法在大姿态下会明显崩溃，LAB 通过边界语义缓解轮廓点歧义。
- 这也是后续 AWing、PIPNet、SCC、SLPT、LDEQ 等方法继续刷 WFLW 的原因。

### 4.3 WFLW 跨阶段代表分数：2018-2023

下表把 WFLW 上不同阶段的代表方法放在一起。它用于观察趋势，不应被理解为严格同一环境下的官方总榜。

| 阶段 | 方法 | 年份 | 主要思想 | WFLW Fullset NME ↓ | FR@0.1 ↓ | 来源可靠性 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 数据集首版 | LAB | 2018 | 边界线辅助 landmark | 5.27 | 7.56 | 官方 WFLW/LAB 页面与论文表 |
| 损失函数改进 | Wing Loss | 2018 | 中小误差放大、pose balancing | 约 5.1 | 约 6.0 | 后续论文表格常用对比 |
| Heatmap loss 成熟 | Adaptive Wing Loss | 2019 | adaptive heatmap loss + weighted loss map | 约 4.36 | 约 2.84 | AWing 论文表格；arXiv 摘要确认 COFW/300W/WFLW 全面实验 |
| 不确定性建模 | LUVLi | 2020 | 位置 + uncertainty + visibility | 约 4.37 | 约 3.1 | LUVLi 论文表格 |
| 结构关系建模 | SCC | 2020 | sparse graph structure coherence + Soft Wing | 约 4.4 | 2.88 | arXiv 摘要明确报告 WFLW FR=2.88 |
| 局部 patch Transformer | SLPT | 2022 | sparse local patch + attention relation | 约 4.1 | 约 2.7 | SLPT 论文表格；arXiv 摘要确认 WFLW/300W/COFW SOTA 级 |
| 视频/递归稳定 | LDEQ | 2023 | deep equilibrium refinement + flicker reduction | 3.92 | 未在摘要列出 | arXiv 摘要明确报告 WFLW NME=3.92 |

趋势解读：

- WFLW fullset NME 从 LAB 的 5.27 到 2023 年 LDEQ 的 3.92，绝对下降约 1.35 个百分点，相对下降约 25.6%。
- 2018-2020 的主要收益来自 heatmap loss、边界/结构约束和更好的训练权重。
- 2021 以后，收益更多来自点间关系、局部 refinement、Transformer/DEQ 和视频稳定，而不是单纯更深的 backbone。
- FR 的下降比 NME 更能说明鲁棒性提升：从 LAB 的 7.56 到 AWing/SCC 约 2.8-2.9，困难样本失败率显著降低。

### 4.4 300W / COFW / AFLW 的可复核锚点

300W 是最经典的 68 点 benchmark，但不同论文 protocol 差异较大。这里不做全榜，只列来源中明确可核对的阶段锚点。

| 数据集 | 方法 | 年份 | 指标 | 分数 | 来源与意义 |
| --- | --- | --- | --- | ---: | --- |
| 300W Fullset | LAB | 2018 | mean error / NME | 3.49% | LAB arXiv 摘要直接报告；说明边界约束在经典 300W 上也超过当时 SOTA |
| COFW | LAB | 2018 | mean error / NME | 3.92% | LAB arXiv 摘要直接报告；COFW 强调遮挡 |
| COFW | LAB | 2018 | failure rate | 0.39% | LAB arXiv 摘要直接报告；遮挡集失败率非常低 |
| AFLW-Full | LAB | 2018 | mean error / NME | 1.25% | LAB arXiv 摘要直接报告；大姿态/多视角相关 |
| COFW | SCC | 2020 | failure rate | 0% | SCC arXiv 摘要直接报告；强调结构一致性降低失败 |
| WFLW | SCC | 2020 | failure rate | 2.88% | SCC arXiv 摘要直接报告；代表 2020 年鲁棒性水平 |

300W 的使用建议：

- 适合做基本 sanity check 和与历史方法对齐。
- 不适合单独判断真实产品效果，因为它对遮挡、模糊、视频 jitter、端侧延迟和跨域泛化覆盖不足。
- 论文 “An Empirical Study of Recent Face Alignment Methods” 专门指出，不同实验设置、检测器和评价细节会影响 300W 横向比较；因此工程选型应同时看 WFLW/COFW/视频/端侧效率。

### 4.5 移动端与实时性 benchmark

移动端方案不能只看 NME。下面是公开论文/官方页面中比较可靠的效率指标锚点。

| 方法/系统 | 年份 | 输出 | 设备/条件 | 公开效率指标 | 适合场景 |
| --- | --- | --- | --- | --- | --- |
| MTCNN | 2016 | face bbox + 5 点 | CPU/GPU 均可实现 | 论文强调实时检测与对齐；速度依赖实现 | 检测 + 粗对齐 |
| PFLD 0.25x | 2019 | 稀疏 2D landmarks | Qualcomm ARM 845 | 模型 2.1MB，单脸 140+ FPS | 端侧 68/98/106 点稀疏定位 |
| MediaPipe Face Mesh | 2019 | 468 个 3D landmarks | mobile GPU | 论文报告 100-1000+ FPS，取决于设备/模型 | AR、滤镜、mesh、表情 |
| PIPNet 轻量版 | 2020 | 稀疏 2D landmarks | 论文实验环境 | CPU 35.7 FPS，GPU 200 FPS | 精度/速度折中，低分辨率解码 |
| 3DDFA-V2 | 2020 | 3DMM/dense alignment | 单 CPU core | 50+ FPS | 3D 姿态、dense face alignment |
| MediaPipe Face Landmarker | 2026 官方文档 | 478 个 3D landmarks + 52 blendshape + transform matrix | Android/iOS/Web/Python | 官方任务化 pipeline，支持 IMAGE/VIDEO/LIVE_STREAM | 当前移动端产品基线 |

移动端排序建议：

- 综合产品落地：MediaPipe Face Landmarker > 自研 PFLD/PIPNet > 重型 HRNet/Transformer。
- 极致小模型：PFLD 优先。
- 稀疏点精度/速度平衡：PIPNet 或轻量 HRNet。
- 需要 3D 姿态和 dense geometry：MediaPipe Face Landmarker 或 3DDFA-V2。

### 4.6 视频稳定性 benchmark

视频 landmark 的评价从单帧 NME 扩展到 jitter/flicker。LDEQ 论文提出 WFLW-V hard subset，包含 500 个视频，并使用 normalized mean flicker（NMF）衡量闪烁。

| 方法/论文 | 年份 | Benchmark | 指标 | 结果 |
| --- | --- | --- | --- | --- |
| LDEQ / Recurrence without Recurrence | 2023 | WFLW | NME | 3.92 |
| LDEQ + RwR | 2023 | WFLW-V hard, 500 videos | NME / NMF | 相比最强手调滤波基线，NME 改善 10%，NMF 改善 13% |

这说明 2023 年以后的研究重点已经明显从“静态图像更准一点”转向“视频中更稳、更少抖动”。

### 4.7 Benchmark 结论

- 2012-2018：WFLW fullset NME 从 ESR 11.13 到 LAB 5.27，深度模型和边界语义带来数量级明显的鲁棒性提升。
- 2018-2020：AWing/LUVLi/SCC 把 WFLW fullset NME 推到 4.3-4.4 区间，FR@0.1 降到约 3%。
- 2022-2023：SLPT/LDEQ 把 WFLW 推到约 4.1/3.92，并开始强调点间关系和视频稳定。
- 300W/COFW 等经典数据集仍有参考价值，但更适合作为回归测试，不足以代表真实移动端体验。
- 移动端 benchmark 的“第一指标”应是端到端延迟、功耗、稳定性和点位语义是否满足下游任务，而不是单一 NME。

## 5. 近十年技术演进

### 阶段 A：级联回归到 CNN 多任务（约 2015-2016）

代表方法：

- 3DDFA / Face Alignment Across Large Poses（2015/2016）
- MTCNN（2016）
- All-in-One CNN（2016）

主要问题：

- 传统 AAM/CLM/SDM 等方法在受控场景可用，但面对大姿态、光照、遮挡和初始化误差不稳定。
- 2D landmark 对 profile face 的语义定义不一致：侧脸中“不可见点”到底该预测到哪里是根本问题。

技术变化：

- CNN 开始替代手工特征和传统回归器。
- 检测与对齐联合训练，利用共享特征和 multi-task learning。
- 3DMM 被引入大姿态对齐，解决 profile face 中可见/不可见点和语义一致性问题。

成果：

- MTCNN 用三级级联网络同时做人脸检测和少量 landmark 定位，并引入 online hard sample mining，在 FDDB/WIDER FACE/AFLW 等任务上取得实时性能。
- 3DDFA 用 3DMM 和 CNN 回归大姿态人脸，对 yaw 接近 90 度的问题给出系统解法。

### 阶段 B：深度级联、Hourglass 与 Heatmap 主导（约 2017-2018）

代表方法：

- DAN（Deep Alignment Network，2017）
- FAN / “How far are we from solving the 2D & 3D Face Alignment problem?”（2017）
- Wing Loss（2017/2018）
- LAB（Look at Boundary，2018）
- Super-FAN（2017/2018）

主要问题：

- 直接坐标回归容易丢空间结构。
- 大姿态、初始化差、低分辨率、遮挡仍会造成失败。
- 传统 landmark 标注本身有歧义，例如脸轮廓点并不是固定解剖点，而是图像边界上的可见轮廓。

技术变化：

- Heatmap regression 成为主流：网络输出每个点的概率热图，再解码坐标。
- Hourglass/stacked hourglass 结构用于保持多尺度空间信息。
- DAN 使用多阶段 refinement，每一阶段利用前一阶段 landmark heatmap 辅助修正。
- Wing Loss 强调中小误差区间，避免 L2/L1 对定位任务的梯度分配不理想。
- LAB 通过边界线预测缓解 landmark 定义歧义，再从边界导出点。

成果：

- DAN 论文报告在公开数据集上把失败率最多降低约 70%。
- LAB 论文报告 300W Fullset NME 3.49%，COFW NME 3.92% 且失败率 0.39%，AFLW-Full NME 1.25%。
- FAN 论文指出强基线在当时的 2D/3D 数据集上已经非常接近饱和，这一判断影响了后续研究方向：从“刷 300W”转向鲁棒性、3D、视频和端侧。

### 阶段 C：结构约束、鲁棒损失与轻量端侧（约 2018-2020）

代表方法：

- Adaptive Wing Loss（2019）
- HRNet / HRNetV2 用于 facial landmark（2019）
- PFLD（2019）
- PIPNet（2020）
- LUVLi（2020）
- 3DDFA-V2（2020）
- MediaPipe Face Mesh（2019）

主要问题：

- Heatmap 高精度但计算量大，分辨率越高越耗算力。
- 前景点和大面积背景像素不平衡，普通 heatmap loss 训练效率低。
- 真实系统需要知道不确定性和可见性，而不只是输出一个坐标。
- 移动端需要极低延迟、小模型和稳定视频表现。

技术变化：

- HRNet 保持全流程高分辨率表示，比 encoder-decoder 式反复降采样/上采样更适合精确定位。
- Adaptive Wing Loss 对 heatmap 前景和困难背景赋予更合适的损失形状与权重。
- PFLD 放弃重型 heatmap，采用轻量单阶段坐标回归，训练时引入姿态/旋转辅助监督，推理时不需要额外分支。
- PIPNet 在低分辨率特征图上同时预测 score 与 offset，避免昂贵上采样，并用邻域回归加强局部几何约束。
- LUVLi 同时预测位置、不确定性和可见性，适合失败检测和下游风险控制。
- 3DDFA-V2 用轻量 backbone 和 3DMM 参数回归，在速度、精度和视频稳定性之间折中。

成果：

- PFLD 报告模型可做到 2.1MB，并在 Qualcomm ARM 845 设备上每张脸 140+ FPS。
- PIPNet 轻量版报告 CPU 35.7 FPS、GPU 200 FPS。
- 3DDFA-V2 报告单 CPU core 超过 50 FPS，并提升视频稳定性。
- MediaPipe Face Mesh 报告 468 顶点 3D face mesh 在移动 GPU 上可达 100-1000+ FPS（取决于设备和模型变体），为 AR 应用提供了非常强的工程基线。

### 阶段 D：关系建模、Transformer、密集 3D 与跨域泛化（约 2021-2026）

代表方法：

- Lite-HRNet（2021，主要为轻量高分辨率表示，可迁移到 landmark）
- SynergyNet（2021）
- SLPT（Sparse Local Patch Transformer，2022）
- Dense landmarks for 3D reconstruction（2022）
- RHT（Reference Heatmap Transformer，2023）
- KeyPosS（2023）
- Generalizable Face Landmarking with Conditional Face Warping（2024）
- Knowledge Distillation for embedded facial landmark detection（2024）

主要问题：

- 经典 benchmark 饱和后，模型在真实分布迁移、风格化脸、端侧部署和低分辨率上仍不稳。
- 热图法存在量化误差和计算代价；坐标回归又容易缺少形状先验。
- 单帧 2D 点不足以支持高质量 AR、虚拟形象和 3D 表情驱动。

技术变化：

- Transformer/attention 用于显式建模 landmark 之间的关系，而不是只依赖卷积的局部感受野。
- 局部 patch + 全局关系成为折中：减少全图 heatmap 计算，又保留点间结构。
- 3DMM 参数、landmark、mesh、blendshape 逐渐融合，目标从“点准”转向“可驱动、可渲染、可稳定跟踪”。
- 知识蒸馏、自训练、伪标签、跨域适配用于提升端侧小模型和非真人/风格化脸上的泛化。

成果：

- SLPT 在 WFLW、300W、COFW 上达到 SOTA 级别，同时强调更低计算复杂度。
- RHT 通过参考 heatmap 和 transformer 模块增强困难场景下的形状约束。
- KeyPosS 用类似 GPS multilateration 的距离定位思想，尝试绕开传统 heatmap/坐标回归的部分缺陷，并强调低分辨率场景。
- 2024 年风格化脸 landmark 工作说明新的难点已经转向 domain generalization，而不只是自然人脸照片。

## 6. 架构演进脉络

### 5.1 传统/浅层：显式形状模型

典型结构：AAM、ASM、CLM、SDM、级联回归。

优点：

- 可解释，参数少。
- 在受控环境和小姿态下有效。

缺点：

- 依赖初始化和手工特征。
- 对大姿态、遮挡、光照、表情泛化差。

### 5.2 CNN 直接回归：端到端但结构弱

典型结构：输入 face crop，CNN backbone，FC 或 global pooling 输出 2K 个坐标。

优点：

- 简单、快、易部署。
- 适合移动端小点数任务。

缺点：

- 空间精度受网络下采样影响。
- 形状约束弱，困难样本容易出现局部点漂移。

代表：PFLD、部分轻量模型。

### 5.3 Heatmap regression：精度主力

典型结构：Hourglass、U-Net、HRNet，输出 K 张 heatmap。

优点：

- 保留空间概率分布，定位精度高。
- 可以处理多峰和不确定区域。

缺点：

- 输出分辨率和 K 个点数直接影响算力/显存。
- argmax/soft-argmax 会引入量化或解码问题。
- 移动端需要剪枝、蒸馏或低分辨率解码。

代表：FAN、AWing、HRNet。

### 5.4 结构/边界/图约束：修正语义歧义

典型结构：预测边界线、局部轮廓、图关系或邻接点，再导出 landmarks。

解决问题：

- 脸轮廓点语义不稳定。
- 局部遮挡和大姿态下单点预测缺少上下文。

代表：LAB、PIPNet neighbor regression、SCC、MMDN。

### 5.5 3DMM/mesh/blendshape：从点定位到脸部几何

典型结构：

- 2D CNN 回归 3DMM 参数。
- CNN/GCN 直接预测 mesh 顶点或 dense landmark。
- 检测器 + mesh 模型 + blendshape 模型 pipeline。

解决问题：

- 大姿态下不可见点语义一致性。
- AR 渲染需要 3D surface、姿态矩阵和表情系数。
- 视频应用需要几何稳定性。

代表：3DDFA、3DDFA-V2、MediaPipe Face Mesh/Face Landmarker、SynergyNet、dense landmarks。

### 5.6 Transformer/attention：显式点间关系

典型结构：

- 从每个 landmark 周围 patch 提特征。
- 用 attention 学习 landmark 间关系。
- 引入参考 heatmap 或先验形状作为查询/条件。

解决问题：

- 卷积局部性限制。
- 难样本中需要利用脸部整体结构修正局部错误。

代表：SLPT、RHT。

## 7. 主要问题与对应解决方案

| 问题 | 表现 | 代表解决思路 |
| --- | --- | --- |
| 大姿态 | 侧脸不可见点、轮廓语义变化 | 3DMM、3D landmark、pose-aware training、anchor/template |
| 遮挡 | 眼镜、手、头发、口罩导致局部点漂移 | heatmap 概率、不确定性/可见性预测、边界/结构约束、遮挡数据集 |
| 低分辨率/模糊 | 热图峰不清晰，argmax 不稳定 | Super-FAN、低分辨率鲁棒解码、KeyPosS、视频融合 |
| 标注歧义 | 轮廓点不是固定解剖点 | LAB 边界线、3D 语义点、dense mesh |
| 视频抖动 | 单帧准但连续帧跳动 | tracking、temporal smoothing、stabilization loss、3DDFA-V2 虚拟视频训练 |
| 移动端算力 | heatmap/HRNet 太重 | PFLD、PIPNet、MediaPipe、蒸馏、量化、NPU/GPU delegate |
| 跨域泛化 | 卡通脸、游戏头像、特殊人群/妆容 | 自训练、domain adaptation、conditional warping、合成数据 |
| 失败可感知 | 输出错误但无置信度 | LUVLi 不确定性/可见性、阈值和失败检测 |

## 8. 是否已经达到天花板

需要分层判断。

### 8.1 经典 2D 静态 benchmark：接近天花板

300W 这类数据集上的主指标已经很难体现真实差距。原因：

- 数据量有限，标注风格固定，模型已充分拟合。
- 指标对微小误差敏感，但对视频稳定、失败可解释、跨域泛化不敏感。
- 强模型之间 NME 差距很小，很多改进来自训练细节、数据增强和 benchmark 适配。

FAN 论文在 2017 年就提出强模型在当时 2D/3D 数据集上可能接近饱和。后续 LAB、AWing、HRNet 等仍有提升，但提升主要集中在困难子集、失败率和训练/解码优化。

### 8.2 真实产品：没有到天花板

真实应用的目标函数更复杂：

- 自拍美颜/相机：需要低延迟、稳定、低功耗。
- AR 滤镜：需要 3D mesh、姿态矩阵、遮挡处理、眼睛/嘴唇细节。
- 直播/短视频：需要视频一致性和极低 jitter。
- 车载/疲劳检测：需要暗光、眼镜、偏头、低帧率鲁棒性。
- 医疗/正畸/面部分析：需要更严格的可解释点定义和跨设备一致性。
- 虚拟人：需要 blendshape、mesh 和表情语义，而不仅是 68 点。

所以结论是：学术 2D 点位 benchmark 接近饱和，端侧 3D 感知、跨域、视频稳定和应用语义远未饱和。

## 9. 移动端当前最佳方案

### 9.1 首选：MediaPipe Face Landmarker / Face Mesh

适用场景：

- AR 滤镜、贴纸、虚拟头像、表情驱动。
- 需要 3D 点、face transform matrix、blendshape。
- 需要 Android/iOS/Web/Python 多平台快速落地。
- 需要较可靠的官方维护和成熟工程 pipeline。

官方 Face Landmarker 当前模型 bundle 包括：

- Face detection model：BlazeFace short-range，移动 GPU 优化。
- Face mesh model：FaceMesh-V2，输出 478 个 3D landmarks。
- Blendshape prediction model：输出 52 个表情 blendshape scores。
- 支持 IMAGE、VIDEO、LIVE_STREAM 模式；单脸时可做 smoothing。

优点：

- 端到端 pipeline 成熟，少踩坑。
- 点数密集，适合 AR/渲染/表情。
- 官方跨平台支持完善。
- 输出不只是点，还有表情系数和变换矩阵。

限制：

- 自定义训练和商业级可控性不如自研模型。
- 点定义和模型细节受官方 bundle 约束。
- 多脸、低端机、长时间运行时仍需做帧率/功耗策略。

推荐工程策略：

- 默认用 MediaPipe Face Landmarker 做基线。
- 针对 Android 使用 GPU/NPU delegate 或 LiteRT/NNAPI 能力；针对 iOS 使用 Core ML/Metal 路径评估。
- 摄像头流中不要每帧全量检测：检测 + 跟踪 + 置信度重检。
- 对输出做轻量时序滤波，但避免过度平滑导致表情延迟。
- 对低端机提供降级：降低输入分辨率、限制 `num_faces=1`、关闭 blendshape 或降低推理频率。

### 9.2 只需要稀疏 2D 点：PFLD / PIPNet / 轻量 HRNet

适用场景：

- 只需要 68/98/106 点。
- 应用是人脸裁剪、粗对齐、美颜基础点位、识别前预对齐。
- 需要模型完全可训练、可量化、可嵌入自有 SDK。

建议：

- 极致轻量：PFLD 或 MobileNet/ShuffleNet backbone 坐标回归。
- 精度/速度折中：PIPNet，低分辨率特征图 + offset 解码。
- 更高精度：Lite-HRNet/小 HRNet + heatmap，但移动端要评估功耗。

### 9.3 需要 3D 姿态/重建但不想用 MediaPipe：3DDFA-V2 / SynergyNet

适用场景：

- 需要 3DMM 参数、头部姿态、稠密对齐。
- 需要较强自研可控性。
- 可接受比纯 2D 点模型更复杂的 pipeline。

建议：

- 从 3DDFA-V2 开始评估：论文报告单 CPU core 50+ FPS，并关注视频稳定。
- 若需要更完整几何，可评估 SynergyNet、dense landmark/3D reconstruction 路线。

## 10. 选型建议

### 10.1 产品快速落地

选 MediaPipe Face Landmarker。

理由：

- 2026 年官方文档仍在维护，最近页面更新时间为 2026-05-28。
- 输出 478 个 3D 点、52 个 blendshape 和变换矩阵，覆盖大多数移动 AR/表情需求。
- 平台适配成本低。

### 10.2 自研 SDK 或模型私有化

首选 PFLD/PIPNet 作为稀疏 2D 基线，再按需求升级：

- 精度不够：加入 Wing/AWing/Soft Wing loss、难例采样、WFLW/COFW 风格数据增强。
- 遮挡不稳：增加可见性/不确定性分支，参考 LUVLi。
- 视频抖动：加入 temporal smoothing、stabilization loss 或 tracking-by-detection。
- 大姿态不稳：增加 3DMM/pose 分支或 3DDFA-V2。

### 10.3 高精度研究或离线处理

选择 HRNet/heatmap/Transformer 类方法：

- HRNet/HRNetV2：强高分辨率表示。
- AWing：成熟 heatmap loss。
- SLPT/RHT：对点间关系、困难场景和结构约束更友好。

代价是移动端部署复杂度和算力需求更高。

## 11. 未来趋势

- 从 2D landmark 转向统一 facial geometry：2D 点、3D 点、mesh、blendshape、pose、expression 一体化。
- 从静态图像指标转向视频和交互指标：jitter、latency、tracking lost、功耗、热降频。
- 更强跨域泛化：真人、卡通、AI 生成脸、游戏头像、医疗图像和不同族群/妆容。
- 自监督/弱监督/合成数据继续重要：高质量密集 3D 标注昂贵，合成数据和伪标签会长期存在。
- 端侧模型会继续向 pipeline 化发展：检测、跟踪、landmark、mesh、表情、质量评估打包，而不是单模型刷榜。
- 隐私和本地推理会变成默认要求：face landmark 虽不是身份识别本身，但仍属于敏感人脸数据处理链条的一部分。

## 12. 代表论文速查表

| 年份 | 论文/系统 | 技术贡献 | 解决的问题 |
| --- | --- | --- | --- |
| 2015/2016 | 3DDFA | CNN + 3DMM 大姿态对齐 | profile face、不可见点、2D 语义不一致 |
| 2016 | MTCNN | 级联多任务检测 + landmark | 检测与粗对齐实时联合 |
| 2017 | DAN | 多阶段全脸 refinement | 初始化差、大姿态 |
| 2017 | FAN / 2D&3D Face Alignment | strong hourglass baseline + LS3D-W | 2D/3D benchmark 饱和判断 |
| 2017/2018 | Wing Loss | 定位任务专用 loss | 中小误差学习不足、pose imbalance |
| 2018 | LAB | 边界感知 landmark | 轮廓点定义歧义、遮挡/困难样本 |
| 2019 | AWing | heatmap 自适应损失 + weighted map | 前景/背景不平衡、heatmap 训练 |
| 2019 | HRNet | 全流程高分辨率表示 | 精确定位、多尺度融合 |
| 2019 | PFLD | 轻量单阶段端侧模型 | 小模型、高 FPS、移动端实时 |
| 2019 | MediaPipe Face Mesh | 468 点 3D mesh 端侧实时 | AR face geometry |
| 2020 | PIPNet | 低分辨率 score+offset + 邻域约束 | heatmap 计算重、跨域泛化 |
| 2020 | LUVLi | 位置 + 不确定性 + 可见性 | 失败检测、遮挡、自遮挡 |
| 2020 | 3DDFA-V2 | 轻量 3DMM 参数回归 + 视频稳定 | 3D dense alignment 工程落地 |
| 2022 | SLPT | sparse local patch transformer | landmark 关系建模、低计算复杂度 |
| 2022 | Dense Landmarks | 大量 dense landmarks + 3D fitting | 3D 重建/表情捕捉 |
| 2023 | RHT | reference heatmap transformer | 困难场景形状约束和 heatmap 精细化 |
| 2023 | KeyPosS | multilateration 解码 | 低分辨率、量化误差 |
| 2024 | Conditional Face Warping | 风格化脸泛化 | stylized/domain generalization |
| 2024 | KD for embedded FLD | 知识蒸馏端侧模型 | 嵌入式精度/效率平衡 |

## 13. 参考文献与资料

1. Wu, Y., Ji, Q. “Facial Landmark Detection: a Literature Survey.” arXiv, 2018. https://arxiv.org/abs/1805.05563
2. Khabarlak, K., Koriashkina, L. “Fast Facial Landmark Detection and Applications: A Survey.” arXiv, 2021. https://arxiv.org/abs/2101.10808
3. Zhu, X. et al. “Face Alignment Across Large Poses: A 3D Solution.” arXiv, 2015. https://arxiv.org/abs/1511.07212
4. Zhang, K. et al. “Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.” arXiv, 2016. https://arxiv.org/abs/1604.02878
5. Kowalski, M. et al. “Deep Alignment Network: A convolutional neural network for robust face alignment.” arXiv, 2017. https://arxiv.org/abs/1706.01789
6. Bulat, A., Tzimiropoulos, G. “How far are we from solving the 2D & 3D Face Alignment problem?” arXiv, 2017. https://arxiv.org/abs/1703.07332
7. Feng, Z.-H. et al. “Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks.” arXiv, 2017. https://arxiv.org/abs/1711.06753
8. Wu, W. et al. “Look at Boundary: A Boundary-Aware Face Alignment Algorithm.” arXiv, 2018. https://arxiv.org/abs/1805.10483
9. Wang, X. et al. “Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression.” arXiv, 2019. https://arxiv.org/abs/1904.07399
10. Sun, K. et al. “High-Resolution Representations for Labeling Pixels and Regions.” arXiv, 2019. https://arxiv.org/abs/1904.04514
11. Guo, X. et al. “PFLD: A Practical Facial Landmark Detector.” arXiv, 2019. https://arxiv.org/abs/1902.10859
12. Kartynnik, Y. et al. “Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.” arXiv, 2019. https://arxiv.org/abs/1907.06724
13. Jin, H. et al. “Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild.” arXiv, 2020. https://arxiv.org/abs/2003.03771
14. Kumar, A. et al. “LUVLi Face Alignment: Estimating Landmarks' Location, Uncertainty, and Visibility Likelihood.” arXiv, 2020. https://arxiv.org/abs/2004.02980
15. Guo, J. et al. “Towards Fast, Accurate and Stable 3D Dense Face Alignment.” arXiv, 2020. https://arxiv.org/abs/2009.09960
16. Xia, J. et al. “Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning.” arXiv, 2022. https://arxiv.org/abs/2203.06541
17. Wood, E. et al. “3D face reconstruction with dense landmarks.” arXiv, 2022. https://arxiv.org/abs/2204.02776
18. Wan, J. et al. “Precise Facial Landmark Detection by Reference Heatmap Transformer.” arXiv, 2023. https://arxiv.org/abs/2303.07840
19. Bao, X. et al. “KeyPosS: Plug-and-Play Facial Landmark Detection through GPS-Inspired True-Range Multilateration.” arXiv, 2023. https://arxiv.org/abs/2305.16437
20. Liang, J. et al. “Generalizable Face Landmarking Guided by Conditional Face Warping.” arXiv, 2024. https://arxiv.org/abs/2404.12322
21. Hong, Z.-W., Lin, Y.-C. “Improving Facial Landmark Detection Accuracy and Efficiency with Knowledge Distillation.” arXiv, 2024. https://arxiv.org/abs/2404.06029
22. Google AI Edge. “Face landmark detection guide.” Last updated 2026-05-28. https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
23. WFLW official project page. “Wider Facial Landmarks in-the-wild.” https://wywu.github.io/projects/LAB/WFLW.html
24. Yang, H. et al. “An Empirical Study of Recent Face Alignment Methods.” arXiv, 2015. https://arxiv.org/abs/1511.05049
25. Zhu, B. et al. “Fast and Accurate: Structure Coherence Component for Face Alignment.” arXiv, 2020. https://arxiv.org/abs/2006.11697
26. Micaelli, P. et al. “Recurrence without Recurrence: Stable Video Landmark Detection with Deep Equilibrium Models.” arXiv, 2023. https://arxiv.org/abs/2304.00600
