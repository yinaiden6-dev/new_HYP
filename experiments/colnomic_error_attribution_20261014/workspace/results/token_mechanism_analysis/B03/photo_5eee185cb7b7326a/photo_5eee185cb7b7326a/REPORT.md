# B03 · photo_5eee185cb7b7326a

按当前评测真值分析，未新增身份人工复核。

包装 `package_099`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000797`，**纯图像实际 Top-1 错误参考** `reference_000799`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.568922 |
| 错误参考 MaxSim | +0.582037 |
| 正确减错误的分差 | -0.013115 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 238 个。正向贡献 +0.022083，负向贡献 -0.035198，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 530 | -0.008885 | -0.013060 |
| nonvisual | 11 | -0.000166 | -0.011746 |
| subject_nontext | 126 | -0.004797 | -0.029656 |
| subject_text | 112 | +0.000733 | +0.005098 |

最主要负差额来自 `background`，贡献 -0.008885。只保留主体 query token 后，本对分差为 -0.013301。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 14 | 23.1% | True | +10.060679 | 71.7% |
| 错误 | 155 | 37.2% | True | +6.749125 | 66.5% |

主体中 112 个 token 与 OCR 行相交，59 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 132 | background | -0.452748 |  | Kids Ear |  |
| 488 | background | -0.341023 | S |  | Alivio para |
| 520 | background | -0.328825 | S | Itching | EYEDROPS |
| 519 | background | -0.299213 | S | Itching | EYEDROPS |
| 164 | background | -0.280517 |  | bottle is missing or broken. |  |
| 135 | background | -0.257383 |  | Kids Ear Relief ear drops help | EYEDROPS |
| 112 | background | -0.249904 |  |  |  |
| 582 | background | -0.241235 |  |  | kids |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 65 | background | +0.400418 | Multi-Care | MULTISYMPTOM | Follow us for special offers |
| 33 | background | +0.382835 | Multi-Care | MULTISYMPTOM | • Enrojecimiento |
| 22 | background | +0.352319 |  |  | unpainted/uncoated area |
| 34 | background | +0.294586 | Multi-Care | MULTISYMPTOM | Similasan |
| 369 | subject_text | +0.281058 | Made in Switzerland |  | Corporstion, IBighlands Ranch, CO 80129 |
| 476 | background | +0.279159 | Comfortis | and comfortable condition. | Feel Good |
| 508 | background | +0.267831 | Comfortis | and comfortable condition. | Feel Good |
| 540 | background | +0.261642 | Comfortis | and comfortable condition. | Feel Good |

## 同包装成功对照

`photo_8a094cfd1e9fa61e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013115 | +0.059681 |
| 只保留主体分差 | -0.013301 | +0.098013 |
| 主体 token 数 | 238 | 260 |

[成功对照 token 表](../photo_8a094cfd1e9fa61e/tokens.csv) · [对照摘要](../photo_8a094cfd1e9fa61e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
