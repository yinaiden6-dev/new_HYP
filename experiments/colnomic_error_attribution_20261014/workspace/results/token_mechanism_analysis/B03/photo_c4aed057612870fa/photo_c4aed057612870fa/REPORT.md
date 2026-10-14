# B03 · photo_c4aed057612870fa

按当前评测真值分析，未新增身份人工复核。

包装 `package_005`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000841`，**纯图像实际 Top-1 错误参考** `reference_003006`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 222 |
| 正确参考 MaxSim | +0.580152 |
| 错误参考 MaxSim | +0.610087 |
| 正确减错误的分差 | -0.029935 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 238 个。正向贡献 +0.025411，负向贡献 -0.055346，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 482 | -0.022191 | -0.033655 |
| nonvisual | 11 | -0.000208 | -0.013796 |
| subject_nontext | 68 | -0.001840 | -0.019781 |
| subject_text | 170 | -0.005696 | -0.024492 |

最主要负差额来自 `background`，贡献 -0.022191。只保留主体 query token 后，本对分差为 -0.023146。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1405 | 69.0% | True | +8.399939 | 74.8% |
| 错误 | 17 | 12.1% | False | 不可评估 | 不可评估 |

主体中 170 个 token 与 OCR 行相交，114 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 501 | background | -0.490311 |  | te mporarily relieves nasal congestion due to the common co ld, | allergic reaction to any other pain reliever/fever reducer |
| 600 | background | -0.444171 |  |  | temporarily relieves these symptoms associated with the common cold or flu: |
| 33 | background | -0.440242 |  | in 24 ho urs |  |
| 27 | background | -0.431297 |  | in 24 ho urs |  |
| 485 | background | -0.410957 |  | pseudoephedrine hydrochloride | allergic reaction to any other pain reliever/fever reducer |
| 400 | background | -0.386558 |  | Sudafed® Congestion | temporarily relieves these symptoms associated with the common cold or flu: |
| 507 | subject_text | -0.385398 | Drug Facts | Drug Facts | Drug Facts |
| 142 | background | -0.380632 |  | under 6years of age |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 240 | background | +0.360238 |  |  | children under 12 years of age: do not use because |
| 428 | subject_text | +0.358814 | Active ingredient (in each tablet) | Active ingredient (in each tablet) |  |
| 140 | background | +0.317118 |  | GOOD | Directionsdo not take more than directedadults and children |
| 448 | subject_text | +0.312785 | Active ingredient (in each tablet) | Active ingredient (in each tablet) |  |
| 241 | background | +0.290300 |  |  | children under 12 years of age: do not use because |
| 120 | background | +0.273750 |  |  | children under 12 years of age: do not use because |
| 237 | background | +0.262342 |  | NEIGHBOR |  |
| 283 | background | +0.247637 |  |  | SUDAFED |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| take 2 tablets every 4 to 6 hours | 4 | +0.000547 |
| do not take more than 8tablets | 7 | +0.000407 |
| do not take more than 4 tablets | 5 | +0.000247 |
| each tablet contains: calcium 20 mg | 7 | -0.000293 |
| Pseudoephedrine HCI 30 mg | 5 | +0.000190 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_fa7137de993a1c70` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.029935 | +0.162037 |
| 只保留主体分差 | -0.023146 | +0.210651 |
| 主体 token 数 | 238 | 238 |

[成功对照 token 表](../photo_fa7137de993a1c70/tokens.csv) · [对照摘要](../photo_fa7137de993a1c70/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
