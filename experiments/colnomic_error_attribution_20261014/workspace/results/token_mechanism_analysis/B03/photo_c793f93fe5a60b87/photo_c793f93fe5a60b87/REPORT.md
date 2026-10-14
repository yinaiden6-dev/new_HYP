# B03 · photo_c793f93fe5a60b87

按当前评测真值分析，未新增身份人工复核。

包装 `package_049`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001182`，**纯图像实际 Top-1 错误参考** `reference_003249`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 112 |
| 正确参考 MaxSim | +0.600665 |
| 错误参考 MaxSim | +0.637532 |
| 正确减错误的分差 | -0.036867 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 72 个。正向贡献 +0.014822，负向贡献 -0.051689，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.031471 | -0.035503 |
| nonvisual | 11 | -0.000132 | -0.008784 |
| subject_nontext | 14 | -0.000361 | -0.018848 |
| subject_text | 58 | -0.004902 | -0.061784 |

最主要负差额来自 `background`，贡献 -0.031471。只保留主体 query token 后，本对分差为 -0.053436。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 162 | 53.5% | True | +6.599674 | 66.2% |
| 错误 | 3 | 0.0% | False | 不可评估 | 不可评估 |

主体中 58 个 token 与 OCR 行相交，48 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 382 | background | -0.521158 |  | skin reactlons. Symptoms may Include: |  |
| 615 | background | -0.471741 |  | skin reactlons. Symptoms may Include: | If a skin reaction occurs, stop use and seek medical |
| 187 | background | -0.462132 |  |  |  |
| 73 | background | -0.437412 |  | damage. In case of overdose, get medical help or |  |
| 575 | background | -0.385994 |  | DISTRIBUTED BY: fred's, Inc. |  |
| 350 | subject_text | -0.379767 | Uses | Uses | Uses |
| 116 | background | -0.367164 |  |  | If a skin reaction occurs, stop use and seek medical |
| 453 | subject_text | -0.350301 | acetaminophen, ask a doctor or pharmacist. | If you are not sure whether a drug contains |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 440 | background | +0.176017 |  | Do not usewith any other drug containing |  |
| 29 | background | +0.172361 |  |  | Purpose |
| 472 | subject_text | +0.170749 | you have ever had an allergic reaction to this | if you have ever had an allergic reaction to this |  |
| 240 | background | +0.157953 |  |  | Purpose |
| 78 | background | +0.152438 |  |  |  |
| 141 | background | +0.151204 |  |  | more than 4,000 mg of acetaminophen in 24 hours |
| 672 | background | +0.145289 |  | if you have ever had an allergic reaction to this | if you are allergic to acetaminophen or any of the |
| 401 | background | +0.137742 |  | do not drive a motor vehicle or operate machinery |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen, USP 500 mg...Paln rellever | 11 | -0.000801 |
| Diphenhydramine HCl 25 mg....ighttime sleep-ald | 7 | -0.000931 |
| more than 4,000 mg of acetaminophen In 24 | 6 | -0.000902 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_375b67daeeeeff5f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.036867 | +0.060983 |
| 只保留主体分差 | -0.053436 | +0.081565 |
| 主体 token 数 | 72 | 66 |

[成功对照 token 表](../photo_375b67daeeeeff5f/tokens.csv) · [对照摘要](../photo_375b67daeeeeff5f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
