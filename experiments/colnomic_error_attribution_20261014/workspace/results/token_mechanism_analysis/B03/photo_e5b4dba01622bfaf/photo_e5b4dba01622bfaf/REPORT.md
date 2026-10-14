# B03 · photo_e5b4dba01622bfaf

按当前评测真值分析，未新增身份人工复核。

包装 `package_044`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001133`，**纯图像实际 Top-1 错误参考** `reference_003667`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.720253 |
| 错误参考 MaxSim | +0.728140 |
| 正确减错误的分差 | -0.007887 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 242 个。正向贡献 +0.029438，负向贡献 -0.037325，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 478 | -0.006708 | -0.010258 |
| nonvisual | 11 | +0.000127 | +0.008438 |
| subject_nontext | 34 | +0.000017 | +0.000374 |
| subject_text | 208 | -0.001323 | -0.004651 |

最主要负差额来自 `background`，贡献 -0.006708。只保留主体 query token 后，本对分差为 -0.003945。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 786 | 75.8% | True | +4.318378 | 54.4% |
| 错误 | 401 | 71.8% | True | +5.139451 | 58.9% |

主体中 208 个 token 与 OCR 行相交，171 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 78 | background | -0.469263 |  |  |  |
| 165 | background | -0.350244 |  |  | Warnings |
| 50 | background | -0.317948 |  |  |  |
| 29 | background | -0.277730 |  | Acetaminophen 160 mg per 5 mL |  |
| 717 | background | -0.258815 |  |  |  |
| 135 | subject_text | -0.250345 | relieves minor aches and pains due to: | relieves minor aches and pains due to: | reduces fever |
| 218 | subject_text | -0.227659 | Symptoms may include:skin reddening blisters rash |  | If a skin reaction occurs, stop use and seek medical help right away. |
| 345 | background | -0.226989 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 220 | subject_nontext | +0.404092 |  | relieves minor aches and pains due to: | more than 5 doses in 24 hours, which is the maximum daily amount |
| 556 | background | +0.354468 |  | syringe tip into bottle opening. |  |
| 273 | subject_text | +0.295647 | drug contains acetaminophen, ask a doctor or pharmacist. | if your child has ever had an allergic reaction to this product or |  |
| 398 | subject_text | +0.291951 | Overdose warning: In case of overdose, get medical help or contact | Overdose warning: In case of overdose, get medical help or contact | For product questions or concerns, contact us |
| 311 | subject_text | +0.285695 | Ask a doctor or pharmacist before use if your child is taking | the blood thinning drug warfarin | When using this product do not exceed recommended dose |
| 411 | subject_text | +0.260721 | a Poison Control Center right away (1-800-222-1222). Quick medical |  | a Poison Control Center right away (1-800-222-1222). Quick medical |
| 257 | subject_text | +0.260272 | Do not use with any other drug containing acetaminophen | Do not use with any other drug containing acetaminophen | if your child has ever had an allergic reaction to this product or |
| 94 | subject_text | +0.258408 | Active ingredient (in each 5 mL) | Active ingredient (in each 5 mL) | 160mg PER 5mL |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Active ingredient (in each 5 mL) | 13 | +0.000092 |
| Acetaminophen 160 mg. | 4 | +0.000139 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_3cfb7f3d355174a8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.007887 | +0.040263 |
| 只保留主体分差 | -0.003945 | +0.062648 |
| 主体 token 数 | 242 | 276 |

[成功对照 token 表](../photo_3cfb7f3d355174a8/tokens.csv) · [对照摘要](../photo_3cfb7f3d355174a8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
