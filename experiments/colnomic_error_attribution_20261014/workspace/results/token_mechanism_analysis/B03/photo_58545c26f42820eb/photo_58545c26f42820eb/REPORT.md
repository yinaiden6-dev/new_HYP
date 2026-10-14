# B03 · photo_58545c26f42820eb

按当前评测真值分析，未新增身份人工复核。

包装 `package_064`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001321`，**纯图像实际 Top-1 错误参考** `reference_001320`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.648129 |
| 错误参考 MaxSim | +0.652794 |
| 正确减错误的分差 | -0.004665 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 156 个。正向贡献 +0.022314，负向贡献 -0.026979，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 564 | -0.001799 | -0.002332 |
| nonvisual | 11 | +0.000119 | +0.007923 |
| subject_nontext | 47 | +0.001649 | +0.025652 |
| subject_text | 109 | -0.004635 | -0.031083 |

最主要负差额来自 `subject_text`，贡献 -0.004635。只保留主体 query token 后，本对分差为 -0.013990。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 101 | 8.2% | False | 不可评估 | 不可评估 |
| 错误 | 72 | 6.8% | False | 不可评估 | 不可评估 |

主体中 109 个 token 与 OCR 行相交，91 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 75 | background | -0.412693 |  |  |  |
| 455 | subject_text | -0.313539 | preep out of reach of children. In case of overdose, get |  | Allergy alert: Acetaminophen may cause severe |
| 317 | subject_text | -0.309064 | Sekzer | Slze | Seltzer |
| 318 | subject_text | -0.292918 | Sekzer | Slze | Seltzer |
| 414 | subject_text | -0.265955 | Ask a doctor or pharmacist before use if you are | • pain, cough, or nasal congestion gets worse or lasts |  |
| 258 | background | -0.254365 |  | Slze | Seltzer |
| 336 | subject_text | -0.249549 | Sekzer |  |  |
| 316 | subject_text | -0.248246 | Sekzer |  | Seltzer |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 467 | subject_nontext | +0.320209 |  | Seltzer |  |
| 156 | background | +0.278972 |  | Slze |  |
| 367 | subject_text | +0.266207 | Phenylephrine hydrochlonide 5 mg |  | Purposes |
| 274 | subject_nontext | +0.230481 |  | Seltzer |  |
| 30 | background | +0.223335 |  |  |  |
| 106 | background | +0.217715 |  |  | Seltzer |
| 175 | background | +0.197863 |  |  | Seltzer |
| 368 | subject_text | +0.193708 | Phenylephrine hydrochlonide 5 mg |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| menophenS25 mg | 2 | -0.000048 |
| hydrobromide 10 mg | 4 | +0.000230 |
| Guaitenesin 200 mg | 3 | +0.000597 |
| 10 mg | 1 | -0.000032 |
| Phenylephrine hydrochlonide 5 mg | 4 | +0.000624 |
| more than 4,000 mg of acetaminophen in 24 hours | 7 | -0.000012 |
| adults and children 12 years and over: take 2 capsules | 5 | -0.000392 |
| with water every 4 hours. Do not exceed 10 capsules | 5 | -0.000392 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_1d2280603a42552a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004665 | +0.067802 |
| 只保留主体分差 | -0.013990 | +0.088872 |
| 主体 token 数 | 156 | 165 |

[成功对照 token 表](../photo_1d2280603a42552a/tokens.csv) · [对照摘要](../photo_1d2280603a42552a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
