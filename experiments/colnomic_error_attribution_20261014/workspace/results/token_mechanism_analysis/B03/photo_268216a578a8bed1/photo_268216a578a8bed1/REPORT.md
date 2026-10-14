# B03 · photo_268216a578a8bed1

按当前评测真值分析，未新增身份人工复核。

包装 `package_064`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001321`，**纯图像实际 Top-1 错误参考** `reference_001338`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.688470 |
| 错误参考 MaxSim | +0.697936 |
| 正确减错误的分差 | -0.009466 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 180 个。正向贡献 +0.033307，负向贡献 -0.042773，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 540 | -0.010061 | -0.013620 |
| nonvisual | 11 | +0.000149 | +0.009894 |
| subject_nontext | 37 | +0.000098 | +0.001928 |
| subject_text | 143 | +0.000348 | +0.001779 |

最主要负差额来自 `background`，贡献 -0.010061。只保留主体 query token 后，本对分差为 +0.001810。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 852 | 69.0% | True | +9.395278 | 76.1% |
| 错误 | 405 | 49.6% | True | +8.226409 | 66.1% |

主体中 143 个 token 与 OCR 行相交，118 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 54 | background | -0.493161 |  |  |  |
| 154 | background | -0.480089 |  | recommended dosage |  |
| 363 | background | -0.476644 |  |  |  |
| 654 | background | -0.476180 |  |  |  |
| 463 | background | -0.474157 |  |  | 7537 |
| 234 | background | -0.472031 |  |  |  |
| 473 | subject_text | -0.443846 | If a skin or general allergic reaction occurs, stop use and | (Mon-Fri 9AM - 5PM EST) | get medical help or contact a Poison Control Center |
| 474 | subject_text | -0.362836 | medical help or contact a Poison Control Center right away. |  | get medical help or contact a Poison Control Center |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 435 | subject_text | +0.474851 | cough comes back or occurs with fever, rash or |  | Headache, Body Ache, |
| 553 | subject_text | +0.354539 | glycerin, lecithin, medium chain triglycerides, polyethylene |  |  |
| 436 | subject_text | +0.346682 | cough comes back or occurs with fever, rash or |  | pain, cough, or nasal congestion gets worse or lasts |
| 53 | background | +0.338810 |  | • helps loosen phlegm (mucus) and thin bronchial | persistent or chronic cough such as occurs with |
| 362 | background | +0.330020 |  | • helps loosen phlegm (mucus) and thin bronchial | persistent or chronic cough such as occurs with |
| 373 | subject_text | +0.320265 | bothersome mucus and make coughs more productive |  |  |
| 371 | subject_text | +0.316266 | bothersome mucus and make coughs more productive |  |  |
| 557 | subject_text | +0.313045 | Inactive ingredients FD&C red #40, gelatin, | glycol, potassium aluminum silicate, povidone, propylene | water, shellac, sodium hydroxide, sorbitol sorbitan |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen 325 mg. | 3 | -0.000346 |
| hydrobromide 10 mg. | 2 | -0.000232 |
| Guaifenesin 200 mg | 2 | -0.000232 |
| Phenylephrine hydrochloride 5 mg | 3 | -0.000064 |
| more than 4,000 mg of acetaminophen in 24 hours | 5 | +0.000313 |
| adults and children 12 years and over: take 2 capsules | 10 | -0.001121 |
| with water every 4 hours. Do not exceed 10 capsules | 6 | -0.000181 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_1d2280603a42552a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.009466 | +0.032292 |
| 只保留主体分差 | +0.001810 | +0.025890 |
| 主体 token 数 | 180 | 165 |

[成功对照 token 表](../photo_1d2280603a42552a/tokens.csv) · [对照摘要](../photo_1d2280603a42552a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
