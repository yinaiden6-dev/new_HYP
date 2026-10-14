# B03 · photo_1a3a7ee95f631a62

按当前评测真值分析，未新增身份人工复核。

包装 `package_029`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001023`，**纯图像实际 Top-1 错误参考** `reference_002460`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 38 |
| 正确参考 MaxSim | +0.584524 |
| 错误参考 MaxSim | +0.601480 |
| 正确减错误的分差 | -0.016956 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 50 个。正向贡献 +0.022093，负向贡献 -0.039049，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 670 | -0.018748 | -0.020455 |
| nonvisual | 11 | -0.000206 | -0.013659 |
| subject_nontext | 12 | +0.001549 | +0.094336 |
| subject_text | 38 | +0.000449 | +0.008638 |

最主要负差额来自 `background`，贡献 -0.018748。只保留主体 query token 后，本对分差为 +0.029205。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 595 | 49.3% | True | +8.847845 | 80.0% |
| 错误 | 6 | 13.7% | False | 不可评估 | 不可评估 |

主体中 38 个 token 与 OCR 行相交，28 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 121 | background | -0.464178 |  |  |  |
| 241 | background | -0.425424 |  | Alka- |  |
| 305 | background | -0.405731 |  | Alka- |  |
| 421 | background | -0.354738 |  |  | Allergies |
| 196 | background | -0.350095 |  | Alka- | Allergies |
| 439 | background | -0.290988 |  |  | Allergies |
| 502 | background | -0.259174 |  |  |  |
| 213 | background | -0.249171 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 513 | subject_text | +0.453379 | Bayer |  |  |
| 114 | background | +0.398649 |  |  |  |
| 395 | subject_nontext | +0.374032 |  |  |  |
| 255 | background | +0.349807 |  |  |  |
| 511 | subject_nontext | +0.318743 |  |  | Indoor & Outdoor |
| 175 | background | +0.309425 |  |  | Temporarily relieves these symptoms due to hay fever or other upper |
| 75 | background | +0.288567 |  | Cold |  |
| 411 | subject_text | +0.271552 | liver damage may occur il you take |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminaphen 250 mg | 3 | -0.000223 |
| Chlorphontranine maleato 2 mg | 2 | -0.000177 |
| Dextromethorphan hydrobramide 10 mg... | 6 | -0.000426 |
| Phenylephrine hydrochloride 5 mg.... | 3 | +0.000041 |
| more than 4,000 mg of acetaminophen in 24 hours | 4 | +0.000103 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_bc13fbd48065accd` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.016956 | +0.219972 |
| 只保留主体分差 | +0.029205 | +0.312894 |
| 主体 token 数 | 50 | 45 |

[成功对照 token 表](../photo_bc13fbd48065accd/tokens.csv) · [对照摘要](../photo_bc13fbd48065accd/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
