# B03 · photo_e6160ef7ce3f2f34

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000146`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.634873 |
| 错误参考 MaxSim | +0.637350 |
| 正确减错误的分差 | -0.002477 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 72 个。正向贡献 +0.021118，负向贡献 -0.023595，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.003791 | -0.004277 |
| nonvisual | 11 | -0.000246 | -0.016336 |
| subject_nontext | 30 | +0.000007 | +0.000160 |
| subject_text | 42 | +0.001553 | +0.027029 |

最主要负差额来自 `background`，贡献 -0.003791。只保留主体 query token 后，本对分差为 +0.015833。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 490 | 56.2% | True | +6.340516 | 69.6% |
| 错误 | 65 | 29.3% | True | +6.982824 | 83.3% |

主体中 42 个 token 与 OCR 行相交，33 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 443 | background | -0.493309 |  |  |  |
| 634 | background | -0.384399 |  | 75 |  |
| 36 | background | -0.326927 |  |  |  |
| 131 | background | -0.296052 |  |  |  |
| 466 | background | -0.208639 |  |  |  |
| 506 | background | -0.206956 |  |  |  |
| 576 | background | -0.193522 |  |  |  |
| 462 | background | -0.181722 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.369432 | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acd and | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acid and |  |
| 183 | background | +0.314770 |  | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acid and |  |
| 108 | background | +0.285644 |  |  | Inject subcutaneously as a single dose in the neck at a dosage |
| 477 | subject_text | +0.275341 | free base in a 50% propylene glycol vehicle, monothioglycerol |  | added to adjust pH. |
| 492 | subject_nontext | +0.245491 |  |  |  |
| 474 | subject_text | +0.231462 | sodium hydroxide added to adjust pH. | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acid and | citric acid (4.8 mg/mL) with hydrochloric acid and sodium hydroxide |
| 481 | background | +0.220458 |  |  |  |
| 231 | background | +0.195871 |  |  | Inject subcutaneously as a single dose in the neck at a dosage |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| of 2.5 mg/kg (1 mL/22 Ib) Body Weight (BW). Do not inject | 5 | +0.000182 |
| more than 4 mL per injection site. | 3 | +0.000187 |
| Table 1. Increxxa 25 Swine Dosing Guide (25 mg/mL) | 5 | +0.000182 |
| of 2.5 mg/kg (1 mL/22 Ib) body weight (BW). Do not inject | 5 | -0.000103 |
| more than 11.5 mL per injection site. | 5 | +0.000108 |
| Table 2. Increxxa 25 Calf Dosing Guide (25 mg/mL) | 5 | +0.000154 |
| Each mL of Increoxa 25 contains 25 mg of tulathromycin as the | 9 | +0.001476 |
| free base in a 50% propylene glycol vehicle, monothioglycerol | 6 | +0.001302 |
| (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acd and | 5 | +0.000926 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002477 | +0.015164 |
| 只保留主体分差 | +0.015833 | +0.034226 |
| 主体 token 数 | 72 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
