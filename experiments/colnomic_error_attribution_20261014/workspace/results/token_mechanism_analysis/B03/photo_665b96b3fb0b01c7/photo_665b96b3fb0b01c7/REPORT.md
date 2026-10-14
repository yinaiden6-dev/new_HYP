# B03 · photo_665b96b3fb0b01c7

按当前评测真值分析，未新增身份人工复核。

包装 `package_085`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000229`，**纯图像实际 Top-1 错误参考** `reference_000233`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.586762 |
| 错误参考 MaxSim | +0.596690 |
| 正确减错误的分差 | -0.009928 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 216 个。正向贡献 +0.019883，负向贡献 -0.029811，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 552 | -0.008602 | -0.012139 |
| nonvisual | 11 | -0.000084 | -0.005915 |
| subject_nontext | 55 | -0.000265 | -0.003749 |
| subject_text | 161 | -0.000978 | -0.004734 |

最主要负差额来自 `background`，贡献 -0.008602。只保留主体 query token 后，本对分差为 -0.004483。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 156 | 47.4% | True | +8.294819 | 74.6% |
| 错误 | 16 | 24.5% | True | +8.728093 | 98.6% |

主体中 161 个 token 与 OCR 行相交，132 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 239 | subject_nontext | -0.413493 |  |  |  |
| 207 | subject_nontext | -0.408473 |  | drug to use by or on the order of a |  |
| 435 | subject_text | -0.393018 | The safe use of enrofloxacin has not |  | The safe use of enrofloxacin has not been |
| 550 | background | -0.356318 |  |  | technical assistance or to obtain a copy of the Safety |
| 461 | subject_text | -0.355435 | (IM) as a single injection at the rate of 1 | For the treatment of Susceptible |  |
| 230 | background | -0.329578 |  |  | technical assistance or to obtain a copy of the Safety |
| 621 | background | -0.328187 |  | DOSAGE: Administer intramuscularly | DOSAGE: Administer intramuscularly (IM) as a |
| 326 | background | -0.326591 |  |  | technical assistance or to obtain a copy of the Safety |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 128 | background | +0.501170 |  |  |  |
| 342 | subject_text | +0.420191 | obtain a copy of the Safety Data Sheet, |  |  |
| 365 | subject_text | +0.338125 | (IM) as a single injection at the rate of 1 |  | to provide 2.5 mg/kg (1.13 mg/lb). The injectable |
| 462 | subject_text | +0.225552 | (1.13 mg/lb). The injectable dose | mL/9.1 kg (20 lb) to provide 2.5 mg/kg |  |
| 333 | subject_text | +0.216148 | mL/9.1 kg (20 lb) to provide 2.5 mg/kg |  |  |
| 364 | subject_text | +0.211475 | DOSAGE: Administer intramuscularly | DOSAGE: Administer intramuscularly |  |
| 60 | background | +0.172925 |  |  |  |
| 92 | background | +0.169924 |  |  | dogs during the rapid growth phase (between 2 |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| enrofloxacin 22.7 mg, n-butyl alcohol | 16 | +0.000851 |
| 30 mg, potassium hydroxide for pH | 16 | -0.000039 |
| mL/9.1 kg (20 lb) to provide 2.5 mg/kg | 13 | +0.000868 |
| (1.13 mg/lb). The injectable dose | 15 | +0.000496 |
| Consultpackage insert fordetails. | 10 | -0.000091 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_20594074f229bd8f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.009928 | +0.001343 |
| 只保留主体分差 | -0.004483 | +0.000349 |
| 主体 token 数 | 216 | 99 |

[成功对照 token 表](../photo_20594074f229bd8f/tokens.csv) · [对照摘要](../photo_20594074f229bd8f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
