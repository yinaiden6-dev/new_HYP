# B03 · photo_41de3eb383934ec3

按当前评测真值分析，未新增身份人工复核。

包装 `package_085`，划分 `test`；用户主题：换行与末尾词位置。正确参考 `reference_000229`，**纯图像实际 Top-1 错误参考** `reference_000233`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.544279 |
| 错误参考 MaxSim | +0.548374 |
| 正确减错误的分差 | -0.004094 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 144 个。正向贡献 +0.019874，负向贡献 -0.023968，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 624 | -0.001229 | -0.001535 |
| nonvisual | 11 | +0.000013 | +0.000933 |
| subject_nontext | 19 | +0.000246 | +0.010086 |
| subject_text | 125 | -0.003124 | -0.019471 |

最主要负差额来自 `subject_text`，贡献 -0.003124。只保留主体 query token 后，本对分差为 -0.015571。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 224 | 61.8% | True | +6.967820 | 65.5% |
| 错误 | 16 | 27.3% | True | +8.516190 | 97.9% |

主体中 125 个 token 与 OCR 行相交，109 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 439 | subject_text | -0.469062 | breeds during the rapid growth phase. |  | The safe use of enrofloxacin has not been |
| 211 | background | -0.378257 |  | Consult package insert for details. |  |
| 377 | subject_text | -0.360238 | events, for technical assistance or to |  | technical assistance or to obtain a copy of the Safety |
| 433 | subject_text | -0.357504 | (IM) as a single injection at the rate of 1 |  |  |
| 441 | subject_text | -0.345618 | events, for technical assistance or to | contact Cronus Pharma LLC at 1-844- | technical assistance or to obtain a copy of the Safety |
| 428 | background | -0.312115 |  |  | to provide 2.5 mg/kg (1.13 mg/lb). The injectable |
| 382 | subject_text | -0.287586 | 50.1-100 lbs | 2 | 50 mL |
| 626 | background | -0.262356 |  | DOSAGE: Administer intramuscularly | DOSAGE: Administer intramuscularly (IM) as a |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 346 | subject_text | +0.409373 | contact Cronus Pharma LLC at 1-844- |  |  |
| 143 | background | +0.272247 |  |  |  |
| 284 | subject_text | +0.271976 | drugs, contact FDA at 1-888-FDA-VETS | drug experience reporting for animal |  |
| 464 | subject_text | +0.241081 | adjustmentand waterforinjection, q.s. |  | enrofloxacin 22.7 mg, n-butyl alcohol 30 mg, |
| 369 | subject_text | +0.216708 | (IM) as a single injection at the rate of 1 | DOSAGE: Administer intramuscularly |  |
| 764 | background | +0.182324 |  | drug experience reporting for animal |  |
| 476 | subject_text | +0.163985 | orhttp://www.fda.gov/reportanimalae | drug experience reporting for animal |  |
| 664 | background | +0.153906 |  | Consult package insert for details. | rapid growth phase. Consult package insert for |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| enrofloxacin 22.7 mg, n-butyi alcohol | 8 | +0.000017 |
| 30 mg, potassium hydroxide for pH | 16 | -0.000035 |
| mL/9.1 kg (20 lb) to provide 2.5 mg/kg | 16 | +0.000122 |
| (1.13 mg/lb). The injectable dose | 8 | +0.000090 |
| Consultpackageinsertfordetails. | 14 | -0.001363 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_20594074f229bd8f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004094 | +0.001343 |
| 只保留主体分差 | -0.015571 | +0.000349 |
| 主体 token 数 | 144 | 99 |

[成功对照 token 表](../photo_20594074f229bd8f/tokens.csv) · [对照摘要](../photo_20594074f229bd8f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
