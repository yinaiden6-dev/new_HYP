# B03 · photo_c798e089f34805e9

按当前评测真值分析，未新增身份人工复核。

包装 `package_028`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001012`，**纯图像实际 Top-1 错误参考** `reference_002223`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.580297 |
| 错误参考 MaxSim | +0.582458 |
| 正确减错误的分差 | -0.002161 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 264 个。正向贡献 +0.025886，负向贡献 -0.028047，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 504 | -0.008221 | -0.012706 |
| nonvisual | 11 | -0.000009 | -0.000648 |
| subject_nontext | 73 | +0.001205 | +0.012858 |
| subject_text | 191 | +0.004864 | +0.019837 |

最主要负差额来自 `background`，贡献 -0.008221。只保留主体 query token 后，本对分差为 +0.017907。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 386 | 73.6% | True | +6.650672 | 63.9% |
| 错误 | 13 | 20.7% | True | +12.143881 | 99.6% |

主体中 191 个 token 与 OCR 行相交，104 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 731 | background | -0.651948 | CVS | Drug Facts | CVS |
| 132 | background | -0.536762 |  |  |  |
| 574 | subject_text | -0.489917 | overdose, get medical help or contact a Poison |  | If pregnant or breast-teeding, aska health professlonal |
| 478 | subject_text | -0.267299 | overdose, get medical help or contact a Poison | Keep out of reach of children. In case of |  |
| 39 | background | -0.249284 | Kids Ear Refief |  | Comfortable relief |
| 536 | subject_text | -0.245508 | are presently taking mineral oil |  | If you are presently taking mineral oll, unless told to do |
| 328 | background | -0.245000 |  |  | Comfortable relief |
| 767 | background | -0.240334 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 505 | subject_text | +0.420076 | you have rectal bleeding or fail to have a | Stop use and ask a doctor if |  |
| 406 | subject_text | +0.416824 | have stomach pain, nausea or vomiting | Ask a doctor before use if you |  |
| 510 | subject_text | +0.390492 | overdose, get medical help or contact a Poison | professional before use. | If pregnant or breast-teeding, aska health professlonal |
| 341 | subject_nontext | +0.368982 |  | have stomach pain, nausea or vomiting |  |
| 342 | subject_text | +0.364952 | have stomach pain, nausea or vomiting | have stomach pain, nausea or vomiting |  |
| 331 | subject_text | +0.319921 | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SEAL | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SEAL |  |
| 438 | subject_text | +0.318703 | have stomach pain, nausea or vomiting | Ask a doctor before use if you |  |
| 315 | subject_text | +0.302283 | you need to use a laxative longer than | a serious condition. | Ask a doctor before use If you have |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Docusate Sodium 100 mg. | 13 | +0.000925 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_b5df0dce47884a73` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002161 | +0.037341 |
| 只保留主体分差 | +0.017907 | +0.094399 |
| 主体 token 数 | 264 | 242 |

[成功对照 token 表](../photo_b5df0dce47884a73/tokens.csv) · [对照摘要](../photo_b5df0dce47884a73/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
