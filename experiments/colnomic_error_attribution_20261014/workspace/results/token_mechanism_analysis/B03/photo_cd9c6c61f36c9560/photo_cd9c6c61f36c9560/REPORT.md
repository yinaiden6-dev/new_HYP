# B03 · photo_cd9c6c61f36c9560

按当前评测真值分析，未新增身份人工复核。

包装 `package_015`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000934`，**纯图像实际 Top-1 错误参考** `reference_003659`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 6 |
| 正确参考 MaxSim | +0.709459 |
| 错误参考 MaxSim | +0.733631 |
| 正确减错误的分差 | -0.024171 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 162 个。正向贡献 +0.039558，负向贡献 -0.063729，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.027318 | -0.035787 |
| nonvisual | 11 | -0.000029 | -0.001937 |
| subject_nontext | 44 | +0.000890 | +0.014781 |
| subject_text | 118 | +0.002286 | +0.014161 |

最主要负差额来自 `background`，贡献 -0.027318。只保留主体 query token 后，本对分差为 +0.014330。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 287 | 73.0% | True | +7.786106 | 66.0% |
| 错误 | 3 | 0.0% | False | 不可评估 | 不可评估 |

主体中 118 个 token 与 OCR 行相交，55 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 151 | background | -0.591038 |  |  |  |
| 392 | subject_text | -0.549481 | For the Temporary Rellef |  | Pain Reliever/Fever Reducer |
| 131 | background | -0.540489 |  |  |  |
| 271 | subject_text | -0.455632 | Pain Reliever/Fever Reducer | Pain Reliever/Fever Reducer | Pain Reliever/Fever Reducer |
| 90 | background | -0.450874 |  | 24 cAPLETS** 650 mg each | 225 CAPLETS* |
| 457 | subject_text | -0.411384 | Actiro Ingredost ef Tylanan | talo 2 capible sory à boue wih weker | Compare to the active ingredient in |
| 132 | background | -0.400931 |  |  |  |
| 261 | background | -0.384975 |  | Pain Reliever/Fever Reducer | Pain Reliever/Fever Reducer |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 256 | subject_text | +0.602934 | SOUNDBODY | SOUNDBODY | 341520001673 |
| 296 | subject_text | +0.586557 | SOUNDBODY | SOUNDBODY | pain gets worse or lasts more than 10 days |
| 257 | subject_text | +0.584608 | SOUNDBODY | SOUNDBODY | 341520001673 |
| 237 | subject_text | +0.579991 | SOUNDBODY | SOUNDBODY | 341520001673 |
| 316 | subject_text | +0.578731 | SOUNDBODY | SOUNDBODY | Forproduct questlons or concerns, contact usat |
| 17 | background | +0.571438 |  |  |  |
| 297 | subject_text | +0.554604 | SOUNDBODY | SOUNDBODY | Forproduct questlons or concerns, contact usat |
| 459 | background | +0.543000 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acataminophem Extended-Release Tablets USP, 650 mg | 31 | -0.004535 |
| 24 CAPLETS** 650 mg each | 17 | -0.001912 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_4917ad5f3fac9f55` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.024171 | +0.054417 |
| 只保留主体分差 | +0.014330 | +0.097190 |
| 主体 token 数 | 162 | 50 |

[成功对照 token 表](../photo_4917ad5f3fac9f55/tokens.csv) · [对照摘要](../photo_4917ad5f3fac9f55/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
