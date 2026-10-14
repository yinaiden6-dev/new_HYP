# B03 · photo_ab435a0418e86576

按当前评测真值分析，未新增身份人工复核。

包装 `package_033`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001031`，**纯图像实际 Top-1 错误参考** `reference_004077`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.724198 |
| 错误参考 MaxSim | +0.739097 |
| 正确减错误的分差 | -0.014899 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 390 个。正向贡献 +0.038421，负向贡献 -0.053321，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 330 | -0.008922 | -0.019763 |
| nonvisual | 11 | -0.000072 | -0.004762 |
| subject_nontext | 162 | -0.002470 | -0.011144 |
| subject_text | 228 | -0.003436 | -0.011016 |

最主要负差额来自 `background`，贡献 -0.008922。只保留主体 query token 后，本对分差为 -0.011069。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 412 | 57.6% | True | +6.014626 | 62.2% |
| 错误 | 29 | 26.7% | True | +9.814785 | 100.0% |

主体中 228 个 token 与 OCR 行相交，155 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 349 | subject_text | -0.615500 | overuse may cause more eye redness | distributed by |  |
| 310 | subject_text | -0.602997 | narrow angle glaucoma |  | narrow angle glaucoma. |
| 309 | subject_text | -0.520849 | narrow angle glaucoma |  | narrow angle glaucoma. |
| 188 | subject_text | -0.482281 | Dextran 700.1%. |  |  |
| 228 | subject_text | -0.468960 | Tetrahydrozoline HCl 0.05%. | Polyethylene glycol 400 1%. | Tetrahydrozoline HCl 0.05%.Redness reliever |
| 375 | subject_text | -0.461909 | becomes cloudyto avoid contamination, do not | touch tip of container to any surface | surface to avoid contamination |
| 229 | subject_text | -0.451562 | Tetrahydrozoline HCl 0.05%. | sodium chloride | Tetrahydrozoline HCl 0.05%.Redness reliever |
| 350 | subject_text | -0.450162 | overuse may cause more eye redness |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 316 | subject_text | +0.562482 | pupils may become enlarged temporarily | Ask a doctor before use if you have |  |
| 554 | subject_text | +0.498163 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | ON THE BOTTLE IS BROKEN OR MISSING |
| 553 | subject_text | +0.473632 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | Stop use and ask a doctor if |
| 115 | subject_text | +0.473044 | BAND ON THE BOTTLE IS BROKEN OR MISSING | DO NOT USE IF CARTONIS OPEN OR BLUE NECK |  |
| 555 | subject_nontext | +0.458093 |  | To open bottle, push cap down and twist | do not touch tip of container to any |
| 537 | subject_text | +0.416920 | To open bottle, push cap down and twist | To open bottle, push cap down and twist | Stop use and ask a doctor if |
| 410 | subject_text | +0.413746 | replace cap after each use |  | surface to avoid contamination |
| 363 | background | +0.383323 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Dextran 700.1%. | 5 | -0.001373 |
| Polyethylene glycol 400 1%. | 12 | -0.000926 |
| Povidone 1 %. | 6 | -0.001322 |
| Tetrahydrozoline HCl 0.05%. | 11 | -0.003084 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_519a92da367dee34` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.014899 | +0.083101 |
| 只保留主体分差 | -0.011069 | +0.104356 |
| 主体 token 数 | 390 | 324 |

[成功对照 token 表](../photo_519a92da367dee34/tokens.csv) · [对照摘要](../photo_519a92da367dee34/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
