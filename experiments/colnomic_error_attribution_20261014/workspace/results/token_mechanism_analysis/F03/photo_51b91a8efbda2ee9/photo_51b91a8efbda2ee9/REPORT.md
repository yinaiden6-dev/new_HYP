# photo_51b91a8efbda2ee9

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001139`，F03 错误参考 `reference_004489`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000950`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 117 |
| 图像 Top-1 参考 | reference_000950 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 4 |
| 图像正负分差 | -0.038539 |
| BM25 正负分差 | +34.577667 |
| 0.5 × 图像 z 分差 | -0.472146 |
| 0.5 × BM25 z 分差 | +0.388192 |
| 最终融合分差 | -0.083953 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4512**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 91 个。正向贡献 +0.020661，负向贡献 -0.059200，相加等于图像分差 -0.038539。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 629 | -0.031622 | -0.036750 |
| nonvisual | 11 | -0.000253 | -0.016781 |
| subject_nontext | 37 | -0.001679 | -0.033171 |
| subject_text | 54 | -0.004985 | -0.067480 |

仅保留主体 query token 的两候选分差：-0.053530。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.031622。主体框内 token 占有效序列 12.4%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 860 | 51.4% | True | +7.736124 | 75.8% |
| 错误 | 8 | 12.9% | False | 不可评估 | 不可评估 |

主体中 54 个 token 与 OCR 相交，47 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 393 | subject_text | -0.519213 | Warnings | Itcontains important information. |  |
| 552 | subject_nontext | -0.471488 |  | cough accompanied by too much phlegm (mucus) | If pregnant or breast-feeding, ask a |
| 82 | background | -0.468279 |  | Itcontains important information. |  |
| 392 | subject_text | -0.452349 | Donot use for chidren under 12 years of age | Itcontains important information. |  |
| 466 | background | -0.410874 |  | Do not use for children under 12 years of age | Stop use and ask a doctor if |
| 86 | background | -0.397644 |  | Itcontains important information. |  |
| 41 | background | -0.396688 |  | CONTROLS COUGH |  |
| 134 | background | -0.387170 |  |  | occurs with smoking, asthma, chronic |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 535 | subject_nontext | +0.304308 |  |  |  |
| 275 | background | +0.245692 |  | Do not use for children under 12 years of age |  |
| 532 | subject_nontext | +0.238734 |  |  |  |
| 710 | background | +0.232425 |  |  |  |
| 288 | background | +0.229065 |  | Do not use for children under 12 years of age |  |
| 357 | subject_text | +0.223344 | De tromethorphan HBr 60 mg.  Cough sup preeant | Dextromethorphan HBr 60 mg..... Cough suppressant |  |
| 315 | background | +0.180549 |  |  |  |
| 267 | background | +0.166822 |  |  | this product can be administered without |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| De tromethorphan HBr 60 mg.  Cough sup preeant | 5 | +0.000253 |
| Gual enesin 1200 mg. | 3 | +0.000043 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_792997f8dbfba097` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.038539 | +0.081908 |
| 仅主体分差 | -0.053530 | +0.099783 |
| 主体 token 数 | 91 | 88 |

[对照 token 数据](../photo_792997f8dbfba097/tokens.csv) · [对照摘要](../photo_792997f8dbfba097/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
