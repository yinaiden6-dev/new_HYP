# photo_0913bcc35e9beb5a

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001357`，F03 错误参考 `reference_004710`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000955`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2216 |
| 图像 Top-1 参考 | reference_000955 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.106005 |
| BM25 正负分差 | +84.995361 |
| 0.5 × 图像 z 分差 | -1.338981 |
| 0.5 × BM25 z 分差 | +1.149827 |
| 最终融合分差 | -0.189154 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4620**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 384 个。正向贡献 +0.020208，负向贡献 -0.126213，相加等于图像分差 -0.106005。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 336 | -0.039729 | -0.086435 |
| nonvisual | 11 | -0.000346 | -0.023004 |
| subject_nontext | 216 | -0.031549 | -0.106768 |
| subject_text | 168 | -0.034381 | -0.149598 |

仅保留主体 query token 的两候选分差：-0.125506。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.039729。主体框内 token 占有效序列 52.5%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1031 | 56.9% | True | +12.254304 | 90.6% |
| 错误 | 24 | 2.4% | False | 不可评估 | 不可评估 |

主体中 168 个 token 与 OCR 相交，102 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 674 | background | -0.559093 |  | to prevent athlete's foot, apply once or twice daily (morning and/or night)this product is not effective on the scalp or nails | Do not use on children under 2 years of age except under the advice and |
| 249 | subject_nontext | -0.555968 |  |  | Directions |
| 474 | subject_text | -0.535963 | Do not use on children under 2 years of age except under the advice and supervision of a doctor. | to prevent athlete's foot, apply once or twice daily (morning and/or night)this product is not effective on the scalp or nails | Do not use on children under 2 years of age except under the advice and |
| 581 | subject_nontext | -0.531741 |  |  | Directions |
| 559 | subject_text | -0.523536 | Questions or comments?Call toll free 1-800-FOR-AVON | apply a thin layer over affected area twice daily (morning and night) | Questions or comments? Call 1-844-474-7464 |
| 523 | background | -0.512011 |  |  | Do not use on children under 2 years of age except under the advice and |
| 313 | subject_nontext | -0.509484 |  |  | When using this product avoid contact with the eyes. |
| 532 | subject_text | -0.505774 | For external use only. |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 206 | subject_nontext | +0.371341 |  |  |  |
| 306 | subject_nontext | +0.255289 |  |  |  |
| 126 | subject_nontext | +0.242337 |  |  |  |
| 566 | subject_nontext | +0.223679 |  |  |  |
| 162 | background | +0.223356 |  |  | Pedikleer |
| 657 | background | +0.220776 |  | Keep out of reach of children. If swallowed, get medical help or contact a Poison Control right away. |  |
| 182 | background | +0.209764 |  |  | Pedikleer |
| 157 | subject_nontext | +0.209605 |  |  | Lot No. & Expiration Date: see box or crimp of tube |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Tolnaftate 1% | 2 | -0.000362 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_e82154bdf3d3c846` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.106005 | +0.193489 |
| 仅主体分差 | -0.125506 | +0.218224 |
| 主体 token 数 | 384 | 294 |

[对照 token 数据](../photo_e82154bdf3d3c846/tokens.csv) · [对照摘要](../photo_e82154bdf3d3c846/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
