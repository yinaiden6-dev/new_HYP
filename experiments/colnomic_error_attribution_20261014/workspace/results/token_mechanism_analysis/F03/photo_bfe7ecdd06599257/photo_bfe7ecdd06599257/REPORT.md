# photo_bfe7ecdd06599257

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000973`，F03 错误参考 `reference_004586`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000992`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 38 |
| 图像 Top-1 参考 | reference_000992 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 4 |
| 图像正负分差 | -0.056442 |
| BM25 正负分差 | +27.679916 |
| 0.5 × 图像 z 分差 | -0.634190 |
| 0.5 × BM25 z 分差 | +0.479784 |
| 最终融合分差 | -0.154406 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4307**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 180 个。正向贡献 +0.016066，负向贡献 -0.072509，相加等于图像分差 -0.056442。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 540 | -0.039954 | -0.054085 |
| nonvisual | 11 | -0.000225 | -0.014973 |
| subject_nontext | 37 | -0.002331 | -0.046056 |
| subject_text | 143 | -0.013932 | -0.071221 |

仅保留主体 query token 的两候选分差：-0.066048。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.039954。主体框内 token 占有效序列 24.6%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 697 | 61.2% | True | +5.842186 | 63.9% |
| 错误 | 21 | 7.4% | False | 不可评估 | 不可评估 |

主体中 143 个 token 与 OCR 相交，108 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 291 | subject_text | -0.549720 | tamporsrlly ralevas minor aches and pains due to: |  | temporarily reduces fever |
| 331 | subject_text | -0.544875 | the common cold | PAINRELIEF | temporarily reduces fever |
| 332 | subject_text | -0.471196 | the common cold | PAINRELIEF | temporarily reduces fever |
| 380 | background | -0.459849 |  |  | hivesfacial swelling |
| 384 | background | -0.444965 |  |  | hivesfacial swelling |
| 571 | background | -0.437113 |  |  | hivesfacial swelling |
| 174 | subject_text | -0.433892 | KEEP CARTON FOR REFERENCE |  | KEEP PACKAGE FOR REFERENCE |
| 293 | subject_text | -0.420927 | tamporsrlly ralevas minor aches and pains due to: |  | *nonsteroldal anti-inflammatory drug |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 81 | background | +0.290291 |  |  |  |
| 391 | subject_text | +0.229702 | hitin) |  |  |
| 52 | background | +0.196878 |  |  |  |
| 71 | background | +0.181452 |  | DO NOT USE IF CARTON IS OPEN OR IF FOIL SEAL | BOTTLE |
| 152 | subject_nontext | +0.179200 |  | Raproon sodium 220 mg..Pan reventvor reducer |  |
| 199 | background | +0.174968 |  |  |  |
| 270 | subject_text | +0.171294 | *nonstarokdal anil-Inilsmmatory drug | Raproon sodium 220 mg..Pan reventvor reducer | hypromellose, magneslum stearate, microcrystalline |
| 483 | background | +0.163141 |  | take a blood thinning (anticoagulant) or |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Ruproon sodium 220 mg...Pain reloverowor reducer | 8 | -0.000443 |
| (aprsan 200 rmg) (NSAID) | 8 | +0.000377 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_38d1d33620ab1ea1` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.056442 | +0.026424 |
| 仅主体分差 | -0.066048 | +0.067129 |
| 主体 token 数 | 180 | 54 |

[对照 token 数据](../photo_38d1d33620ab1ea1/tokens.csv) · [对照摘要](../photo_38d1d33620ab1ea1/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
