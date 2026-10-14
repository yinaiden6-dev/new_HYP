# photo_6f1045004ede2c6e

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_003333`，F03 错误参考 `reference_003332`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_002079`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 11 |
| 图像 Top-1 参考 | reference_002079 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.001505 |
| BM25 正负分差 | -0.744263 |
| 0.5 × 图像 z 分差 | -0.012217 |
| 0.5 × BM25 z 分差 | -0.005568 |
| 最终融合分差 | -0.017785 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 300 个。正向贡献 +0.010668，负向贡献 -0.012172，相加等于图像分差 -0.001505。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 420 | -0.001003 | -0.001746 |
| nonvisual | 11 | -0.000131 | -0.008679 |
| subject_nontext | 65 | -0.000430 | -0.004840 |
| subject_text | 235 | +0.000060 | +0.000185 |

仅保留主体 query token 的两候选分差：-0.000904。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.001003。主体框内 token 占有效序列 41.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 619 | 65.9% | True | +4.666683 | 54.3% |
| 错误 | 616 | 65.9% | True | +4.437270 | 55.5% |

主体中 235 个 token 与 OCR 相交，161 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 620 | subject_text | -0.164197 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 |  |
| 542 | background | -0.128408 |  |  |  |
| 451 | subject_nontext | -0.126891 |  |  |  |
| 349 | background | -0.123776 |  |  |  |
| 250 | background | -0.119334 |  |  |  |
| 257 | subject_text | -0.115939 | For external use only • Do not use on damaged | For external use only • Do not use on damaged | For external use only • Do not use on damaged |
| 81 | background | -0.115878 |  |  |  |
| 204 | background | -0.114638 |  | • Helps prevent sunburn |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 331 | subject_text | +0.172580 | Directions | medical help or contact a Poison Control Center right away. | medical help or contact a Poison Control Center right away. |
| 354 | subject_text | +0.143647 | For sunscreen use: • apply liberally 15 minutes |  |  |
| 614 | subject_text | +0.132849 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 |
| 46 | background | +0.128698 |  |  |  |
| 274 | subject_text | +0.125511 | or broken skin • When using this product keep |  | elf |
| 278 | subject_text | +0.123710 | or broken skin • When using this product keep | For external use only • Do not use on damaged |  |
| 689 | background | +0.120920 |  |  |  |
| 27 | background | +0.110758 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Octinoxate 6.0% | 4 | -0.000010 |
| Octisalate 4.0% | 4 | +0.000063 |
| Oxybenzone 3.5% | 9 | +0.000198 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_51f7a3bb218030b8` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001505 | +0.019705 |
| 仅主体分差 | -0.000904 | +0.014700 |
| 主体 token 数 | 300 | 319 |

[对照 token 数据](../photo_51f7a3bb218030b8/tokens.csv) · [对照摘要](../photo_51f7a3bb218030b8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
