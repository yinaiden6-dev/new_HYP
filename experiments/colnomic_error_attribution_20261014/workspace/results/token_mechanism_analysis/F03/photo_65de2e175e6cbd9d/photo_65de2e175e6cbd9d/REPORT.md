# photo_65de2e175e6cbd9d

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000935`，F03 错误参考 `reference_003797`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_004066`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 5 |
| 图像 Top-1 参考 | reference_004066 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.000846 |
| BM25 正负分差 | -11.765236 |
| 0.5 × 图像 z 分差 | -0.020662 |
| 0.5 × BM25 z 分差 | -0.490559 |
| 最终融合分差 | -0.511221 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 119 个。正向贡献 +0.030757，负向贡献 -0.031603，相加等于图像分差 -0.000846。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 601 | -0.000671 | -0.000816 |
| nonvisual | 11 | +0.000000 | +0.000015 |
| subject_nontext | 47 | -0.000615 | -0.009570 |
| subject_text | 72 | +0.000440 | +0.004465 |

仅保留主体 query token 的两候选分差：-0.001078。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.000671。主体框内 token 占有效序列 16.3%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 168 | 54.1% | True | +4.350083 | 58.8% |
| 错误 | 64 | 7.9% | False | 不可评估 | 不可评估 |

主体中 72 个 token 与 OCR 相交，53 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 182 | background | -0.442357 |  | ORG0220-F | ITEM 323865 W10383-0321-L |
| 353 | subject_nontext | -0.322667 |  |  | walgreens.com ©2021 Walgreen Co. |
| 366 | background | -0.321465 |  |  | walgreens.com ©2021 Walgreen Co. |
| 362 | background | -0.320930 |  |  | walgreens.com ©2021 Walgreen Co. |
| 402 | background | -0.291576 |  |  | walgreens.com ©2021 Walgreen Co. |
| 393 | subject_text | -0.274231 | MADEIN INDIA |  | walgreens.com ©2021 Walgreen Co. |
| 459 | background | -0.261180 |  |  |  |
| 534 | background | -0.244131 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.494695 | ASPIRIN FREE | ITEM 696611W10383-0420-L |  |
| 667 | background | +0.433600 |  | ITEM 696611W10383-0420-L |  |
| 660 | background | +0.407368 |  | ITEM 696611W10383-0420-L |  |
| 663 | background | +0.393940 |  | ITEM 696611W10383-0420-L |  |
| 483 | background | +0.367758 |  | ITEM 696611W10383-0420-L |  |
| 235 | subject_text | +0.351645 | This product is not manufactured or distributed | "This product is not manufactured or distributed |  |
| 155 | background | +0.343318 |  | "This product is not manufactured or distributed |  |
| 344 | background | +0.331582 |  | ITEM 696611W10383-0420-L |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% | 8 | -0.000740 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_7f4bc7d166b40bfc` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000846 | +0.085909 |
| 仅主体分差 | -0.001078 | +0.101169 |
| 主体 token 数 | 119 | 288 |

[对照 token 数据](../photo_7f4bc7d166b40bfc/tokens.csv) · [对照摘要](../photo_7f4bc7d166b40bfc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
