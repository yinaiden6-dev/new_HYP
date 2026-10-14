# photo_4920adb7ad190a5f

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000100`，F03 错误参考 `reference_000104`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_000100`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_000100 |
| BM25 真值排名 | 5 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.018254 |
| BM25 正负分差 | -8.914711 |
| 0.5 × 图像 z 分差 | +0.381876 |
| 0.5 × BM25 z 分差 | -1.388716 |
| 最终融合分差 | -1.006841 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.7843**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 456 个。正向贡献 +0.039507，负向贡献 -0.021253，相加等于图像分差 +0.018254。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 312 | +0.001424 | +0.003557 |
| nonvisual | 11 | -0.000036 | -0.002548 |
| subject_nontext | 319 | +0.012289 | +0.030010 |
| subject_text | 137 | +0.004577 | +0.026023 |

仅保留主体 query token 的两候选分差：+0.028812。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 -0.000036。主体框内 token 占有效序列 58.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 159 | 58.3% | True | +9.064513 | 78.5% |
| 错误 | 40 | 14.4% | False | 不可评估 | 不可评估 |

主体中 137 个 token 与 OCR 相交，27 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 119 | subject_nontext | -0.309687 |  |  |  |
| 371 | subject_nontext | -0.275145 |  |  |  |
| 636 | subject_nontext | -0.270978 |  |  |  |
| 129 | background | -0.250182 |  |  |  |
| 565 | subject_nontext | -0.216606 |  |  |  |
| 307 | subject_nontext | -0.214572 |  | LOT |  |
| 403 | subject_nontext | -0.208522 |  |  |  |
| 404 | subject_nontext | -0.196906 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 603 | subject_text | +0.652001 | the reach of children | Keep this and all drugs out of the reach of children |  |
| 379 | subject_nontext | +0.573611 |  | Keep this and all drugs out of the reach of children |  |
| 667 | subject_text | +0.493021 | the reach of children | Keep this and all drugs out of the reach of children |  |
| 155 | subject_nontext | +0.473884 |  | Keep this and all drugs out of the reach of children |  |
| 570 | subject_text | +0.471552 | blets (11.4 mg each) | PETIQ | www.capstarpet.com |
| 443 | subject_nontext | +0.467035 |  | Keep this and all drugs out of the reach of children |  |
| 475 | subject_nontext | +0.456144 |  | Keep this and all drugs out of the reach of children |  |
| 634 | subject_text | +0.417873 | blets (11.4 mg each) | PETIQ | www.capstarpet.com |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 11.4mg | 7 | +0.000363 |
| blets (11.4 mg each) | 20 | +0.001515 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_2d3809ffa08d826a` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.018254 | +0.056790 |
| 仅主体分差 | +0.028812 | +0.075831 |
| 主体 token 数 | 456 | 390 |

[对照 token 数据](../photo_2d3809ffa08d826a/tokens.csv) · [对照摘要](../photo_2d3809ffa08d826a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
