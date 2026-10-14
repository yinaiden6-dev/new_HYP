# photo_04a4b44735fe9ba8

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001983`，F03 错误参考 `reference_001939`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_001939 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.013976 |
| BM25 正负分差 | -12.830643 |
| 0.5 × 图像 z 分差 | -0.152716 |
| 0.5 × BM25 z 分差 | -0.438652 |
| 最终融合分差 | -0.591367 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 304 个。正向贡献 +0.022219，负向贡献 -0.036195，相加等于图像分差 -0.013976。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 464 | -0.007263 | -0.012194 |
| nonvisual | 11 | -0.000040 | -0.002817 |
| subject_nontext | 100 | -0.001063 | -0.008282 |
| subject_text | 204 | -0.005610 | -0.021422 |

仅保留主体 query token 的两候选分差：-0.017099。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.007263。主体框内 token 占有效序列 39.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 379 | 65.5% | True | +2.133823 | 47.0% |
| 错误 | 105 | 38.5% | True | +5.505915 | 60.1% |

主体中 204 个 token 与 OCR 相交，69 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 591 | subject_text | -0.348332 | WHEN TAKEN AS DIRECTED SEE CRUG FACTS PANEL |  | *WHEN TAKEN AS DIRECTED. |
| 60 | background | -0.307123 |  | Actual |  |
| 59 | background | -0.298116 |  | Actual |  |
| 636 | subject_nontext | -0.273903 |  |  |  |
| 29 | background | -0.273841 |  | Loratadine tablets USP 10 mg, antihistamine |  |
| 102 | background | -0.254627 |  | Loratadine tablets USP 10 mg, antihistamine |  |
| 103 | background | -0.254146 |  | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL |  |
| 589 | subject_text | -0.251573 | WHEN TAKEN AS DIRECTED SEE CRUG FACTS PANEL | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL | • No synthetic colors |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 358 | background | +0.396071 |  |  |  |
| 365 | subject_text | +0.391402 | Compare to the active ingredient in Claritin | Compare to the active ingredient in Claritin |  |
| 397 | subject_text | +0.384259 | 24 hour |  |  |
| 364 | subject_nontext | +0.322046 |  | Compare to the active ingredient in Claritin |  |
| 378 | subject_nontext | +0.283935 |  | Compare to the active ingredient in Claritin |  |
| 141 | subject_text | +0.276907 | Walgreensfree+pure | Walgreensfree+pure | Walgreens |
| 410 | subject_nontext | +0.252512 |  | Compare to the active ingredient in Claritin |  |
| 677 | background | +0.247316 |  |  | Compare to the active ingredient in Claritin® Tablets" |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Loratadine tablets USP 10 mg, antihistamine | 25 | -0.001304 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
