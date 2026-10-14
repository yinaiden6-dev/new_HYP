# photo_2bdbf879ee1d7dd8

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001983`，F03 错误参考 `reference_001939`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_001939 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.008512 |
| BM25 正负分差 | -13.788834 |
| 0.5 × 图像 z 分差 | -0.123017 |
| 0.5 × BM25 z 分差 | -0.338048 |
| 最终融合分差 | -0.461065 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 320 个。正向贡献 +0.021959，负向贡献 -0.030471，相加等于图像分差 -0.008512。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 448 | +0.000257 | +0.000446 |
| nonvisual | 11 | +0.000032 | +0.002273 |
| subject_nontext | 89 | +0.001162 | +0.010171 |
| subject_text | 231 | -0.009963 | -0.033599 |

仅保留主体 query token 的两候选分差：-0.021425。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.009963。主体框内 token 占有效序列 41.1%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 293 | 57.2% | True | +3.842097 | 53.3% |
| 错误 | 86 | 37.4% | True | +5.854686 | 61.9% |

主体中 231 个 token 与 OCR 相交，75 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 230 | background | -0.471335 | NDC 31722-414-31 |  | NDC 0363-1612-95 |
| 229 | background | -0.362129 |  |  | NDC 0363-1612-95 |
| 246 | subject_text | -0.328570 | Relief | 200000004729 | NDC 0363-1612-95 |
| 550 | background | -0.325711 |  | 200000004729 |  |
| 602 | subject_nontext | -0.308771 |  | Actual |  |
| 180 | subject_text | -0.289280 | Walgreensfree+pure | free+pure | free+pure |
| 28 | background | -0.282878 |  | Loratadine tablets USP 10 mg, antihistamine |  |
| 621 | subject_text | -0.276267 | WHEN TAKEN AS DIRECTED SEE DRUG FACTS PANEL | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL | • No synthetic colors |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 365 | subject_text | +0.338088 | Loratadine tablets USP 10 mg, antihistamine | Compare to the active ingredient in Claritin |  |
| 397 | subject_text | +0.324807 | Compare to the active ingredient in Claritin |  |  |
| 401 | subject_text | +0.295693 | Compare to the active ingredient in Claritin |  | Compare to the active ingredient in Claritin® Tablets" |
| 442 | subject_nontext | +0.276977 |  |  |  |
| 72 | background | +0.222827 | proved by FDA under ANADA # 200-785 | 200000004729 |  |
| 143 | subject_text | +0.217738 | Walgreensfree+pure | Allergy Relief | Walgreens |
| 750 | background | +0.195486 |  |  | 24 hour |
| 603 | subject_text | +0.176861 | 20 | 20 |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Loratadine tablets USP 10 mg, antihistamine | 28 | -0.001181 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
