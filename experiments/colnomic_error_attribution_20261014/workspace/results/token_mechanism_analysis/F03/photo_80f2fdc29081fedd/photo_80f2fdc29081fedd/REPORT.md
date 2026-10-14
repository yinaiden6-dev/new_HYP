# photo_80f2fdc29081fedd

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_003233`，F03 错误参考 `reference_001680`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003666`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 297 |
| 图像 Top-1 参考 | reference_003666 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 22 |
| 图像正负分差 | -0.031719 |
| BM25 正负分差 | +8.201416 |
| 0.5 × 图像 z 分差 | -0.546260 |
| 0.5 × BM25 z 分差 | +0.074259 |
| 最终融合分差 | -0.472001 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.1197**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 88 个。正向贡献 +0.009428，负向贡献 -0.041147，相加等于图像分差 -0.031719。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 632 | -0.027045 | -0.031281 |
| nonvisual | 11 | -0.000093 | -0.006149 |
| subject_nontext | 16 | -0.001106 | -0.050512 |
| subject_text | 72 | -0.003476 | -0.035286 |

仅保留主体 query token 的两候选分差：-0.038055。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.027045。主体框内 token 占有效序列 12.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 356 | 45.8% | True | +14.392938 | 87.1% |
| 错误 | 6 | 6.2% | False | 不可评估 | 不可评估 |

主体中 72 个 token 与 OCR 相交，64 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 450 | subject_nontext | -0.337225 |  |  |  |
| 489 | subject_text | -0.308001 | (in each tablet) |  |  |
| 357 | subject_text | -0.277661 | i    |  |  |
| 51 | background | -0.267707 |  |  |  |
| 106 | background | -0.239133 |  |  |  |
| 221 | background | -0.207161 |  | Reye's syndrome: Children and teenagers who have or |  |
| 88 | background | -0.206839 |  | Drug Facts (continued) | Drug Facts (continued) |
| 236 | background | -0.204328 |  |  | stomach bleeding. The chance is higher if you |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 257 | background | +0.319430 |  |  |  |
| 377 | subject_text | +0.314571 | i    |  |  |
| 571 | background | +0.207879 |  |  |  |
| 580 | background | +0.187910 |  |  | kidney disease |
| 651 | background | +0.157697 |  |  |  |
| 552 | background | +0.139573 |  | you have high blood pressure, heart disease, liver | Ask a doctor or pharmacist before use if you are taking a |
| 397 | subject_text | +0.122520 | pam |  | doctor because it may cause problems in the |
| 577 | background | +0.110428 |  | Aspirin 325 mg |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Aspirin 325 mg (NSAIO) | 2 | -0.000404 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_59498b6acfbfbf4f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.031719 | +0.072915 |
| 仅主体分差 | -0.038055 | +0.114786 |
| 主体 token 数 | 88 | 78 |

[对照 token 数据](../photo_59498b6acfbfbf4f/tokens.csv) · [对照摘要](../photo_59498b6acfbfbf4f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
