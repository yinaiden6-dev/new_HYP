# photo_900c769d86302c63

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000929`，F03 错误参考 `reference_004336`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_002207`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 19 |
| 图像 Top-1 参考 | reference_002207 |
| BM25 真值排名 | 16 |
| 融合真值排名 | 19 |
| 图像正负分差 | -0.021619 |
| BM25 正负分差 | -13.103745 |
| 0.5 × 图像 z 分差 | -0.422932 |
| 0.5 × BM25 z 分差 | -0.431836 |
| 最终融合分差 | -0.854769 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 225 个。正向贡献 +0.025491，负向贡献 -0.047110，相加等于图像分差 -0.021619。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 495 | -0.019362 | -0.028594 |
| nonvisual | 11 | -0.000171 | -0.011372 |
| subject_nontext | 185 | +0.000317 | +0.001252 |
| subject_text | 40 | -0.002402 | -0.043903 |

仅保留主体 query token 的两候选分差：-0.006776。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.019362。主体框内 token 占有效序列 30.8%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 88 | 26.1% | True | +9.824098 | 78.8% |
| 错误 | 50 | 3.1% | False | 不可评估 | 不可评估 |

主体中 40 个 token 与 OCR 相交，23 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 513 | subject_text | -0.475794 | Wooneocket, Ri 02895 | 1G71R 17 C4 | Distributed by: CVS Pharmacy, Inc. |
| 393 | subject_nontext | -0.463546 |  |  |  |
| 374 | subject_nontext | -0.443387 |  | Healthcare, L.P., distributor of Flonase® Allergy Relief. |  |
| 706 | background | -0.432951 |  |  | Distributed by: CVS Pharmacy, Inc. |
| 631 | subject_text | -0.426023 | PAPER | One Bottle | PAPER |
| 722 | background | -0.413695 |  |  | Distributed by: CVS Pharmacy, Inc. |
| 612 | subject_nontext | -0.388422 |  |  | PAPER |
| 312 | subject_nontext | -0.376583 |  |  | PAPER |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 290 | subject_nontext | +0.457618 |  |  |  |
| 229 | subject_nontext | +0.376936 |  |  |  |
| 316 | subject_nontext | +0.352601 |  |  |  |
| 616 | subject_nontext | +0.341966 |  |  |  |
| 521 | background | +0.297115 |  |  |  |
| 433 | subject_nontext | +0.296558 |  | 100% money back | CVS.com/returnpolicy |
| 711 | background | +0.272793 |  |  |  |
| 476 | subject_nontext | +0.270993 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100%money | 5 | -0.000104 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_7247f008f2067eb2` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.021619 | +0.319692 |
| 仅主体分差 | -0.006776 | +0.354524 |
| 主体 token 数 | 225 | 375 |

[对照 token 数据](../photo_7247f008f2067eb2/tokens.csv) · [对照摘要](../photo_7247f008f2067eb2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
