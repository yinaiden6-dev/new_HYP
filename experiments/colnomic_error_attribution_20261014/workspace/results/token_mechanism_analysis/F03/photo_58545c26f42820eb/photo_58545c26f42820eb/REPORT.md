# photo_58545c26f42820eb

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001321`，F03 错误参考 `reference_001320`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_001320 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 3 |
| 图像正负分差 | -0.004665 |
| BM25 正负分差 | -4.388916 |
| 0.5 × 图像 z 分差 | -0.061962 |
| 0.5 × BM25 z 分差 | -0.018122 |
| 最终融合分差 | -0.080085 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 156 个。正向贡献 +0.022314，负向贡献 -0.026979，相加等于图像分差 -0.004665。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 564 | -0.001799 | -0.002332 |
| nonvisual | 11 | +0.000119 | +0.007923 |
| subject_nontext | 47 | +0.001649 | +0.025652 |
| subject_text | 109 | -0.004635 | -0.031083 |

仅保留主体 query token 的两候选分差：-0.013990。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.004635。主体框内 token 占有效序列 21.3%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 101 | 8.2% | False | 不可评估 | 不可评估 |
| 错误 | 72 | 6.8% | False | 不可评估 | 不可评估 |

主体中 109 个 token 与 OCR 相交，91 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 75 | background | -0.412693 |  |  |  |
| 455 | subject_text | -0.313539 | preep out of reach of children. In case of overdose, get |  | Allergy alert: Acetaminophen may cause severe |
| 317 | subject_text | -0.309064 | Sekzer | Slze | Seltzer |
| 318 | subject_text | -0.292918 | Sekzer | Slze | Seltzer |
| 414 | subject_text | -0.265955 | Ask a doctor or pharmacist before use if you are | • pain, cough, or nasal congestion gets worse or lasts |  |
| 258 | background | -0.254365 |  | Slze | Seltzer |
| 336 | subject_text | -0.249549 | Sekzer |  |  |
| 316 | subject_text | -0.248246 | Sekzer |  | Seltzer |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 467 | subject_nontext | +0.320209 |  | Seltzer |  |
| 156 | background | +0.278972 |  | Slze |  |
| 367 | subject_text | +0.266207 | Phenylephrine hydrochlonide 5 mg |  | Purposes |
| 274 | subject_nontext | +0.230481 |  | Seltzer |  |
| 30 | background | +0.223335 |  |  |  |
| 106 | background | +0.217715 |  |  | Seltzer |
| 175 | background | +0.197863 |  |  | Seltzer |
| 368 | subject_text | +0.193708 | Phenylephrine hydrochlonide 5 mg |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| menophenS25 mg | 2 | -0.000048 |
| hydrobromide 10 mg | 4 | +0.000230 |
| Guaitenesin 200 mg | 3 | +0.000597 |
| 10 mg | 1 | -0.000032 |
| Phenylephrine hydrochlonide 5 mg | 4 | +0.000624 |
| more than 4,000 mg of acetaminophen in 24 hours | 7 | -0.000012 |
| adults and children 12 years and over: take 2 capsules | 5 | -0.000392 |
| with water every 4 hours. Do not exceed 10 capsules | 5 | -0.000392 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_1d2280603a42552a` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004665 | +0.067802 |
| 仅主体分差 | -0.013990 | +0.088872 |
| 主体 token 数 | 156 | 165 |

[对照 token 数据](../photo_1d2280603a42552a/tokens.csv) · [对照摘要](../photo_1d2280603a42552a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
