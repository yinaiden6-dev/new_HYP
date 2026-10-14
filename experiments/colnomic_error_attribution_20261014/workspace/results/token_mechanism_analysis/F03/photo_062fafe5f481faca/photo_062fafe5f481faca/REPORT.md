# photo_062fafe5f481faca

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001971`，F03 错误参考 `reference_001973`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_001973 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.002851 |
| BM25 正负分差 | -5.033241 |
| 0.5 × 图像 z 分差 | -0.025432 |
| 0.5 × BM25 z 分差 | -0.410252 |
| 最终融合分差 | -0.435683 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 231 个。正向贡献 +0.027931，负向贡献 -0.030782，相加等于图像分差 -0.002851。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 537 | -0.000097 | -0.000141 |
| nonvisual | 11 | -0.000160 | -0.011356 |
| subject_nontext | 95 | +0.000276 | +0.002260 |
| subject_text | 136 | -0.002868 | -0.016430 |

仅保留主体 query token 的两候选分差：-0.008744。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.002868。主体框内 token 占有效序列 29.7%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 290 | 74.7% | True | +6.138643 | 63.6% |
| 错误 | 43 | 17.1% | True | +6.056289 | 85.6% |

主体中 136 个 token 与 OCR 相交，32 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 17 | background | -0.465384 |  | (*capsule-shapedtablets) |  |
| 425 | background | -0.421009 |  |  |  |
| 528 | subject_text | -0.403711 | ("capsule-shapedtablets) |  | CAPLETS* |
| 431 | subject_text | -0.357139 | CONT AINS 2 MEDICINES - Acetaminophen + Ituprofen • DUALACTION |  |  |
| 721 | background | -0.350306 |  |  |  |
| 163 | background | -0.347216 |  |  |  |
| 73 | background | -0.338700 |  | •Acetaminophen+Ibuprofen •DUALACTION | Acto ingrodlent (it euch capix) |
| 16 | background | -0.336689 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 461 | subject_text | +0.351552 | 144 | 144 | 216 |
| 493 | subject_text | +0.326196 | 144 | 144 | 216 |
| 71 | background | +0.322543 |  | 144 | 216 |
| 462 | subject_text | +0.287290 | 144 | 144 |  |
| 212 | subject_nontext | +0.247747 |  |  |  |
| 267 | subject_nontext | +0.242654 |  |  |  |
| 750 | background | +0.232826 |  | snachbeedngwaningapplistoyou |  |
| 438 | subject_text | +0.227567 | CONT AINS 2 MEDICINES - Acetaminophen + Ituprofen • DUALACTION |  | -Acetaminophen + Ibuprofen |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| ACE TAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | 50 | -0.002276 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_ca454f8a9d866abb` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002851 | +0.001656 |
| 仅主体分差 | -0.008744 | -0.010628 |
| 主体 token 数 | 231 | 264 |

[对照 token 数据](../photo_ca454f8a9d866abb/tokens.csv) · [对照摘要](../photo_ca454f8a9d866abb/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
