# photo_1db26288dca18777

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001971`，F03 错误参考 `reference_001973`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001971`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001971 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.001554 |
| BM25 正负分差 | -7.616821 |
| 0.5 × 图像 z 分差 | +0.018506 |
| 0.5 × BM25 z 分差 | -0.620718 |
| 最终融合分差 | -0.602212 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.9710**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 476 个。正向贡献 +0.028806，负向贡献 -0.027252，相加等于图像分差 +0.001554。缓存复现误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 292 | +0.003720 | +0.009923 |
| nonvisual | 11 | -0.000049 | -0.003477 |
| subject_nontext | 206 | +0.003753 | +0.014192 |
| subject_text | 270 | -0.005869 | -0.016934 |

仅保留主体 query token 的两候选分差：-0.003463。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.005869。主体框内 token 占有效序列 61.1%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 257 | 54.2% | True | +7.116487 | 66.4% |
| 错误 | 42 | 17.1% | True | +6.576807 | 87.0% |

主体中 270 个 token 与 OCR 相交，72 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 144 | background | -0.393573 |  | •Acetaminophen+Ibuprofen •DUALACTION | Acto ingrodlent (it euch capix) |
| 498 | subject_text | -0.343157 | CONTAINS 2 MEDICINES |  |  |
| 497 | subject_text | -0.333784 | CONTAINS 2 MEDICINES | 1 |  |
| 527 | subject_text | -0.324096 | CONTAINS 2 MEDICINES |  |  |
| 420 | background | -0.322984 |  |  |  |
| 496 | subject_text | -0.322365 | CONTAINS 2 MEDICINES |  |  |
| 132 | background | -0.313173 |  |  |  |
| 528 | subject_text | -0.294236 | CONTAINS 2 MEDICINES |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 556 | subject_text | +0.423114 | 144 | 144 |  |
| 557 | subject_text | +0.374669 | 144 | 144 |  |
| 76 | background | +0.325402 |  |  |  |
| 266 | subject_nontext | +0.289409 |  |  |  |
| 193 | subject_nontext | +0.268322 |  |  |  |
| 587 | subject_text | +0.267875 | 144 | snachbeedngwaningapplistoyou | 216 |
| 538 | subject_text | +0.259355 | • Acetaminophen + Ibuprofen • DUALACTION |  | -Acetaminophen + Ibuprofen |
| 87 | background | +0.241265 |  | 144 |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| ACE TAMINOPHEN 250 mg & IBUPROFEN (NS AID) 125 mg TAE | 79 | -0.001591 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_ca454f8a9d866abb` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.001554 | +0.001656 |
| 仅主体分差 | -0.003463 | -0.010628 |
| 主体 token 数 | 476 | 264 |

[对照 token 数据](../photo_ca454f8a9d866abb/tokens.csv) · [对照摘要](../photo_ca454f8a9d866abb/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
