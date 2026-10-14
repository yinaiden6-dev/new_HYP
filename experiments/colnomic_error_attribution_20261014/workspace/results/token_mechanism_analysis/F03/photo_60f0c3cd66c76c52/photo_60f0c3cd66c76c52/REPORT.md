# photo_60f0c3cd66c76c52

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001007`，F03 错误参考 `reference_003109`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001007`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001007 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 3 |
| 图像正负分差 | +0.000638 |
| BM25 正负分差 | -38.263779 |
| 0.5 × 图像 z 分差 | +0.014371 |
| 0.5 × BM25 z 分差 | -1.547079 |
| 最终融合分差 | -1.532707 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.9908**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 192 个。正向贡献 +0.048668，负向贡献 -0.048030，相加等于图像分差 +0.000638。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 528 | +0.003895 | +0.005392 |
| nonvisual | 11 | +0.000668 | +0.044421 |
| subject_nontext | 93 | -0.000763 | -0.005998 |
| subject_text | 99 | -0.003162 | -0.023346 |

仅保留主体 query token 的两候选分差：-0.014943。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.003162。主体框内 token 占有效序列 26.3%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 114 | 30.4% | True | +9.095827 | 75.0% |
| 错误 | 10 | 12.7% | False | 不可评估 | 不可评估 |

主体中 99 个 token 与 OCR 相交，59 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 323 | background | -0.384785 |  |  |  |
| 29 | background | -0.371270 |  |  |  |
| 615 | background | -0.357252 |  |  |  |
| 429 | subject_text | -0.355505 | where proofof age cannot be verified. |  | age cannot be verified |
| 606 | background | -0.352754 |  |  |  |
| 328 | subject_nontext | -0.343486 |  | where proofof age cannot be verified. |  |
| 47 | background | -0.327181 |  | 1. You must be motivated to quit. |  |
| 421 | background | -0.319352 |  | each lacenge conlairs:sodiun, 17 ing |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 468 | subject_nontext | +0.534333 |  | waking e p, use 2 mg nicolfine lozange |  |
| 712 | background | +0.530228 |  | waking e p, use 2 mg nicolfine lozange |  |
| 686 | background | +0.529436 |  |  |  |
| 632 | background | +0.527387 |  | waking e p, use 2 mg nicolfine lozange |  |
| 413 | subject_text | +0.525572 | 3. Use Long Enough - Use Habitrol lozenges for the | Habitrol | Nicotine Lozenge |
| 649 | background | +0.511071 |  |  |  |
| 723 | background | +0.510930 |  |  |  |
| 716 | background | +0.506689 |  |  |  |

## 成功对照

`photo_9a8d63f5711209a8` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.000638 | +0.011489 |
| 仅主体分差 | -0.014943 | -0.002273 |
| 主体 token 数 | 192 | 228 |

[对照 token 数据](../photo_9a8d63f5711209a8/tokens.csv) · [对照摘要](../photo_9a8d63f5711209a8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
