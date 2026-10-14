# photo_578a5c367aa237fe

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_000512`，F03 错误参考 `reference_000510`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000337`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 16 |
| 图像 Top-1 参考 | reference_000337 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.025228 |
| BM25 正负分差 | -29.325012 |
| 0.5 × 图像 z 分差 | -0.436813 |
| 0.5 × BM25 z 分差 | -0.738811 |
| 最终融合分差 | -1.175625 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 170 个。正向贡献 +0.011028，负向贡献 -0.036256，相加等于图像分差 -0.025228。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 550 | -0.018363 | -0.024406 |
| nonvisual | 11 | -0.000369 | -0.024498 |
| subject_nontext | 31 | -0.001897 | -0.044744 |
| subject_text | 139 | -0.004599 | -0.024185 |

仅保留主体 query token 的两候选分差：-0.027934。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.018363。主体框内 token 占有效序列 23.3%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 963 | 68.8% | True | +9.921557 | 88.2% |
| 错误 | 825 | 69.3% | True | +10.748172 | 85.9% |

主体中 139 个 token 与 OCR 相交，95 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 287 | subject_text | -0.265361 | HUMAN WARNINGS:Keepoutofreachofchildren. Notforhuman use. | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. | HUMAN WARNINGS: Keepoutofreachofctildren. Notforhuman use. |
| 616 | background | -0.265282 |  |  |  |
| 288 | subject_text | -0.253411 | HUMAN WARNINGS:Keepoutofreachofchildren. Notforhuman use. | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. | HUMAN WARNINGS: Keepoutofreachofctildren. Notforhuman use. |
| 657 | background | -0.245877 |  | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. |  |
| 268 | background | -0.241103 |  | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. | HUMAN WARNINGS: Keepoutofreachofctildren. Notforhuman use. |
| 120 | background | -0.237777 |  |  | S tbre at c ontrolled room temper ature 25C (77°F) with exc ursions tetwen 15 °-3 0C (59º -86° F) permited. |
| 180 | background | -0.233896 |  | FREE | and early pregnancy loss. |
| 304 | background | -0.233218 |  | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 144 | background | +0.203697 |  |  |  |
| 133 | background | +0.148460 |  | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide tfe capsules. Do not hardle the |  |
| 151 | background | +0.122128 |  | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide tfe capsules. Do not hardle the |  |
| 614 | background | +0.116753 |  |  |  |
| 324 | background | +0.109741 |  |  |  |
| 164 | background | +0.096200 |  | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. |  |
| 13 | background | +0.094082 |  |  | and early pregnancy loss. |
| 52 | background | +0.090002 |  |  |  |

## 成功对照

`photo_b111a943691a2aa7` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.025228 | +0.004759 |
| 仅主体分差 | -0.027934 | +0.006582 |
| 主体 token 数 | 170 | 190 |

[对照 token 数据](../photo_b111a943691a2aa7/tokens.csv) · [对照摘要](../photo_b111a943691a2aa7/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
