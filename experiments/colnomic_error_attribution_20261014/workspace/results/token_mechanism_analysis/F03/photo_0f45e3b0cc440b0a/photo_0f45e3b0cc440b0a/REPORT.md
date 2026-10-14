# photo_0f45e3b0cc440b0a

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000973`，F03 错误参考 `reference_004475`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003726`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 678 |
| 图像 Top-1 参考 | reference_003726 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 3 |
| 图像正负分差 | -0.044576 |
| BM25 正负分差 | -14.245529 |
| 0.5 × 图像 z 分差 | -1.138033 |
| 0.5 × BM25 z 分差 | -1.540541 |
| 最终融合分差 | -2.678574 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 100 个。正向贡献 +0.036426，负向贡献 -0.081002，相加等于图像分差 -0.044576。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 620 | -0.037457 | -0.044163 |
| nonvisual | 11 | -0.000306 | -0.020356 |
| subject_nontext | 51 | +0.000816 | +0.011689 |
| subject_text | 49 | -0.007628 | -0.113802 |

仅保留主体 query token 的两候选分差：-0.049802。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.037457。主体框内 token 占有效序列 13.7%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 0 | 0.0% | False | 不可评估 | 不可评估 |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 49 个 token 与 OCR 相交，37 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 313 | subject_text | -0.496220 | NOT100%SATISFIED? | COHAPARS TO ALEVE* CAPLETS | DISTRIBUTED BY: MIDWOOD BRANDS, LLC |
| 322 | background | -0.483255 |  | aspirin, Increase ths risk of heart attack, heart | DISTRIBUTED BY: MIDWOOD BRANDS, LLC |
| 693 | background | -0.442900 |  |  |  |
| 498 | background | -0.442441 |  | gluten-containing grain (wheat, barley |  |
| 336 | subject_text | -0.439995 | within 30 days to any Family Dollar store |  | Family Dollar store for a refund (with receipt) or exchange. |
| 546 | background | -0.439632 |  |  shock = akdn reddening  raeh = blstore |  |
| 526 | background | -0.436964 |  |  shock = akdn reddening  raeh = blstore |  |
| 716 | background | -0.436543 |  |  | wellness |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 230 | subject_nontext | +0.358594 |  |  |  |
| 234 | subject_nontext | +0.325819 |  |  |  |
| 277 | subject_nontext | +0.321737 |  |  |  |
| 237 | subject_nontext | +0.312753 |  |  |  |
| 449 | background | +0.291946 |  |  |  |
| 566 | background | +0.291171 |  |  |  |
| 413 | subject_nontext | +0.287493 |  |  |  |
| 378 | subject_nontext | +0.275551 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| NOT100%SATISFIED? | 6 | -0.001748 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_38d1d33620ab1ea1` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.044576 | +0.111841 |
| 仅主体分差 | -0.049802 | +0.119547 |
| 主体 token 数 | 100 | 54 |

[对照 token 数据](../photo_38d1d33620ab1ea1/tokens.csv) · [对照摘要](../photo_38d1d33620ab1ea1/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
