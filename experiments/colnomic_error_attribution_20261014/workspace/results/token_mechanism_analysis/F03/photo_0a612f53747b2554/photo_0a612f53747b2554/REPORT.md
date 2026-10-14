# photo_0a612f53747b2554

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000929`，F03 错误参考 `reference_000930`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_002389`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 3 |
| 图像 Top-1 参考 | reference_002389 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.002653 |
| BM25 正负分差 | +3.538269 |
| 0.5 × 图像 z 分差 | -0.031068 |
| 0.5 × BM25 z 分差 | +0.021692 |
| 最终融合分差 | -0.009376 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4111**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 160 个。正向贡献 +0.028843，负向贡献 -0.031495，相加等于图像分差 -0.002653。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 560 | +0.001406 | +0.001835 |
| nonvisual | 11 | -0.000328 | -0.021823 |
| subject_nontext | 28 | -0.000382 | -0.009980 |
| subject_text | 132 | -0.003348 | -0.018540 |

仅保留主体 query token 的两候选分差：-0.017042。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.003348。主体框内 token 占有效序列 21.9%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1100 | 66.8% | True | +6.052051 | 69.2% |
| 错误 | 552 | 59.9% | True | +3.983060 | 55.1% |

主体中 132 个 token 与 OCR 相交，117 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 438 | background | -0.482456 |  | 7 days of starting use or you get new symptoms such as severe facial pain or thick nasal | Week 1 - use 2 sprays in each nostril once daily |
| 409 | subject_text | -0.424557 | Directions read the Quick Start Guide for how to: |  | ircons |
| 389 | subject_text | -0.400125 | If pregnant or breast-feeding, ask a health professional before use. | Do not usein children under 4 years of age |  |
| 449 | subject_text | -0.370827 | Week 2 through 6 months - use 1 or 2 sprays in each nostril once daily, as needed to | Week 1 - use 2 sprays in each nostril once daily |  |
| 398 | background | -0.325772 |  |  | ircons |
| 509 | subject_text | -0.310270 | *This product is not manufactured or distributed by Glaxo Smithkline Consumer | shake gently before each use | an adti use |
| 629 | background | -0.284752 |  |  | Week 1 - use 2 sprays in each nostril once daily |
| 511 | subject_text | -0.278305 | *This product is not manufactured or distributed by Glaxo Smithkline Consumer | do not share this bottle with anyone else as this may spread germs | an adult should supervise use use 1 spray in each nostril once daily |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 615 | background | +0.334783 |  | prime the bottleuse the spraydean the spray nozzle | prime the bottleuse the spraydean the spray nozzle |
| 149 | background | +0.278694 |  |  |  |
| 110 | background | +0.267602 |  |  |  |
| 286 | background | +0.247935 |  |  | NasalSpray |
| 271 | subject_text | +0.233797 | Uses Temporarily relieves these symptoms of hay fever or other upper respiratory allergies: |  | allergens like pollen, mold, dust or pet dander. |
| 649 | background | +0.231595 |  | the growth rate of some children may be slower while using this product. Children |  |
| 124 | background | +0.224651 |  |  |  |
| 47 | background | +0.223802 |  |  |  |

## 成功对照

`photo_cdd1aa422a4f6dc0` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002653 | +0.047626 |
| 仅主体分差 | -0.017042 | +0.067377 |
| 主体 token 数 | 160 | 150 |

[对照 token 数据](../photo_cdd1aa422a4f6dc0/tokens.csv) · [对照摘要](../photo_cdd1aa422a4f6dc0/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
