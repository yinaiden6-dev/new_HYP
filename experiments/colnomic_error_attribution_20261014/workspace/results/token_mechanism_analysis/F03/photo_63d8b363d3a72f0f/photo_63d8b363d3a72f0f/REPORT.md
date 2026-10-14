# photo_63d8b363d3a72f0f

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000995`，F03 错误参考 `reference_000852`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_002203`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 4 |
| 图像 Top-1 参考 | reference_002203 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.001206 |
| BM25 正负分差 | -70.363708 |
| 0.5 × 图像 z 分差 | -0.037500 |
| 0.5 × BM25 z 分差 | -1.739295 |
| 最终融合分差 | -1.776796 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 195 个。正向贡献 +0.025748，负向贡献 -0.026955，相加等于图像分差 -0.001206。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 573 | +0.000226 | +0.000307 |
| nonvisual | 11 | +0.000167 | +0.011793 |
| subject_nontext | 72 | -0.000003 | -0.000029 |
| subject_text | 123 | -0.001596 | -0.010107 |

仅保留主体 query token 的两候选分差：-0.006386。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.001596。主体框内 token 占有效序列 25.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 152 | 47.1% | True | +5.960571 | 64.9% |
| 错误 | 90 | 8.6% | False | 不可评估 | 不可评估 |

主体中 123 个 token 与 OCR 相交，89 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 40 | background | -0.439932 |  | basiccare | Walgreens |
| 551 | background | -0.374859 | mian | 1 BOTTLE |  |
| 300 | background | -0.366728 |  | Do not eat late at night or just before bedtime. | Tips for Managing Heartburn |
| 552 | background | -0.360085 | mian |  |  |
| 519 | background | -0.343019 | mian | If pregnant or breast-feeding, ask a health care | Walgreens |
| 41 | background | -0.313339 |  | One 14-Day Course of Treatment | Walgreens |
| 280 | subject_text | -0.310535 | how2recycle_info |  | how2recycle,info |
| 376 | subject_text | -0.303595 | PAPER | BOX |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 150 | background | +0.345056 | ALLERGYRELIEF | Allergy alert: |  |
| 118 | background | +0.332611 | ALLERGYRELIEF | Allergy alert: |  |
| 532 | subject_text | +0.221852 | Wear loose-fitting clothing around | • Raise the head of your bed | Wear loose fitting clothing around |
| 197 | background | +0.195963 |  |  |  |
| 43 | background | +0.193299 | 1713 |  |  |
| 121 | background | +0.187402 |  |  |  |
| 42 | background | +0.181059 |  |  |  |
| 734 | background | +0.179135 | COMPLETE PF | One 14-Day Course of Treatment |  |

## 成功对照

`photo_ab6fd7a33337ccee` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001206 | +0.087824 |
| 仅主体分差 | -0.006386 | +0.211470 |
| 主体 token 数 | 195 | 209 |

[对照 token 数据](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
