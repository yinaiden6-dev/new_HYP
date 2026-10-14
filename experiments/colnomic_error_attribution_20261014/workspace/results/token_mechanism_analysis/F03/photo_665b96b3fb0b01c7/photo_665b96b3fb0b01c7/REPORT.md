# photo_665b96b3fb0b01c7

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000229`，F03 错误参考 `reference_000233`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000233 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.009928 |
| BM25 正负分差 | -0.156189 |
| 0.5 × 图像 z 分差 | -0.250283 |
| 0.5 × BM25 z 分差 | -0.003029 |
| 最终融合分差 | -0.253313 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 216 个。正向贡献 +0.019883，负向贡献 -0.029811，相加等于图像分差 -0.009928。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 552 | -0.008602 | -0.012139 |
| nonvisual | 11 | -0.000084 | -0.005915 |
| subject_nontext | 55 | -0.000265 | -0.003749 |
| subject_text | 161 | -0.000978 | -0.004734 |

仅保留主体 query token 的两候选分差：-0.004483。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.008602。主体框内 token 占有效序列 27.7%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 156 | 47.4% | True | +8.294819 | 74.6% |
| 错误 | 16 | 24.5% | True | +8.728093 | 98.6% |

主体中 161 个 token 与 OCR 相交，132 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 239 | subject_nontext | -0.413493 |  |  |  |
| 207 | subject_nontext | -0.408473 |  | drug to use by or on the order of a |  |
| 435 | subject_text | -0.393018 | The safe use of enrofloxacin has not |  | The safe use of enrofloxacin has not been |
| 550 | background | -0.356318 |  |  | technical assistance or to obtain a copy of the Safety |
| 461 | subject_text | -0.355435 | (IM) as a single injection at the rate of 1 | For the treatment of Susceptible |  |
| 230 | background | -0.329578 |  |  | technical assistance or to obtain a copy of the Safety |
| 621 | background | -0.328187 |  | DOSAGE: Administer intramuscularly | DOSAGE: Administer intramuscularly (IM) as a |
| 326 | background | -0.326591 |  |  | technical assistance or to obtain a copy of the Safety |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 128 | background | +0.501170 |  |  |  |
| 342 | subject_text | +0.420191 | obtain a copy of the Safety Data Sheet, |  |  |
| 365 | subject_text | +0.338125 | (IM) as a single injection at the rate of 1 |  | to provide 2.5 mg/kg (1.13 mg/lb). The injectable |
| 462 | subject_text | +0.225552 | (1.13 mg/lb). The injectable dose | mL/9.1 kg (20 lb) to provide 2.5 mg/kg |  |
| 333 | subject_text | +0.216148 | mL/9.1 kg (20 lb) to provide 2.5 mg/kg |  |  |
| 364 | subject_text | +0.211475 | DOSAGE: Administer intramuscularly | DOSAGE: Administer intramuscularly |  |
| 60 | background | +0.172925 |  |  |  |
| 92 | background | +0.169924 |  |  | dogs during the rapid growth phase (between 2 |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| enrofloxacin 22.7 mg, n-butyl alcohol | 16 | +0.000851 |
| 30 mg, potassium hydroxide for pH | 16 | -0.000039 |
| mL/9.1 kg (20 lb) to provide 2.5 mg/kg | 13 | +0.000868 |
| (1.13 mg/lb). The injectable dose | 15 | +0.000496 |
| Consultpackage insert fordetails. | 10 | -0.000091 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_20594074f229bd8f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.009928 | +0.001343 |
| 仅主体分差 | -0.004483 | +0.000349 |
| 主体 token 数 | 216 | 99 |

[对照 token 数据](../photo_20594074f229bd8f/tokens.csv) · [对照摘要](../photo_20594074f229bd8f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。

![末尾文字局部](focal_text.png)
