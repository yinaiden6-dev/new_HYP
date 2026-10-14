# photo_41de3eb383934ec3

按当前真值分析，未新增身份复核。 用户指定主题：换行与末尾词位置。

正确参考 `reference_000229`，F03 错误参考 `reference_000233`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000233 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.004094 |
| BM25 正负分差 | -13.054565 |
| 0.5 × 图像 z 分差 | -0.105170 |
| 0.5 × BM25 z 分差 | -0.168726 |
| 最终融合分差 | -0.273896 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 144 个。正向贡献 +0.019874，负向贡献 -0.023968，相加等于图像分差 -0.004094。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 624 | -0.001229 | -0.001535 |
| nonvisual | 11 | +0.000013 | +0.000933 |
| subject_nontext | 19 | +0.000246 | +0.010086 |
| subject_text | 125 | -0.003124 | -0.019471 |

仅保留主体 query token 的两候选分差：-0.015571。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.003124。主体框内 token 占有效序列 18.5%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 224 | 61.8% | True | +6.967820 | 65.5% |
| 错误 | 16 | 27.3% | True | +8.516190 | 97.9% |

主体中 125 个 token 与 OCR 相交，109 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 439 | subject_text | -0.469062 | breeds during the rapid growth phase. |  | The safe use of enrofloxacin has not been |
| 211 | background | -0.378257 |  | Consult package insert for details. |  |
| 377 | subject_text | -0.360238 | events, for technical assistance or to |  | technical assistance or to obtain a copy of the Safety |
| 433 | subject_text | -0.357504 | (IM) as a single injection at the rate of 1 |  |  |
| 441 | subject_text | -0.345618 | events, for technical assistance or to | contact Cronus Pharma LLC at 1-844- | technical assistance or to obtain a copy of the Safety |
| 428 | background | -0.312115 |  |  | to provide 2.5 mg/kg (1.13 mg/lb). The injectable |
| 382 | subject_text | -0.287586 | 50.1-100 lbs | 2 | 50 mL |
| 626 | background | -0.262356 |  | DOSAGE: Administer intramuscularly | DOSAGE: Administer intramuscularly (IM) as a |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 346 | subject_text | +0.409373 | contact Cronus Pharma LLC at 1-844- |  |  |
| 143 | background | +0.272247 |  |  |  |
| 284 | subject_text | +0.271976 | drugs, contact FDA at 1-888-FDA-VETS | drug experience reporting for animal |  |
| 464 | subject_text | +0.241081 | adjustmentand waterforinjection, q.s. |  | enrofloxacin 22.7 mg, n-butyl alcohol 30 mg, |
| 369 | subject_text | +0.216708 | (IM) as a single injection at the rate of 1 | DOSAGE: Administer intramuscularly |  |
| 764 | background | +0.182324 |  | drug experience reporting for animal |  |
| 476 | subject_text | +0.163985 | orhttp://www.fda.gov/reportanimalae | drug experience reporting for animal |  |
| 664 | background | +0.153906 |  | Consult package insert for details. | rapid growth phase. Consult package insert for |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| enrofloxacin 22.7 mg, n-butyi alcohol | 8 | +0.000017 |
| 30 mg, potassium hydroxide for pH | 16 | -0.000035 |
| mL/9.1 kg (20 lb) to provide 2.5 mg/kg | 16 | +0.000122 |
| (1.13 mg/lb). The injectable dose | 8 | +0.000090 |
| Consultpackageinsertfordetails. | 14 | -0.001363 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_20594074f229bd8f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004094 | +0.001343 |
| 仅主体分差 | -0.015571 | +0.000349 |
| 主体 token 数 | 144 | 99 |

[对照 token 数据](../photo_20594074f229bd8f/tokens.csv) · [对照摘要](../photo_20594074f229bd8f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。

![末尾文字局部](focal_text.png)
