# photo_bc5564b0e01eeb18

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000229`，F03 错误参考 `reference_000233`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000233 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.001996 |
| BM25 正负分差 | -12.197815 |
| 0.5 × 图像 z 分差 | -0.054025 |
| 0.5 × BM25 z 分差 | -0.184579 |
| 最终融合分差 | -0.238603 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 136 个。正向贡献 +0.020258，负向贡献 -0.022254，相加等于图像分差 -0.001996。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 632 | -0.001035 | -0.001275 |
| nonvisual | 11 | -0.000074 | -0.005235 |
| subject_nontext | 29 | -0.001024 | -0.027510 |
| subject_text | 107 | +0.000137 | +0.000995 |

仅保留主体 query token 的两候选分差：-0.005083。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.001035。主体框内 token 占有效序列 17.5%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 189 | 58.0% | True | +6.711442 | 61.8% |
| 错误 | 55 | 33.3% | True | +7.353903 | 65.2% |

主体中 107 个 token 与 OCR 相交，74 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 504 | subject_text | -0.408278 | East Brunswick, NJ 08816. | Contact No: 1-844-227-6687 |  |
| 443 | subject_text | -0.274143 | August 2024 | Code: 4206162/TS/DRUGS/2023 |  |
| 343 | subject_nontext | -0.248309 |  |  |  |
| 280 | subject_nontext | -0.240779 |  |  |  |
| 345 | subject_text | -0.233403 | Contact No:1-844-227-6687 |  | Code: 4206162/TS/DRUGS/2023 |
| 382 | background | -0.232950 | 50.1-100 lbs |  | 50 mL |
| 500 | subject_text | -0.213826 | more than 90 days after initial puncture |  |  |
| 474 | subject_text | -0.207852 | Code:4206162/TS/DRUGS/2023 |  | August 2024 |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 354 | background | +0.403134 |  NG MANDG LLG |  |  |
| 270 | subject_text | +0.367741 | The Use of this Product in Cats may result |  | The Use of this Product in Cats may result in |
| 408 | subject_text | +0.361686 | East Brunswick, NJ 08816. |  |  |
| 173 | background | +0.320777 |  | puncture a maximum of 20 times. Any |  |
| 375 | subject_text | +0.300485 | Cronus Pharma LLC, |  |  |
| 367 | subject_text | +0.268055 | in Retinal Toxicity. Keep out of reach of | The Use of this Product in Cats may result |  |
| 274 | subject_text | +0.219809 | Use within 90 days of first puncture and |  | a maximum of 50 times. Any product remaining |
| 498 | subject_text | +0.212104 | (59-86°F). |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| complete details. | 4 | -0.000227 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_20594074f229bd8f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001996 | +0.001343 |
| 仅主体分差 | -0.005083 | +0.000349 |
| 主体 token 数 | 136 | 99 |

[对照 token 数据](../photo_20594074f229bd8f/tokens.csv) · [对照摘要](../photo_20594074f229bd8f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。

![末尾文字局部](focal_text.png)
