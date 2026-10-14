# photo_e5b4dba01622bfaf

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001133`，F03 错误参考 `reference_003667`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 3 |
| 图像 Top-1 参考 | reference_003667 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.007887 |
| BM25 正负分差 | -5.541199 |
| 0.5 × 图像 z 分差 | -0.073064 |
| 0.5 × BM25 z 分差 | -0.033929 |
| 最终融合分差 | -0.106993 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 242 个。正向贡献 +0.029438，负向贡献 -0.037325，相加等于图像分差 -0.007887。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 478 | -0.006708 | -0.010258 |
| nonvisual | 11 | +0.000127 | +0.008438 |
| subject_nontext | 34 | +0.000017 | +0.000374 |
| subject_text | 208 | -0.001323 | -0.004651 |

仅保留主体 query token 的两候选分差：-0.003945。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.006708。主体框内 token 占有效序列 33.1%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 786 | 75.8% | True | +4.318378 | 54.4% |
| 错误 | 401 | 71.8% | True | +5.139451 | 58.9% |

主体中 208 个 token 与 OCR 相交，171 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 78 | background | -0.469263 |  |  |  |
| 165 | background | -0.350244 |  |  | Warnings |
| 50 | background | -0.317948 |  |  |  |
| 29 | background | -0.277730 |  | Acetaminophen 160 mg per 5 mL |  |
| 717 | background | -0.258815 |  |  |  |
| 135 | subject_text | -0.250345 | relieves minor aches and pains due to: | relieves minor aches and pains due to: | reduces fever |
| 218 | subject_text | -0.227659 | Symptoms may include:skin reddening blisters rash |  | If a skin reaction occurs, stop use and seek medical help right away. |
| 345 | background | -0.226989 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 220 | subject_nontext | +0.404092 |  | relieves minor aches and pains due to: | more than 5 doses in 24 hours, which is the maximum daily amount |
| 556 | background | +0.354468 |  | syringe tip into bottle opening. |  |
| 273 | subject_text | +0.295647 | drug contains acetaminophen, ask a doctor or pharmacist. | if your child has ever had an allergic reaction to this product or |  |
| 398 | subject_text | +0.291951 | Overdose warning: In case of overdose, get medical help or contact | Overdose warning: In case of overdose, get medical help or contact | For product questions or concerns, contact us |
| 311 | subject_text | +0.285695 | Ask a doctor or pharmacist before use if your child is taking | the blood thinning drug warfarin | When using this product do not exceed recommended dose |
| 411 | subject_text | +0.260721 | a Poison Control Center right away (1-800-222-1222). Quick medical |  | a Poison Control Center right away (1-800-222-1222). Quick medical |
| 257 | subject_text | +0.260272 | Do not use with any other drug containing acetaminophen | Do not use with any other drug containing acetaminophen | if your child has ever had an allergic reaction to this product or |
| 94 | subject_text | +0.258408 | Active ingredient (in each 5 mL) | Active ingredient (in each 5 mL) | 160mg PER 5mL |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Active ingredient (in each 5 mL) | 13 | +0.000092 |
| Acetaminophen 160 mg. | 4 | +0.000139 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_3cfb7f3d355174a8` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.007887 | +0.040263 |
| 仅主体分差 | -0.003945 | +0.062648 |
| 主体 token 数 | 242 | 276 |

[对照 token 数据](../photo_3cfb7f3d355174a8/tokens.csv) · [对照摘要](../photo_3cfb7f3d355174a8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
