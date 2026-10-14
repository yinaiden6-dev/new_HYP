# photo_7f3f0a27f2d5c4c3

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000995`，F03 错误参考 `reference_003677`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_003677 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.007253 |
| BM25 正负分差 | +8.344513 |
| 0.5 × 图像 z 分差 | -0.116940 |
| 0.5 × BM25 z 分差 | +0.084971 |
| 最终融合分差 | -0.031969 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4208**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 264 个。正向贡献 +0.021531，负向贡献 -0.028784，相加等于图像分差 -0.007253。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 504 | -0.005189 | -0.008020 |
| nonvisual | 11 | -0.000088 | -0.006266 |
| subject_nontext | 56 | -0.001708 | -0.023766 |
| subject_text | 208 | -0.000267 | -0.001000 |

仅保留主体 query token 的两候选分差：-0.005829。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.005189。主体框内 token 占有效序列 33.9%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 237 | 67.7% | True | +6.602379 | 67.8% |
| 错误 | 12 | 26.8% | True | +14.404310 | 100.0% |

主体中 208 个 token 与 OCR 相交，159 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 172 | background | -0.374941 |  |  | chest pain or shoulder pain with shortness of breath; sweating; pain |
| 541 | subject_text | -0.351687 | Repeated 14-Day Courses (if needed) | do not take for more than 14 days or more | swallow whole. Do not chew, crush, or suck tablets. |
| 573 | subject_text | -0.260353 | Repeated 14-Day Courses (if needed) |  | you may repeat a 14-day course every 4 months |
| 339 | subject_text | -0.247747 | In case of overdose, get medical help or contact a |  | contact a Poison Control Center right away (1-800-222-1222). |
| 478 | subject_text | -0.224493 | you may repeat a 14-day course every 4 months | you may repeat a 14-day course every 4 months | you may repeat a 14-day course every 4 months |
| 493 | subject_text | -0.224243 | taking a prescription drug. Acid reducers may |  | drug. Acid reducers may interact with certain prescription drugs. |
| 566 | subject_text | -0.220925 | 24 hours), every day for 14 days it may take |  | this product is to be used once a day (every 24 hours), every day for |
| 461 | subject_text | -0.213520 | taking a prescription drug. Acid reducers may |  | drug. Acid reducers may interact with certain prescription drugs. |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.404565 | eating in the morning | Lansoprazole 15 mg.. |  |
| 507 | subject_text | +0.401854 | swallow whole. Do not crush or chew capsules. |  | swallow 1 tablet with a glass of water before eating in the morning |
| 347 | subject_text | +0.366098 | swallow whole. Do not crush or chew capsules. |  |  |
| 411 | subject_text | +0.330280 | swallow whole. Do not crush or chew capsules. |  | If an allergic reaction occurs, stop use and seek medical help right away. |
| 281 | subject_text | +0.313626 | swallow 1 capsule with a glass of water before | 14 CAPSULES |  |
| 315 | subject_text | +0.298983 | swallow whole. Do not crush or chew capsules. | Capsules 15 mg |  |
| 472 | subject_text | +0.270922 | 14-Day Course of Treatment | 14-Day Course of Treatment | do not take more than 1 tablet a day |
| 159 | background | +0.253777 | (milbemycin cxome/praziquantel) |  |  |

## 成功对照

`photo_ab6fd7a33337ccee` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.007253 | +0.070055 |
| 仅主体分差 | -0.005829 | +0.171679 |
| 主体 token 数 | 264 | 209 |

[对照 token 数据](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
