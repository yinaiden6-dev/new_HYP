# B03 · photo_7f3f0a27f2d5c4c3

按当前评测真值分析，未新增身份人工复核。

包装 `package_026`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000995`，**纯图像实际 Top-1 错误参考** `reference_003677`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.575587 |
| 错误参考 MaxSim | +0.582839 |
| 正确减错误的分差 | -0.007253 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 264 个。正向贡献 +0.021531，负向贡献 -0.028784，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 504 | -0.005189 | -0.008020 |
| nonvisual | 11 | -0.000088 | -0.006266 |
| subject_nontext | 56 | -0.001708 | -0.023766 |
| subject_text | 208 | -0.000267 | -0.001000 |

最主要负差额来自 `background`，贡献 -0.005189。只保留主体 query token 后，本对分差为 -0.005829。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 237 | 67.7% | True | +6.602379 | 67.8% |
| 错误 | 12 | 26.8% | True | +14.404310 | 100.0% |

主体中 208 个 token 与 OCR 行相交，159 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 172 | background | -0.374941 |  |  | chest pain or shoulder pain with shortness of breath; sweating; pain |
| 541 | subject_text | -0.351687 | Repeated 14-Day Courses (if needed) | do not take for more than 14 days or more | swallow whole. Do not chew, crush, or suck tablets. |
| 573 | subject_text | -0.260353 | Repeated 14-Day Courses (if needed) |  | you may repeat a 14-day course every 4 months |
| 339 | subject_text | -0.247747 | In case of overdose, get medical help or contact a |  | contact a Poison Control Center right away (1-800-222-1222). |
| 478 | subject_text | -0.224493 | you may repeat a 14-day course every 4 months | you may repeat a 14-day course every 4 months | you may repeat a 14-day course every 4 months |
| 493 | subject_text | -0.224243 | taking a prescription drug. Acid reducers may |  | drug. Acid reducers may interact with certain prescription drugs. |
| 566 | subject_text | -0.220925 | 24 hours), every day for 14 days it may take |  | this product is to be used once a day (every 24 hours), every day for |
| 461 | subject_text | -0.213520 | taking a prescription drug. Acid reducers may |  | drug. Acid reducers may interact with certain prescription drugs. |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.404565 | eating in the morning | Lansoprazole 15 mg.. |  |
| 507 | subject_text | +0.401854 | swallow whole. Do not crush or chew capsules. |  | swallow 1 tablet with a glass of water before eating in the morning |
| 347 | subject_text | +0.366098 | swallow whole. Do not crush or chew capsules. |  |  |
| 411 | subject_text | +0.330280 | swallow whole. Do not crush or chew capsules. |  | If an allergic reaction occurs, stop use and seek medical help right away. |
| 281 | subject_text | +0.313626 | swallow 1 capsule with a glass of water before | 14 CAPSULES |  |
| 315 | subject_text | +0.298983 | swallow whole. Do not crush or chew capsules. | Capsules 15 mg |  |
| 472 | subject_text | +0.270922 | 14-Day Course of Treatment | 14-Day Course of Treatment | do not take more than 1 tablet a day |
| 159 | background | +0.253777 | (milbemycin cxome/praziquantel) |  |  |

## 同包装成功对照

`photo_ab6fd7a33337ccee` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.007253 | +0.070055 |
| 只保留主体分差 | -0.005829 | +0.171679 |
| 主体 token 数 | 264 | 209 |

[成功对照 token 表](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
