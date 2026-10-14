# B03 · photo_59d5e8c638df29f0

按当前评测真值分析，未新增身份人工复核。

包装 `package_054`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001401`，**纯图像实际 Top-1 错误参考** `reference_003429`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 114 |
| 正确参考 MaxSim | +0.615038 |
| 错误参考 MaxSim | +0.689737 |
| 正确减错误的分差 | -0.074698 |
| F03 的正确参考排名（辅助对照） | 23 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 221 个。正向贡献 +0.022571，负向贡献 -0.097269，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 499 | -0.050650 | -0.074199 |
| nonvisual | 11 | -0.000861 | -0.057200 |
| subject_nontext | 81 | -0.005719 | -0.051613 |
| subject_text | 140 | -0.017468 | -0.091208 |

最主要负差额来自 `background`，贡献 -0.050650。只保留主体 query token 后，本对分差为 -0.076696。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1159 | 77.3% | True | +8.526282 | 71.8% |
| 错误 | 8 | 1.4% | False | 不可评估 | 不可评估 |

主体中 140 个 token 与 OCR 行相交，98 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 296 | subject_text | -0.577039 | Reye's syndrome: Chidren and fenagers who have or are recoverng rum chicken pox or fu-lis symptoms should not | ORIGINAL | who have or are recovering from chicken pox |
| 303 | background | -0.560014 |  |  | consult a doctor because these symptoms |
| 309 | subject_text | -0.514283 | use this product When using ths product I changes in behavior wth nausea and vomiting cotur consut a doctor | Reye's syndrome: Children and teenagers who have or are recovering from chicken pox or flu-like symptoms should not | rare but serious illness. |
| 130 | background | -0.510751 |  |  |  |
| 90 | background | -0.488431 |  | upset stomach with headache from overindulgence in food or drink |  |
| 669 | background | -0.470836 |  |  |  |
| 553 | background | -0.427819 |  | Seltzer | If pregnant or breast-feeding, ask a health |
| 501 | background | -0.417981 |  | problems in the unborn child or complications during delivery. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 213 | subject_nontext | +0.415483 |  |  |  |
| 412 | subject_text | +0.308291 | you have high blood pressure, heart discase, Iver ornhosis, or kidney disease |  you have a history of stomach problems, such as heartburn | heart disease, liver cirrhosis, or kidney disease |
| 284 | background | +0.281881 |  |  | Uses for the temporary relief of minor |
| 415 | subject_text | +0.280466 | you have high blood pressure, heart discase, Iver ornhosis, or kidney disease |  | heart disease, liver cirrhosis, or kidney disease |
| 242 | background | +0.263552 |  | Sodium bicarbonate (heat-treated) 1916 mg |  |
| 540 | subject_nontext | +0.257032 |  |  |  |
| 92 | background | +0.256870 |  |  | ibuprofen, naproxen, or others)have 3 or |
| 395 | subject_text | +0.253468 | if you have ever had an alergc reaction to this product or any of ts angradents | if you have ever had an allergic reaction to this product or any of its ingredients | Bi |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Anthydrous citric acid 1000 mg | 4 | +0.000039 |
| Aspirin 325 mg (NSAID) | 5 | +0.000094 |
| Sodium bicartborate (heat-treated) 1916 mg | 5 | +0.000446 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_4f2addadc5937695` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.074698 | +0.199863 |
| 只保留主体分差 | -0.076696 | +0.251257 |
| 主体 token 数 | 221 | 221 |

[成功对照 token 表](../photo_4f2addadc5937695/tokens.csv) · [对照摘要](../photo_4f2addadc5937695/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
