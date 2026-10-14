# B03 · photo_0913bcc35e9beb5a

按当前评测真值分析，未新增身份人工复核。

包装 `package_067`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001357`，**纯图像实际 Top-1 错误参考** `reference_000955`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2216 |
| 正确参考 MaxSim | +0.540828 |
| 错误参考 MaxSim | +0.648772 |
| 正确减错误的分差 | -0.107944 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 384 个。正向贡献 +0.015044，负向贡献 -0.122988，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 336 | -0.040185 | -0.087427 |
| nonvisual | 11 | -0.000025 | -0.001646 |
| subject_nontext | 216 | -0.034294 | -0.116061 |
| subject_text | 168 | -0.033440 | -0.145503 |

最主要负差额来自 `background`，贡献 -0.040185。只保留主体 query token 后，本对分差为 -0.128942。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1031 | 56.9% | True | +12.254304 | 90.6% |
| 错误 | 29 | 5.3% | False | 不可评估 | 不可评估 |

主体中 168 个 token 与 OCR 行相交，102 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 674 | background | -0.563675 |  | to prevent athlete's foot, apply once or twice daily (morning and/or night)this product is not effective on the scalp or nails |  |
| 249 | subject_nontext | -0.554083 |  |  |  |
| 633 | background | -0.551981 |  | Keep out of reach of children. If swallowed, get medical help or contact a Poison Control right away. |  |
| 581 | subject_nontext | -0.550060 |  |  |  |
| 569 | subject_nontext | -0.534158 |  |  |  |
| 297 | subject_text | -0.530896 | for athlete's foot: pay special attention to spaces between the toes; wear well-fitting, ventilated shoes and change shoes and socks at least once daily | Keep out of reach of children. If swallowed, get medical help or contact a Poison Control right away. |  |
| 483 | background | -0.513597 |  |  |  |
| 313 | subject_nontext | -0.511277 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 206 | subject_nontext | +0.437908 |  |  | Antifungal Cream |
| 331 | subject_text | +0.380555 | Uses proven cinically eective in the treatment of most athlete's foot (tinea peis) and ringworm (nea corporis)helps prevent most atlete's f with daiy ue |  |  |
| 126 | subject_nontext | +0.337673 |  |  | Antifungal Cream |
| 566 | subject_nontext | +0.189687 |  |  |  |
| 360 | subject_text | +0.186569 | * This product is not manufactured or distributed by Bayer, owner of the registered trademark Tinactin |  | WNED |
| 106 | background | +0.170655 |  |  |  |
| 142 | background | +0.168768 |  |  | Antifungal Cream |
| 190 | subject_nontext | +0.152064 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Tolnaftate 1% | 2 | -0.000081 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_e82154bdf3d3c846` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.107944 | +0.194455 |
| 只保留主体分差 | -0.128942 | +0.226989 |
| 主体 token 数 | 384 | 294 |

[成功对照 token 表](../photo_e82154bdf3d3c846/tokens.csv) · [对照摘要](../photo_e82154bdf3d3c846/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
