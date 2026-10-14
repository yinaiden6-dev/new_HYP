# B03 · photo_9bb993b443dfe2ff

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_003797`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.645726 |
| 错误参考 MaxSim | +0.659627 |
| 正确减错误的分差 | -0.013901 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 156 个。正向贡献 +0.026786，负向贡献 -0.040687，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 564 | -0.011299 | -0.014644 |
| nonvisual | 11 | +0.000055 | +0.003651 |
| subject_nontext | 119 | -0.001003 | -0.006160 |
| subject_text | 37 | -0.001655 | -0.032692 |

最主要负差额来自 `background`，贡献 -0.011299。只保留主体 query token 后，本对分差为 -0.012453。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 125 | 22.6% | True | +8.109859 | 71.7% |
| 错误 | 55 | 3.8% | False | 不可评估 | 不可评估 |

主体中 37 个 token 与 OCR 行相交，24 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | subject_text | -0.599109 | This product is not manufactured or distributed | Drug Facts (continued) |  |
| 350 | subject_text | -0.549424 | This product is not manufactured or distributed | "This product is not manufactured or distributed |  |
| 231 | background | -0.474546 |  | Drug Facts (continued) |  |
| 230 | background | -0.447269 |  | hypromellose, isopropyl alcoho, microcrystaline |  |
| 498 | subject_nontext | -0.370986 |  | Drug Facts (continued) | 200 WILMOT RD. DEERELD, IL 60015 |
| 392 | subject_nontext | -0.369024 |  | call 1-877-770-3183 Mon-Fri 8:00 AM EST to 5:00 PM PST | Walgreens Pharmacist Survey |
| 171 | background | -0.367987 |  |  |  |
| 612 | background | -0.341108 |  |  | 200 WILMOT RD. DEERELD, IL 60015 |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 399 | subject_text | +0.506291 | walgreens.com ©2020 Walgreen Co. | ITEM 696611W10383-0420-L | temporarilly rellieves minor aches and pains due to: |
| 599 | background | +0.455624 |  | ITEM 696611W10383-0420-L |  |
| 401 | background | +0.444768 |  | ITEM 696611W10383-0420-L |  |
| 600 | background | +0.425222 |  | ITEM 696611W10383-0420-L |  |
| 373 | subject_text | +0.425036 | Tylenol• and Extra Strength Tylenol• PM. | Compare to the active ingredlents |  |
| 659 | background | +0.412877 |  | ITEM 696611W10383-0420-L |  |
| 254 | background | +0.320946 |  | Reliever PM | Overdose warning: In case of overdose, get medical hellp or |
| 589 | background | +0.302481 |  | Exp. | Exp. |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% | 3 | +0.000155 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_77b67ea38a19f6b3` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013901 | +0.079191 |
| 只保留主体分差 | -0.012453 | +0.105753 |
| 主体 token 数 | 156 | 143 |

[成功对照 token 表](../photo_77b67ea38a19f6b3/tokens.csv) · [对照摘要](../photo_77b67ea38a19f6b3/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
