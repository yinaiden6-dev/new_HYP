# B03 · photo_3035bdfaed71d4db

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_004350`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 44 |
| 正确参考 MaxSim | +0.629171 |
| 错误参考 MaxSim | +0.654968 |
| 正确减错误的分差 | -0.025797 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 162 个。正向贡献 +0.019797，负向贡献 -0.045594，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.019295 | -0.025277 |
| nonvisual | 11 | -0.000041 | -0.002745 |
| subject_nontext | 66 | -0.002119 | -0.023473 |
| subject_text | 96 | -0.004341 | -0.033059 |

最主要负差额来自 `background`，贡献 -0.019295。只保留主体 query token 后，本对分差为 -0.029153。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 242 | 55.2% | True | +13.126230 | 94.3% |
| 错误 | 7 | 17.4% | False | 不可评估 | 不可评估 |

主体中 96 个 token 与 OCR 行相交，65 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 137 | background | -0.385093 |  |  |  |
| 378 | subject_nontext | -0.372879 |  |  | For the temporary relief of minor arthritis pain |
| 256 | subject_text | -0.365103 | minor pain of arthritis | Pain | minor pain of arthritis |
| 276 | subject_text | -0.360414 | minor pain of arthritis | Pain | minor pain of arthritis |
| 553 | subject_nontext | -0.328958 |  |  | (in each geltab) |
| 715 | background | -0.304740 |  | Purposes | Uses |
| 535 | subject_text | -0.296640 | Uses |  | Uses |
| 533 | subject_text | -0.268236 | (in each gelcap) |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 233 | subject_nontext | +0.285681 |  |  |  |
| 575 | subject_nontext | +0.263034 |  |  |  |
| 514 | subject_text | +0.246961 | Acetaminophen, USP 500 mg... | ACETAMINOPHEN, USP 500 mg / | (prescription or nonprescription). If you are not |
| 635 | background | +0.235605 |  |  |  |
| 685 | background | +0.224465 |  | ACETAMINOPHEN, USP 500 mg / | in Tylenol®Arthritis Pain† |
| 548 | background | +0.212886 |  |  |  |
| 658 | background | +0.211005 |  |  |  |
| 661 | background | +0.205863 |  |  | (prescription or nonprescription). If you are not |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen, USP 500 mg...Pain reliever fever rexducer | 14 | +0.000003 |
| Diphenhydramine HCI, USP 25 mg.Nightime sleep aid | 14 | +0.000055 |
| Acetaminophen, USP 500 mg... | 5 | +0.000798 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7f4bc7d166b40bfc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.025797 | +0.168379 |
| 只保留主体分差 | -0.029153 | +0.216202 |
| 主体 token 数 | 162 | 288 |

[成功对照 token 表](../photo_7f4bc7d166b40bfc/tokens.csv) · [对照摘要](../photo_7f4bc7d166b40bfc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
