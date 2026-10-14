# B03 · photo_39034fd7db4f883a

按当前评测真值分析，未新增身份人工复核。

包装 `package_040`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001112`，**纯图像实际 Top-1 错误参考** `reference_001130`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4399 |
| 正确参考 MaxSim | +0.372963 |
| 错误参考 MaxSim | +0.417433 |
| 正确减错误的分差 | -0.044470 |
| F03 的正确参考排名（辅助对照） | 360 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 91 个。正向贡献 +0.006895，负向贡献 -0.051365，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 677 | -0.038716 | -0.044549 |
| nonvisual | 11 | -0.000315 | -0.022337 |
| subject_nontext | 11 | -0.000517 | -0.036648 |
| subject_text | 80 | -0.004921 | -0.047916 |

最主要负差额来自 `background`，贡献 -0.038716。只保留主体 query token 后，本对分差为 -0.046554。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 644 | 64.9% | True | +11.140132 | 88.6% |
| 错误 | 6 | 36.4% | False | 不可评估 | 不可评估 |

主体中 80 个 token 与 OCR 行相交，64 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 505 | subject_text | -0.284242 | medical help or contact a Poison Control Center right |  |  |
| 31 | background | -0.259681 |  |  |  |
| 257 | background | -0.256595 | Sitzplatze an Bord |  |  |
| 324 | background | -0.222568 | Flow |  |  |
| 34 | background | -0.210547 |  |  |  |
| 17 | background | -0.184345 |  |  |  |
| 612 | background | -0.182163 | & tanzende | less severe symptoms. |  |
| 525 | background | -0.176925 | bierundbierli.at |  | Help Relieves Pollen/Dust Mites Allergy. |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 354 | background | +0.519841 | 安 |  | NDC Package Code |
| 353 | background | +0.491612 |  |  |  |
| 684 | background | +0.433515 |  |  |  |
| 443 | subject_text | +0.306633 | Drug Facts (continued) | Drug Facts (continued) |  |
| 282 | background | +0.130835 | eleganten Alltagsoutfits. |  |  |
| 244 | background | +0.119674 | KLYO Wien |  |  |
| 506 | subject_text | +0.116301 | pregnant: ask a health protessional betore use. | Drug Facts (continued) |  |
| 327 | background | +0.093869 | AKTUELL | Actual Size |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| tablet in 24 hours. A 5 mg | 3 | -0.000326 |
| not take more than one 10 mg | 5 | -0.000429 |
| one 10 mg tablet once daily; do | 4 | -0.000262 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_2df687d8a2f89f40` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.044470 | +0.164800 |
| 只保留主体分差 | -0.046554 | +0.223653 |
| 主体 token 数 | 91 | 312 |

[成功对照 token 表](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
