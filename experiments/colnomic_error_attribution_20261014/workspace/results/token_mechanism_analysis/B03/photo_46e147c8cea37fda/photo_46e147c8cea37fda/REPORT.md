# B03 · photo_46e147c8cea37fda

按当前评测真值分析，未新增身份人工复核。

包装 `package_120`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_001983`，**纯图像实际 Top-1 错误参考** `reference_001939`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.668786 |
| 错误参考 MaxSim | +0.677677 |
| 正确减错误的分差 | -0.008891 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 304 个。正向贡献 +0.021822，负向贡献 -0.030713，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 464 | +0.000280 | +0.000469 |
| nonvisual | 11 | +0.000030 | +0.002106 |
| subject_nontext | 109 | -0.000371 | -0.002651 |
| subject_text | 195 | -0.008829 | -0.035272 |

最主要负差额来自 `subject_text`，贡献 -0.008829。只保留主体 query token 后，本对分差为 -0.023575。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 397 | 62.1% | True | +2.242550 | 45.7% |
| 错误 | 103 | 37.7% | True | +5.219506 | 59.7% |

主体中 195 个 token 与 OCR 行相交，57 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 634 | subject_nontext | -0.322319 |  | Actual |  |
| 149 | background | -0.288423 |  | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL |  |
| 662 | subject_text | -0.261029 | WHEN TAKEN AS DIRECTED SEE DRUG FACTS PANEL |  |  |
| 653 | subject_text | -0.249714 | WHEN TAKEN AS DIRECTED SEE DRUG FACTS PANEL | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL | • No synthetic colors |
| 268 | subject_text | -0.227605 | NON-DROWSY | NON-DROWSY |  |
| 667 | subject_text | -0.224777 | 20 | Loratadine tablets USP 10 mg, antihistamine | tablets |
| 147 | background | -0.221960 |  | WHEN TAKEN AS DIRECTED, SEE DRUG FACTS PANEL |  |
| 155 | background | -0.218931 |  | Loratadine tablets USP 10 mg, antihistamine |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 410 | subject_nontext | +0.410218 |  |  |  |
| 461 | subject_text | +0.393378 | Compare to the active ingredient in Claritin |  |  |
| 474 | subject_nontext | +0.301064 |  | Compare to the active ingredient in Claritin |  |
| 123 | background | +0.272391 |  |  | Contains 1 bottle with 45 tablets |
| 428 | subject_nontext | +0.269025 |  |  |  |
| 571 | subject_nontext | +0.255944 |  | Actual |  |
| 174 | subject_nontext | +0.230271 |  |  | Walgreens |
| 635 | subject_text | +0.224091 | 20 | 20 |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Loratadine tablets USP 10 mg, antihistamine | 25 | -0.001530 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7e0b721a9c25b555` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008891 | +0.008533 |
| 只保留主体分差 | -0.023575 | +0.001896 |
| 主体 token 数 | 304 | 208 |

[成功对照 token 表](../photo_7e0b721a9c25b555/tokens.csv) · [对照摘要](../photo_7e0b721a9c25b555/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
