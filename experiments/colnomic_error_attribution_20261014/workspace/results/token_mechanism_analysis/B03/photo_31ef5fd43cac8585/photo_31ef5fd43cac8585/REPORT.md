# B03 · photo_31ef5fd43cac8585

按当前评测真值分析，未新增身份人工复核。

包装 `package_105`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000142`，**纯图像实际 Top-1 错误参考** `reference_000143`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.733366 |
| 错误参考 MaxSim | +0.736622 |
| 正确减错误的分差 | -0.003256 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 112 个。正向贡献 +0.022570，负向贡献 -0.025826，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 608 | -0.004184 | -0.005030 |
| nonvisual | 11 | -0.000137 | -0.009091 |
| subject_nontext | 35 | +0.000171 | +0.003563 |
| subject_text | 77 | +0.000894 | +0.008489 |

最主要负差额来自 `background`，贡献 -0.004184。只保留主体 query token 后，本对分差为 +0.006950。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 956 | 60.2% | True | +2.631323 | 46.8% |
| 错误 | 920 | 60.8% | True | +3.415556 | 53.2% |

主体中 77 个 token 与 OCR 行相交，69 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 424 | background | -0.391193 |  |  |  |
| 420 | background | -0.325383 |  |  |  |
| 579 | background | -0.300189 |  | 6 months of age has not been evaluated. |  |
| 710 | background | -0.282209 |  | 6 months of age has not been evaluated. |  |
| 425 | background | -0.269377 |  |  | For additional information about adverse drug |
| 587 | background | -0.250514 |  | 6 months of age has not been evaluated. |  |
| 421 | background | -0.233343 |  |  |  |
| 486 | background | -0.207973 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 133 | background | +0.341320 |  |  | CREDELIO CAT is also indicated for the treatment and |
| 75 | background | +0.251525 |  | The safe use of CREDELIO CAT in breeding, pregnant or | The effectiveness of CREDELIO CAT against Ixodes |
| 147 | background | +0.249142 |  |  | Manufactured for Elanco US Inc. |
| 471 | subject_text | +0.247387 | SH104223A | SH104223A |  |
| 294 | subject_text | +0.244769 | Dosage and Administration: CREDELIO CAT chewable | recommended minimum dosage of 2.7 mg/lb (6 mg/kg). | CREDELIO CAT must be administered with food. |
| 674 | background | +0.244571 |  | recommended minimum dosage of 2.7 mg/lb (6 mg/kg). |  |
| 78 | background | +0.231570 |  | The safe use of CREDELIO CAT in breeding, pregnant or | The effectiveness of CREDELIO CAT against Ixodes |
| 308 | background | +0.230947 |  | Warnings: Not for human use. Keep this and all |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| recommended minimum dosage of 2.7 mg/lb (6 mg/kg). | 6 | +0.000386 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_831ae64dd7297736` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003256 | +0.006583 |
| 只保留主体分差 | +0.006950 | +0.026738 |
| 主体 token 数 | 112 | 98 |

[成功对照 token 表](../photo_831ae64dd7297736/tokens.csv) · [对照摘要](../photo_831ae64dd7297736/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
