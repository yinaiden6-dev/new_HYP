# B03 · photo_ace6fcbc6d37331a

按当前评测真值分析，未新增身份人工复核。

包装 `package_021`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000973`，**纯图像实际 Top-1 错误参考** `reference_004066`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 855 |
| 正确参考 MaxSim | +0.499921 |
| 错误参考 MaxSim | +0.555095 |
| 正确减错误的分差 | -0.055174 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 45 个。正向贡献 +0.023248，负向贡献 -0.078422，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 675 | -0.053089 | -0.057494 |
| nonvisual | 11 | -0.000305 | -0.020260 |
| subject_nontext | 26 | -0.000705 | -0.019815 |
| subject_text | 19 | -0.001075 | -0.041360 |

最主要负差额来自 `background`，贡献 -0.053089。只保留主体 query token 后，本对分差为 -0.028912。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 5 | 17.4% | False | 不可评估 | 不可评估 |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 19 个 token 与 OCR 行相交，12 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 313 | subject_text | -0.546684 | NOT 100%SATISFIED? | aspirin, Increase ths risk of heart attack, heart | SATISFACTION |
| 570 | background | -0.528667 |  |  |  |
| 322 | background | -0.513486 |  |  |  |
| 314 | subject_text | -0.503523 | Return package and unusad product |  | SATISFACTION |
| 687 | background | -0.503413 |  |  |  |
| 325 | background | -0.502745 |  |  |  |
| 382 | background | -0.498547 |  |  |  |
| 80 | background | -0.497405 |  | 300000365 | SATISFACTION |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 614 | background | +0.296019 |  | CC | 627T0322 |
| 93 | background | +0.293431 |  |  |  |
| 405 | background | +0.279629 |  | CC | 627T0322 |
| 154 | background | +0.273022 |  |  |  |
| 33 | background | +0.268741 |  |  |  |
| 332 | subject_text | +0.252363 | within 30 days to any Family Dollar store |  | more than B so tgels in 24 hours, which is themaximum daily |
| 31 | background | +0.245883 |  |  |  |
| 659 | background | +0.226139 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| NOT 100%SATISFIED? | 3 | -0.001457 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_60b643cb7fb93b4e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.055174 | +0.092912 |
| 只保留主体分差 | -0.028912 | +0.152078 |
| 主体 token 数 | 45 | 50 |

[成功对照 token 表](../photo_60b643cb7fb93b4e/tokens.csv) · [对照摘要](../photo_60b643cb7fb93b4e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
