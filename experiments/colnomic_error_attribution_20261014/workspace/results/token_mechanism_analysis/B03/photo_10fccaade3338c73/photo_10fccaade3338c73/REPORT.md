# B03 · photo_10fccaade3338c73

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_004359`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.556513 |
| 错误参考 MaxSim | +0.559886 |
| 正确减错误的分差 | -0.003373 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 121 个。正向贡献 +0.028665，负向贡献 -0.032037，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 647 | -0.005915 | -0.007121 |
| nonvisual | 11 | +0.000177 | +0.012531 |
| subject_nontext | 33 | -0.000097 | -0.002299 |
| subject_text | 88 | +0.002462 | +0.021796 |

最主要负差额来自 `background`，贡献 -0.005915。只保留主体 query token 后，本对分差为 +0.015225。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 133 | 48.4% | True | +5.118594 | 61.2% |
| 错误 | 9 | 7.4% | False | 不可评估 | 不可评估 |

主体中 88 个 token 与 OCR 行相交，68 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 378 | background | -0.342448 | sseplatz 1, 1020 Vienna |  |  |
| 56 | background | -0.338290 | rmacy | non-nicotine stop smoking drugtaking prescription medicine for |  |
| 19 | background | -0.335950 |  | lozenge for a longer perlod to keep from smoking, talk to your | during the first six weeks. |
| 119 | background | -0.330386 | rmacy |  | © 2022 CVS/pharmacy |
| 62 | background | -0.309714 |  | If you are pregnant or breast-feeding, only use this medicine |  |
| 237 | background | -0.304514 |  | 1. You must be motivated to quit |  |
| 318 | background | -0.301598 | COIE |  |  |
| 257 | background | -0.283913 | erview Plan |  | © 2022 CVS/pharmacy |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 533 | subject_text | +0.485806 | Satisfaction guaranteed- | Love it or your money back. |  |
| 501 | subject_text | +0.461394 | Satisfaction guaranteed- | Love it or your money back. |  |
| 262 | background | +0.408490 |  | the enclosed User's Guide. |  |
| 207 | background | +0.348653 |  | Love it or your money back. |  |
| 182 | background | +0.325347 | MetroU2 |  |  |
| 405 | subject_text | +0.323162 | Satisfaction guaranteed- | Satisfaction guaranteed – | CVS.com/returnpolicy |
| 47 | background | +0.318507 |  | Love it or your money back. |  |
| 57 | background | +0.311386 |  |  | during the first six weeks. |

## 同包装成功对照

`photo_d2377339b5c178d4` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003373 | +0.006482 |
| 只保留主体分差 | +0.015225 | +0.030441 |
| 主体 token 数 | 121 | 135 |

[成功对照 token 表](../photo_d2377339b5c178d4/tokens.csv) · [对照摘要](../photo_d2377339b5c178d4/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
