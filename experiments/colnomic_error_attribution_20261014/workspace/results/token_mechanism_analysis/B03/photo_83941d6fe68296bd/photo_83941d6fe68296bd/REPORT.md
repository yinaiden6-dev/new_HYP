# B03 · photo_83941d6fe68296bd

按当前评测真值分析，未新增身份人工复核。

包装 `package_026`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000995`，**纯图像实际 Top-1 错误参考** `reference_003684`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 600 |
| 正确参考 MaxSim | +0.491811 |
| 错误参考 MaxSim | +0.525474 |
| 正确减错误的分差 | -0.033663 |
| F03 的正确参考排名（辅助对照） | 110 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 169 个。正向贡献 +0.023884，负向贡献 -0.057548，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 599 | -0.040399 | -0.052538 |
| nonvisual | 11 | -0.000241 | -0.017083 |
| subject_nontext | 124 | +0.001289 | +0.008096 |
| subject_text | 45 | +0.005688 | +0.098464 |

最主要负差额来自 `background`，贡献 -0.040399。只保留主体 query token 后，本对分差为 +0.032158。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 179 | 52.2% | True | +7.206017 | 70.2% |
| 错误 | 39 | 64.4% | True | +10.992059 | 96.2% |

主体中 45 个 token 与 OCR 行相交，9 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 618 | background | -0.561735 | Eye Drops | Questions or comments?1-800-719-9260 |  |
| 134 | background | -0.561557 |  | Acid Reducer | Eye Itch Relief |
| 617 | background | -0.538569 | Eye Drops | • May Take 1 to 4 Days |  |
| 685 | background | -0.536685 | 1FLOZ | Lansoprazole 15 mg.. | STERILE 0.34 FL OZ (10 mL) |
| 553 | background | -0.520476 | OMEOT c | Questions or comments?1-800-719-9260 | STERILE 0.34 FL OZ (10 mL) |
| 552 | background | -0.512762 | OMEOT c | basiccare | STERILE 0.34 FL OZ (10 mL) |
| 694 | background | -0.500465 |  | NDC 72288-343-01 |  |
| 43 | background | -0.488604 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 250 | subject_nontext | +0.522142 |  |  | W3ORG0821-F2 |
| 251 | subject_nontext | +0.510794 |  |  | W3ORG0821-F2 |
| 443 | subject_text | +0.468845 | 3T3D5 BN C4 | 3T3D5 BN C4 | W3ORG0821-F2 |
| 411 | subject_text | +0.465158 | 3T3D5 BN C4 | 3T3D5 BN C4 | 1191716082 |
| 442 | subject_text | +0.431248 | 3T3D5 BN C4 | 3T3D5 BN C4 | W3ORG0821-F2 |
| 315 | subject_nontext | +0.351765 |  |  |  |
| 185 | background | +0.336490 |  |  |  |
| 410 | subject_text | +0.297430 | 3T3D5 BN C4 | 3T3D5 BN C4 | NC |

## 同包装成功对照

`photo_ab6fd7a33337ccee` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.033663 | +0.070883 |
| 只保留主体分差 | +0.032158 | +0.250529 |
| 主体 token 数 | 169 | 209 |

[成功对照 token 表](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
