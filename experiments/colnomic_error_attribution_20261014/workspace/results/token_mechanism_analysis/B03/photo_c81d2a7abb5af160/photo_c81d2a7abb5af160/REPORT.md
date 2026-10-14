# B03 · photo_c81d2a7abb5af160

按当前评测真值分析，未新增身份人工复核。

包装 `package_114`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_004816`，**纯图像实际 Top-1 错误参考** `reference_005119`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.591109 |
| 错误参考 MaxSim | +0.593395 |
| 正确减错误的分差 | -0.002286 |
| F03 的正确参考排名（辅助对照） | 7 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 72 个。正向贡献 +0.101848，负向贡献 -0.104134，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 696 | -0.000697 | -0.000780 |
| nonvisual | 11 | +0.000031 | +0.002219 |
| subject_nontext | 17 | -0.000281 | -0.012858 |
| subject_text | 55 | -0.001340 | -0.018979 |

最主要负差额来自 `subject_text`，贡献 -0.001340。只保留主体 query token 后，本对分差为 -0.017534。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 5 | 3.7% | False | 不可评估 | 不可评估 |
| 错误 | 6 | 26.3% | False | 不可评估 | 不可评估 |

主体中 55 个 token 与 OCR 行相交，21 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 184 | background | -0.684331 | Enoxaparin |  |  |
| 152 | background | -0.663760 | Enoxaparin |  |  |
| 762 | background | -0.642029 |  |  | TEAR OFF TO OPEN BEE HERE |
| 272 | background | -0.635679 |  |  | TEAR OFF TO OPEN BEE HERE |
| 731 | background | -0.629014 |  |  |  |
| 736 | background | -0.627306 |  |  | TEAR OFF TO OPEN BEE HERE |
| 463 | background | -0.623616 |  |  | TEAR OFF TO OPEN BEE HERE |
| 460 | background | -0.621895 |  |  | TEAR OFF TO OPEN BEE HERE |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 371 | background | +0.607272 | (levothyroxine sodium) tablets | (levothyroxine sodium) tablets |  |
| 402 | background | +0.602995 | 150 mcg per tablet, USP | (levothyroxine sodium) tablets | Sodium Injection, USP |
| 364 | background | +0.593583 | Euthyrox® |  | TEAR OFF TO OPEN BOE HERE |
| 365 | background | +0.583455 | Euthyrox® | Euthyrox® | Enoxaparin |
| 597 | background | +0.582229 |  | blister packaging until ready to use. |  |
| 660 | background | +0.565226 | 90 Tablets (6 x 15 tablets per blister pack) | 90 Tablets (6 × 15 tablets per blister pack) |  |
| 397 | background | +0.555045 | (levothyroxine sodium) tablets |  |  |
| 735 | background | +0.551604 |  |  | Enoxaparin |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 140 mg | 8 | +0.000346 |
| 5 Capsules | 6 | +0.000516 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_0eb729e9f7b895d9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002286 | +0.178932 |
| 只保留主体分差 | -0.017534 | +0.240894 |
| 主体 token 数 | 72 | 242 |

[成功对照 token 表](../photo_0eb729e9f7b895d9/tokens.csv) · [对照摘要](../photo_0eb729e9f7b895d9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
