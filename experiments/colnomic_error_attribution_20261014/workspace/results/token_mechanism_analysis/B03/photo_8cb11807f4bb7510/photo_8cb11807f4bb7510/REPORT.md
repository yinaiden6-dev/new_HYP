# B03 · photo_8cb11807f4bb7510

按当前评测真值分析，未新增身份人工复核。

包装 `package_097`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000792`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.584083 |
| 错误参考 MaxSim | +0.592406 |
| 正确减错误的分差 | -0.008323 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 216 个。正向贡献 +0.066824，负向贡献 -0.075148，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 552 | -0.036193 | -0.051077 |
| nonvisual | 11 | -0.000979 | -0.069327 |
| subject_nontext | 37 | -0.002242 | -0.047206 |
| subject_text | 179 | +0.031091 | +0.135308 |

最主要负差额来自 `background`，贡献 -0.036193。只保留主体 query token 后，本对分差为 +0.104044。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 330 | 74.7% | True | +3.599834 | 53.3% |
| 错误 | 23 | 17.0% | True | +17.445779 | 89.1% |

主体中 179 个 token 与 OCR 行相交，138 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 96 | background | -0.614057 | No Drip | Well | No Drip |
| 158 | background | -0.613857 | NASAL |  | NASAL MIST |
| 126 | background | -0.600386 | NASAL | Active ingredients ....Purpose | NASAL MIST |
| 189 | background | -0.592464 | NASAL |  |  |
| 157 | background | -0.585675 | NASAL |  | ORIGINAL |
| 159 | background | -0.583974 | NASAL | STERILE • HOMEOPATHIC | ASAL L |
| 125 | background | -0.580502 | NASAL | ORG1012-F |  |
| 155 | background | -0.578831 | congestion relief |  | Fast, powerful |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 637 | background | +0.562565 |  | Walgreens | CVS Pharmacy, Inc. |
| 463 | subject_text | +0.527804 | According to homeopathic principles, the active | According to homeopathic principles, the active |  |
| 695 | background | +0.503819 |  | Hepar sulphuris 12X. | povidone, purified water, sodium |
| 528 | subject_text | +0.499777 | excessive watery (clear) discharge |  |  |
| 627 | background | +0.494333 |  | For external use only. | temporarily relieves nasal congestion due to: |
| 244 | background | +0.490846 | ht0S.1-800-757-0195 |  |  |
| 623 | background | +0.481480 |  |  |  |
| 685 | background | +0.448037 | Defore sun exposure | ...redness, stinging |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% SARANTDT | 27 | +0.007948 |
| 100% | 10 | +0.003792 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_ae99ad8e6f14e09e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008323 | +0.045028 |
| 只保留主体分差 | +0.104044 | +0.185633 |
| 主体 token 数 | 216 | 200 |

[成功对照 token 表](../photo_ae99ad8e6f14e09e/tokens.csv) · [对照摘要](../photo_ae99ad8e6f14e09e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
