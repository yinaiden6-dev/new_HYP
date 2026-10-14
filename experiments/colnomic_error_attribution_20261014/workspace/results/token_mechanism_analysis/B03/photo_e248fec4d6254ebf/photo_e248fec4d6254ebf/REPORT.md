# B03 · photo_e248fec4d6254ebf

按当前评测真值分析，未新增身份人工复核。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_004705`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 11 |
| 正确参考 MaxSim | +0.582327 |
| 错误参考 MaxSim | +0.612284 |
| 正确减错误的分差 | -0.029957 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 85 个。正向贡献 +0.015899，负向贡献 -0.045856，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 635 | -0.027608 | -0.031782 |
| nonvisual | 11 | -0.000346 | -0.023011 |
| subject_nontext | 17 | -0.000457 | -0.019655 |
| subject_text | 68 | -0.001546 | -0.016619 |

最主要负差额来自 `background`，贡献 -0.027608。只保留主体 query token 后，本对分差为 -0.017227。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 793 | 54.1% | True | +13.563202 | 98.6% |
| 错误 | 4 | 11.1% | False | 不可评估 | 不可评估 |

主体中 68 个 token 与 OCR 行相交，61 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 411 | background | -0.286822 |  |  | Microcristallina), |
| 649 | background | -0.282278 |  |  |  |
| 140 | background | -0.268927 |  |  |  |
| 63 | background | -0.265841 |  |  |  |
| 397 | background | -0.247337 |  |  | AYCONTAINJron |
| 650 | background | -0.247051 |  |  |  |
| 511 | background | -0.241870 |  |  |  |
| 338 | background | -0.241528 |  |  | Microcristallina), |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 613 | subject_nontext | +0.191416 |  |  |  |
| 275 | background | +0.171959 |  | alkanes, squalene, parafn, neopentyl glycol | USES: Apply |
| 436 | subject_text | +0.171752 | sun, especilly from 10 a.m.-2 p.m. • wear |  | Mineral Moisturizing Lip Tint SPF 8 |
| 663 | background | +0.170836 |  |  |  |
| 295 | subject_text | +0.163413 | Questions or comments? | alkanes, squalene, parafn, neopentyl glycol | USES: Apply |
| 630 | background | +0.163106 |  |  |  |
| 643 | background | +0.144797 |  |  |  |
| 660 | background | +0.143909 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Titanium donide 4.15% | 3 | +0.000062 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
