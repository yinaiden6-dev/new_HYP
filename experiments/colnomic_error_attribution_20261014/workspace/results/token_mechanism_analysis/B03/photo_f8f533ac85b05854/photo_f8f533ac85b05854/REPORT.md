# B03 · photo_f8f533ac85b05854

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_003312`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 27 |
| 正确参考 MaxSim | +0.631551 |
| 错误参考 MaxSim | +0.641573 |
| 正确减错误的分差 | -0.010022 |
| F03 的正确参考排名（辅助对照） | 6 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 85 个。正向贡献 +0.029706，负向贡献 -0.039728，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 635 | -0.009856 | -0.011346 |
| nonvisual | 11 | -0.000225 | -0.014942 |
| subject_nontext | 27 | +0.000811 | +0.021945 |
| subject_text | 58 | -0.000752 | -0.009477 |

最主要负差额来自 `background`，贡献 -0.009856。只保留主体 query token 后，本对分差为 +0.000504。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 917 | 58.1% | True | +8.511767 | 75.3% |
| 错误 | 11 | 0.4% | False | 不可评估 | 不可评估 |

主体中 58 个 token 与 OCR 行相交，54 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 217 | background | -0.429361 |  |  |  |
| 533 | subject_text | -0.409401 | Other Information |  |  |
| 328 | background | -0.376563 |  |  |  |
| 150 | background | -0.360747 |  | japonica flower extract, salicornia herbacea extract, |  |
| 227 | background | -0.352885 |  |  |  |
| 336 | subject_text | -0.310451 | When using this product keep out of eyes. Rinse with |  |  |
| 200 | background | -0.301583 |  |  |  |
| 325 | background | -0.294309 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 593 | subject_nontext | +0.451488 |  |  | (ci 77491) • iron oxides (ci 77492) • iron oxides |
| 548 | background | +0.412561 |  | Call toll free 1-888-795-4747 | (ci 77491) • iron oxides (ci 77492) • iron oxides |
| 552 | subject_text | +0.407329 | Call toil free 1-888-795-4747 |  | Other information |
| 553 | subject_text | +0.395361 | Questions or comments? |  |  |
| 547 | background | +0.386663 |  | Call toll free 1-888-795-4747 |  |
| 399 | background | +0.289674 |  |  |  |
| 718 | background | +0.274958 |  |  |  |
| 398 | background | +0.274086 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Titanium diaxide 4.15% | 3 | -0.000144 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
