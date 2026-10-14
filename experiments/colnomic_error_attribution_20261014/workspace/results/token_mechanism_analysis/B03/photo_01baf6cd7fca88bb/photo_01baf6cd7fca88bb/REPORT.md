# B03 · photo_01baf6cd7fca88bb

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_002441`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 390 |
| 正确参考 MaxSim | +0.606274 |
| 错误参考 MaxSim | +0.682995 |
| 正确减错误的分差 | -0.076721 |
| F03 的正确参考排名（辅助对照） | 8 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 306 个。正向贡献 +0.024170，负向贡献 -0.100891，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 414 | -0.035233 | -0.062211 |
| nonvisual | 11 | -0.000465 | -0.030905 |
| subject_nontext | 72 | -0.006308 | -0.064041 |
| subject_text | 234 | -0.034715 | -0.108448 |

最主要负差额来自 `background`，贡献 -0.035233。只保留主体 query token 后，本对分差为 -0.097999。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1160 | 69.6% | True | +8.726092 | 68.9% |
| 错误 | 4 | 14.6% | False | 不可评估 | 不可评估 |

主体中 234 个 token 与 OCR 行相交，198 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 213 | subject_text | -0.565927 | water to remove. Stop use and ask a doctor if rash | Control Center right away. |  |
| 229 | subject_text | -0.563104 | swallowed, get medical help or contact a Poison |  | children. If swallowed, get |
| 219 | background | -0.533184 |  |  | children. If swallowed, get |
| 194 | subject_text | -0.514696 | When using this product keep out of eyes. Rinse with | Directions For sunscreen useapply liberally | When using this product |
| 650 | subject_text | -0.505884 | Other Information |  | Other information |
| 652 | subject_text | -0.489256 | Other Information |  | Protect the product in this |
| 107 | background | -0.482086 |  | Uses • helps prevent sunburn • if used as | Helps prevent sunburn. |
| 103 | background | -0.479732 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 603 | background | +0.439882 |  |  |  |
| 607 | background | +0.427706 |  |  | Avobenzone 3.0 % |
| 680 | background | +0.353365 |  |  | Control Center right away. |
| 52 | subject_nontext | +0.347679 |  |  |  |
| 713 | subject_nontext | +0.322948 |  |  | YOU MAY REPORT SERIOUS |
| 701 | background | +0.316157 |  |  |  |
| 695 | subject_nontext | +0.312753 |  | Call toll free 1-888-795-4747 | Control Center right away. |
| 682 | background | +0.309853 |  | Call toll free 1-888-795-4747 | Control Center right away. |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Titanium dioxide 4.15% | 4 | -0.001180 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
