# B03 · photo_2bb7a4894380d509

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_003452`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 10 |
| 正确参考 MaxSim | +0.605224 |
| 错误参考 MaxSim | +0.622401 |
| 正确减错误的分差 | -0.017177 |
| F03 的正确参考排名（辅助对照） | 9 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 288 个。正向贡献 +0.024069，负向贡献 -0.041246，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 432 | -0.013298 | -0.022501 |
| nonvisual | 11 | +0.000096 | +0.006406 |
| subject_nontext | 273 | -0.002529 | -0.006772 |
| subject_text | 15 | -0.001447 | -0.070502 |

最主要负差额来自 `background`，贡献 -0.013298。只保留主体 query token 后，本对分差为 -0.010091。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 44 | 23.3% | True | +10.334254 | 89.2% |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 15 个 token 与 OCR 行相交，6 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 611 | subject_text | -0.496319 | Distbypar bareMineasNew York |  | Dist. by bareMinerals New York, NY 10022,USA |
| 311 | subject_nontext | -0.463102 |  | FORMULÉ SANS HUILES QUI | Dist. by bareMinerals New York, NY 10022,USA |
| 241 | background | -0.454661 |  | FABRIQUÉ AUX ÉTATS-UNIS | Dist. by bareMinerals New York, NY 10022,USA |
| 491 | subject_nontext | -0.436719 |  |  | Dist. by bareMinerals New York, NY 10022,USA |
| 381 | background | -0.429957 |  |  | Dist. by bareMinerals New York, NY 10022,USA |
| 178 | background | -0.428784 |  | FABRIQUÉ AUX ÉTATS-UNIS | Dist. by bareMinerals New York, NY 10022,USA |
| 105 | background | -0.396501 |  | FABRIQUÉ AUX ÉTATS-UNIS | Dist. by bareMinerals New York, NY 10022,USA |
| 665 | background | -0.392432 |  | FABRIQUÉ AUX ÉTATS-UNIS | Dist. by bareMinerals New York, NY 10022,USA |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 370 | subject_nontext | +0.301942 |  |  | FOUNDATION GENTLY CLEARS AND |
| 229 | subject_nontext | +0.288535 |  |  |  |
| 111 | subject_nontext | +0.280747 |  |  |  |
| 32 | subject_nontext | +0.279083 |  |  | MADE IN USA |
| 310 | subject_nontext | +0.276455 |  |  |  |
| 91 | subject_nontext | +0.267630 |  |  | MADE IN USA |
| 156 | subject_nontext | +0.255714 |  |  |  |
| 214 | subject_nontext | +0.255355 |  |  |  |

## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
