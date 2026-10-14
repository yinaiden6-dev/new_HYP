# B03 · photo_206a9b7d61884264

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_003299`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 10 |
| 正确参考 MaxSim | +0.726289 |
| 错误参考 MaxSim | +0.738053 |
| 正确减错误的分差 | -0.011764 |
| F03 的正确参考排名（辅助对照） | 10 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 270 个。正向贡献 +0.010054，负向贡献 -0.021818，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 450 | -0.004111 | -0.006678 |
| nonvisual | 11 | -0.000030 | -0.001967 |
| subject_nontext | 179 | -0.005377 | -0.021958 |
| subject_text | 91 | -0.002247 | -0.018046 |

最主要负差额来自 `subject_nontext`，贡献 -0.005377。只保留主体 query token 后，本对分差为 -0.020640。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 248 | 34.9% | True | +8.457510 | 67.3% |
| 错误 | 268 | 36.1% | True | +5.853434 | 62.3% |

主体中 91 个 token 与 OCR 行相交，34 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 436 | subject_nontext | -0.267091 |  |  |  |
| 216 | subject_nontext | -0.224591 |  |  |  |
| 77 | subject_nontext | -0.212991 |  |  |  |
| 172 | subject_text | -0.205851 | bareMineral | Dist. by/par bareMinerals, New York | bareMinerals |
| 234 | subject_nontext | -0.188155 |  |  |  |
| 150 | subject_text | -0.186705 | bareMineral | Dist. by/par bareMinerals, New York |  |
| 316 | subject_nontext | -0.186478 |  |  |  |
| 356 | subject_nontext | -0.185598 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 397 | subject_nontext | +0.214432 |  |  | COUVRANCE MOYENNE TIENT |
| 427 | background | +0.197902 |  |  |  |
| 665 | background | +0.168009 |  |  | bareMinerals |
| 285 | background | +0.150216 |  |  |  |
| 466 | background | +0.149072 |  |  |  |
| 657 | subject_text | +0.147963 | 一 |  |  |
| 103 | background | +0.140312 |  |  |  |
| 537 | subject_nontext | +0.118866 |  |  |  |

## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
