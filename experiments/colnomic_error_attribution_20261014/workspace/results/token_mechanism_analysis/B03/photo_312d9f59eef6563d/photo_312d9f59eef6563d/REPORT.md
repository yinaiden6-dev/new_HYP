# B03 · photo_312d9f59eef6563d

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_003260`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 690 |
| 正确参考 MaxSim | +0.509812 |
| 错误参考 MaxSim | +0.576240 |
| 正确减错误的分差 | -0.066428 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 90 个。正向贡献 +0.010125，负向贡献 -0.076554，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 630 | -0.056304 | -0.065331 |
| nonvisual | 11 | -0.001265 | -0.084086 |
| subject_nontext | 84 | -0.008114 | -0.070607 |
| subject_text | 6 | -0.000745 | -0.090822 |

最主要负差额来自 `background`，贡献 -0.056304。只保留主体 query token 后，本对分差为 -0.071955。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 17 | 1.0% | False | 不可评估 | 不可评估 |
| 错误 | 5 | 49.1% | False | 不可评估 | 不可评估 |

主体中 6 个 token 与 OCR 行相交，3 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 668 | background | -0.435688 |  |  |  |
| 706 | background | -0.312206 |  |  |  |
| 572 | subject_nontext | -0.310825 |  |  |  |
| 652 | subject_nontext | -0.308789 |  | For external use only |  |
| 157 | background | -0.300833 |  |  |  |
| 719 | background | -0.284277 |  |  |  |
| 603 | background | -0.281004 |  |  |  |
| 493 | subject_nontext | -0.278152 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 271 | background | +0.227385 |  |  | Jack Black |
| 92 | background | +0.192178 |  |  |  |
| 455 | subject_nontext | +0.190527 |  |  |  |
| 511 | subject_nontext | +0.184024 |  |  |  |
| 375 | subject_nontext | +0.167522 |  |  |  |
| 435 | subject_nontext | +0.166978 |  |  |  |
| 153 | background | +0.162943 |  |  | Lightweight, emollient moisturizer is an advanced facial treatment and broad-spectrum sunscreen in one. |
| 300 | background | +0.158667 |  |  |  |

## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
