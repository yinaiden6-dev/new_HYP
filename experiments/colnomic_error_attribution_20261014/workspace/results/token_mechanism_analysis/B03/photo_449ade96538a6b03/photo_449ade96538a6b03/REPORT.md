# B03 · photo_449ade96538a6b03

本查询与相同正负图像已有身份歧义标记，不能直接作为模型失败证据。

包装 `package_076`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_003298`，**纯图像实际 Top-1 错误参考** `reference_003301`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 9 |
| 正确参考 MaxSim | +0.749476 |
| 错误参考 MaxSim | +0.758272 |
| 正确减错误的分差 | -0.008797 |
| F03 的正确参考排名（辅助对照） | 9 |
| 基于测量的诊断 | 同一候选对的身份歧义待复核 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 256 个。正向贡献 +0.012389，负向贡献 -0.021186，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 464 | -0.007651 | -0.012054 |
| nonvisual | 11 | -0.000241 | -0.015990 |
| subject_nontext | 67 | -0.000017 | -0.000181 |
| subject_text | 189 | -0.000888 | -0.003435 |

最主要负差额来自 `background`，贡献 -0.007651。只保留主体 query token 后，本对分差为 -0.002584。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 242 | 68.6% | True | +8.740378 | 62.5% |
| 错误 | 238 | 66.3% | True | +8.665479 | 65.2% |

主体中 189 个 token 与 OCR 行相交，152 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 91 | subject_text | -0.361135 | PORE-CLOGGING OILS |  |  |
| 179 | background | -0.222237 |  |  |  |
| 101 | background | -0.221571 |  |  | (see Directions), decreases the risk of skin cancer and |
| 180 | background | -0.191067 |  |  |  |
| 61 | background | -0.180217 |  | FULL OF FRESH, | CLEAN COVERAGE |
| 548 | background | -0.178810 |  |  |  |
| 300 | background | -0.177882 |  |  | (see Directions), decreases the risk of skin cancer and |
| 566 | background | -0.174845 |  | PORE-CLOGGING OILS |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 158 | background | +0.392804 |  |  | THAT'S CLEAN, FRESH AND |
| 289 | subject_text | +0.351225 | OPTHAMOLOGIST TESTED |  | OPTHAMOLOGIST TESTED |
| 393 | subject_text | +0.189021 | DE LA PEAU |  |  |
| 96 | subject_nontext | +0.124412 |  |  |  |
| 574 | subject_nontext | +0.111977 |  |  |  |
| 702 | background | +0.107050 |  |  |  |
| 254 | subject_nontext | +0.106580 |  | bareMinerals |  |
| 632 | subject_text | +0.099649 | TESTÉ SOUS CONTRÔLE |  | OPTHAMOLOGIST TESTED |

## 成功对照

此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
