# B03 · photo_d6f42eda7b050220

按当前评测真值分析，未新增身份人工复核。

包装 `package_077`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003310`，**纯图像实际 Top-1 错误参考** `reference_000977`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 745 |
| 正确参考 MaxSim | +0.492737 |
| 错误参考 MaxSim | +0.540236 |
| 正确减错误的分差 | -0.047499 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 217 个。正向贡献 +0.024275，负向贡献 -0.071774，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 551 | -0.041984 | -0.059356 |
| nonvisual | 11 | +0.000021 | +0.001517 |
| subject_nontext | 79 | -0.000207 | -0.002044 |
| subject_text | 138 | -0.005330 | -0.030086 |

最主要负差额来自 `background`，贡献 -0.041984。只保留主体 query token 后，本对分差为 -0.019877。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 251 | 64.5% | True | +12.281473 | 85.0% |
| 错误 | 6 | 6.1% | False | 不可评估 | 不可评估 |

主体中 138 个 token 与 OCR 行相交，84 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 727 | background | -0.536894 |  |  | COMPLETE PF |
| 690 | background | -0.531632 | ystane | CLEANS | Systane |
| 540 | subject_text | -0.528229 | Use hstps prokact ajanat caritise |  |  |
| 145 | background | -0.518540 | Eye Drops | Blue, Red, Yellow Dyes | LUBRICANT EYE DROPS |
| 723 | background | -0.516640 | LUBRICANT EYE DROPS |  | Systane |
| 50 | background | -0.508962 |  | Blue, Red, Yellow Dyes |  |
| 177 | background | -0.506898 | Eye Drops |  | LUBRICANT EYE DROPS |
| 144 | background | -0.499020 | grittineas, dryness & |  | for use as a lubricant to prevent further irritation or |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 596 | background | +0.458556 | 24 | ADA | For external use only |
| 437 | subject_text | +0.421474 | ADA | ADA | 0 |
| 392 | subject_text | +0.337973 | 3077205922 | 3077205922 | In the U.S. call 1-800-757-9195 |
| 747 | background | +0.289632 |  |  |  |
| 170 | background | +0.283794 |  | NaturalPeppermint |  |
| 424 | subject_text | +0.266339 | 3077205922 | 3077205922 | In the U.S. call 1-800-757-9195 |
| 106 | background | +0.212955 | Mint Mini |  |  |
| 434 | subject_text | +0.207275 | Halps | Use helpe protectagainstcavities | Purpose |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Sodum fiunnde 0.243% 0.13% wy fmrida ianj | 11 | -0.000853 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_134131912f9105af` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.047499 | +0.059273 |
| 只保留主体分差 | -0.019877 | +0.090627 |
| 主体 token 数 | 217 | 176 |

[成功对照 token 表](../photo_134131912f9105af/tokens.csv) · [对照摘要](../photo_134131912f9105af/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
