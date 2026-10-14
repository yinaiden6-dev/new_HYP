# B03 · photo_9d9c1aba85ca10cc

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_001219`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.608362 |
| 错误参考 MaxSim | +0.614880 |
| 正确减错误的分差 | -0.006517 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 209 个。正向贡献 +0.052344，负向贡献 -0.058861，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 511 | -0.011775 | -0.016845 |
| nonvisual | 11 | -0.000158 | -0.010532 |
| subject_nontext | 72 | +0.003162 | +0.032102 |
| subject_text | 137 | +0.002255 | +0.012030 |

最主要负差额来自 `background`，贡献 -0.011775。只保留主体 query token 后，本对分差为 +0.018944。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 300 | 42.4% | True | +7.483935 | 70.4% |
| 错误 | 6 | 8.9% | False | 不可评估 | 不可评估 |

主体中 137 个 token 与 OCR 行相交，80 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 408 | background | -0.531889 |  |  | Inactive ingredients ammonium hydroxide, bck |
| 473 | subject_text | -0.519574 | call 1-877-770-3183 Mon-Fri 8:00 AM EST to 5:00 PM PST |  |  |
| 405 | background | -0.501672 |  |  | Inactive ingredients ammonium hydroxide, bck |
| 371 | subject_text | -0.479015 | swold high hunidilty |  |  |
| 721 | background | -0.472675 |  |  | Inactive ingredients ammonium hydroxide, bck |
| 101 | background | -0.453427 |  | Other information | Inactive ingredients ammonium hydroxide, bck |
| 591 | subject_nontext | -0.449692 |  |  | Inactive ingredients ammonium hydroxide, bck |
| 474 | subject_text | -0.396848 | call 1-877-770-3183 Mon-Fri 8:00 AM EST to 5:00 PM PST |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 551 | subject_text | +0.533704 | Free shipping+ 20% off regular price | Free shipping + 20% off regular price |  |
| 550 | subject_text | +0.519621 | Free shipping+ 20% off regular price | Free shipping + 20% off regular price | Liver warning: This product contains acetaminophen, Severe |
| 512 | subject_text | +0.506757 | Buy this item often? | Buy this item often? | (1-800-222-1222). Quick medica attention is critical for |
| 534 | subject_text | +0.503353 | AUTO-REORDER | Buy this item often? |  |
| 632 | background | +0.500301 |  | WALGREENS_RED MATCH |  |
| 527 | background | +0.499532 |  | Buy this item often? |  |
| 572 | subject_text | +0.499492 | Leam mors at Walgreens.com/AutoReorder | 200 WILMOT RD., DEERFIELD, IL 60015 | TM & ©2024 Target Brands, Inc. |
| 276 | subject_nontext | +0.497855 |  | Walgreens | Minneapolis, MN 55403 |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Free shipping+ 20% off regular price | 7 | +0.003296 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7f4bc7d166b40bfc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006517 | +0.116648 |
| 只保留主体分差 | +0.018944 | +0.145108 |
| 主体 token 数 | 209 | 288 |

[成功对照 token 表](../photo_7f4bc7d166b40bfc/tokens.csv) · [对照摘要](../photo_7f4bc7d166b40bfc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
