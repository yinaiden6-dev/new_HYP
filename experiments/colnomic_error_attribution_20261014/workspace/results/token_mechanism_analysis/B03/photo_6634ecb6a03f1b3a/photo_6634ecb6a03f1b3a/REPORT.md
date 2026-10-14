# B03 · photo_6634ecb6a03f1b3a

按当前评测真值分析，未新增身份人工复核。

包装 `package_057`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001410`，**纯图像实际 Top-1 错误参考** `reference_001426`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.639953 |
| 错误参考 MaxSim | +0.644572 |
| 正确减错误的分差 | -0.004619 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 63 个。正向贡献 +0.027306，负向贡献 -0.031925，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 657 | -0.005319 | -0.005919 |
| nonvisual | 11 | +0.000037 | +0.002461 |
| subject_nontext | 42 | +0.000681 | +0.011852 |
| subject_text | 21 | -0.000017 | -0.000606 |

最主要负差额来自 `background`，贡献 -0.005319。只保留主体 query token 后，本对分差为 +0.007700。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 53 | 14.8% | False | 不可评估 | 不可评估 |
| 错误 | 6 | 4.6% | False | 不可评估 | 不可评估 |

主体中 21 个 token 与 OCR 行相交，10 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 109 | background | -0.397371 |  | Redicare LLC, Congers, NY 10920 |  |
| 249 | background | -0.381060 |  |  |  |
| 319 | background | -0.356236 |  |  |  |
| 101 | background | -0.328818 |  |  |  |
| 248 | background | -0.316212 |  | *This product is not manufactured or distributed by GlaxoSmithKline |  |
| 295 | background | -0.302115 |  |  |  |
| 235 | background | -0.289839 |  | Redicare LLC, Congers, NY 10920 |  |
| 198 | background | -0.277294 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 556 | background | +0.391260 |  |  |  |
| 436 | subject_nontext | +0.375682 |  |  |  |
| 557 | background | +0.373226 |  |  |  |
| 617 | background | +0.319144 |  |  | Manufactured for Redicare LLC, |
| 56 | background | +0.316994 |  |  |  |
| 492 | background | +0.291400 |  | *This product is not manufactured or distributed by GlaxoSmithKline |  |
| 570 | background | +0.287623 |  |  | Made in China |
| 550 | background | +0.286569 |  |  | If swallowed, get medical help or contact a Poison Control Center right away. |

## 同包装成功对照

`photo_49600b91aa952055` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004619 | +0.187546 |
| 只保留主体分差 | +0.007700 | +0.235581 |
| 主体 token 数 | 63 | 56 |

[成功对照 token 表](../photo_49600b91aa952055/tokens.csv) · [对照摘要](../photo_49600b91aa952055/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
