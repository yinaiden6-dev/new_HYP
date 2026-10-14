# B03 · photo_4b09c283ed00faca

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 290 |
| 正确参考 MaxSim | +0.498406 |
| 错误参考 MaxSim | +0.588421 |
| 正确减错误的分差 | -0.090015 |
| F03 的正确参考排名（辅助对照） | 6 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 180 个。正向贡献 +0.029958，负向贡献 -0.119973，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 588 | -0.095400 | -0.126388 |
| nonvisual | 11 | -0.001449 | -0.102618 |
| subject_nontext | 20 | -0.001000 | -0.038960 |
| subject_text | 160 | +0.007834 | +0.038139 |

最主要负差额来自 `background`，贡献 -0.095400。只保留主体 query token 后，本对分差为 +0.029573。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 479 | 76.2% | True | +7.576145 | 78.7% |
| 错误 | 8 | 30.1% | True | +9.703862 | 86.9% |

主体中 160 个 token 与 OCR 行相交，145 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 704 | background | -0.637867 |  |  |  |
| 711 | background | -0.636505 | 1FLOZ |  | 1 FL OZ (30 mL) |
| 617 | background | -0.632392 | congestion relief |  | congestion relief |
| 712 | background | -0.631278 | 1FLOZ |  | 1 FL OZ (30 mL) |
| 684 | background | -0.614210 |  |  | 7 |
| 683 | background | -0.609099 |  | or rapid heartbeat you have symptoms of an allergic reaction |  |
| 199 | background | -0.604905 |  | 370030275458 | 1 FL OZ (30 mL) |
| 618 | background | -0.597017 | congestion relief | Usereduces withdrawal symptoms, including nicotine | congestion relief |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 507 | subject_text | +0.478588 | before use. No studles have been done to show if this product |  | congestion relief |
| 292 | background | +0.447305 |  |  | 1-800-shop-CVS |
| 508 | subject_text | +0.392064 | endosed User's Gulde for complete directions and other important | R |  |
| 733 | background | +0.391154 |  | diabeteshistory of seizures |  |
| 51 | background | +0.366273 | prenaninating (veal) calves, and swinn) |  |  |
| 540 | subject_text | +0.357326 | endosed User's Gulde for complete directions and other important |  | Other information |
| 343 | subject_text | +0.350896 | or rapid heartbeat you have symptoms of an allergic reaction |  |  |
| 400 | subject_text | +0.340561 | nicotine replacement medicine. This medicine is belleved to be | nicotine replacement medicine. This medicine is believed to be |  |

## 同包装成功对照

`photo_80eb2d50ed90738e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.090015 | +0.132694 |
| 只保留主体分差 | +0.029573 | +0.260755 |
| 主体 token 数 | 180 | 273 |

[成功对照 token 表](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
