# B03 · photo_aabee66f53cb53e8

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_003367`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.549339 |
| 错误参考 MaxSim | +0.571073 |
| 正确减错误的分差 | -0.021734 |
| F03 的正确参考排名（辅助对照） | 4 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 121 个。正向贡献 +0.080952，负向贡献 -0.102686，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 647 | -0.057431 | -0.069148 |
| nonvisual | 11 | +0.000140 | +0.009904 |
| subject_nontext | 9 | +0.002454 | +0.212440 |
| subject_text | 112 | +0.033103 | +0.230244 |

最主要负差额来自 `background`，贡献 -0.057431。只保留主体 query token 后，本对分差为 +0.228919。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 225 | 52.9% | True | +2.695987 | 46.3% |
| 错误 | 25 | 40.2% | True | +15.619954 | 99.2% |

主体中 112 个 token 与 OCR 行相交，76 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 673 | background | -0.627317 |  |  |  |
| 231 | background | -0.620830 |  | 370030275458 | 1 FL OZ (30 mL) |
| 748 | background | -0.619466 |  |  |  |
| 522 | background | -0.610022 | NASALL |  |  |
| 648 | background | -0.603732 | UFRMABMAL | R |  |
| 714 | background | -0.602938 | 一 | or rapid heartbeat you have symptoms of an allergic reaction |  |
| 294 | background | -0.600899 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | CVS |
| 619 | background | -0.596506 | congestion relief | Usereduces withdrawal symptoms, including nicotine | congestion relief |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 471 | subject_text | +0.596039 | TAMPER EVIDENT FEATURE: Do not use if | Love it or your money back. | Fast, powerful |
| 502 | subject_text | +0.551829 | Love it or your money back. | Love it or your money back. | Money Back Guarantee |
| 503 | subject_text | +0.533222 | TAMPER EVIDENT FEATURE: Do not use if | Love it or your money back. |  |
| 274 | subject_text | +0.507867 | Uso Enough - Use at least 9 mini lozenges | 20 LOZENGES, 2 mg EACH |  |
| 273 | subject_text | +0.506608 | Uso Enough - Use at least 9 mini lozenges | 20 LOZENGES, 2 mg EACH |  |
| 435 | subject_text | +0.503818 | Use Long Enough - Use mini lozenges for the | 3. Use Long Enough - Use mini lozenges for the |  |
| 463 | subject_text | +0.491571 | TO INCREASE YOUR | TO INCREASE YOUR |  |
| 727 | background | +0.487484 |  | Love it or your money back. |  |

## 同包装成功对照

`photo_80eb2d50ed90738e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.021734 | +0.132694 |
| 只保留主体分差 | +0.228919 | +0.260755 |
| 主体 token 数 | 121 | 273 |

[成功对照 token 表](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
