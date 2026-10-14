# B03 · photo_a0c829b7849000b2

按当前评测真值分析，未新增身份人工复核。

包装 `package_015`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000934`，**纯图像实际 Top-1 错误参考** `reference_004416`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 40 |
| 正确参考 MaxSim | +0.550439 |
| 错误参考 MaxSim | +0.563193 |
| 正确减错误的分差 | -0.012753 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 50 个。正向贡献 +0.026554，负向贡献 -0.039307，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 670 | -0.011965 | -0.013055 |
| nonvisual | 11 | +0.000286 | +0.019017 |
| subject_nontext | 13 | -0.000089 | -0.004983 |
| subject_text | 37 | -0.000985 | -0.019466 |

最主要负差额来自 `background`，贡献 -0.011965。只保留主体 query token 后，本对分差为 -0.015701。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 506 | 56.4% | True | +6.582652 | 75.5% |
| 错误 | 8 | 15.5% | True | +11.356993 | 91.8% |

主体中 37 个 token 与 OCR 行相交，31 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 679 | background | -0.287575 |  |  |  |
| 202 | background | -0.275917 |  |  |  |
| 662 | background | -0.269878 |  |  |  |
| 313 | background | -0.252177 |  |  | PARENTS: |
| 33 | background | -0.250568 |  |  |  |
| 362 | background | -0.246649 |  |  |  |
| 370 | background | -0.232119 |  |  | Active ingredients (in each softgel) |
| 237 | background | -0.230973 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 193 | background | +0.233329 |  |  |  |
| 32 | background | +0.228810 |  |  |  |
| 216 | background | +0.227452 |  |  |  |
| 199 | background | +0.222722 |  |  |  |
| 235 | background | +0.215873 |  |  | Parkinson's disease), or for 2 weeks after stopping the MAOl drug. If you |
| 693 | background | +0.207316 |  |  | cough comes back or occurs with rash or headache that lasts. These could be |
| 535 | subject_text | +0.199214 | 00002710 | 200000002710 | NDC 81522-898-16 |
| 336 | background | +0.197780 |  |  |  |

## 同包装成功对照

`photo_4917ad5f3fac9f55` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.012753 | +0.159071 |
| 只保留主体分差 | -0.015701 | +0.199119 |
| 主体 token 数 | 50 | 50 |

[成功对照 token 表](../photo_4917ad5f3fac9f55/tokens.csv) · [对照摘要](../photo_4917ad5f3fac9f55/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
