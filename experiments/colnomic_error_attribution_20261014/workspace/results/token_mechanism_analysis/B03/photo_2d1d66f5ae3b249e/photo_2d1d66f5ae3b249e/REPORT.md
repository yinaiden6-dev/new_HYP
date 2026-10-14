# B03 · photo_2d1d66f5ae3b249e

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000146`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.654657 |
| 错误参考 MaxSim | +0.667915 |
| 正确减错误的分差 | -0.013258 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 72 个。正向贡献 +0.023431，负向贡献 -0.036689，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.013598 | -0.015340 |
| nonvisual | 11 | -0.000021 | -0.001374 |
| subject_nontext | 24 | -0.000706 | -0.021518 |
| subject_text | 48 | +0.001067 | +0.016255 |

最主要负差额来自 `background`，贡献 -0.013598。只保留主体 query token 后，本对分差为 +0.003664。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 418 | 54.0% | True | +1.402271 | 47.9% |
| 错误 | 170 | 26.1% | True | +7.095678 | 58.3% |

主体中 48 个 token 与 OCR 行相交，32 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 308 | background | -0.659864 |  |  | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 174 | background | -0.601064 |  |  | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 307 | background | -0.595123 |  |  | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 303 | background | -0.547351 |  |  | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 495 | subject_nontext | -0.450330 |  | STORAGE CONDITIONS: Store at or below 25°C (77°F). | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 305 | background | -0.449213 |  |  | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 252 | background | -0.433221 |  | Elanco | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |
| 493 | subject_nontext | -0.330625 |  |  | FOR USE IN ANIMALS ONLY. NOT FOR HUMAN USE |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 513 | subject_nontext | +0.306379 |  |  |  |
| 327 | background | +0.292014 |  |  |  |
| 323 | background | +0.270694 |  |  |  |
| 207 | background | +0.220608 |  |  |  |
| 522 | background | +0.213359 |  |  |  |
| 565 | background | +0.213358 |  |  |  |
| 555 | background | +0.208266 |  |  | Distributed by: Elanco US, Inc. Greenfield, IN 46140 |
| 586 | background | +0.207275 |  |  |  |

## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013258 | +0.015164 |
| 只保留主体分差 | +0.003664 | +0.034226 |
| 主体 token 数 | 72 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
