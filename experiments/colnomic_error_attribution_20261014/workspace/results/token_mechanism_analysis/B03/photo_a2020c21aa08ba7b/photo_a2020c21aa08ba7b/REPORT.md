# B03 · photo_a2020c21aa08ba7b

按当前评测真值分析，未新增身份人工复核。

包装 `package_064`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001321`，**纯图像实际 Top-1 错误参考** `reference_001326`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 9 |
| 正确参考 MaxSim | +0.587345 |
| 错误参考 MaxSim | +0.611215 |
| 正确减错误的分差 | -0.023870 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 108 个。正向贡献 +0.038132，负向贡献 -0.062002，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 612 | -0.020145 | -0.024063 |
| nonvisual | 11 | -0.000018 | -0.001193 |
| subject_nontext | 59 | -0.002720 | -0.033700 |
| subject_text | 49 | -0.000987 | -0.014720 |

最主要负差额来自 `background`，贡献 -0.020145。只保留主体 query token 后，本对分差为 -0.025089。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 86 | 14.6% | False | 不可评估 | 不可评估 |
| 错误 | 21 | 0.6% | False | 不可评估 | 不可评估 |

主体中 49 个 token 与 OCR 行相交，26 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 155 | background | -0.500730 |  | Slze | Dist. by: Bayer HealthCare LLC |
| 292 | subject_text | -0.497649 | Pat.: patents.livewell.bayer.com | recommended dosage |  |
| 92 | background | -0.496694 |  | recommended dosage | Alka- |
| 352 | subject_nontext | -0.486939 |  | recommended dosage |  |
| 162 | background | -0.482275 |  | recommended dosage |  |
| 152 | background | -0.481460 |  | recommended dosage | Alka- |
| 412 | background | -0.461728 |  | recommended dosage |  |
| 226 | background | -0.454465 |  | recommended dosage |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 311 | subject_nontext | +0.498748 |  | Pat.: patents.livewell.bayer.com |  |
| 252 | background | +0.453628 |  | Pat.: patents.livewell.bayer.com |  |
| 359 | subject_text | +0.441350 | PAPER | PAPER | POWERFAST FiZZ |
| 313 | subject_nontext | +0.423290 |  | Pat.: patents.livewell.bayer.com | PARENTS |
| 280 | subject_text | +0.416734 | Bayer, the Bayer Cross, Alka-Seltzer Plus, and PowerMax are registered trademarks of Bayer. | Bayer, the Bayer Cross, Alka-Seltzer Plus, and PowerMax are registered trademarks of Bayer. |  |
| 492 | background | +0.398687 |  | Pat.: patents.livewell.bayer.com |  |
| 232 | background | +0.392711 |  | PAPER |  |
| 172 | background | +0.387382 |  | Pat.: patents.livewell.bayer.com |  |

## 同包装成功对照

`photo_47430c79b3238e4a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.023870 | +0.108781 |
| 只保留主体分差 | -0.025089 | +0.142660 |
| 主体 token 数 | 108 | 108 |

[成功对照 token 表](../photo_47430c79b3238e4a/tokens.csv) · [对照摘要](../photo_47430c79b3238e4a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
