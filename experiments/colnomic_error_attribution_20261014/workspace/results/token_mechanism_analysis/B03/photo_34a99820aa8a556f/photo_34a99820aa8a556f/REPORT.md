# B03 · photo_34a99820aa8a556f

按当前评测真值分析，未新增身份人工复核。

包装 `package_070`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001640`，**纯图像实际 Top-1 错误参考** `reference_001849`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.651189 |
| 错误参考 MaxSim | +0.659712 |
| 正确减错误的分差 | -0.008523 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 256 个。正向贡献 +0.038983，负向贡献 -0.047506，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 464 | -0.009898 | -0.015593 |
| nonvisual | 11 | -0.000273 | -0.018116 |
| subject_nontext | 158 | -0.001873 | -0.008665 |
| subject_text | 98 | +0.003520 | +0.026255 |

最主要负差额来自 `background`，贡献 -0.009898。只保留主体 query token 后，本对分差为 +0.004703。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 274 | 48.0% | True | +3.705798 | 54.4% |
| 错误 | 8 | 3.8% | False | 不可评估 | 不可评估 |

主体中 98 个 token 与 OCR 行相交，41 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 355 | subject_text | -0.551270 | Call toll-free for US & CA | Exclusively produced for: |  |
| 536 | background | -0.550962 |  | Distributed in the US by: |  |
| 706 | background | -0.509079 |  | DESERV |  |
| 516 | background | -0.503149 |  |  |  |
| 358 | subject_text | -0.497793 | Vancouver, BC V6E 2E9 CANADA |  | produced for: 300 - 1055 West Hastings Street, |
| 556 | background | -0.480235 |  | Mada |  |
| 219 | subject_nontext | -0.474122 |  |  | produced for: 300 - 1055 West Hastings Street, |
| 335 | subject_text | -0.466472 | Call toll-free for US & CA |  | Bremerton,WA 98337 |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 387 | subject_text | +0.469356 | Hecho en Canada | Hecho en Canada | • Helps relieve hemorrhoids |
| 227 | subject_text | +0.408992 | Made in Canada | Made in Canada |  |
| 246 | subject_text | +0.394271 | Made in Canada | Made in Canada |  |
| 367 | subject_text | +0.383316 | Hecho en Canada | Hecho en Canada | DESERVING |
| 67 | background | +0.377280 |  | Hecho en Canada | produced for: 300 - 1055 West Hastings Street, |
| 366 | subject_nontext | +0.375827 |  | Made in CanadaHecho en Canada | in the US by: |
| 368 | subject_text | +0.371430 | 27118"33011 | 211 | 2711835009 |
| 347 | subject_text | +0.305325 | Hecho en Canada | Made in CanadaHecho en Canada |  |

## 同包装成功对照

`photo_ee5f8169ea7ffcd2` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008523 | +0.181504 |
| 只保留主体分差 | +0.004703 | +0.198491 |
| 主体 token 数 | 256 | 255 |

[成功对照 token 表](../photo_ee5f8169ea7ffcd2/tokens.csv) · [对照摘要](../photo_ee5f8169ea7ffcd2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
