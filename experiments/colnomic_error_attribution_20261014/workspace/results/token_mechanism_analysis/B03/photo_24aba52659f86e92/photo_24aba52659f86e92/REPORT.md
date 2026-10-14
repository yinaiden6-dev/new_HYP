# B03 · photo_24aba52659f86e92

按当前评测真值分析，未新增身份人工复核。

包装 `package_095`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000377`，**纯图像实际 Top-1 错误参考** `reference_000379`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.778784 |
| 错误参考 MaxSim | +0.785642 |
| 正确减错误的分差 | -0.006859 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 384 个。正向贡献 +0.023135，负向贡献 -0.029994，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 336 | -0.000553 | -0.001204 |
| nonvisual | 11 | -0.000065 | -0.004287 |
| subject_nontext | 241 | -0.005333 | -0.016175 |
| subject_text | 143 | -0.000908 | -0.004642 |

最主要负差额来自 `subject_nontext`，贡献 -0.005333。只保留主体 query token 后，本对分差为 -0.011880。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 148 | 48.7% | True | +6.714045 | 63.6% |
| 错误 | 134 | 56.6% | True | +6.266430 | 63.2% |

主体中 143 个 token 与 OCR 行相交，50 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 389 | subject_text | -0.446061 | www.InterceptorPet.com |  | www.InterceptorPet.com |
| 99 | background | -0.394159 |  |  | www.InterceptorPet.com |
| 89 | background | -0.384383 |  |  | www.InterceptorPet.com |
| 159 | subject_nontext | -0.326801 |  |  |  |
| 469 | subject_nontext | -0.320474 |  |  | www.InterceptorPet.com |
| 177 | subject_nontext | -0.306397 |  |  |  |
| 187 | subject_nontext | -0.287599 |  |  |  |
| 589 | subject_nontext | -0.274493 |  |  | www.InterceptorPet.com |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 250 | subject_text | +0.328489 | Interceptor |  | (milbemycin oxime) |
| 377 | subject_text | +0.324262 | To report adverse effects access medical information, or obtain additional product information | To report adverse effects, access medical information, or obtain additional product information |  |
| 416 | subject_nontext | +0.319179 |  |  |  |
| 390 | subject_text | +0.305748 | www.InterceptorPet.com |  |  |
| 70 | background | +0.297267 |  |  |  |
| 50 | background | +0.250204 |  |  | If you have any questions concerning the use of this product, contact your veterinarian. |
| 46 | background | +0.248241 |  |  | Interceptor™ |
| 661 | background | +0.243951 |  |  | SH103154E |

## 同包装成功对照

`photo_73d46bde9ac4b4c5` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006859 | +0.011400 |
| 只保留主体分差 | -0.011880 | +0.017619 |
| 主体 token 数 | 384 | 425 |

[成功对照 token 表](../photo_73d46bde9ac4b4c5/tokens.csv) · [对照摘要](../photo_73d46bde9ac4b4c5/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
