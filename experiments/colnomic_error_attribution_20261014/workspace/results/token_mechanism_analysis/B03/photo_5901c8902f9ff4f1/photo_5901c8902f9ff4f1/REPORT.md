# B03 · photo_5901c8902f9ff4f1

按当前评测真值分析，未新增身份人工复核。

包装 `package_111`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000512`，**纯图像实际 Top-1 错误参考** `reference_000508`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.666344 |
| 错误参考 MaxSim | +0.675060 |
| 正确减错误的分差 | -0.008717 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 91 个。正向贡献 +0.020725，负向贡献 -0.029441，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 629 | -0.008297 | -0.009643 |
| nonvisual | 11 | +0.000076 | +0.005072 |
| subject_nontext | 18 | -0.001053 | -0.042759 |
| subject_text | 73 | +0.000557 | +0.005578 |

最主要负差额来自 `background`，贡献 -0.008297。只保留主体 query token 后，本对分差为 -0.003983。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 180 | 31.0% | True | +8.739523 | 82.4% |
| 错误 | 85 | 22.3% | True | +9.524116 | 86.8% |

主体中 73 个 token 与 OCR 行相交，47 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 516 | subject_nontext | -0.362663 |  | VETORYLCAPSULES |  |
| 695 | background | -0.338807 |  |  |  |
| 692 | background | -0.328450 |  |  |  |
| 694 | background | -0.319623 |  | VETORYLCAPSULES |  |
| 197 | background | -0.315977 |  |  |  |
| 561 | background | -0.301073 |  | VETORYLCAPSULES |  |
| 703 | background | -0.299887 |  |  |  |
| 193 | background | -0.287880 |  | S tbre at c ontrolled room temper ature 25°C (77°F) with exc ursions betwe n 15 °-3 0C (59º -86° F) permitted. | Capsules as both drugs have the potential to inhibit aldosterone, increasing the likelihood of hyperkalemia. |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 260 | background | +0.384037 |  | 120 | 20 |
| 379 | background | +0.296472 | 120 | (trilostane) 120 mg | 200 |
| 580 | background | +0.275660 |  | (trilostane) 120 mg |  |
| 259 | background | +0.255121 |  | (trilostane) 120 mg |  |
| 79 | background | +0.235925 |  | 120 |  |
| 73 | background | +0.235499 |  |  |  |
| 400 | subject_text | +0.223526 | 120 | (trilostane) 120 mg |  |
| 599 | background | +0.190832 |  | (trilostane) 120 mg |  |

## 同包装成功对照

`photo_1b6a28535da238b9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008717 | +0.010059 |
| 只保留主体分差 | -0.003983 | +0.014152 |
| 主体 token 数 | 91 | 66 |

[成功对照 token 表](../photo_1b6a28535da238b9/tokens.csv) · [对照摘要](../photo_1b6a28535da238b9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
