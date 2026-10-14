# B03 · photo_5885af0890eb6878

按当前评测真值分析，未新增身份人工复核。

包装 `package_064`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001321`，**纯图像实际 Top-1 错误参考** `reference_001312`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.613628 |
| 错误参考 MaxSim | +0.614223 |
| 正确减错误的分差 | -0.000595 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 156 个。正向贡献 +0.019482，负向贡献 -0.020077，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 564 | -0.001641 | -0.002127 |
| nonvisual | 11 | -0.000193 | -0.012836 |
| subject_nontext | 64 | +0.001001 | +0.011429 |
| subject_text | 92 | +0.000238 | +0.001892 |

最主要负差额来自 `background`，贡献 -0.001641。只保留主体 query token 后，本对分差为 +0.005805。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 62 | 5.0% | False | 不可评估 | 不可评估 |
| 错误 | 69 | 5.3% | False | 不可评估 | 不可评估 |

主体中 92 个 token 与 OCR 行相交，55 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 292 | subject_text | -0.348868 | USONLY | how2recycle.info |  |
| 270 | subject_text | -0.222058 | owrifo |  |  |
| 118 | background | -0.153906 |  | persistent or chronic cough such as occurs with |  |
| 155 | background | -0.153238 |  |  |  |
| 306 | background | -0.151988 |  |  |  |
| 156 | background | -0.149559 |  | Seltzer |  |
| 98 | background | -0.143828 |  |  |  |
| 218 | background | -0.135736 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 630 | background | +0.219981 |  |  |  |
| 394 | subject_text | +0.200581 | Acetaminophen/Pain reliever-fever reducer |  |  |
| 434 | subject_nontext | +0.179750 |  | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax@ Gels |  |
| 431 | subject_nontext | +0.171589 |  |  |  |
| 611 | background | +0.157519 |  |  |  |
| 287 | background | +0.152607 |  | Bayer | Bayer, the Bayer Cross, Alka-Seltzer Plus, and PowerMax are registered trademarks of Bayer. |
| 710 | background | +0.149005 |  |  |  |
| 412 | subject_text | +0.145379 | COUGH, MUCUS & CONGESTION | COUGH, MUCUS & CONGESTION |  |

## 同包装成功对照

`photo_1d2280603a42552a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000595 | +0.028870 |
| 只保留主体分差 | +0.005805 | +0.031832 |
| 主体 token 数 | 156 | 165 |

[成功对照 token 表](../photo_1d2280603a42552a/tokens.csv) · [对照摘要](../photo_1d2280603a42552a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
