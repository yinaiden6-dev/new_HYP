# B03 · photo_07e45f3cfc816f44

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_004321`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 48 |
| 正确参考 MaxSim | +0.625647 |
| 错误参考 MaxSim | +0.656814 |
| 正确减错误的分差 | -0.031166 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 132 个。正向贡献 +0.016226，负向贡献 -0.047392，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 588 | -0.023159 | -0.028791 |
| nonvisual | 11 | -0.000262 | -0.017404 |
| subject_nontext | 76 | -0.001139 | -0.010957 |
| subject_text | 56 | -0.006606 | -0.086231 |

最主要负差额来自 `background`，贡献 -0.023159。只保留主体 query token 后，本对分差为 -0.042891。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 166 | 27.8% | True | +16.337839 | 96.1% |
| 错误 | 9 | 5.8% | False | 不可评估 | 不可评估 |

主体中 56 个 token 与 OCR 行相交，43 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 436 | subject_text | -0.543645 | the common coldtoothache | 75 TOTAL GELCAPS | headachethe common cold |
| 415 | subject_text | -0.483095 | Uses temporarty relieves minor aches and pains | GELCAPS |  |
| 454 | subject_text | -0.463807 | temporarlly reduces fever | Tylenol and Extra Strength Tylenol• PM. |  |
| 455 | subject_text | -0.459576 | temporarlly reduces fever | Reliever PM |  |
| 289 | background | -0.443222 |  |  |  |
| 458 | subject_text | -0.403478 | premenstrual and menatrual crampa | EXTRA STRENGTH | premenstrual and menstruall cramps |
| 435 | subject_text | -0.396233 | the common coldtoothache |  | headachethe common cold |
| 316 | background | -0.392534 |  | 75 TOTAL GELCAPS | headachethe common cold |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 422 | background | +0.231768 |  | PAIN RELIEVER/NIGHTTIME SLEEP-AID | medical hellp right away. |
| 471 | subject_nontext | +0.217700 |  | ACETAMINOPHEN, USP 500 mg / |  |
| 291 | background | +0.184043 |  | ACETAMINOPHEN, USP 500 mg / | Acetaminophen, Extended-Release Tablets USP, 650 mg |
| 591 | background | +0.174027 |  | ACETAMINOPHEN, USP 500 mg / | 24 CAPLETS† |
| 496 | subject_nontext | +0.173793 |  |  |  |
| 271 | background | +0.160370 |  | ACETAMINOPHEN, USP 500 mg / | 8HR MUSCLE ACHES & PAIN |
| 250 | background | +0.156237 |  |  | medical hellp right away. |
| 275 | background | +0.155886 |  | acetaminophen. Severe liver damage may occur |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen, USP 500 mg.....Pain reliever | 6 | +0.000254 |
| Diphenhydramine HC, USP 25 mg..Nighttime sleep aid | 6 | +0.000254 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_77b67ea38a19f6b3` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.031166 | +0.169751 |
| 只保留主体分差 | -0.042891 | +0.246799 |
| 主体 token 数 | 132 | 143 |

[成功对照 token 表](../photo_77b67ea38a19f6b3/tokens.csv) · [对照摘要](../photo_77b67ea38a19f6b3/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
