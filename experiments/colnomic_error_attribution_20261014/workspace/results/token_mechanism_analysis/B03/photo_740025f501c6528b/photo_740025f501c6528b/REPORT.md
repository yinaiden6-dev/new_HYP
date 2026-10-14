# B03 · photo_740025f501c6528b

按当前评测真值分析，未新增身份人工复核。

包装 `package_029`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001023`，**纯图像实际 Top-1 错误参考** `reference_004316`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 160 |
| 正确参考 MaxSim | +0.613731 |
| 错误参考 MaxSim | +0.686117 |
| 正确减错误的分差 | -0.072386 |
| F03 的正确参考排名（辅助对照） | 21 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 187 个。正向贡献 +0.024239，负向贡献 -0.096625，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 533 | -0.055688 | -0.076375 |
| nonvisual | 11 | -0.000351 | -0.023326 |
| subject_nontext | 81 | -0.004550 | -0.041060 |
| subject_text | 106 | -0.011798 | -0.081359 |

最主要负差额来自 `background`，贡献 -0.055688。只保留主体 query token 后，本对分差为 -0.063903。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 695 | 59.0% | True | +8.882153 | 83.6% |
| 错误 | 8 | 21.1% | True | +15.317191 | 89.7% |

主体中 106 个 token 与 OCR 行相交，87 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 105 | background | -0.561310 |  |  |  |
| 110 | background | -0.554366 |  |  |  |
| 111 | background | -0.553754 |  | Cold |  |
| 656 | background | -0.501350 |  | If a skin or general allergic reaction occurs, stop use and seek | 20 TABLETS |
| 420 | background | -0.489583 |  |  |  |
| 657 | background | -0.487205 |  | If a skin or general allergic reaction occurs, stop use and seek | 20 TABLETS |
| 275 | subject_nontext | -0.477427 |  | Phenylephrine hydrochloride 5 mg....... Nasal decongestant |  |
| 294 | subject_text | -0.443928 | Pain relievemMever reducer |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 256 | subject_nontext | +0.473225 |  |  |  |
| 370 | subject_nontext | +0.406389 |  |  | Body aches & pains, Fever, Sore throat |
| 250 | subject_nontext | +0.398426 |  |  |  |
| 90 | background | +0.323674 |  |  | Body aches & pains, Fever, Sore throat |
| 57 | background | +0.310573 |  |  |  |
| 54 | background | +0.294726 |  |  | Body aches & pains, Fever, Sore throat |
| 236 | subject_nontext | +0.290988 |  |  |  |
| 703 | background | +0.289332 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen 250 mg | 6 | -0.001306 |
| Dextromethorphan hydrobromide 10 mg.Cough suppressant | 12 | -0.000305 |
| Chlorphentamina maleato 2 mg | 5 | -0.001136 |
| Phenyfephrine hydrochloride 5 mg. Nasal decongestant | 13 | +0.001479 |
| more than 4,000 mg of acataminophen in 24 hours | 9 | -0.002057 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_079284acc905a73f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.072386 | +0.197774 |
| 只保留主体分差 | -0.063903 | +0.302592 |
| 主体 token 数 | 187 | 120 |

[成功对照 token 表](../photo_079284acc905a73f/tokens.csv) · [对照摘要](../photo_079284acc905a73f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
