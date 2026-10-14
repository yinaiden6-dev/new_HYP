# B03 · photo_0c63c2a740b336ea

按当前评测真值分析，未新增身份人工复核。

包装 `package_040`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001112`，**纯图像实际 Top-1 错误参考** `reference_003821`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 7 |
| 正确参考 MaxSim | +0.618569 |
| 错误参考 MaxSim | +0.630062 |
| 正确减错误的分差 | -0.011493 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 312 个。正向贡献 +0.028873，负向贡献 -0.040366，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 456 | -0.007358 | -0.012570 |
| nonvisual | 11 | +0.000161 | +0.011393 |
| subject_nontext | 102 | -0.001175 | -0.008972 |
| subject_text | 210 | -0.003121 | -0.011577 |

最主要负差额来自 `background`，贡献 -0.007358。只保留主体 query token 后，本对分差为 -0.010726。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 624 | 68.9% | True | +6.802026 | 66.2% |
| 错误 | 28 | 18.8% | True | +5.956310 | 64.5% |

主体中 210 个 token 与 OCR 行相交，141 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 81 | background | -0.551568 |  | Active ingredient (in each tablet) Purpose | less severe symptoms. |
| 325 | background | -0.460075 |  |  |  |
| 49 | background | -0.435523 |  |  | less severe symptoms. |
| 331 | subject_text | -0.401533 | if pregnant: ask a health professional before use | be careful when driving a motor vehicle or operating | Keep out of reach of children. In case of overdose, get |
| 267 | subject_text | -0.358973 | if pregnant: ask a health professional before use |  | Keep out of reach of children. In case of overdose, get |
| 226 | background | -0.344907 |  |  | cellulose, polyethylene glycol, titanium dioxide |
| 405 | subject_nontext | -0.344387 |  | Other information | Other information |
| 313 | subject_text | -0.334478 | polyethyfene glycol, povidone, titanium dioxide |  | cellulose, polyethylene glycol, titanium dioxide |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 573 | subject_text | +0.496634 | P.O. BOX 99. PLEASANTON. CA 04560-0000 |  | FC005407 |
| 446 | subject_nontext | +0.471640 |  |  | Camber Consumer Care Inc, Piscataway, NJ 08854, USA. |
| 283 | subject_text | +0.448548 | This product is not manufactured or distributed by Johason |  | contains no ingredient made from a gluten-containing |
| 251 | subject_text | +0.436806 | This product is not manufactured or distributed by Johason |  | contains no ingredient made from a gluten-containing |
| 541 | subject_text | +0.432052 | P.O. BOX 99. PLEASANTON. CA 04560-0000 |  | FC005407 |
| 29 | background | +0.422101 | Inducer |  | Questions or comments? |
| 539 | subject_text | +0.421989 | This product is not manufactured or distributed by Johason |  | croscarmellose sodium, hypromellose, lactose |
| 413 | subject_text | +0.398449 | P.O. BOX 99. PLEASANTON. CA 04560-0000 | DISTRIBUTED BY: BETTER LIVING BRANDS LLC | contains no ingredient made from a gluten-containing |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| one 10 mg tablet once daily; do | 14 | -0.001343 |
| not take more than one 10 mg | 7 | -0.000312 |
| tablet in 24 hours. A 5 mg | 12 | -0.000613 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_2df687d8a2f89f40` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.011493 | +0.023698 |
| 只保留主体分差 | -0.010726 | +0.041633 |
| 主体 token 数 | 312 | 312 |

[成功对照 token 表](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
