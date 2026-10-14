# B03 · photo_2db3d27da013553c

按当前评测真值分析，未新增身份人工复核。

包装 `package_050`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003087`，**纯图像实际 Top-1 错误参考** `reference_001680`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 29 |
| 正确参考 MaxSim | +0.594712 |
| 错误参考 MaxSim | +0.605385 |
| 正确减错误的分差 | -0.010673 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 108 个。正向贡献 +0.024005，负向贡献 -0.034678，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 612 | -0.010015 | -0.011963 |
| nonvisual | 11 | -0.000257 | -0.017083 |
| subject_nontext | 43 | -0.001139 | -0.019361 |
| subject_text | 65 | +0.000739 | +0.008306 |

最主要负差额来自 `background`，贡献 -0.010015。只保留主体 query token 后，本对分差为 -0.002710。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 190 | 43.0% | True | +12.118803 | 82.2% |
| 错误 | 5 | 9.5% | False | 不可评估 | 不可评估 |

主体中 65 个 token 与 OCR 行相交，54 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 229 | subject_nontext | -0.386036 |  | PENTREXCILINA |  |
| 234 | subject_text | -0.322305 | Purpose |  |  |
| 80 | background | -0.217583 |  |  |  |
| 126 | background | -0.212359 |  |  | Inactive ingredientscarnauba wax, corn |
| 260 | background | -0.194378 |  |  | CLOSE ANGLE |
| 488 | background | -0.191768 |  |  |  |
| 442 | background | -0.191193 |  |  |  |
| 349 | subject_nontext | -0.189481 |  |  | Do not use |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 91 | background | +0.310512 |  |  | take 2 caplets every 6 hours not to exceed |
| 250 | subject_nontext | +0.294611 |  |  |  |
| 430 | subject_text | +0.276908 | taking sodium restricted diet |  |  |
| 254 | subject_text | +0.245763 | Nasal decongestant | Cold& | Purpose |
| 674 | background | +0.234591 |  | Liver warning: This product contains acetaminophen. |  |
| 433 | subject_text | +0.232689 | smoking, asthma chronic bronchitis, or emphysema |  | Ask a doctor or pharmacist before use if you are taking a |
| 706 | background | +0.229663 |  | 6 fl oz (177 ml) | BAYER |
| 322 | background | +0.228505 |  | Tos | if you are allergic to aspirin or any other pain reliever/fever |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| (in each 20 mL) | 2 | -0.000136 |
| Acetaminophen 500 mg | 2 | -0.000136 |
| Dextromethorphan HBr 26.66 mg | 3 | -0.000233 |
| Guaitenesin USP 400 mg | 2 | -0.000136 |
| Phenylephrine HCl 10 mg | 3 | +0.000145 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_8bd2c2ece5146d67` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010673 | +0.229639 |
| 只保留主体分差 | -0.002710 | +0.321379 |
| 主体 token 数 | 108 | 130 |

[成功对照 token 表](../photo_8bd2c2ece5146d67/tokens.csv) · [对照摘要](../photo_8bd2c2ece5146d67/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
