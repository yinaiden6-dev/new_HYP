# B03 · photo_0500abf430ca34bc

按当前评测真值分析，未新增身份人工复核。

包装 `package_027`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_001007`，**纯图像实际 Top-1 错误参考** `reference_004360`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 665 |
| 正确参考 MaxSim | +0.515890 |
| 错误参考 MaxSim | +0.641298 |
| 正确减错误的分差 | -0.125407 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 238 个。正向贡献 +0.005221，负向贡献 -0.130628，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 482 | -0.080023 | -0.121363 |
| nonvisual | 11 | -0.000259 | -0.017205 |
| subject_nontext | 42 | -0.005789 | -0.100760 |
| subject_text | 196 | -0.039336 | -0.146707 |

最主要负差额来自 `background`，贡献 -0.080023。只保留主体 query token 后，本对分差为 -0.138599。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1557 | 75.9% | True | +13.446237 | 91.9% |
| 错误 | 12 | 4.9% | False | 不可评估 | 不可评估 |

主体中 196 个 token 与 OCR 行相交，167 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 383 | background | -0.587759 |  |  | Questions or comments? |
| 364 | background | -0.568625 |  |  |  |
| 166 | background | -0.563621 |  | STOP SMOKING AID | Drug Facts |
| 67 | background | -0.558559 |  |  | Warnings |
| 154 | subject_text | -0.552874 | Drug Facts(continued) |  |  |
| 68 | background | -0.547758 |  |  | Warnings |
| 205 | background | -0.540970 |  |  | Warnings |
| 354 | subject_text | -0.532174 | ka ap container tigh ty cboad and protac t fron fight | (5 bottles Inside - contalners of 27) | Inactive ingredients |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 548 | background | +0.171293 |  | 1. You must be motivated to quit. | Ask a doctor before use if you have |
| 436 | subject_text | +0.118558 | Distributed by: Dr. Reddy's Laboratbries, Inc. |  | Visit MinuteClinic® for |
| 455 | subject_text | +0.116057 | Issued:01/21 | Issued:01/21 | 1 to 2 hours |
| 116 | background | +0.111335 |  |  | it is important to complete treatment. If you feel you |
| 726 | nonvisual | +0.093192 |  |  |  |
| 450 | subject_text | +0.087214 | wakag ap, use 2 rmg nosine iozngs |  |  |
| 57 | background | +0.087203 |  | Original |  |
| 98 | background | +0.081768 |  |  | if you smoke your first cigarette more than 30 minutes after waking up, use 2 mg nicotine |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| waking up, une 4 mg niootine lazenge acoording to the | 6 | -0.000637 |
| Micctine podacrikes, 4 mg. | 3 | -0.000543 |
| wakag ap, use 2 rmg nosine iozngs | 7 | -0.000154 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_0a8643c0c2c88704` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.125407 | +0.134963 |
| 只保留主体分差 | -0.138599 | +0.187224 |
| 主体 token 数 | 238 | 270 |

[成功对照 token 表](../photo_0a8643c0c2c88704/tokens.csv) · [对照摘要](../photo_0a8643c0c2c88704/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
