# B03 · photo_64625e0a749be4a5

按当前评测真值分析，未新增身份人工复核。

包装 `package_042`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001129`，**纯图像实际 Top-1 错误参考** `reference_003973`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 19 |
| 正确参考 MaxSim | +0.522315 |
| 错误参考 MaxSim | +0.533137 |
| 正确减错误的分差 | -0.010822 |
| F03 的正确参考排名（辅助对照） | 4 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 120 个。正向贡献 +0.016841，负向贡献 -0.027663，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.008776 | -0.010551 |
| nonvisual | 11 | -0.000232 | -0.016415 |
| subject_nontext | 5 | +0.000102 | +0.015955 |
| subject_text | 115 | -0.001916 | -0.012981 |

最主要负差额来自 `background`，贡献 -0.008776。只保留主体 query token 后，本对分差为 -0.011775。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 652 | 71.9% | True | +8.200064 | 71.7% |
| 错误 | 8 | 23.9% | True | +18.110134 | 100.0% |

主体中 115 个 token 与 OCR 行相交，107 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 368 | subject_text | -0.332436 | 1 lozenge every |  |  |
| 402 | subject_text | -0.275799 | and allow the lozenge to slowly dissolve. Minimize swallowing. |  | Nicotine Polacrilex Lozenge, |
| 448 | background | -0.247284 | Pharmacy | If you are pregnant or breast-feeding, only use this medicine | Ask a doctor or pharmacist before use if you are |
| 336 | subject_text | -0.228717 | 1 lozenge every |  |  |
| 646 | background | -0.218336 |  |  | the other until completely dissolved |
| 661 | background | -0.205463 |  |  | the other until completely dissolved |
| 343 | subject_text | -0.199602 | Other information store at 20-25℃ (68-77°F) | Questions or comments?call tll-free 1-866-751-9303 | the lozenge for a longer period to keep from smoking, talk to your |
| 764 | background | -0.190304 | Ent |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 491 | background | +0.258681 |  | if you smoke your first cigarette within 30 minutes of |  |
| 396 | background | +0.252782 | MAGNES | Inactive ingredients acesulfame potassium, calcium | Inactive ingredients acacia, acesulfame potassium, calcium |
| 129 | background | +0.237924 | 1-5 June 20 |  |  |
| 98 | background | +0.237665 | 1-5 June 20 |  |  |
| 97 | background | +0.220352 | 1-5 June 20 |  |  |
| 427 | background | +0.208233 | CALCIU | Inactive ingredients acesulfame potassium, calcium | Inactive ingredients acacia, acesulfame potassium, calcium |
| 428 | background | +0.192429 | MAGNES | Inactive ingredients acesulfame potassium, calcium | Inactive ingredients acacia, acesulfame potassium, calcium |
| 314 | subject_text | +0.185353 | This product is not manufactured or distributed by GlapSrithKine |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| waking up, use 4 mg nicotine lozenge if you smoke your | 9 | +0.000216 |
| 2 mg nicotine lozenge aocording to the following 12 week | 12 | +0.000261 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_d2377339b5c178d4` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010822 | +0.006043 |
| 只保留主体分差 | -0.011775 | +0.014886 |
| 主体 token 数 | 120 | 135 |

[成功对照 token 表](../photo_d2377339b5c178d4/tokens.csv) · [对照摘要](../photo_d2377339b5c178d4/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
