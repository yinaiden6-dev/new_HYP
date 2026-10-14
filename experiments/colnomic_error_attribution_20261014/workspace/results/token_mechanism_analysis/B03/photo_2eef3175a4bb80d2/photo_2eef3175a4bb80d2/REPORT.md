# B03 · photo_2eef3175a4bb80d2

按当前评测真值分析，未新增身份人工复核。

包装 `package_110`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000506`，**纯图像实际 Top-1 错误参考** `reference_000157`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.559324 |
| 错误参考 MaxSim | +0.571344 |
| 正确减错误的分差 | -0.012020 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 171 个。正向贡献 +0.017094，负向贡献 -0.029114，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 597 | -0.010454 | -0.013640 |
| nonvisual | 11 | -0.000008 | -0.000584 |
| subject_nontext | 77 | -0.001333 | -0.013483 |
| subject_text | 94 | -0.000225 | -0.001868 |

最主要负差额来自 `background`，贡献 -0.010454。只保留主体 query token 后，本对分差为 -0.007098。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 73 | 62.2% | True | +11.789275 | 74.7% |
| 错误 | 21 | 2.2% | False | 不可评估 | 不可评估 |

主体中 94 个 token 与 OCR 行相交，30 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 427 | subject_text | -0.434176 | Net Contents: 500 mL |  | Approved by FDA under ANADA # 200-128 |
| 291 | background | -0.359149 | Bupivacaine Hydrochloride | Vetrimycin™ 200 |  |
| 323 | background | -0.257081 | Bupivacaine Hydrochloride | Vetrimycin™ 200 |  |
| 662 | background | -0.253457 |  | preruminating (veal) calves; and swine. | For animal use only |
| 631 | background | -0.226971 |  |  |  |
| 531 | subject_nontext | -0.223715 |  |  | For animal use only |
| 466 | subject_text | -0.202571 | Approved by FDA under ANADA # 200-306 |  |  |
| 630 | background | -0.198975 |  | Each mL contains 200 mg of oxytetracycline. | For animal use only |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 349 | subject_text | +0.276215 | NDC 13985-035-05 | NDC 13985-035-05 |  |
| 521 | background | +0.238813 | STERILE | Vetrimycin™ 200 is a sterile, preconstituted solution | 500 mL |
| 498 | subject_text | +0.204688 | Approved by FDA under ANADA # 200-306 | Approved by FDA under ANADA # 200-306 | Net Contents: 500 mL (16.91 oz) |
| 277 | subject_nontext | +0.194603 |  |  |  |
| 381 | subject_text | +0.193233 | NDC 13985-035-05 | NDC 13985-035-05 |  |
| 407 | subject_text | +0.188112 | Each mL contains 200 mg of oxytetracycline. |  |  |
| 520 | background | +0.172038 | STERILE | Vetrimycin™ 200 is a sterile, preconstituted solution | 500 mL |
| 486 | background | +0.168674 |  |  | At the first signs of pneumonia,* administer |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Net Contents: 500 mL | 8 | -0.000680 |
| Each mL contains 200 mg of oxytetracycline. | 8 | +0.000495 |
| (oxytetracycline injection) 200 mg/mL | 8 | +0.000836 |
| 500 mL | 2 | +0.000106 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_4f3036c68b1b3b4e` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.012020 | +0.002674 |
| 只保留主体分差 | -0.007098 | +0.031366 |
| 主体 token 数 | 171 | 169 |

[成功对照 token 表](../photo_4f3036c68b1b3b4e/tokens.csv) · [对照摘要](../photo_4f3036c68b1b3b4e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
