# B03 · photo_cc46823d6b68b813

按当前评测真值分析，未新增身份人工复核。

包装 `package_005`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000841`，**纯图像实际 Top-1 错误参考** `reference_000861`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 643 |
| 正确参考 MaxSim | +0.558447 |
| 错误参考 MaxSim | +0.626282 |
| 正确减错误的分差 | -0.067835 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 374 个。正向贡献 +0.019322，负向贡献 -0.087157，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 346 | -0.029268 | -0.061835 |
| nonvisual | 11 | -0.000127 | -0.008432 |
| subject_nontext | 85 | -0.006838 | -0.058805 |
| subject_text | 289 | -0.031602 | -0.079935 |

最主要负差额来自 `subject_text`，贡献 -0.031602。只保留主体 query token 后，本对分差为 -0.075132。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1139 | 80.8% | True | +9.318657 | 84.9% |
| 错误 | 82 | 46.6% | True | +6.180389 | 65.1% |

主体中 289 个 token 与 OCR 行相交，184 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 330 | subject_text | -0.573845 | sympto ms do not improve within 7 days or occur with a fever | Other information | symptomsdo not improve within 7 daysor occur with a fever |
| 295 | subject_text | -0.555204 | oxidase inhibitor (MAOl) (certain drugs fordepression, |  |  |
| 349 | subject_text | -0.536267 | sympto ms do not improve within 7 days or occur with a fever | Active ingredient (in each tablet) | symptomsdo not improve within 7 daysor occur with a fever |
| 289 | subject_text | -0.487778 | If pre gnant orbre ast-fe eding, ask a health pro fession al | for 2 wee ks after stopping the MAOI d rug. If you do not know if | symptomsdo not improve within 7 daysor occur with a fever |
| 338 | subject_text | -0.466698 | te mporar ly relieves nasal conge stion due to the com m on co ld | Sudafed® Congestion | temporarily relie ves nasal congestion due to the com mon cold |
| 227 | subject_text | -0.465604 | he lp or contact a Poison Cont rol Center right a way | te mporarily relieves nasal congestion due to the common co ld, |  |
| 269 | subject_text | -0.446471 | If pre gnant orbre ast-fe eding, ask a health pro fession al | :43262 29 C7 | symptomsdo not improve within 7 daysor occur with a fever |
| 544 | background | -0.420192 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 241 | subject_text | +0.350474 | Important: Re ad all product information be fore using. Keep this box for important information | Important: Re ad all pro duct information before using. Keep this box for important information |  |
| 99 | background | +0.319333 |  | 24 Tablets \|30 mg Each | Pseudoephedrine Hydrochloride Extended Release Tablets, 120 mg |
| 100 | background | +0.290865 |  | 24 Tablets \|30 mg Each | LANDOVER,MD 20785-1-877-846-9949 |
| 377 | subject_text | +0.280786 | 6 to 11 years |  | under 12 yearsof age |
| 221 | subject_text | +0.252707 | Important: Re ad all product information be fore using. Keep this box for important information | Important: Re ad all pro duct information before using. Keep this box for important information | Other information |
| 281 | subject_text | +0.227794 | Important: Re ad all product information be fore using. Keep this box for important information | Nasal Decongestant |  |
| 620 | background | +0.197061 |  | 24 Tablets \|30 mg Each | Pseudoephedrine Hydrochloride Extended Release Tablets, 120 mg |
| 539 | subject_text | +0.196404 | do not take more than 8 tablets |  | adults andchildren |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Pseudoephedrine HCl 30 img. | 5 | -0.000071 |
| eachtabletcontains: calcium 20mg | 14 | -0.002958 |
| take 2 tablets every 4 to 6 hours | 11 | -0.001789 |
| do not take more than 4 tablets | 10 | -0.001030 |
| do not take more than 8 tablets | 5 | -0.000464 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_fd6c39a0ef3bd92a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.067835 | +0.060388 |
| 只保留主体分差 | -0.075132 | +0.085904 |
| 主体 token 数 | 374 | 352 |

[成功对照 token 表](../photo_fd6c39a0ef3bd92a/tokens.csv) · [对照摘要](../photo_fd6c39a0ef3bd92a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
