# B03 · photo_5e415d5f852b3694

按当前评测真值分析，未新增身份人工复核。

包装 `package_050`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003087`，**纯图像实际 Top-1 错误参考** `reference_000825`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.644825 |
| 错误参考 MaxSim | +0.654701 |
| 正确减错误的分差 | -0.009876 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 224 个。正向贡献 +0.043238，负向贡献 -0.053114，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 496 | -0.005760 | -0.008488 |
| nonvisual | 11 | -0.000029 | -0.001919 |
| subject_nontext | 78 | -0.002509 | -0.023511 |
| subject_text | 146 | -0.001579 | -0.007905 |

最主要负差额来自 `background`，贡献 -0.005760。只保留主体 query token 后，本对分差为 -0.013339。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 501 | 67.5% | True | +6.408398 | 63.7% |
| 错误 | 121 | 9.3% | False | 不可评估 | 不可评估 |

主体中 146 个 token 与 OCR 行相交，101 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 251 | subject_text | -0.470590 | dose (overdose) may cause liver damage. In case of | even if you do not notice any signs or symptoms. | get medical help or contact a Poison Control Center |
| 254 | subject_text | -0.450614 | dose (overdose) may cause liver damage. In case of | PENTREXCILINA |  |
| 634 | subject_nontext | -0.447779 |  |  | San Diego, CA |
| 252 | subject_text | -0.414000 | dose (overdose) may cause liver damage. In case of |  |  |
| 249 | subject_nontext | -0.387847 |  |  |  |
| 181 | background | -0.382896 |  |  |  |
| 653 | subject_text | -0.382561 | FDA Registered Facility |  | FDA Registered Facility |
| 502 | background | -0.377406 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 151 | subject_text | +0.422754 | Drug Facts (continued) | redness or swelling is present |  |
| 482 | background | +0.409444 |  |  |  |
| 415 | subject_nontext | +0.371612 |  |  |  |
| 532 | subject_text | +0.356522 | Call 619-600-5632 MON to FRI. 9 a.m. to 6 p.m. PTZ |  |  |
| 463 | background | +0.350977 |  |  |  |
| 92 | background | +0.343774 |  | redness or swelling is present |  |
| 428 | background | +0.339412 |  |  | Anhydrous Citric Acid, Cherry Flavor, Glycerin, Menthol |
| 235 | subject_text | +0.337127 | Overdose warning Taking more than the recommended | PENTREXCILINA | Effective for |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 20 mL in dosing cup | 3 | +0.000579 |
| each 20 mL contains: | 6 | +0.000448 |
| sodium 20 mg | 3 | +0.000199 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_1089688627444cf6` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.009876 | +0.071800 |
| 只保留主体分差 | -0.013339 | +0.108605 |
| 主体 token 数 | 224 | 112 |

[成功对照 token 表](../photo_1089688627444cf6/tokens.csv) · [对照摘要](../photo_1089688627444cf6/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
