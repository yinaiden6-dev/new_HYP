# B03 · photo_6c1be6707e17adb6

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000138`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.700015 |
| 错误参考 MaxSim | +0.730228 |
| 正确减错误的分差 | -0.030213 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 220 个。正向贡献 +0.035165，负向贡献 -0.065379，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 500 | -0.010801 | -0.015791 |
| nonvisual | 11 | +0.000186 | +0.012327 |
| subject_nontext | 71 | -0.005403 | -0.055625 |
| subject_text | 149 | -0.014195 | -0.069642 |

最主要负差额来自 `subject_text`，贡献 -0.014195。只保留主体 query token 后，本对分差为 -0.065118。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 548 | 69.9% | True | +6.540431 | 71.7% |
| 错误 | 15 | 10.4% | False | 不可评估 | 不可评估 |

主体中 149 个 token 与 OCR 行相交，114 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 313 | subject_text | -0.570269 | Animal Weight (Pounds) |  | Animal Weight (Pounds) |
| 322 | background | -0.566697 |  |  | Animal Weight (Pounds) |
| 473 | subject_text | -0.562856 | Table 2. Increxxa 25 Calf Dosing Guide (25 mg/mL) | Injectable Solution is also indicated for the treatment of bovine | Animal Weight (Pounds) |
| 392 | subject_text | -0.525673 | 8 |  |  |
| 575 | subject_text | -0.520858 | sodum hydroxide added to adjust pH. | Approved by FDA under ANADA # 200-665 |  |
| 13 | background | -0.450855 |  |  | Animal Weight (Pounds) |
| 90 | background | -0.442060 |  |  | for the control of SRD associated in groups of pigs |
| 332 | subject_text | -0.439760 | 10 |  | 75 |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 86 | background | +0.410026 |  |  |  |
| 98 | background | +0.383608 |  | Increxxa25 | RESPIRmycin 25 |
| 311 | subject_text | +0.359179 | Animal Weight (Pounds) | more than 11.5 mL per injection site. | RESPIRmycin 25 |
| 208 | background | +0.354435 |  | Increxxa 25 | RESPIRmycin 25 |
| 461 | background | +0.349546 |  | Elanco | RESPIRmycin 25 |
| 465 | background | +0.346857 |  | more than 11.5 mL per injection site. | RESPIRmycin 25 |
| 541 | background | +0.319619 |  | Each mL of Increxa 25 contains 25 mg of tulathromycin as the | RESPIRmycin 25 |
| 217 | subject_text | +0.315821 | swine respiratory disease (SRD) and for the control of SRD in | Increxxa 25 | RESPIRmycin 25 |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| of 2.5 mg kg (1 mL/22 Ib) Body Weight (BW). Do not inject | 18 | -0.001728 |
| more than 4 mL per injection site. | 5 | -0.000546 |
| Table 1. Increxxa 25 Swine Dosing Guide (25 mg'mL) | 16 | -0.001516 |
| of 2.5 mg kg (1 mL/22 lb) bcdy weight (BW). Do not inject | 18 | -0.001499 |
| more than 11.5 mL per injection site. | 6 | -0.000324 |
| Table 2. Increxxa 25 Calf Dosing Guide (25 mg/mL) | 14 | -0.002199 |
| Each mL of Increxxa 25 contains 25 mg of tulathromycin as the | 18 | -0.000293 |
| free base in a 50% propylene glycol vehicle, moncthiogtycerol | 9 | -0.000522 |
| (5 mg/mL), ctric acid (4.8 mg/mL) with hydrochloric acd and | 18 | -0.002127 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.030213 | +0.039794 |
| 只保留主体分差 | -0.065118 | +0.046755 |
| 主体 token 数 | 220 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
