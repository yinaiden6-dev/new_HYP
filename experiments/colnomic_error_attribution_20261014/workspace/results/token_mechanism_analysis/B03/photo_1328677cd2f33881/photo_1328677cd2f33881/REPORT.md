# B03 · photo_1328677cd2f33881

按当前评测真值分析，未新增身份人工复核。

包装 `package_064`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001321`，**纯图像实际 Top-1 错误参考** `reference_004422`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.698164 |
| 错误参考 MaxSim | +0.698472 |
| 正确减错误的分差 | -0.000308 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 468 个。正向贡献 +0.053352，负向贡献 -0.053660，相加等于原图像分差。与原评分缓存的最大误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 252 | -0.006426 | -0.018642 |
| nonvisual | 11 | +0.000092 | +0.006127 |
| subject_nontext | 84 | +0.001679 | +0.014612 |
| subject_text | 384 | +0.004347 | +0.008275 |

最主要负差额来自 `background`，贡献 -0.006426。只保留主体 query token 后，本对分差为 +0.009413。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 912 | 83.6% | True | +9.095682 | 71.9% |
| 错误 | 36 | 10.2% | False | 不可评估 | 不可评估 |

主体中 384 个 token 与 OCR 行相交，290 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 629 | background | -0.540302 |  |  | conditions, or Parkinson's disease), or for 2 weeks after |
| 542 | subject_nontext | -0.452229 |  |  | prescription monoamine oxidase inhibitor (MAOl) (certain |
| 569 | subject_text | -0.450074 | disease), or for 2 weeks after stopping the MAOI drug |  | conditions, or Parkinson's disease), or for 2 weeks after |
| 440 | subject_text | -0.437753 | medical help or contact a Poison Control Center right away. |  | dose can cause liver damage. In case of overdose, get |
| 628 | background | -0.430668 |  | • with any other drug containing acetaminophen | conditions, or Parkinson's disease), or for 2 weeks after |
| 243 | background | -0.406718 |  |  | such as occurs with smoking, asthma, or emphysema |
| 218 | subject_text | -0.406584 |  thyroid disease | • thyroid disease | equate |
| 543 | background | -0.401515 |  |  | carefully read each label to ensure correct dosing |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 115 | subject_text | +0.558589 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels | COUGH, MUCUS & CONGESTION |  |
| 116 | subject_text | +0.551896 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels | COUGH, MUCUS & CONGESTION | 4,000 mg of acetaminophen in 24 hours with other |
| 118 | subject_text | +0.551335 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels | COUGH, MUCUS & CONGESTION | Dextromethorphan HBr 10 mg.....…Cough suppressant |
| 658 | background | +0.527947 |  |  |  |
| 117 | subject_text | +0.519610 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels |  | Pain reliever/fever reducer |
| 109 | subject_text | +0.513438 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels | Slze | Ask a doctor before use if you have |
| 129 | subject_text | +0.490998 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels |  |  |
| 139 | subject_text | +0.476469 | Alka-Seltzer Plus® Cough, Mucus & Congestion Maximum Strength PowerMax® Gels | Bayer, the Bayer Cross, Alka-Seltzer Plus, and PowerMax are registered trademarks of Bayer. | Dextromethorphan HBr 10 mg.....…Cough suppressant |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen 325 mg | 4 | -0.000656 |
| hydrobromide 10 mg. | 4 | -0.000281 |
| Guaifenesin 200 mg | 4 | +0.000428 |
| Phenylephrine hydrochloride 5 mg.Nasal decongestant | 9 | -0.000182 |
| Phenylephrine hydrochloride 5 mg | 6 | -0.000011 |
| • more than 4,000 mg of acetaminophen in 24 hours | 16 | -0.001345 |
| adults and children 12 years and over: take 2 capsules | 8 | -0.000754 |
| with water every 4 hours. Do not exceed 10 capsules | 8 | -0.000754 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_1d2280603a42552a` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000308 | +0.161353 |
| 只保留主体分差 | +0.009413 | +0.209334 |
| 主体 token 数 | 468 | 165 |

[成功对照 token 表](../photo_1d2280603a42552a/tokens.csv) · [对照摘要](../photo_1d2280603a42552a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
