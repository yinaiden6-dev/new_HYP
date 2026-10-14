# B03 · photo_cc01a8280f094648

按当前评测真值分析，未新增身份人工复核。

包装 `package_026`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000995`，**纯图像实际 Top-1 错误参考** `reference_003677`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.593815 |
| 错误参考 MaxSim | +0.597635 |
| 正确减错误的分差 | -0.003820 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 210 个。正向贡献 +0.024998，负向贡献 -0.028818，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.001837 | -0.002565 |
| nonvisual | 11 | +0.000142 | +0.010066 |
| subject_nontext | 35 | -0.000494 | -0.011000 |
| subject_text | 175 | -0.001631 | -0.007259 |

最主要负差额来自 `background`，贡献 -0.001837。只保留主体 query token 后，本对分差为 -0.007883。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 295 | 73.5% | True | +7.480057 | 66.0% |
| 错误 | 13 | 31.4% | True | +7.355783 | 76.7% |

主体中 175 个 token 与 OCR 行相交，133 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 543 | subject_text | -0.338323 | stomach pain | One 14-Day Course of Treatment | stomach pain |
| 473 | subject_text | -0.321272 | pain with shortness of breath; sweating; pain | spreading to arms, neck or shoulders; or | chest pain or shoulder pain with shortness of breath; sweating; pain |
| 344 | subject_text | -0.311843 | sweating or dizziness chest pain or shoulder | black stoolsheartburn with lightheadedness, | omeprazole may cause severe skin reactions. Symptoms may include: |
| 762 | background | -0.309760 |  | spreading to arms, neck or shoulders; or | chest pain or shoulder pain with shortness of breath; sweating; pain |
| 531 | subject_text | -0.306630 | Allergy alert: |  |  |
| 639 | background | -0.296075 |  |  | stomach pain |
| 499 | subject_text | -0.270182 | Allergy alert: |  |  |
| 595 | background | -0.236560 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 573 | subject_text | +0.451473 | This may be a sign of a more serious condition. |  |  |
| 590 | background | +0.382148 |  | Lansoprazole 15 mg.. |  |
| 494 | subject_text | +0.379135 | Lansoprazole 15 mg. | Delayed-Release | 0meprazole 20 mg....... |
| 398 | subject_text | +0.290368 | 1402024049402 | Capsules 15 mg |  |
| 486 | background | +0.247631 |  |  |  |
| 24 | background | +0.247464 | TAOLETS UBP, 10 mg-Antihiglamine |  |  |
| 276 | subject_nontext | +0.237513 |  |  | omeprazole may cause severe skin reactions. Symptoms may include: |
| 334 | subject_text | +0.231180 | Acid Reducer | Reducer |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Lansoprazole 15 mg. | 7 | +0.000936 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_ab6fd7a33337ccee` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003820 | +0.070055 |
| 只保留主体分差 | -0.007883 | +0.171679 |
| 主体 token 数 | 210 | 209 |

[成功对照 token 表](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
