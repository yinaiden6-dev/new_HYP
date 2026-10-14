# B03 · photo_3487dd8994f438f3

按当前评测真值分析，未新增身份人工复核。

包装 `package_047`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001139`，**纯图像实际 Top-1 错误参考** `reference_001680`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.587804 |
| 错误参考 MaxSim | +0.594371 |
| 正确减错误的分差 | -0.006567 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 96 个。正向贡献 +0.032647，负向贡献 -0.039214，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 624 | -0.007733 | -0.009059 |
| nonvisual | 11 | -0.000225 | -0.014975 |
| subject_nontext | 28 | +0.000784 | +0.020463 |
| subject_text | 68 | +0.000608 | +0.006534 |

最主要负差额来自 `background`，贡献 -0.007733。只保留主体 query token 后，本对分差为 +0.010597。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 822 | 51.5% | True | +4.889283 | 70.5% |
| 错误 | 9 | 23.4% | True | +8.154249 | 89.4% |

主体中 68 个 token 与 OCR 行相交，56 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 192 | background | -0.471138 |  | Itcontains important information. | temporarily reduces fever |
| 512 | subject_text | -0.461672 | Warnings | Itcontains important information. | temporarily reduces fever |
| 372 | background | -0.453383 |  | Itcontains important information. | temporarily reduces fever |
| 572 | background | -0.383526 |  | Itcontains important information. | temporarily reduces fever |
| 352 | background | -0.318241 |  | Itcontains important information. |  |
| 719 | background | -0.282777 |  |  |  |
| 712 | background | -0.272544 |  | Itcontains important information. |  |
| 193 | background | -0.245292 |  | Itcontains important information. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 548 | subject_text | +0.546074 | PARENTS: | PARENTS | BAYER |
| 549 | subject_text | +0.502196 | Leam about teninee abuse | PARENTS |  |
| 634 | background | +0.452318 |  |  |  |
| 258 | background | +0.451405 |  |  |  |
| 579 | background | +0.428506 |  |  | professional before use |
| 278 | background | +0.424185 |  | Dextromethorphan HBr 60 mg..... Cough suppressant | NSAIDs (aspirin, ibuprofen, naproxen, or others) |
| 655 | background | +0.420010 |  |  | professional before use |
| 568 | background | +0.415239 | PARENTS: |  | Ask a doctor or pharmacist before use if you are taking a |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Gual enesin 1200 mg.Expectorant | 6 | +0.000186 |
| De tromethorphan Her 60 mg... Coughsuppreeant | 7 | +0.000778 |
| Gualenesin 1200 mg | 3 | -0.000122 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_c0ed1211516e6abd` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006567 | +0.025764 |
| 只保留主体分差 | +0.010597 | +0.056976 |
| 主体 token 数 | 96 | 96 |

[成功对照 token 表](../photo_c0ed1211516e6abd/tokens.csv) · [对照摘要](../photo_c0ed1211516e6abd/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
