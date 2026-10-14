# B03 · photo_40d4735eb62e06cf

按当前评测真值分析，未新增身份人工复核。

包装 `package_013`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000929`，**纯图像实际 Top-1 错误参考** `reference_000930`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.695574 |
| 错误参考 MaxSim | +0.705709 |
| 正确减错误的分差 | -0.010135 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 350 个。正向贡献 +0.031635，负向贡献 -0.041770，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 370 | -0.005113 | -0.010102 |
| nonvisual | 11 | -0.000320 | -0.021248 |
| subject_nontext | 64 | +0.000573 | +0.006549 |
| subject_text | 286 | -0.005275 | -0.013483 |

最主要负差额来自 `subject_text`，贡献 -0.005275。只保留主体 query token 后，本对分差为 -0.009820。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1119 | 79.2% | True | +6.275792 | 63.3% |
| 错误 | 592 | 71.2% | True | +6.494746 | 56.9% |

主体中 286 个 token 与 OCR 行相交，229 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 528 | subject_text | -0.490643 | Week 1 - use 2 sprays in each nostril once daily | 7 days of starting use or you get new symptoms such as severe facial pain or thick nasal | Week 1 - use 2 sprays in each nostril once daily |
| 376 | subject_text | -0.428500 | do not share this bottle with anyone else as this may spread germs |  | remember to tell your doctor about all the medicines you take, including this one |
| 488 | subject_text | -0.410638 | Directionsread the Quick Start Guide for how to: |  | ircons |
| 550 | subject_text | -0.358931 | Week 2 through 6 m onths - use 1 or 2 sprays in each nostril once daily, as needed to | Talk to your child's doctor if your child needs to use the spray for longer than | Week 1 - use 2 sprays in each nostril once daily |
| 156 | subject_nontext | -0.347325 |  |  |  |
| 114 | background | -0.344478 |  |  |  |
| 649 | background | -0.340774 |  | Do not usein children under 4 years of age |  |
| 33 | background | -0.317010 |  | pet dander. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 454 | subject_text | +0.491606 | productyou have severe or frequent nosebleeds | right away.you get new changes to your vision that develop after starting this | has, chicken pox, measles or tuberculosis your symptoms do not get better within |
| 363 | background | +0.478587 |  |  | See below for important information about use in children. Children 4-11: do not use |
| 689 | background | +0.438011 |  | Directionsread the Quick Start Guide for how to: | u iv |
| 690 | background | +0.406433 |  | Directionsread the Quick Start Guide for how to: | remember to tell your doctor about all the medicines you take, including this one |
| 54 | background | +0.396195 |  |  | See below for important information about use in children. Children 4-11: do not use |
| 498 | subject_nontext | +0.337989 |  | Directionsread the Quick Start Guide for how to: | nasal congestionrunny nosesneezingitchy noseitchy, watery eyes |
| 316 | subject_text | +0.321393 | Ask a doctor before use if you have or had glaucoma or cataracts |  | inside your nose. you get an allergic reaction to this product. Seek medical help |
| 499 | subject_nontext | +0.305932 |  | Directionsread the Quick Start Guide for how to: | remember to tell your doctor about all the medicines you take, including this one |

## 同包装成功对照

`photo_7247f008f2067eb2` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010135 | +0.055801 |
| 只保留主体分差 | -0.009820 | +0.082910 |
| 主体 token 数 | 350 | 375 |

[成功对照 token 表](../photo_7247f008f2067eb2/tokens.csv) · [对照摘要](../photo_7247f008f2067eb2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
