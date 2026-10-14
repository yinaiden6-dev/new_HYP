# photo_40d4735eb62e06cf

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000929`，F03 错误参考 `reference_000930`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000930 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.010135 |
| BM25 正负分差 | -0.453918 |
| 0.5 × 图像 z 分差 | -0.107287 |
| 0.5 × BM25 z 分差 | -0.002896 |
| 最终融合分差 | -0.110183 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 350 个。正向贡献 +0.031635，负向贡献 -0.041770，相加等于图像分差 -0.010135。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 370 | -0.005113 | -0.010102 |
| nonvisual | 11 | -0.000320 | -0.021248 |
| subject_nontext | 64 | +0.000573 | +0.006549 |
| subject_text | 286 | -0.005275 | -0.013483 |

仅保留主体 query token 的两候选分差：-0.009820。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.005275。主体框内 token 占有效序列 47.9%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1119 | 79.2% | True | +6.275792 | 63.3% |
| 错误 | 592 | 71.2% | True | +6.494746 | 56.9% |

主体中 286 个 token 与 OCR 相交，229 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 528 | subject_text | -0.490643 | Week 1 - use 2 sprays in each nostril once daily | 7 days of starting use or you get new symptoms such as severe facial pain or thick nasal | Week 1 - use 2 sprays in each nostril once daily |
| 376 | subject_text | -0.428500 | do not share this bottle with anyone else as this may spread germs |  | remember to tell your doctor about all the medicines you take, including this one |
| 488 | subject_text | -0.410638 | Directionsread the Quick Start Guide for how to: |  | ircons |
| 550 | subject_text | -0.358931 | Week 2 through 6 m onths - use 1 or 2 sprays in each nostril once daily, as needed to | Talk to your child's doctor if your child needs to use the spray for longer than | Week 1 - use 2 sprays in each nostril once daily |
| 156 | subject_nontext | -0.347325 |  |  |  |
| 114 | background | -0.344478 |  |  |  |
| 649 | background | -0.340774 |  | Do not usein children under 4 years of age |  |
| 33 | background | -0.317010 |  | pet dander. |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 454 | subject_text | +0.491606 | productyou have severe or frequent nosebleeds | right away.you get new changes to your vision that develop after starting this | has, chicken pox, measles or tuberculosis your symptoms do not get better within |
| 363 | background | +0.478587 |  |  | See below for important information about use in children. Children 4-11: do not use |
| 689 | background | +0.438011 |  | Directionsread the Quick Start Guide for how to: | u iv |
| 690 | background | +0.406433 |  | Directionsread the Quick Start Guide for how to: | remember to tell your doctor about all the medicines you take, including this one |
| 54 | background | +0.396195 |  |  | See below for important information about use in children. Children 4-11: do not use |
| 498 | subject_nontext | +0.337989 |  | Directionsread the Quick Start Guide for how to: | nasal congestionrunny nosesneezingitchy noseitchy, watery eyes |
| 316 | subject_text | +0.321393 | Ask a doctor before use if you have or had glaucoma or cataracts |  | inside your nose. you get an allergic reaction to this product. Seek medical help |
| 499 | subject_nontext | +0.305932 |  | Directionsread the Quick Start Guide for how to: | remember to tell your doctor about all the medicines you take, including this one |

## 成功对照

`photo_7247f008f2067eb2` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010135 | +0.055801 |
| 仅主体分差 | -0.009820 | +0.082910 |
| 主体 token 数 | 350 | 375 |

[对照 token 数据](../photo_7247f008f2067eb2/tokens.csv) · [对照摘要](../photo_7247f008f2067eb2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
