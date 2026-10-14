# B03 · photo_0a612f53747b2554

按当前评测真值分析，未新增身份人工复核。

包装 `package_013`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000929`，**纯图像实际 Top-1 错误参考** `reference_002389`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.667990 |
| 错误参考 MaxSim | +0.670892 |
| 正确减错误的分差 | -0.002902 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 160 个。正向贡献 +0.033447，负向贡献 -0.036349，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 560 | +0.001855 | +0.002422 |
| nonvisual | 11 | -0.000420 | -0.027931 |
| subject_nontext | 28 | -0.000473 | -0.012350 |
| subject_text | 132 | -0.003864 | -0.021400 |

最主要负差额来自 `subject_text`，贡献 -0.003864。只保留主体 query token 后，本对分差为 -0.019816。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1100 | 66.8% | True | +6.052051 | 69.2% |
| 错误 | 8 | 13.9% | False | 不可评估 | 不可评估 |

主体中 132 个 token 与 OCR 行相交，117 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 389 | subject_text | -0.450469 | If pregnant or breast-feeding, ask a health professional before use. | Do not usein children under 4 years of age | If pregnant or breast-feeding, ask a |
| 409 | subject_text | -0.431234 | Directions read the Quick Start Guide for how to: |  |  |
| 438 | background | -0.415172 |  | 7 days of starting use or you get new symptoms such as severe facial pain or thick nasal | ADULTS AND CHILDREN 12 YEARS OF |
| 449 | subject_text | -0.409739 | Week 2 through 6 months - use 1 or 2 sprays in each nostril once daily, as needed to | Week 1 - use 2 sprays in each nostril once daily | R |
| 436 | subject_text | -0.362037 | do not use more than directed | Directionsread the Quick Start Guide for how to: | use this product only once a day |
| 552 | subject_nontext | -0.291818 |  | Talk to your child's doctor if your child needs to use the spray for longer than | should use for the shortest amount of |
| 212 | background | -0.287641 |  | Keep out of reach of children. In case of overdose, get medical help or contact a Poison | remember to tell your doctor about all the |
| 356 | subject_text | -0.269362 | remember to tell your doctor about all the medicines you take, induding this one | do not share this bottle with anyone else as this may spread germs | remember to tell your doctor about all the |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 417 | subject_text | +0.539044 | Keep out of reach of children. In case of overdose, get medical help or contact a Poison | Keep out of reach of children. In case of overdose, get medical help or contact a Poison |  |
| 403 | background | +0.386295 |  | Keep out of reach of children. In case of overdose, get medical help or contact a Poison |  |
| 129 | background | +0.384159 |  |  | ADULTS AND CHILDREN 12 YEARS OF |
| 69 | background | +0.372496 |  | right away.you get new changes to your vision that develop after starting this |  |
| 589 | background | +0.354964 |  |  |  |
| 167 | background | +0.316797 |  |  | ADULTS AND CHILDREN 12 YEARS OF |
| 407 | background | +0.310172 |  | Keep out of reach of children. In case of overdose, get medical help or contact a Poison |  |
| 613 | background | +0.281500 |  | prime the bottleuse the spraydean the spray nozzle | clean the spray nozzle |

## 同包装成功对照

`photo_cdd1aa422a4f6dc0` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002902 | +0.072623 |
| 只保留主体分差 | -0.019816 | +0.098508 |
| 主体 token 数 | 160 | 150 |

[成功对照 token 表](../photo_cdd1aa422a4f6dc0/tokens.csv) · [对照摘要](../photo_cdd1aa422a4f6dc0/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
