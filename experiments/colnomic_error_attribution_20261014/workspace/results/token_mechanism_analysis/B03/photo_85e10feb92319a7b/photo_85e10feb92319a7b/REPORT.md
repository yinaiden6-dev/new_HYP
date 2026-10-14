# B03 · photo_85e10feb92319a7b

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_111`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000512`，**纯图像实际 Top-1 错误参考** `reference_000337`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 16 |
| 正确参考 MaxSim | +0.530401 |
| 错误参考 MaxSim | +0.565092 |
| 正确减错误的分差 | -0.034692 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 84 个。正向贡献 +0.042876，负向贡献 -0.077568，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 636 | -0.034325 | -0.039452 |
| nonvisual | 11 | -0.000203 | -0.013517 |
| subject_nontext | 17 | +0.000080 | +0.003424 |
| subject_text | 67 | -0.000243 | -0.002654 |

最主要负差额来自 `background`，贡献 -0.034325。只保留主体 query token 后，本对分差为 -0.001424。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 898 | 56.1% | True | +12.341897 | 88.1% |
| 错误 | 6 | 12.5% | False | 不可评估 | 不可评估 |

主体中 67 个 token 与 OCR 行相交，53 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 248 | background | -0.553441 |  | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide tfe capsules. Do not hardle the |  |
| 197 | background | -0.538956 |  |  |  |
| 67 | background | -0.533798 |  | Dechra |  |
| 187 | background | -0.522950 |  | VETORYLCAPSULES |  |
| 567 | background | -0.521122 |  | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide tfe capsules. Do not hardle the |  |
| 68 | background | -0.508816 |  |  |  |
| 188 | background | -0.496336 |  |  |  |
| 208 | background | -0.471184 |  |  | CONTRAINDICATIONS: Do not use in |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 447 | subject_text | +0.461032 | blood volume ard renal per fusi on. Potas sium sparing dur etic s(e g. spi ronola ctore) should rot be us ed wih VETORYL | VETORYLCAPSULES |  |
| 458 | subject_text | +0.456502 | have aldos erore -lowering efe dts which may be add fve, im pair ing the patient's ability to mai rtain normal de ctrolytes, | VETORYLCAPSULES | 21071645 |
| 695 | background | +0.449609 |  | VETORYLCAPSULES |  |
| 646 | background | +0.443614 |  | VETORYLCAPSULES |  |
| 446 | background | +0.412362 | blood volume ard renal per fusi on. Potas sium sparing dur etic s(e g. spi ronola ctore) should rot be us ed wih VETORYL | VETORYLCAPSULES |  |
| 658 | background | +0.393340 |  | VETORYLCAPSULES |  |
| 659 | background | +0.392126 |  | VETORYLCAPSULES |  |
| 686 | background | +0.388680 |  |  |  |

## 同包装成功对照

`photo_1b6a28535da238b9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.034692 | +0.209194 |
| 只保留主体分差 | -0.001424 | +0.298266 |
| 主体 token 数 | 84 | 66 |

[成功对照 token 表](../photo_1b6a28535da238b9/tokens.csv) · [对照摘要](../photo_1b6a28535da238b9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
