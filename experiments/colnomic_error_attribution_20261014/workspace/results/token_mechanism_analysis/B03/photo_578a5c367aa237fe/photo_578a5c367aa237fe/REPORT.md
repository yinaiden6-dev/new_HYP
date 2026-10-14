# B03 · photo_578a5c367aa237fe

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_111`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000512`，**纯图像实际 Top-1 错误参考** `reference_000337`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 16 |
| 正确参考 MaxSim | +0.520943 |
| 错误参考 MaxSim | +0.571336 |
| 正确减错误的分差 | -0.050393 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 170 个。正向贡献 +0.053434，负向贡献 -0.103827，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 550 | -0.031534 | -0.041912 |
| nonvisual | 11 | -0.000057 | -0.003769 |
| subject_nontext | 31 | -0.003315 | -0.078165 |
| subject_text | 139 | -0.015487 | -0.081445 |

最主要负差额来自 `background`，贡献 -0.031534。只保留主体 query token 后，本对分差为 -0.080847。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 963 | 68.8% | True | +9.921557 | 88.2% |
| 错误 | 7 | 20.1% | False | 不可评估 | 不可评估 |

主体中 139 个 token 与 OCR 行相交，95 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 292 | subject_text | -0.630863 | HUMAN WARNINGS:Keepoutofreachofchildren. Notforhuman use. |  |  |
| 247 | background | -0.595182 |  |  |  |
| 487 | background | -0.590091 |  | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide tfe capsules. Do not hardle the |  |
| 167 | background | -0.561219 |  |  |  |
| 291 | subject_text | -0.549270 | HUMAN WARNINGS:Keepoutofreachofchildren. Notforhuman use. | CAUTION: Federal (USA) law restricts this drug to use by or on the |  |
| 16 | background | -0.529703 |  | hyperadrenocorticism in dogs. |  |
| 166 | background | -0.529151 |  | and early pregnancy loss. |  |
| 434 | subject_text | -0.522258 | S bre at c antrolled room temper ature 25℃ (77°F) with exc ursions tetwen 15 °3 0℃ (59" 86° A permited. |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 333 | subject_text | +0.509059 | CoNTRA NDICATONS: The use is c ontraindicated in dogs hat have demonstrated hyper sens itivity b tilos ane. |  |  |
| 623 | background | +0.500601 |  |  |  |
| 602 | background | +0.498040 |  | VETORYLCAPSULES |  |
| 702 | background | +0.487779 |  |  | C |
| 606 | background | +0.478890 |  | VETORYLCAPSULES |  |
| 607 | background | +0.470886 |  | VETORYLCAPSULES | C |
| 579 | background | +0.470579 |  | VETORYLCAPSULES |  |
| 79 | background | +0.452420 |  |  |  |

## 同包装成功对照

`photo_b111a943691a2aa7` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.050393 | +0.172996 |
| 只保留主体分差 | -0.080847 | +0.229867 |
| 主体 token 数 | 170 | 190 |

[成功对照 token 表](../photo_b111a943691a2aa7/tokens.csv) · [对照摘要](../photo_b111a943691a2aa7/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
