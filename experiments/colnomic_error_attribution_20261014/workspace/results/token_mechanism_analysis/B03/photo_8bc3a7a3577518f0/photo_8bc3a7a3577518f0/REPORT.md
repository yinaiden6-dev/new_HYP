# B03 · photo_8bc3a7a3577518f0

按当前评测真值分析，未新增身份人工复核。

包装 `package_049`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001182`，**纯图像实际 Top-1 错误参考** `reference_003251`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 1004 |
| 正确参考 MaxSim | +0.546763 |
| 错误参考 MaxSim | +0.712963 |
| 正确减错误的分差 | -0.166201 |
| F03 的正确参考排名（辅助对照） | 16 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 190 个。正向贡献 +0.006727，负向贡献 -0.172928，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 530 | -0.099997 | -0.137920 |
| nonvisual | 11 | -0.000495 | -0.032905 |
| subject_nontext | 40 | -0.008609 | -0.157321 |
| subject_text | 150 | -0.057100 | -0.278269 |

最主要负差额来自 `background`，贡献 -0.099997。只保留主体 query token 后，本对分差为 -0.252806。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 491 | 76.1% | True | +13.662335 | 79.1% |
| 错误 | 9 | 12.8% | False | 不可评估 | 不可评估 |

主体中 150 个 token 与 OCR 行相交，121 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 478 | subject_text | -0.561013 | recommended dose (overdose) may cause liver |  |  |
| 551 | subject_text | -0.552283 | do not take more than directed |  | (overd ose) may cause liver damage. In case of overd ose, get medical help or contact a |
| 670 | background | -0.529336 |  | if you have ever had an allergic reaction to this | (overd ose) may cause liver damage. In case of overd ose, get medical help or contact a |
| 278 | subject_text | -0.528064 | Ask a doctor or pharmacist before use If you are | minor aches and pains with accompanying | Ask a doctor or pharmacist before use if you are taking the blood thinning drug warfarin |
| 488 | background | -0.526253 |  | if you have ever had an allergic reaction to this | (overd ose) may cause liver damage. In case of overd ose, get medical help or contact a |
| 453 | subject_text | -0.524090 | Keep out of the reach of children. |  | If pregnant or breast-feeding, ask a health care professional before use. Keep out of |
| 479 | subject_nontext | -0.515020 |  | Uses |  |
| 510 | subject_text | -0.514467 | critical for adults as well as for children even If |  | Poison Control Center right away (1-800-222-1222). Quick medical attent ion is critical for adults as |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 9 | background | +0.218693 |  |  |  |
| 11 | background | +0.187806 |  |  |  |
| 116 | background | +0.167451 |  | sleeplessness persists continuously for more | more than 10 daysfever gets worse or lasts for more than 3 daysredness or swelling is present |
| 103 | background | +0.164496 |  | sleeplessness persists continuously for more | more than 10 daysfever gets worse or lasts for more than 3 daysredness or swelling is present |
| 199 | background | +0.163398 |  |  | Drug Facts |
| 190 | background | +0.156252 |  |  |  |
| 169 | background | +0.133462 |  |  |  |
| 47 | background | +0.124401 |  |  |  |

## 同包装成功对照

`photo_441c254874dcdd8d` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.166201 | +0.094601 |
| 只保留主体分差 | -0.252806 | +0.127914 |
| 主体 token 数 | 190 | 190 |

[成功对照 token 表](../photo_441c254874dcdd8d/tokens.csv) · [对照摘要](../photo_441c254874dcdd8d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
