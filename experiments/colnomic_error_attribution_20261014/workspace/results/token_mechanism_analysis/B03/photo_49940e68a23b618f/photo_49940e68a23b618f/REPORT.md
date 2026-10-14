# B03 · photo_49940e68a23b618f

按当前评测真值分析，未新增身份人工复核。

包装 `package_119`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001971`，**纯图像实际 Top-1 错误参考** `reference_001973`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 29 |
| 正确参考 MaxSim | +0.630068 |
| 错误参考 MaxSim | +0.659880 |
| 正确减错误的分差 | -0.029812 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 286 个。正向贡献 +0.018390，负向贡献 -0.048202，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 482 | -0.017621 | -0.028479 |
| nonvisual | 11 | -0.000037 | -0.002631 |
| subject_nontext | 100 | -0.003294 | -0.025663 |
| subject_text | 186 | -0.008860 | -0.037106 |

最主要负差额来自 `background`，贡献 -0.017621。只保留主体 query token 后，本对分差为 -0.033105。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 283 | 40.1% | True | +10.667932 | 83.6% |
| 错误 | 11 | 26.6% | True | +13.536251 | 97.2% |

主体中 186 个 token 与 OCR 行相交，108 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 396 | subject_text | -0.428091 | GRE |  | Compare to the atme ingedires |
| 165 | background | -0.426767 |  |  | Acto ingrodlent (it euch capix) |
| 299 | subject_text | -0.417006 | pnd |  | mn e |
| 372 | subject_nontext | -0.406167 |  | do sol ble iroe fban 6 cap bis i 34 lo ms,  abood lec td lyadocbr | Wamihp |
| 141 | subject_text | -0.387225 | DrugFacts | READ AND NEEP CARTON FOR CON FLETE WAFNING S AND INFOIMATION | Acto ingrodlent (it euch capix) |
| 317 | subject_text | -0.386721 | huaaonrealohdcdhiseoydywheurghtspodid |  | mn e |
| 613 | background | -0.359841 |  | do sol ble iroe fban 6 cap bis i 34 lo ms,  abood lec td lyadocbr |  |
| 123 | subject_nontext | -0.353465 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 490 | subject_nontext | +0.253240 |  |  | ("capsule-shaped tablets) |
| 494 | subject_text | +0.230704 | 7 | 144 |  |
| 493 | subject_text | +0.212060 | 7 | 144 |  |
| 435 | subject_text | +0.210958 | ain Reliever |  | ACETAMINOPHEN 250 mg |
| 669 | background | +0.201894 |  | NDC0363-9131-44 |  |
| 614 | background | +0.195332 |  |  |  |
| 393 | background | +0.181200 |  |  |  |
| 611 | background | +0.172312 |  | I'sinrodinocou, sip ue andsek nedicl hdp igit any |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| ACETAMINOPHEN 25O mg & IBUPROFEN (NSAID) 125 mg TABLETS/ | 31 | -0.000790 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_ca454f8a9d866abb` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.029812 | +0.001656 |
| 只保留主体分差 | -0.033105 | -0.010628 |
| 主体 token 数 | 286 | 264 |

[成功对照 token 表](../photo_ca454f8a9d866abb/tokens.csv) · [对照摘要](../photo_ca454f8a9d866abb/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
