# B03 · photo_f30f6769bf236e52

按当前评测真值分析，未新增身份人工复核。

包装 `package_049`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001182`，**纯图像实际 Top-1 错误参考** `reference_001219`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 591 |
| 正确参考 MaxSim | +0.543617 |
| 错误参考 MaxSim | +0.635182 |
| 正确减错误的分差 | -0.091565 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 210 个。正向贡献 +0.017470，负向贡献 -0.109035，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 510 | -0.065600 | -0.094027 |
| nonvisual | 11 | -0.000154 | -0.010204 |
| subject_nontext | 60 | -0.006844 | -0.083380 |
| subject_text | 150 | -0.018967 | -0.092434 |

最主要负差额来自 `background`，贡献 -0.065600。只保留主体 query token 后，本对分差为 -0.089847。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 535 | 70.6% | True | +6.666894 | 70.8% |
| 错误 | 9 | 32.5% | True | +10.133734 | 89.0% |

主体中 150 个 token 与 OCR 行相交，114 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 494 | subject_text | -0.591645 | Questions or comments: | product or any of Its Ingredlents | Questions or comments? |
| 403 | background | -0.584154 |  | product or any of Its Ingredlents | Questions or comments? |
| 483 | background | -0.569889 |  | minor aches and pains with accompanying | ca 1-877-770-3183 Mon-Fi 8:00 AM EST to 5:00 PM PST |
| 402 | background | -0.553993 |  |  | Questions or comments? |
| 487 | background | -0.547866 |  | Drug Facts (continued) | ca 1-877-770-3183 Mon-Fi 8:00 AM EST to 5:00 PM PST |
| 635 | background | -0.547101 |  | 300000186 |  |
| 674 | background | -0.546710 |  | Extra Strength TYLENOL® PM Rapid Release Geis | ca 1-877-770-3183 Mon-Fi 8:00 AM EST to 5:00 PM PST |
| 594 | background | -0.539678 |  |  | Questions or comments? |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 453 | subject_text | +0.508479 | Extra Strength TYLENOL® PM Rapid Release Gels | PAIN RELIEF PM | take 2 gelcaps every 6 hours whilke |
| 511 | subject_text | +0.362427 | DISTRIBUTED BY: fred's, Inc. | DISTRIBUTED BY: fred's, Inc. |  |
| 513 | subject_text | +0.343670 | DISTRIBUTED BY: fred's, Inc. | DISTRIBUTED BY: fred's, Inc. |  |
| 716 | background | +0.339191 |  | DISTRIBUTED BY: fred's, Inc. |  |
| 687 | background | +0.290415 |  | DISTRIBUTED BY: fred's, Inc. |  |
| 472 | subject_text | +0.287790 | 100% | 100% |  |
| 652 | background | +0.280237 |  | This product Is not affillated with the makers/owners of | Acetaminophen |
| 461 | background | +0.274884 |  | This product Is not affillated with the makers/owners of | This product is not affiliated with the makers/owners of |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% | 5 | -0.000117 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_441c254874dcdd8d` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.091565 | +0.140427 |
| 只保留主体分差 | -0.089847 | +0.131961 |
| 主体 token 数 | 210 | 190 |

[成功对照 token 表](../photo_441c254874dcdd8d/tokens.csv) · [对照摘要](../photo_441c254874dcdd8d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
