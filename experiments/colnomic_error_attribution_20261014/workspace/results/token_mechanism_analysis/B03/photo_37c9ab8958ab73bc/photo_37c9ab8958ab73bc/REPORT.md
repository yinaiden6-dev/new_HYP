# B03 · photo_37c9ab8958ab73bc

按当前评测真值分析，未新增身份人工复核。

包装 `package_049`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001182`，**纯图像实际 Top-1 错误参考** `reference_002727`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 355 |
| 正确参考 MaxSim | +0.590142 |
| 错误参考 MaxSim | +0.649418 |
| 正确减错误的分差 | -0.059276 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 72 个。正向贡献 +0.022189，负向贡献 -0.081465，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.051687 | -0.058307 |
| nonvisual | 11 | -0.000201 | -0.013385 |
| subject_nontext | 19 | -0.000549 | -0.021112 |
| subject_text | 53 | -0.006839 | -0.094326 |

最主要负差额来自 `background`，贡献 -0.051687。只保留主体 query token 后，本对分差为 -0.075006。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 199 | 57.6% | True | +11.168757 | 83.3% |
| 错误 | 3 | 0.0% | False | 不可评估 | 不可评估 |

主体中 53 个 token 与 OCR 行相交，46 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 453 | subject_text | -0.512498 | contact a Polson Control Center right away. | DISTRIBUTED BY: fred's, Inc. | Quick medical attention is critical for adults |
| 447 | background | -0.489042 |  |  |  |
| 443 | background | -0.488518 |  |  |  |
| 333 | subject_text | -0.477900 | trouble urtnating due to an enkarged prostate gland |  | trouble urinating due to an enlarged |
| 413 | subject_text | -0.471641 | fever get worse or lasts more than 3 days |  |  |
| 632 | background | -0.461038 |  |  | may canse liver danage. In case of overdose, |
| 543 | background | -0.441017 |  |  | prostate gland = glaucoma |
| 446 | background | -0.439855 |  |  | Overdose warning |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 532 | subject_text | +0.418084 | 531000003 |  |  |
| 542 | background | +0.392319 |  |  |  |
| 533 | subject_text | +0.349958 | 531000003 | FREE AREA | Contains No Aspirin |
| 541 | background | +0.315037 |  | FREE AREA | Contains No Aspirin |
| 722 | background | +0.314135 |  |  | Contains No Aspirin |
| 127 | background | +0.255434 |  |  |  |
| 314 | subject_nontext | +0.251430 |  |  |  |
| 712 | background | +0.231837 |  | 300000186 | ACETAMINOPHEN 500mg |

## 同包装成功对照

`photo_375b67daeeeeff5f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.059276 | +0.111672 |
| 只保留主体分差 | -0.075006 | +0.133162 |
| 主体 token 数 | 72 | 66 |

[成功对照 token 表](../photo_375b67daeeeeff5f/tokens.csv) · [对照摘要](../photo_375b67daeeeeff5f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
