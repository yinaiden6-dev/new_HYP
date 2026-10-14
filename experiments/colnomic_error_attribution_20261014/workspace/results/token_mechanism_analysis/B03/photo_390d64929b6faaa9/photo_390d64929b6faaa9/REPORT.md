# B03 · photo_390d64929b6faaa9

按当前评测真值分析，未新增身份人工复核。

包装 `package_054`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001401`，**纯图像实际 Top-1 错误参考** `reference_004751`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 18 |
| 正确参考 MaxSim | +0.588659 |
| 错误参考 MaxSim | +0.604282 |
| 正确减错误的分差 | -0.015624 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 88 个。正向贡献 +0.019644，负向贡献 -0.035267，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 632 | -0.015059 | -0.017418 |
| nonvisual | 11 | +0.000001 | +0.000074 |
| subject_nontext | 55 | -0.000447 | -0.005945 |
| subject_text | 33 | -0.000118 | -0.002610 |

最主要负差额来自 `background`，贡献 -0.015059。只保留主体 query token 后，本对分差为 -0.004694。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 712 | 59.9% | True | +12.793465 | 88.4% |
| 错误 | 6 | 23.9% | False | 不可评估 | 不可评估 |

主体中 33 个 token 与 OCR 行相交，28 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 336 | subject_nontext | -0.297121 |  | Keep out of reach of chilldren. In case of overdose, get medical help or contact a Poison Control Center right away. | READ AND KEEP CARTON FOR COMPLETE |
| 695 | background | -0.282665 |  | Sodium bicarbonate / Antacid | even one used on skin if you have sleeplessness |
| 405 | background | -0.263267 |  |  | breathingweakness in one part or side ofbody |
| 124 | background | -0.259788 |  | Seltzer |  |
| 295 | background | -0.257119 |  | need relief from heartburn, upset | even one used on skin if you have sleeplessness |
| 115 | background | -0.245484 |  |  | even one used on skin if you have sleeplessness |
| 707 | background | -0.241773 |  | Seltzer |  |
| 546 | background | -0.241337 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 603 | background | +0.331936 |  |  |  |
| 433 | subject_text | +0.262460 | feel taint vomt blood have bloody or black aools have stomach pain that does not get better | you experience any of the following signs of stomach bleeding |  |
| 570 | background | +0.239256 |  |  |  |
| 350 | subject_nontext | +0.216209 |  |  |  |
| 420 | background | +0.178910 |  |  |  |
| 65 | background | +0.174377 |  |  | microcrystalline cellulose, polyethylene glycol, polyvinyl |
| 88 | background | +0.169401 |  |  |  |
| 428 | background | +0.168582 |  |  | or lasts more than 10 days sleeplessness persists |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Anhydroua citric acid 1000 mg | 2 | -0.000242 |
| Asprin 325 mg (NSA/D) | 1 | -0.000222 |
| Sodum bicartborate (heat-trealed) 1916 mg | 4 | -0.000257 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_60a129173d7c7794` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.015624 | +0.211563 |
| 只保留主体分差 | -0.004694 | +0.300937 |
| 主体 token 数 | 88 | 77 |

[成功对照 token 表](../photo_60a129173d7c7794/tokens.csv) · [对照摘要](../photo_60a129173d7c7794/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
