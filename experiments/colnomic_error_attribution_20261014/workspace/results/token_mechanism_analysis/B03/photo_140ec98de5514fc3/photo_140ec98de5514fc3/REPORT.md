# B03 · photo_140ec98de5514fc3

按当前评测真值分析，未新增身份人工复核。

包装 `package_022`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000976`，**纯图像实际 Top-1 错误参考** `reference_000891`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 190 |
| 正确参考 MaxSim | +0.618635 |
| 错误参考 MaxSim | +0.646553 |
| 正确减错误的分差 | -0.027919 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 90 个。正向贡献 +0.010921，负向贡献 -0.038840，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 630 | -0.024581 | -0.028521 |
| nonvisual | 11 | -0.000408 | -0.027136 |
| subject_nontext | 55 | -0.002166 | -0.028783 |
| subject_text | 35 | -0.000764 | -0.015960 |

最主要负差额来自 `background`，贡献 -0.024581。只保留主体 query token 后，本对分差为 -0.023796。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 210 | 45.4% | True | +6.989539 | 73.2% |
| 错误 | 6 | 20.9% | False | 不可评估 | 不可评估 |

主体中 35 个 token 与 OCR 行相交，20 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 286 | background | -0.207105 |  |  | • for the temporary relief of minor aches and |
| 372 | subject_nontext | -0.205738 |  | Exp. |  |
| 157 | background | -0.203464 |  |  |  |
| 577 | background | -0.199146 |  |  |  |
| 585 | background | -0.191242 |  |  |  |
| 699 | background | -0.176760 |  |  |  |
| 229 | background | -0.174732 |  |  |  |
| 170 | background | -0.167850 |  | Q |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 39 | background | +0.259553 |  |  | Drug Facts |
| 13 | background | +0.254701 |  |  | Drug Facts |
| 161 | background | +0.239070 |  |  | Ask a doctor before usa if |
| 433 | subject_nontext | +0.155510 |  |  |  |
| 16 | background | +0.142431 |  |  |  |
| 703 | background | +0.125226 |  |  | Questions or Comments? |
| 314 | subject_text | +0.122866 | De not maw if Indilvidunl bileder unit ia epan or tom |  | Reye's syndrome: Children and teenagers |
| 454 | subject_nontext | +0.120010 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Catirizaine HCl 10 mg | 4 | -0.000269 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_0cf112e57c488bf5` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.027919 | +0.179655 |
| 只保留主体分差 | -0.023796 | +0.287998 |
| 主体 token 数 | 90 | 90 |

[成功对照 token 表](../photo_0cf112e57c488bf5/tokens.csv) · [对照摘要](../photo_0cf112e57c488bf5/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
