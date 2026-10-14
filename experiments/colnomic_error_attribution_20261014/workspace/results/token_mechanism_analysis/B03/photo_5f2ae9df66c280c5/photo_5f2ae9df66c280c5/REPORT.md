# B03 · photo_5f2ae9df66c280c5

按当前评测真值分析，未新增身份人工复核。

包装 `package_022`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000976`，**纯图像实际 Top-1 错误参考** `reference_002511`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 1685 |
| 正确参考 MaxSim | +0.484844 |
| 错误参考 MaxSim | +0.588713 |
| 正确减错误的分差 | -0.103869 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 224 个。正向贡献 +0.004787，负向贡献 -0.108656，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 496 | -0.067122 | -0.098924 |
| nonvisual | 11 | -0.000168 | -0.011155 |
| subject_nontext | 80 | -0.008244 | -0.075333 |
| subject_text | 144 | -0.028335 | -0.143839 |

最主要负差额来自 `background`，贡献 -0.067122。只保留主体 query token 后，本对分差为 -0.119373。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1218 | 66.8% | True | +11.212577 | 92.2% |
| 错误 | 7 | 24.2% | False | 不可评估 | 不可评估 |

主体中 144 个 token 与 OCR 行相交，99 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 467 | subject_nontext | -0.461361 |  |  |  |
| 251 | background | -0.362397 |  |  |  |
| 422 | background | -0.352874 |  |  |  |
| 318 | subject_text | -0.341143 | Quadns ? cal 1-a77-770-31s: Waadays 0:00 AM t0 4:30 PM EST. |  |  |
| 661 | background | -0.340434 |  |  dronsiness may occur  avoid alcoholic drinics  alcohol, sedadivea, and tranqultars may Increeae drowsinees |  |
| 247 | background | -0.332303 |  |  |  |
| 223 | background | -0.322595 |  | Do mot mae I you have aver had an allergic meaction to this product or any ol ts ingredents or to an anthistamina containing |  |
| 596 | background | -0.317095 |  |  dronsiness may occur  avoid alcoholic drinics  alcohol, sedadivea, and tranqultars may Increeae drowsinees |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 308 | subject_nontext | +0.259125 |  | Kea t of naach of cids. In cae ol verdos, get medcal he or conact a Polaon Contrl Canter rght ay. 1-800-222-1222 |  |
| 289 | subject_nontext | +0.215517 |  |  |  |
| 288 | subject_nontext | +0.197466 |  | Kea t of naach of cids. In cae ol verdos, get medcal he or conact a Polaon Contrl Canter rght ay. 1-800-222-1222 |  |
| 307 | subject_nontext | +0.166221 |  |  |  |
| 108 | background | +0.162201 |  | Kea t of naach of cids. In cae ol verdos, get medcal he or conact a Polaon Contrl Canter rght ay. 1-800-222-1222 |  |
| 36 | background | +0.151823 |  |  |  |
| 270 | background | +0.129393 |  |  |  |
| 229 | background | +0.128262 |  |  | COLDTAC |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| A 5 mg product may bo sapproprtate for loes sovore symptons. | 8 | -0.001603 |
| one 10 ng bablet ance da ly; do not take nare than ana 10 mg tablst In 24 hours. | 12 | -0.002222 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_1d7f3a9eaf659041` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.103869 | +0.293207 |
| 只保留主体分差 | -0.119373 | +0.406199 |
| 主体 token 数 | 224 | 195 |

[成功对照 token 表](../photo_1d7f3a9eaf659041/tokens.csv) · [对照摘要](../photo_1d7f3a9eaf659041/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
