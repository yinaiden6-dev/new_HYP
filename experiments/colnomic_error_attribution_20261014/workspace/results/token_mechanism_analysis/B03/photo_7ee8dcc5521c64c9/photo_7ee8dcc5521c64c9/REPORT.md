# B03 · photo_7ee8dcc5521c64c9

按当前评测真值分析，未新增身份人工复核。

包装 `package_049`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001182`，**纯图像实际 Top-1 错误参考** `reference_003249`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 166 |
| 正确参考 MaxSim | +0.625578 |
| 错误参考 MaxSim | +0.711505 |
| 正确减错误的分差 | -0.085926 |
| F03 的正确参考排名（辅助对照） | 9 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 220 个。正向贡献 +0.014356，负向贡献 -0.100283，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 500 | -0.058985 | -0.086237 |
| nonvisual | 11 | -0.000253 | -0.016821 |
| subject_nontext | 67 | -0.005615 | -0.061261 |
| subject_text | 153 | -0.021073 | -0.100683 |

最主要负差额来自 `background`，贡献 -0.058985。只保留主体 query token 后，本对分差为 -0.088677。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 188 | 73.2% | True | +8.037643 | 71.3% |
| 错误 | 7 | 8.8% | False | 不可评估 | 不可评估 |

主体中 153 个 token 与 OCR 行相交，124 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 322 | background | -0.542474 |  | Keep out of the reach of children. |  |
| 369 | subject_text | -0.537095 | Liver marminge This product contains |  |  |
| 657 | background | -0.526965 |  |  | If a skin reaction occurs, stop use and seek medical |
| 477 | subject_text | -0.518851 | U |  | If a skin reaction occurs, stop use and seek medical |
| 399 | background | -0.515159 |  | Pain Reliever/Nighttime Sleep Aid |  |
| 409 | subject_text | -0.505224 | nours | more than 4,000 mg of acetaminophen In 24 |  |
| 459 | background | -0.505000 |  | skin reactlons. Symptoms may Include: | If a skin reaction occurs, stop use and seek medical |
| 326 | background | -0.501522 |  | Keep out of the reach of children. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 552 | subject_text | +0.301049 | In chlidren under 12 years of age | if you have ever had an allergic reaction to this |  |
| 56 | background | +0.243961 |  | if you have ever had an allergic reaction to this | NIGHTTIME SLEEP-AID |
| 277 | subject_text | +0.226077 | Purpose |  | Pain reliever |
| 129 | background | +0.209826 |  | skin reactlons. Symptoms may Include: | if you are allergic to acetaminophen or any of the |
| 438 | subject_nontext | +0.182590 |  | skin reactlons. Symptoms may Include: | if you are allergic to acetaminophen or any of the |
| 132 | background | +0.182415 |  | product or any of Its Ingredlents | inactive ingredients in this product |
| 9 | background | +0.173452 |  |  |  |
| 43 | background | +0.169592 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acatamlinophen, Usp 500 mg | 6 | -0.000325 |
| Diphenhydramina HC/ 25 mg.. Miphitime sleap-ald | 17 | -0.001439 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_441c254874dcdd8d` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.085926 | +0.070027 |
| 只保留主体分差 | -0.088677 | +0.089461 |
| 主体 token 数 | 220 | 190 |

[成功对照 token 表](../photo_441c254874dcdd8d/tokens.csv) · [对照摘要](../photo_441c254874dcdd8d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
