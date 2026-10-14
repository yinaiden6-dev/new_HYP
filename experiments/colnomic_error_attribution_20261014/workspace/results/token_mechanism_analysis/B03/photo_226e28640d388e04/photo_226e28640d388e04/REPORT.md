# B03 · photo_226e28640d388e04

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000146`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.691692 |
| 错误参考 MaxSim | +0.702011 |
| 正确减错误的分差 | -0.010319 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 110 个。正向贡献 +0.012273，负向贡献 -0.022592，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 610 | -0.009142 | -0.010956 |
| nonvisual | 11 | -0.000103 | -0.006815 |
| subject_nontext | 53 | -0.000289 | -0.003984 |
| subject_text | 57 | -0.000785 | -0.010066 |

最主要负差额来自 `background`，贡献 -0.009142。只保留主体 query token 后，本对分差为 -0.007135。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 26 | 43.9% | True | +7.758801 | 74.5% |
| 错误 | 34 | 45.4% | True | +7.987203 | 69.1% |

主体中 57 个 token 与 OCR 行相交，10 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 166 | background | -0.332187 |  | Product of China. |  |
| 35 | background | -0.274215 |  | Increxxa25 |  |
| 75 | background | -0.257892 |  |  |  |
| 315 | subject_text | -0.180831 | (tulathromycin injection) | (tulathromycin injection) | Injectable Solution |
| 76 | background | -0.162516 |  |  |  |
| 57 | background | -0.161845 |  |  |  |
| 17 | background | -0.158184 |  |  |  |
| 81 | background | -0.156408 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 442 | background | +0.260442 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 422 | background | +0.250809 |  | Net Contents: 250 mL |  |
| 426 | background | +0.229161 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 215 | subject_nontext | +0.226418 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 435 | background | +0.215574 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 519 | background | +0.201336 |  |  |  |
| 235 | subject_text | +0.188001 | Net Contents: 250 mL | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 286 | background | +0.185878 |  | Net Contents: 250 mL |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Net Contents: 250 mL | 5 | +0.000477 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010319 | +0.015164 |
| 只保留主体分差 | -0.007135 | +0.034226 |
| 主体 token 数 | 110 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
