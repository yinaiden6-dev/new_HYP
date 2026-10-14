# B03 · photo_5623815155f4d189

按当前评测真值分析，未新增身份人工复核。

包装 `package_113`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000157`，**纯图像实际 Top-1 错误参考** `reference_000506`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.641193 |
| 错误参考 MaxSim | +0.658976 |
| 正确减错误的分差 | -0.017783 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 216 个。正向贡献 +0.047800，负向贡献 -0.065584，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 504 | -0.003292 | -0.004775 |
| nonvisual | 11 | +0.000055 | +0.003661 |
| subject_nontext | 67 | -0.007139 | -0.077889 |
| subject_text | 149 | -0.007408 | -0.036342 |

最主要负差额来自 `subject_text`，贡献 -0.007408。只保留主体 query token 后，本对分差为 -0.049229。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 903 | 60.1% | True | +7.741511 | 75.0% |
| 错误 | 9 | 32.8% | True | +10.383066 | 94.9% |

主体中 149 个 token 与 OCR 行相交，127 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 556 | subject_text | -0.548591 | Distributed by: MWI | V1 504020 |  |
| 358 | subject_nontext | -0.517636 |  | injection site (1-2 mL per site in small calves) |  |
| 536 | subject_nontext | -0.510512 |  |  |  |
| 518 | subject_nontext | -0.505085 |  |  |  |
| 516 | subject_nontext | -0.498714 |  |  | freezing. Use within 60 days of first puncture and |
| 537 | subject_nontext | -0.484405 |  | V1 504020 |  |
| 296 | subject_text | -0.483627 | discoloration may necessitate trimming of the |  |  |
| 394 | subject_text | -0.480858 | Exceeding |  | Exceeding the highest recommended level of drug per |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 634 | subject_text | +0.545499 | NoCoating | NoCoating | STORAGE CONDITIONS: |
| 593 | subject_nontext | +0.534836 |  | EXP. DATE: |  |
| 613 | subject_text | +0.532018 | EXP. DATE: | EXP. DATE: |  |
| 531 | subject_text | +0.484252 | TAKETIME | TAKE TIME |  |
| 635 | subject_text | +0.480046 | NoCoating | Coating |  |
| 610 | subject_text | +0.453453 | LOTNO | LOT NO. |  |
| 300 | background | +0.445024 |  |  |  |
| 532 | subject_text | +0.431568 | TAKETIME | TAKE TIME |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| antibiofic oxytetracycline Each mL contains 200 mg | 16 | +0.000001 |
| and on a w/u basis, 40.0% 2-pyrrolidone, 5.0% | 14 | +0.001739 |
| povidone, 18% magnesium oxide, 02% sodium | 8 | +0.001339 |
| exceeding 10 mL subcutaneously per injection site in | 9 | -0.000870 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_63103452c5081a78` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.017783 | +0.028276 |
| 只保留主体分差 | -0.049229 | +0.035520 |
| 主体 token 数 | 216 | 230 |

[成功对照 token 表](../photo_63103452c5081a78/tokens.csv) · [对照摘要](../photo_63103452c5081a78/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
