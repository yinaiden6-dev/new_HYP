# B03 · photo_a0a282b24735a4d5

按当前评测真值分析，未新增身份人工复核。

包装 `package_100`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000409`，**纯图像实际 Top-1 错误参考** `reference_000411`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.621883 |
| 错误参考 MaxSim | +0.627525 |
| 正确减错误的分差 | -0.005642 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 230 个。正向贡献 +0.028141，负向贡献 -0.033783，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 490 | -0.004961 | -0.007401 |
| nonvisual | 11 | -0.000139 | -0.009257 |
| subject_nontext | 77 | -0.002136 | -0.020273 |
| subject_text | 153 | +0.001594 | +0.007617 |

最主要负差额来自 `background`，贡献 -0.004961。只保留主体 query token 后，本对分差为 -0.001720。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 415 | 65.4% | True | +6.911283 | 61.7% |
| 错误 | 8 | 28.4% | True | +10.620566 | 97.4% |

主体中 153 个 token 与 OCR 行相交，117 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 457 | subject_text | -0.377011 | reach of children. Consult a physician in | oral suspension) | Keep Meloxidyl in a secure location out of reach of dogs, cats, and |
| 194 | subject_text | -0.293596 | provided dosing syringes. | provided dosing syringes. |  |
| 133 | background | -0.293265 |  | reach of children. Consult a physician in |  |
| 176 | subject_nontext | -0.288131 |  |  |  |
| 254 | subject_text | -0.258973 | provided dosing syringes. | Only administer with |  |
| 719 | background | -0.244867 |  | insert or client information sheet | sheet with prescription. Meloxidyl Oral Suspension should be |
| 174 | subject_text | -0.242746 | provided dosing syringes. | insert or client information sheet |  |
| 78 | background | -0.226667 |  | NDC 55529-041-02 |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 314 | subject_text | +0.473889 | For technical assistance or to report |  |  |
| 143 | background | +0.439736 |  |  |  |
| 323 | background | +0.427228 |  |  |  |
| 547 | background | +0.366899 |  | treatment should be discontinued |  |
| 533 | subject_text | +0.322322 | veterinarian sought Refer to the product | treatment should be discontinued | Refer to the package insert for complete dosage and |
| 333 | subject_text | +0.285248 | during administration. |  |  |
| 541 | background | +0.278799 |  | treatment should be discontinued |  |
| 453 | subject_text | +0.263715 | reach of children. Consult a physician in |  | sheet with prescription. Meloxidyl Oral Suspension should be |

## 同包装成功对照

`photo_add4d3fb17dace03` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005642 | +0.046307 |
| 只保留主体分差 | -0.001720 | +0.067831 |
| 主体 token 数 | 230 | 230 |

[成功对照 token 表](../photo_add4d3fb17dace03/tokens.csv) · [对照摘要](../photo_add4d3fb17dace03/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
