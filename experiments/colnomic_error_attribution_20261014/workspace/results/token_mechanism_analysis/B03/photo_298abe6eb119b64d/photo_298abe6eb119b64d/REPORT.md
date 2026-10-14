# B03 · photo_298abe6eb119b64d

按当前评测真值分析，未新增身份人工复核。

包装 `package_084`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_004975`，**纯图像实际 Top-1 错误参考** `reference_000395`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 14 |
| 正确参考 MaxSim | +0.509438 |
| 错误参考 MaxSim | +0.516377 |
| 正确减错误的分差 | -0.006939 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 120 个。正向贡献 +0.074924，负向贡献 -0.081863，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 648 | -0.043074 | -0.051782 |
| nonvisual | 11 | +0.001002 | +0.070980 |
| subject_nontext | 45 | +0.004335 | +0.075042 |
| subject_text | 75 | +0.030798 | +0.319885 |

最主要负差额来自 `background`，贡献 -0.043074。只保留主体 query token 后，本对分差为 +0.228069。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 209 | 48.9% | True | +3.647693 | 56.9% |
| 错误 | 8 | 33.2% | True | +11.074097 | 94.9% |

主体中 75 个 token 与 OCR 行相交，33 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 64 | background | -0.635014 | INTERCEPTOR | Discard unused portion. | INTERCEPTOR |
| 157 | background | -0.633602 |  | NaOH or HCl. |  |
| 413 | background | -0.623892 |  |  |  |
| 293 | background | -0.623631 |  |  |  |
| 442 | subject_nontext | -0.620080 |  | NDC 72603-327-10 |  |
| 131 | background | -0.615397 |  | NaOH or HCl. |  |
| 450 | background | -0.611904 |  |  |  |
| 701 | background | -0.604891 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 485 | background | +0.578026 |  |  |  |
| 674 | background | +0.562661 |  |  |  |
| 414 | background | +0.558871 |  |  |  |
| 660 | subject_text | +0.552090 | Discard unused portion. | Discard unused portion. | INTERCEPTOR |
| 663 | subject_nontext | +0.547536 |  | Each mL contains 2.5 mg bupivacaine |  |
| 440 | subject_nontext | +0.541308 |  |  |  |
| 540 | background | +0.539349 |  |  |  |
| 484 | background | +0.532997 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Each mL contains 2.5 mg bupivacaine | 9 | +0.003743 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_709e8a3ee42539bc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006939 | +0.022158 |
| 只保留主体分差 | +0.228069 | +0.224435 |
| 主体 token 数 | 120 | 153 |

[成功对照 token 表](../photo_709e8a3ee42539bc/tokens.csv) · [对照摘要](../photo_709e8a3ee42539bc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
