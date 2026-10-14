# B03 · photo_0956810f30cd4324

按当前评测真值分析，未新增身份人工复核。

包装 `package_084`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_004975`，**纯图像实际 Top-1 错误参考** `reference_004974`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.489124 |
| 错误参考 MaxSim | +0.491189 |
| 正确减错误的分差 | -0.002065 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 238 个。正向贡献 +0.020914，负向贡献 -0.022979，相加等于原图像分差。与原评分缓存的最大误差 3e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 530 | -0.002893 | -0.004252 |
| nonvisual | 11 | -0.000226 | -0.016019 |
| subject_nontext | 134 | -0.000888 | -0.005163 |
| subject_text | 104 | +0.001942 | +0.014548 |

最主要负差额来自 `background`，贡献 -0.002893。只保留主体 query token 后，本对分差为 +0.003450。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 229 | 27.3% | True | +15.579030 | 87.8% |
| 错误 | 30 | 2.9% | False | 不可评估 | 不可评估 |

主体中 104 个 token 与 OCR 行相交，32 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 178 | background | -0.461636 |  | Preservative-Free |  |
| 341 | subject_nontext | -0.287744 |  |  |  |
| 369 | subject_nontext | -0.271764 |  | Preservative-Free | PRESERVATIVE FREE |
| 489 | background | -0.211960 |  |  |  |
| 417 | background | -0.195600 |  |  |  |
| 333 | subject_nontext | -0.160346 |  |  |  |
| 549 | background | -0.158749 |  |  |  |
| 398 | subject_nontext | -0.155587 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 307 | subject_nontext | +0.322721 |  | NORTHSTAR® | E. Windsor, NJ 08520 |
| 510 | background | +0.269629 | Pharmacy |  |  |
| 431 | subject_text | +0.268523 | 10 x 10 ml. Single-Dose Vials |  |  |
| 400 | subject_text | +0.257884 | 10 x 10 ml. Single-Dose Vials |  | 25 x 30 mL Single-Dose Vials |
| 397 | subject_nontext | +0.250700 |  |  |  |
| 541 | background | +0.236075 | Pharmacy |  |  |
| 410 | subject_nontext | +0.226732 |  | NDC 72603-327-10 |  |
| 756 | background | +0.219165 |  | 0.25% 25 mg/10 mL (2.5 mg/mL) |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 10 x 10 ml. Single-Dose Vials | 8 | +0.000366 |
| 0.25% 25 mg/10 mL (2.5 mg/mL) | 23 | +0.001050 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_faec4f7b9134de7d` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002065 | +0.013620 |
| 只保留主体分差 | +0.003450 | +0.036646 |
| 主体 token 数 | 238 | 238 |

[成功对照 token 表](../photo_faec4f7b9134de7d/tokens.csv) · [对照摘要](../photo_faec4f7b9134de7d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
