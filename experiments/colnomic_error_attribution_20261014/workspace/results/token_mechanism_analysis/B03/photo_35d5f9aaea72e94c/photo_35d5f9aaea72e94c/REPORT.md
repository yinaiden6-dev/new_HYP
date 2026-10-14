# B03 · photo_35d5f9aaea72e94c

按当前评测真值分析，未新增身份人工复核。

包装 `package_119`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001971`，**纯图像实际 Top-1 错误参考** `reference_001973`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.551829 |
| 错误参考 MaxSim | +0.563397 |
| 正确减错误的分差 | -0.011568 |
| F03 的正确参考排名（辅助对照） | 66 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 231 个。正向贡献 +0.017677，负向贡献 -0.029245，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 537 | -0.008404 | -0.012191 |
| nonvisual | 11 | -0.000104 | -0.007374 |
| subject_nontext | 103 | -0.000520 | -0.003933 |
| subject_text | 128 | -0.002540 | -0.015460 |

最主要负差额来自 `background`，贡献 -0.008404。只保留主体 query token 后，本对分差为 -0.010320。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 242 | 63.0% | True | +7.155518 | 68.0% |
| 错误 | 35 | 7.5% | False | 不可评估 | 不可评估 |

主体中 128 个 token 与 OCR 行相交，37 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 360 | background | -0.346440 | has not been establisbed for this product ia |  | mn e |
| 559 | subject_text | -0.278049 | CONTAINS 2 MEDICINES | 1 |  |
| 398 | subject_text | -0.259906 | Gonpae tb tba actre gediers |  | Acto ingrodlent (it euch capix) |
| 253 | background | -0.247308 | mg, antihistamine |  |  |
| 575 | background | -0.225891 |  |  |  |
| 285 | background | -0.206013 | ingredient in Claritin |  |  |
| 553 | background | -0.203219 |  |  |  |
| 753 | background | -0.191230 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 589 | subject_text | +0.415258 | 144 | 144 | 216 |
| 588 | subject_text | +0.316769 | 144 | 144 | 216 |
| 348 | subject_nontext | +0.188895 |  | NDC0363-9131-44 | VALUE SIZE |
| 499 | subject_text | +0.183158 | ACETAMINOPHEN 250 mg & IBUPROFEN (NS AID) 125 mg TABLETS / |  | ACETAMINOPHEN 250 mg |
| 177 | background | +0.178092 |  | E7578 |  |
| 9 | background | +0.176609 |  |  |  |
| 591 | subject_text | +0.171523 | CAPLETS* | CAPLETS* |  |
| 137 | background | +0.162795 |  | 144 |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| ACETAMINOPHEN 250 mg & IBUPROFEN (NS AID) 125 mg TABLETS / | 38 | -0.001276 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_ca454f8a9d866abb` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.011568 | +0.001656 |
| 只保留主体分差 | -0.010320 | -0.010628 |
| 主体 token 数 | 231 | 264 |

[成功对照 token 表](../photo_ca454f8a9d866abb/tokens.csv) · [对照摘要](../photo_ca454f8a9d866abb/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
