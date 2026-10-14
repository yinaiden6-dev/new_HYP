# B03 · photo_3409192d1c80c20f

按当前评测真值分析，未新增身份人工复核。

包装 `package_092`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000304`，**纯图像实际 Top-1 错误参考** `reference_000303`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.684553 |
| 错误参考 MaxSim | +0.686491 |
| 正确减错误的分差 | -0.001937 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 168 个。正向贡献 +0.034930，负向贡献 -0.036867，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 552 | +0.000851 | +0.001126 |
| nonvisual | 11 | -0.000132 | -0.008787 |
| subject_nontext | 48 | -0.001158 | -0.017643 |
| subject_text | 120 | -0.001497 | -0.009120 |

最主要负差额来自 `subject_text`，贡献 -0.001497。只保留主体 query token 后，本对分差为 -0.011555。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 686 | 65.0% | True | +7.656130 | 73.5% |
| 错误 | 175 | 58.0% | True | +6.496547 | 67.5% |

主体中 120 个 token 与 OCR 行相交，107 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 492 | subject_text | -0.451389 | Consult package insert for |  | Consult package insert for |
| 554 | subject_text | -0.448545 | For product questions, to report adverse |  | reactions, or for a copy of the Safety Data |
| 475 | subject_text | -0.427425 | tablets should be stored in a tightly sealed | Storage Conditions: Dronta Plus Taste |  |
| 476 | subject_text | -0.411563 | tablets should be stored in a tightly sealed |  | tablets should be stored in a tightly sealed |
| 676 | background | -0.393845 |  |  | tablets should be stored in a tightly sealed |
| 412 | subject_text | -0.358070 | Contraindications: DO NOT USE IN |  |  |
| 313 | subject_text | -0.357029 | Large Dogs (45 bs, and greater): |  |  |
| 672 | background | -0.342380 |  | Storage Conditions: Dronta Plus Taste |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 311 | subject_text | +0.462578 | Large Dogs (45 bs, and greater): |  |  |
| 543 | background | +0.451738 |  | reactions, or for a copy of the Safety |  |
| 547 | background | +0.444276 |  | reactions, or for a copy of the Safety |  |
| 212 | subject_text | +0.436919 | For rermoval of Hookworms (Ancyfostoma |  | Echinococcus granulosus, and removal |
| 546 | background | +0.431816 |  | For product questions, to report adverse | Greenfield, IN 46140 U.S.A |
| 174 | subject_text | +0.411268 | pyrantel base as pyrantel pamoate, and |  |  |
| 185 | background | +0.405439 |  |  |  |
| 181 | background | +0.393604 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| contains: 136,0 mg praziguantel 136,0 mg | 12 | +0.001728 |
| further details. | 4 | -0.001328 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_e564c98bb10f3824` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001937 | +0.008259 |
| 只保留主体分差 | -0.011555 | +0.011968 |
| 主体 token 数 | 168 | 150 |

[成功对照 token 表](../photo_e564c98bb10f3824/tokens.csv) · [对照摘要](../photo_e564c98bb10f3824/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
