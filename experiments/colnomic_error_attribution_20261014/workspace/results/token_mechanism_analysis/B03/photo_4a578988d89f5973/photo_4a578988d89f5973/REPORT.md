# B03 · photo_4a578988d89f5973

按当前评测真值分析，未新增身份人工复核。

包装 `package_014`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000930`，**纯图像实际 Top-1 错误参考** `reference_004327`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.676232 |
| 错误参考 MaxSim | +0.677576 |
| 正确减错误的分差 | -0.001344 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 232 个。正向贡献 +0.037861，负向贡献 -0.039205，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 488 | +0.001096 | +0.001642 |
| nonvisual | 11 | -0.000053 | -0.003520 |
| subject_nontext | 80 | -0.003278 | -0.029952 |
| subject_text | 152 | +0.000891 | +0.004284 |

最主要负差额来自 `subject_nontext`，贡献 -0.003278。只保留主体 query token 后，本对分差为 -0.007521。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 516 | 76.1% | True | +4.160145 | 54.6% |
| 错误 | 144 | 24.3% | True | +4.038094 | 62.2% |

主体中 152 个 token 与 OCR 行相交，80 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 594 | subject_nontext | -0.443905 |  | 24 |  |
| 413 | subject_text | -0.421322 | helps relieve a broad range of unconfortable | Allergy Relief | symptoms like congestion |
| 16 | background | -0.416950 |  | Questions or comments? |  |
| 416 | subject_text | -0.403368 | helps relieve a broad range of unconfortable |  | helps relleve a broad |
| 617 | subject_nontext | -0.399811 |  |  |  |
| 232 | subject_text | -0.382853 | store between 4º and 30℃ |  | store between 4° and 30℃ |
| 329 | background | -0.363869 |  | Questions or comments? |  |
| 417 | subject_text | -0.355218 | helps relieve a broad range of unconfortable |  | helps relleve a broad |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 94 | background | +0.461436 |  |  | ADULTS AND CHILDREN 12 YEARS OF AGE AND OLDER |
| 315 | subject_text | +0.374250 | microcrystalline cellulose, phenylethyl | Inactive ingredients |  |
| 226 | background | +0.369910 |  |  | Talk to your child's doctorif your child needs to use the spray for longer than |
| 695 | background | +0.362933 |  | Inactive ingredients |  |
| 222 | background | +0.323113 |  |  | 24 |
| 114 | subject_nontext | +0.295148 |  | store between 4° and 30℃ | store between 4° and 30℃ |
| 673 | subject_text | +0.285629 | 0113-0285-085 | 0113-0285-085 | 1Q31R 17 C1 |
| 507 | background | +0.282913 |  | Inactive ingredients |  |

## 同包装成功对照

`photo_85d3b99db68ac04f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001344 | +0.049299 |
| 只保留主体分差 | -0.007521 | +0.091113 |
| 主体 token 数 | 232 | 153 |

[成功对照 token 表](../photo_85d3b99db68ac04f/tokens.csv) · [对照摘要](../photo_85d3b99db68ac04f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
