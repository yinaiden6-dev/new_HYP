# B03 · photo_96d22a840ba148c4

按当前评测真值分析，未新增身份人工复核。

包装 `package_005`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000841`，**纯图像实际 Top-1 错误参考** `reference_005194`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 20 |
| 正确参考 MaxSim | +0.541888 |
| 错误参考 MaxSim | +0.564717 |
| 正确减错误的分差 | -0.022829 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 150 个。正向贡献 +0.038172，负向贡献 -0.061001，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 570 | -0.024733 | -0.031719 |
| nonvisual | 11 | -0.000828 | -0.055044 |
| subject_nontext | 95 | +0.000427 | +0.003282 |
| subject_text | 55 | +0.002306 | +0.030647 |

最主要负差额来自 `background`，贡献 -0.024733。只保留主体 query token 后，本对分差为 +0.013316。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 205 | 39.0% | True | +6.006047 | 58.8% |
| 错误 | 4 | 3.9% | False | 不可评估 | 不可评估 |

主体中 55 个 token 与 OCR 行相交，21 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 295 | subject_text | -0.569477 | GLUTENFREE | Inactive ingredients camauba wax, dibasic calcium | GLUTEN FREE! |
| 489 | background | -0.553349 |  |  |  |
| 275 | subject_text | -0.552914 | GLUTENFREE |  | GLUTEN FREE! |
| 271 | subject_nontext | -0.551532 |  |  |  |
| 378 | background | -0.540332 |  |  |  |
| 249 | background | -0.534587 |  |  |  |
| 501 | background | -0.531779 |  |  |  |
| 655 | background | -0.530711 |  |  | GLUTEN FREE! |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 192 | subject_nontext | +0.371445 |  |  |  |
| 474 | subject_text | +0.348237 | ABC#:10038886 | ABC#: 10038886 | 395259 |
| 514 | subject_text | +0.347100 | ABC#:10038886 | 24 Tablets \|30 mg Each | 395259 |
| 222 | background | +0.342971 |  | Questions or comments?1-800-719-9260 |  |
| 231 | subject_nontext | +0.335128 |  |  |  |
| 226 | background | +0.331585 |  | Questions or comments?1-800-719-9260 |  |
| 212 | subject_nontext | +0.326067 |  |  |  |
| 596 | background | +0.318851 |  | When using this product do not exc eedrecomme nde d do sage |  |

## 同包装成功对照

`photo_5d0b3972b2685095` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.022829 | +0.290764 |
| 只保留主体分差 | +0.013316 | +0.318906 |
| 主体 token 数 | 150 | 224 |

[成功对照 token 表](../photo_5d0b3972b2685095/tokens.csv) · [对照摘要](../photo_5d0b3972b2685095/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
