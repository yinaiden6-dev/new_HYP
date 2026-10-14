# B03 · photo_2b13d05bc35513ee

按当前评测真值分析，未新增身份人工复核。

包装 `package_101`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_000412`，**纯图像实际 Top-1 错误参考** `reference_000028`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.703696 |
| 错误参考 MaxSim | +0.703741 |
| 正确减错误的分差 | -0.000044 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 200 个。正向贡献 +0.040330，负向贡献 -0.040374，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 520 | -0.004329 | -0.006085 |
| nonvisual | 11 | -0.000203 | -0.013522 |
| subject_nontext | 77 | +0.005751 | +0.054596 |
| subject_text | 123 | -0.001263 | -0.007505 |

最主要负差额来自 `background`，贡献 -0.004329。只保留主体 query token 后，本对分差为 +0.016404。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 609 | 64.2% | True | +3.698335 | 55.1% |
| 错误 | 12 | 31.2% | True | +14.779162 | 100.0% |

主体中 123 个 token 与 OCR 行相交，87 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 314 | subject_text | -0.524385 | excursions permitted between 15 and |  | 30°℃ (59° and 86°F) [see USP |
| 330 | subject_text | -0.494289 | Controlled Room Temperature]. |  | Use contents within 6 months of first |
| 373 | subject_text | -0.389492 | 7015 College Boulevard, Sute 525 |  |  |
| 191 | subject_text | -0.358519 | failure and death. Do not administer |  | additional injectable or oral |
| 715 | background | -0.355248 |  |  |  |
| 79 | background | -0.336354 |  | has been associated with acute renal |  |
| 507 | background | -0.312151 |  |  |  |
| 346 | background | -0.295016 |  |  | Use contents within 6 months of first |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 370 | subject_text | +0.554140 | Manufactured for: | Dechra |  |
| 457 | subject_nontext | +0.488524 |  |  | 5 mg/mL Solution for Injection |
| 568 | background | +0.483740 |  | Dechra |  |
| 399 | background | +0.471006 |  |  | 30°℃ (59° and 86°F) [see USP |
| 552 | background | +0.466495 |  | Dechra Veterinary Products |  |
| 157 | subject_nontext | +0.444974 |  |  | Caution: Federal law restricts this |
| 377 | subject_text | +0.429031 | Rew. January 2024 | Rev. January 2024 | Rev. October2024 |
| 149 | subject_nontext | +0.425049 |  |  |  |

## 同包装成功对照

`photo_7bc115b787ac85b4` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000044 | +0.049822 |
| 只保留主体分差 | +0.016404 | +0.056720 |
| 主体 token 数 | 200 | 171 |

[成功对照 token 表](../photo_7bc115b787ac85b4/tokens.csv) · [对照摘要](../photo_7bc115b787ac85b4/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
