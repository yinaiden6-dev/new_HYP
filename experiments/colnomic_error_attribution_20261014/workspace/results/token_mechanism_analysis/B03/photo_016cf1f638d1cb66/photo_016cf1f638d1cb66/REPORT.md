# B03 · photo_016cf1f638d1cb66

按当前评测真值分析，未新增身份人工复核。

包装 `package_088`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_005041`，**纯图像实际 Top-1 错误参考** `reference_000486`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 8 |
| 正确参考 MaxSim | +0.522111 |
| 错误参考 MaxSim | +0.527291 |
| 正确减错误的分差 | -0.005180 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 155 个。正向贡献 +0.053794，负向贡献 -0.058974，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 613 | -0.031831 | -0.040451 |
| nonvisual | 11 | -0.000160 | -0.011324 |
| subject_nontext | 77 | -0.000563 | -0.005697 |
| subject_text | 78 | +0.027374 | +0.273385 |

最主要负差额来自 `background`，贡献 -0.031831。只保留主体 query token 后，本对分差为 +0.134744。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 98 | 15.8% | True | +8.642220 | 63.8% |
| 错误 | 5 | 9.6% | False | 不可评估 | 不可评估 |

主体中 78 个 token 与 OCR 行相交，33 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 288 | background | -0.658558 | Elanco |  | Elanco |
| 319 | background | -0.647094 | Elanco |  | Elanco |
| 287 | background | -0.606130 | Elanco |  | Elanco |
| 127 | background | -0.548205 |  | 0000 |  |
| 719 | background | -0.543568 | nhalation Anesthetic | 0000 | For otic use in dogs only. |
| 320 | background | -0.454612 | Elanco |  | Elanco |
| 286 | background | -0.453632 | Elanco | Manufactured by: GP Pharm, S.A. |  |
| 477 | subject_nontext | -0.452660 |  | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 391 | subject_text | +0.582026 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 |  |
| 396 | subject_text | +0.561869 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 | continue for 7 consecutive days. |
| 393 | subject_text | +0.531915 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 | For complete product information, |
| 367 | subject_text | +0.526332 | Keep out of the reach of children. Store the packaged tray in the outer carton. Store refrigerated, 2-8C (36-46'F). | One carton contains one packaged tray which contains: |  |
| 392 | subject_text | +0.516541 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 | Call your doctor for medical advice about side effects. You may report side effects to FDA at 1-800-FDA-1088 |  |
| 380 | subject_text | +0.504093 | See package insert for dosage information. For single use only. Reconstitute only with the diluent provided. Use immediately after reconstitution. | See package insert for dosage information. For single use only. Reconstitute only with the diluent provided. Use immediately after reconstitution. |  |
| 671 | background | +0.500901 |  | See package insert for dosage information. For single use only. Reconstitute only with the diluent provided. Use immediately after reconstitution. |  |
| 366 | subject_text | +0.492262 | Keep out of the reach of children. Store the packaged tray in the outer carton. Store refrigerated, 2-8C (36-46'F). |  |  |

## 同包装成功对照

`photo_be1805e5ca1f87a1` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005180 | +0.052802 |
| 只保留主体分差 | +0.134744 | +0.214020 |
| 主体 token 数 | 155 | 155 |

[成功对照 token 表](../photo_be1805e5ca1f87a1/tokens.csv) · [对照摘要](../photo_be1805e5ca1f87a1/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
