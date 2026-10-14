# B03 · photo_0cfa49485997e1d4

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_003251`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 296 |
| 正确参考 MaxSim | +0.589642 |
| 错误参考 MaxSim | +0.664878 |
| 正确减错误的分差 | -0.075236 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 180 个。正向贡献 +0.022644，负向贡献 -0.097880，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 540 | -0.053279 | -0.072124 |
| nonvisual | 11 | -0.000268 | -0.017782 |
| subject_nontext | 53 | -0.003778 | -0.052112 |
| subject_text | 127 | -0.017911 | -0.103096 |

最主要负差额来自 `background`，贡献 -0.053279。只保留主体 query token 后，本对分差为 -0.088084。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 436 | 54.2% | True | +15.847732 | 83.4% |
| 错误 | 6 | 20.4% | False | 不可评估 | 不可评估 |

主体中 127 个 token 与 OCR 行相交，100 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 510 | subject_text | -0.539765 | croecarmelloee sodium, FD&C blue #2, FD&C red #40, |  | Inactive ingredients colloidal silicon dioxide*, croscarmellose sodium*, D&C Yellow |
| 495 | subject_text | -0.524147 | see end panal for expinaion date and lot number | "This product is not manufactured or distributed | Inactive ingredients colloidal silicon dioxide*, croscarmellose sodium*, D&C Yellow |
| 595 | subject_text | -0.520917 | call1-877-770-3183 Mon-Fri 8:00 AM EST to 5:00 PM PST |  |  |
| 372 | subject_text | -0.498964 | right away. Quick medical attantion ls critical for |  | Poison Control Center right away (1-800-222-1222). Quick medical attent ion is critical for adults as |
| 603 | background | -0.486352 |  |  |  |
| 202 | background | -0.481707 |  |  |  |
| 376 | subject_text | -0.460301 | right away. Quick medical attantion ls critical for |  | Poison Control Center right away (1-800-222-1222). Quick medical attent ion is critical for adults as |
| 710 | background | -0.454076 |  | Tylenol and Extra Strength Tylenol• PM. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 310 | subject_text | +0.404288 | If pregnant or breast-feeding, ask a health | If pregnant or breast-feeding, ask a health |  |
| 630 | subject_nontext | +0.393567 |  |  |  |
| 472 | subject_text | +0.350645 | Other inforrnation  store between 20*-25C | store at 20°-25°C (68°-77°F). See USP Controlled |  |
| 162 | background | +0.287291 |  |  |  |
| 257 | subject_nontext | +0.271458 |  |  | Pain Relief PM |
| 179 | background | +0.262103 |  |  |  |
| 11 | background | +0.244014 |  |  |  |
| 298 | subject_nontext | +0.240363 |  |  |  |

## 同包装成功对照

`photo_7f4bc7d166b40bfc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.075236 | +0.142336 |
| 只保留主体分差 | -0.088084 | +0.159852 |
| 主体 token 数 | 180 | 288 |

[成功对照 token 表](../photo_7f4bc7d166b40bfc/tokens.csv) · [对照摘要](../photo_7f4bc7d166b40bfc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
