# B03 · photo_b8aaaa8b5f65ec7d

按当前评测真值分析，未新增身份人工复核。

包装 `package_026`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000995`，**纯图像实际 Top-1 错误参考** `reference_001786`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.545378 |
| 错误参考 MaxSim | +0.549115 |
| 正确减错误的分差 | -0.003737 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 210 个。正向贡献 +0.041834，负向贡献 -0.045571，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.023975 | -0.033470 |
| nonvisual | 11 | -0.000148 | -0.010476 |
| subject_nontext | 31 | +0.000081 | +0.002047 |
| subject_text | 179 | +0.020304 | +0.088364 |

最主要负差额来自 `background`，贡献 -0.023975。只保留主体 query token 后，本对分差为 +0.075622。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 394 | 57.7% | True | +6.316658 | 60.6% |
| 错误 | 14 | 23.7% | True | +11.088070 | 95.7% |

主体中 179 个 token 与 OCR 行相交，127 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 521 | background | -0.510291 | Walgreens |  |  |
| 520 | background | -0.491402 | Walgreens | GLUTEN | PHARMACIST |
| 291 | background | -0.417762 | Bupivacaine Hydrochloride | spreading to arms, neck or shoulders; or | FEXOFENADINE HYDROCHLORIDE TABLETS USP |
| 522 | background | -0.400494 | Walgreens |  | walgreens.com ©2018 Walgreen Co. |
| 244 | background | -0.396397 |  | KEEP THE CARTON. IT CONTAINS | PRESCRIPTION STRENGTH |
| 76 | background | -0.372720 |  |  |  |
| 120 | background | -0.360596 | TABLETS USP, 10 mg - Antihistamine |  |  |
| 529 | subject_text | -0.355775 | store at 20-25°C (68-77°F) |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 317 | subject_text | +0.514840 | GLUTEN | GLUTEN |  |
| 543 | subject_text | +0.507020 | © 2023 AMAZON.COM, INC. OR ITS AFFILUATES | 410 TERRY AVENUE N., SEATTLE, WA 98109 | 24 |
| 511 | subject_text | +0.463008 | © 2023 AMAZON.COM, INC. OR ITS AFFILUATES | amazon | TAMPER EVIDENT: DO NOT USE IF |
| 542 | subject_text | +0.462345 | AMAZON.COM SERVICES LLC | 410 TERRY AVENUE N., SEATTLE, WA 98109 | 11917129563 |
| 544 | subject_text | +0.453314 | ALL RIGHTS RESERVED | ALL RIGHTS RESERVED. | 150072897 |
| 347 | subject_text | +0.441104 | AROUND THE CENTER OF EACH CAPSULE IS | 14 CAPSULES | Directions |
| 735 | background | +0.430167 |  | 410 TERRY AVENUE N., SEATTLE, WA 98109 |  |
| 442 | subject_text | +0.428929 | AROUND THE CENTER OF EACH CAPSULE IS | BROKEN OR MISSING OR BLACK BAND | safety sealed: do not use if carton is opened or if printed foil inner seal on bottle is tom or missing |

## 同包装成功对照

`photo_ab6fd7a33337ccee` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003737 | +0.070894 |
| 只保留主体分差 | +0.075622 | +0.231209 |
| 主体 token 数 | 210 | 209 |

[成功对照 token 表](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
