# B03 · photo_2f4051894f10c596

按当前评测真值分析，未新增身份人工复核。

包装 `package_096`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000395`，**纯图像实际 Top-1 错误参考** `reference_000396`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 3 |
| 正确参考 MaxSim | +0.591956 |
| 错误参考 MaxSim | +0.592785 |
| 正确减错误的分差 | -0.000829 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 171 个。正向贡献 +0.018722，负向贡献 -0.019551，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 597 | -0.000292 | -0.000381 |
| nonvisual | 11 | +0.000053 | +0.003753 |
| subject_nontext | 65 | -0.001063 | -0.012734 |
| subject_text | 106 | +0.000473 | +0.003475 |

最主要负差额来自 `subject_nontext`，贡献 -0.001063。只保留主体 query token 后，本对分差为 -0.002686。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 150 | 63.3% | True | +7.968358 | 65.9% |
| 错误 | 59 | 18.0% | True | +9.465687 | 77.4% |

主体中 106 个 token 与 OCR 行相交，71 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 28 | background | -0.226386 |  | INTERCEPTOR | Approved by FDA under NADA # 141-338 |
| 442 | subject_nontext | -0.199287 |  |  |  |
| 181 | background | -0.174604 |  |  | Caution: Federal (USA) law restrids this drug to use by or on the order of a |
| 403 | subject_text | -0.161319 | Precautions: Treatment with fewer than 6 monthly doses after |  |  |
| 694 | background | -0.158696 |  |  |  |
| 64 | background | -0.152275 |  | To report adverse effects, access medical information, or obtain |  |
| 471 | subject_text | -0.148781 | To report adverse effects, access medical information, or obtain |  |  |
| 150 | background | -0.148394 |  | For more information, please visit www.InterceptorPet.com |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 572 | subject_text | +0.427941 | Greenfield, IN 46140, USA | Product of France | Approved by FDA under NADA # 141-338 |
| 243 | background | +0.208124 |  | Precautions: Treatment with fewer than 6 monthly doses after |  |
| 480 | subject_nontext | +0.207849 |  | 727804564599 |  |
| 131 | background | +0.175508 |  |  |  |
| 247 | background | +0.150875 |  |  |  |
| 418 | background | +0.147696 |  |  |  |
| 511 | subject_text | +0.140175 | 727804564599 | 727804564599 |  |
| 218 | background | +0.139112 |  |  | (milbemycin oxime/praziquantel) |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 25.1-50 lbs | 2 | -0.000003 |
| 50.1-100 lbs | 4 | -0.000041 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_8ae269507838f198` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000829 | +0.001762 |
| 只保留主体分差 | -0.002686 | -0.002953 |
| 主体 token 数 | 171 | 152 |

[成功对照 token 表](../photo_8ae269507838f198/tokens.csv) · [对照摘要](../photo_8ae269507838f198/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
