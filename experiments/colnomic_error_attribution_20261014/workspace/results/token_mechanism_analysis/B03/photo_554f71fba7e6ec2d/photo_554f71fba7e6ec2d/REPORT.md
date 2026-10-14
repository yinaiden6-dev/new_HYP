# B03 · photo_554f71fba7e6ec2d

按当前评测真值分析，未新增身份人工复核。

包装 `package_110`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000506`，**纯图像实际 Top-1 错误参考** `reference_000157`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.641954 |
| 错误参考 MaxSim | +0.645693 |
| 正确减错误的分差 | -0.003739 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 210 个。正向贡献 +0.024253，负向贡献 -0.027992，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.005037 | -0.007033 |
| nonvisual | 11 | -0.000003 | -0.000213 |
| subject_nontext | 111 | +0.000752 | +0.005276 |
| subject_text | 99 | +0.000549 | +0.004322 |

最主要负差额来自 `background`，贡献 -0.005037。只保留主体 query token 后，本对分差为 +0.004826。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 88 | 44.8% | True | +11.732696 | 75.7% |
| 错误 | 13 | 28.7% | True | +9.798239 | 95.7% |

主体中 99 个 token 与 OCR 行相交，35 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 658 | subject_text | -0.371219 | Net Contents: 500 mL |  | Approved by FDA under ANADA # 200-128 |
| 627 | subject_nontext | -0.329694 |  |  | For animal use only |
| 237 | subject_text | -0.305131 | A |  |  |
| 590 | subject_nontext | -0.294469 |  |  | For animal use only |
| 363 | background | -0.250352 |  |  | For animal use only |
| 53 | background | -0.214480 | ch Sto |  |  |
| 347 | background | -0.208227 |  | Caution: Federal law restricts this drug | Caution: Federal law restricts |
| 369 | subject_text | -0.207218 | Caution: Federal law restricts this drug | Caution: Federal law restricts this drug | NoCoating |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 469 | subject_nontext | +0.238579 |  | (oxytetracycline injection) 200 mg/mL |  |
| 80 | subject_text | +0.230583 | NDC 13985-035-05 | NDC 13985-035-05 | NDC 13985-944-50 |
| 181 | subject_text | +0.227977 | (oxytetracycline injection) 200 mg/mL | (oxytetracycline injection) 200 mg/mL |  |
| 626 | subject_nontext | +0.219881 |  |  |  |
| 79 | subject_text | +0.189111 | NDC 13985-035-05 | NDC 13985-035-05 |  |
| 225 | background | +0.186995 | má den |  |  |
| 262 | background | +0.184835 |  |  |  |
| 110 | subject_nontext | +0.179221 |  | NDC 13985-035-05 |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| (oxytetracycline injection) 200 mg/mL | 9 | +0.001219 |
| Each mL contains 200 mg of oxytetracycline. | 8 | +0.000352 |
| Net Contents: 500 mL | 9 | -0.000367 |
| 500 mL | 3 | +0.000041 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_bdfa0b3f68166b32` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003739 | +0.006288 |
| 只保留主体分差 | +0.004826 | +0.016977 |
| 主体 token 数 | 210 | 256 |

[成功对照 token 表](../photo_bdfa0b3f68166b32/tokens.csv) · [对照摘要](../photo_bdfa0b3f68166b32/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
