# B03 · photo_65dbff1a3b69197f

按当前评测真值分析，未新增身份人工复核。

包装 `package_073`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_003276`，**纯图像实际 Top-1 错误参考** `reference_002163`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.732755 |
| 错误参考 MaxSim | +0.732807 |
| 正确减错误的分差 | -0.000051 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 108 个。正向贡献 +0.032878，负向贡献 -0.032929，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 612 | -0.006745 | -0.008056 |
| nonvisual | 11 | -0.001042 | -0.069223 |
| subject_nontext | 29 | +0.003908 | +0.098498 |
| subject_text | 79 | +0.003828 | +0.035419 |

最主要负差额来自 `background`，贡献 -0.006745。只保留主体 query token 后，本对分差为 +0.052357。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 636 | 59.2% | True | +2.586854 | 47.2% |
| 错误 | 23 | 10.5% | False | 不可评估 | 不可评估 |

主体中 79 个 token 与 OCR 行相交，63 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 133 | background | -0.359284 |  | of the tip of a finger) on the area 1 to |  |
| 284 | background | -0.306260 |  |  | Polymyxin B 10,000 units as polymyxin B sulfate |
| 442 | background | -0.300551 |  | clean the affected area | Questions or comments?1-800-719-9260 |
| 481 | background | -0.288207 |  | Other information |  |
| 26 | background | -0.272512 |  | Ask a doctor before use if you have | over large areas of the body |
| 393 | subject_text | -0.263882 | you need to use longer than 1 week |  | you need to use longer than 1 week |
| 105 | background | -0.249696 |  | swallowed, get medical help or contact |  |
| 9 | background | -0.235508 |  | Drug Facts |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 573 | subject_nontext | +0.560001 |  | Coating |  |
| 574 | subject_nontext | +0.539718 |  | Coating/Print Omit Area | oil, poloxamer 188, propylene glycol, purified water, white petrolatum |
| 575 | subject_nontext | +0.521066 |  | Coating/Print Omit Area | 100% money back |
| 583 | background | +0.482447 |  | Coating | 80964 17 C5 |
| 204 | background | +0.410630 |  | Coating/Print Omit Area | Inactive ingredients emulsifying wax, methylparaben, mineral |
| 530 | background | +0.409051 |  | light mineral oil, white petrolatum | oil, poloxamer 188, propylene glycol, purified water, white petrolatum |
| 684 | background | +0.321644 |  | Coating/Print Omit Area |  |
| 592 | subject_text | +0.313220 | EXP | Coating/Print Omit Area |  |

## 同包装成功对照

`photo_e589720459db3cc8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000051 | +0.102374 |
| 只保留主体分差 | +0.052357 | +0.178953 |
| 主体 token 数 | 108 | 90 |

[成功对照 token 表](../photo_e589720459db3cc8/tokens.csv) · [对照摘要](../photo_e589720459db3cc8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
