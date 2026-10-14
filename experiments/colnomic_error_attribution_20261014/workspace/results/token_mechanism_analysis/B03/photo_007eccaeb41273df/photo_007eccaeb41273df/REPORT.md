# B03 · photo_007eccaeb41273df

按当前评测真值分析，未新增身份人工复核。

包装 `package_015`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000934`，**纯图像实际 Top-1 错误参考** `reference_001218`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 80 |
| 正确参考 MaxSim | +0.617425 |
| 错误参考 MaxSim | +0.650635 |
| 正确减错误的分差 | -0.033210 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 162 个。正向贡献 +0.021273，负向贡献 -0.054483，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 558 | -0.028123 | -0.036842 |
| nonvisual | 11 | -0.000299 | -0.019850 |
| subject_nontext | 46 | -0.000636 | -0.010113 |
| subject_text | 116 | -0.004152 | -0.026166 |

最主要负差额来自 `background`，贡献 -0.028123。只保留主体 query token 后，本对分差为 -0.021607。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 464 | 72.9% | True | +5.436174 | 61.1% |
| 错误 | 10 | 51.3% | True | +8.652444 | 92.6% |

主体中 116 个 token 与 OCR 行相交，60 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 400 | background | -0.425075 |  |  |  |
| 409 | background | -0.404496 |  |  |  |
| 195 | subject_text | -0.331872 | No nat use It fo amer aond is brolin or anoing | Bo mit ese t te isnr seed is broban er aisang | Tamper evident: do not use if print |
| 715 | background | -0.302612 |  |  |  |
| 241 | background | -0.299512 |  | lemporíy niswes naino achas and painas du to |  |
| 511 | background | -0.277410 |  | Pain Reliever/Fever Reducer |  |
| 569 | background | -0.265296 |  |  |  |
| 714 | background | -0.263475 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 151 | subject_nontext | +0.326960 |  |  |  |
| 242 | background | +0.324637 |  |  |  |
| 92 | background | +0.310670 |  |  |  |
| 158 | subject_nontext | +0.289986 |  |  |  |
| 198 | subject_nontext | +0.277367 |  |  |  |
| 35 | background | +0.269842 |  |  |  |
| 354 | subject_text | +0.255883 | Uuar weming: Tha product cortaine sostanhopher Severs | Lüaer maming: Thia prouct crtans scetanrophen Severa | Do not use • with any other drug containing acetaminophen |
| 166 | background | +0.240172 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| taminaphan USP, a50 mg... | 4 | +0.000178 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_4917ad5f3fac9f55` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.033210 | +0.124820 |
| 只保留主体分差 | -0.021607 | +0.173414 |
| 主体 token 数 | 162 | 50 |

[成功对照 token 表](../photo_4917ad5f3fac9f55/tokens.csv) · [对照摘要](../photo_4917ad5f3fac9f55/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
