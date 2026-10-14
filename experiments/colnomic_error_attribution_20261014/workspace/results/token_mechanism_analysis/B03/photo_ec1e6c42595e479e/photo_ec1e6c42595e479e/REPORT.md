# B03 · photo_ec1e6c42595e479e

按当前评测真值分析，未新增身份人工复核。

包装 `package_041`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_001113`，**纯图像实际 Top-1 错误参考** `reference_000693`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 19 |
| 正确参考 MaxSim | +0.591481 |
| 错误参考 MaxSim | +0.601983 |
| 正确减错误的分差 | -0.010503 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 156 个。正向贡献 +0.034561，负向贡献 -0.045064，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 564 | -0.014517 | -0.018815 |
| nonvisual | 11 | -0.000542 | -0.036032 |
| subject_nontext | 36 | +0.000882 | +0.017916 |
| subject_text | 120 | +0.003674 | +0.022381 |

最主要负差额来自 `background`，贡献 -0.014517。只保留主体 query token 后，本对分差为 +0.021351。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 678 | 79.0% | True | +9.278979 | 84.3% |
| 错误 | 19 | 48.3% | True | +14.104845 | 95.5% |

主体中 120 个 token 与 OCR 行相交，83 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 380 | background | -0.237760 |  |  |  |
| 502 | background | -0.233972 |  |  |  |
| 466 | background | -0.212362 |  |  |  |
| 493 | background | -0.209784 |  |  |  |
| 371 | background | -0.199288 |  |  |  |
| 42 | background | -0.194999 |  | PACKAGE - 50 mL |  |
| 470 | background | -0.186609 |  |  |  |
| 657 | background | -0.185940 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 162 | background | +0.409097 |  |  |  |
| 136 | subject_text | +0.407315 | Questions?Calltoll-free(877)322-8633 | Questions? Call toll-free (877) 322-8633 | OtherInformation |
| 161 | background | +0.382913 |  |  |  |
| 515 | background | +0.355075 |  | Uses Temporarily protects and helps relieve minor skin irritation and | 5XHPUS |
| 485 | background | +0.348398 |  |  |  |
| 45 | background | +0.336436 |  |  |  |
| 123 | background | +0.331591 |  |  | Questions or Comments? 1-844-KADESH1 |
| 160 | background | +0.327312 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Colloidal Oatmeal2% | 4 | +0.000702 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_03df1f5abf1d6a3d` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010503 | +0.137576 |
| 只保留主体分差 | +0.021351 | +0.237132 |
| 主体 token 数 | 156 | 144 |

[成功对照 token 表](../photo_03df1f5abf1d6a3d/tokens.csv) · [对照摘要](../photo_03df1f5abf1d6a3d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
