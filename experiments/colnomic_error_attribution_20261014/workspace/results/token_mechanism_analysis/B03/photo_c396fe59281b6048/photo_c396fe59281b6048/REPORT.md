# B03 · photo_c396fe59281b6048

按当前评测真值分析，未新增身份人工复核。

包装 `package_110`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000506`，**纯图像实际 Top-1 错误参考** `reference_000157`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.631534 |
| 错误参考 MaxSim | +0.641176 |
| 正确减错误的分差 | -0.009642 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 242 个。正向贡献 +0.021106，负向贡献 -0.030749，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 526 | -0.008567 | -0.012687 |
| nonvisual | 11 | -0.000013 | -0.000914 |
| subject_nontext | 122 | -0.002147 | -0.013710 |
| subject_text | 120 | +0.001085 | +0.007041 |

最主要负差额来自 `background`，贡献 -0.008567。只保留主体 query token 后，本对分差为 -0.003420。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 74 | 46.7% | True | +11.566598 | 77.2% |
| 错误 | 17 | 3.0% | False | 不可评估 | 不可评估 |

主体中 120 个 token 与 OCR 行相交，40 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 696 | background | -0.376966 | at the two nam entrance do |  | For animal use only |
| 285 | subject_nontext | -0.361886 |  | Net Contents: 500 mL | Approved by FDA under ANADA # 200-128 |
| 222 | subject_nontext | -0.266061 |  | Net Contents: 500 mL | Approved by FDA under ANADA # 200-128 |
| 734 | background | -0.260847 |  |  | For animal use only |
| 212 | subject_nontext | -0.247168 |  | preruminating (veal) calves; and swine. | For animal use only |
| 316 | subject_nontext | -0.230033 |  | VETone |  |
| 382 | subject_text | -0.223587 | Net Contents: 500 mL | Net Contents: 500 mL | Approved by FDA under ANADA # 200-128 |
| 662 | background | -0.222043 |  | Approved by FDA under ANADA # 200-306 | order of a licensed veterinarian |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 500 | subject_text | +0.344171 | preruminating (veal) calves; and swine. | dairy cattle; calves, including | Vetrimycin 200 (oxytetracycline injection) is a |
| 239 | subject_text | +0.307784 | (oxytetracycline injection) 200 mg/mL |  | 500 mL |
| 47 | background | +0.260242 |  |  | 500 mL |
| 238 | subject_text | +0.241242 | (oxytetracycline injection) 200 mg/mL | (oxytetracycline injection) 200 mg/mL |  |
| 114 | background | +0.221216 |  | Vetrimycin™ 200 |  |
| 78 | background | +0.205422 |  |  |  vET On |
| 426 | subject_text | +0.185621 | NDC 13985-035-05 | 31398503505 | NDC 13985-944-50 |
| 274 | subject_text | +0.183915 | Each mL contains 200 mg of oxytetracycline. | (oxytetracycline injection) 200 mg/mL | Each mL contains 200 mg of |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 500 mL | 4 | -0.000143 |
| (oxytetracycline injection) 200 mg/mL | 19 | +0.002239 |
| Each mL contains 200 mg of oxytetracycline. | 9 | +0.000421 |
| Net Contents: 500 mL | 5 | -0.000234 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_bdfa0b3f68166b32` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.009642 | +0.006288 |
| 只保留主体分差 | -0.003420 | +0.016977 |
| 主体 token 数 | 242 | 256 |

[成功对照 token 表](../photo_bdfa0b3f68166b32/tokens.csv) · [对照摘要](../photo_bdfa0b3f68166b32/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。

![末尾说明文字局部](focal_text.png)
