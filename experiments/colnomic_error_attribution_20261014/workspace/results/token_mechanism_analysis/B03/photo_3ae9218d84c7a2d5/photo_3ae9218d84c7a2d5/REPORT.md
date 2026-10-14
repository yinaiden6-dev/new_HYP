# B03 · photo_3ae9218d84c7a2d5

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000146`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.729648 |
| 错误参考 MaxSim | +0.735747 |
| 正确减错误的分差 | -0.006099 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 210 个。正向贡献 +0.020245，负向贡献 -0.026344，相加等于原图像分差。与原评分缓存的最大误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 510 | -0.007033 | -0.010081 |
| nonvisual | 11 | -0.000076 | -0.005048 |
| subject_nontext | 85 | -0.000796 | -0.006844 |
| subject_text | 125 | +0.001806 | +0.010559 |

最主要负差额来自 `background`，贡献 -0.007033。只保留主体 query token 后，本对分差为 +0.003515。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 73 | 53.4% | True | +12.567017 | 75.1% |
| 错误 | 41 | 45.1% | True | +12.337908 | 75.6% |

主体中 125 个 token 与 OCR 行相交，34 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 179 | background | -0.345173 |  | Product of China. |  |
| 287 | background | -0.261978 |  | more than 11.5 mL per injection site. |  |
| 525 | background | -0.240353 |  | Elanco |  |
| 114 | background | -0.228247 |  | Increxxa25 |  |
| 221 | background | -0.209026 |  |  |  |
| 14 | background | -0.200741 |  | Increxxa 25 |  |
| 271 | subject_text | -0.182867 | (tulathromycin injection) | 25 mg of tulathromycin/mL | (tulathromycin injection) |
| 290 | subject_nontext | -0.181924 |  |  | Injectable Solution |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 557 | subject_text | +0.281704 | Approved by FDA under ANADA # 200-665 | Approved by FDA under ANADA # 200-665 | Approved by FDA under ANADA # 200-665 |
| 533 | subject_text | +0.279350 | Net Contents: 250 mL | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 714 | background | +0.266340 |  | Net Contents: 250 mL |  |
| 708 | background | +0.247826 |  |  |  |
| 410 | subject_text | +0.204744 | ruminating cattle. | ruminating cattle. |  |
| 475 | subject_nontext | +0.199805 |  |  |  |
| 38 | background | +0.192308 |  |  |  |
| 411 | subject_text | +0.184419 | ruminating cattle. | treatment with Increxxa 25 Injectable Solution. |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 25 mg of tulathromycin/mL | 9 | +0.000036 |
| Net Contents: 250 mL | 6 | +0.000669 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006099 | +0.015164 |
| 只保留主体分差 | +0.003515 | +0.034226 |
| 主体 token 数 | 210 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
