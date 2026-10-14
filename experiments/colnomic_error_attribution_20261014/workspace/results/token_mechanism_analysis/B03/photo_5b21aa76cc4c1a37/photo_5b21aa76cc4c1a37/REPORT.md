# B03 · photo_5b21aa76cc4c1a37

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000146`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.736661 |
| 错误参考 MaxSim | +0.742832 |
| 正确减错误的分差 | -0.006172 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 66 个。正向贡献 +0.016449，负向贡献 -0.022621，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 654 | -0.005771 | -0.006451 |
| nonvisual | 11 | +0.000005 | +0.000330 |
| subject_nontext | 13 | -0.000000 | -0.000021 |
| subject_text | 53 | -0.000405 | -0.005588 |

最主要负差额来自 `background`，贡献 -0.005771。只保留主体 query token 后，本对分差为 -0.004491。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 33 | 22.4% | True | +14.876285 | 98.5% |
| 错误 | 23 | 16.4% | True | +14.194926 | 98.5% |

主体中 53 个 token 与 OCR 行相交，29 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 86 | background | -0.230313 |  |  |  |
| 179 | background | -0.222957 |  |  |  |
| 321 | background | -0.220399 |  |  |  |
| 27 | background | -0.209898 |  | Increxxa 25 |  |
| 625 | background | -0.186575 |  |  |  |
| 342 | background | -0.173973 |  |  |  |
| 225 | background | -0.172003 |  |  |  |
| 490 | background | -0.153766 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 439 | background | +0.270922 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 433 | subject_text | +0.248026 | Net Contents: 250 mL | Net Contents: 250 mL |  |
| 542 | background | +0.239090 |  | Net Contents: 250 mL |  |
| 317 | background | +0.210706 |  | Net Contents: 250 mL |  |
| 547 | background | +0.207249 |  | Net Contents: 250 mL |  |
| 313 | background | +0.193430 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 123 | background | +0.180950 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 115 | background | +0.177044 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Net Contents: 250 mL | 3 | +0.000318 |
| 25 mg of tulathromycin/mL | 7 | +0.000088 |
| Net Contents: 250mL | 4 | +0.000059 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006172 | +0.015164 |
| 只保留主体分差 | -0.004491 | +0.034226 |
| 主体 token 数 | 66 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
