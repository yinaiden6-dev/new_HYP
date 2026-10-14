# B03 · photo_0e87e0717ab1c215

按当前评测真值分析，未新增身份人工复核。

包装 `package_107`，划分 `train`；用户主题：细小文字差异。正确参考 `reference_000147`，**纯图像实际 Top-1 错误参考** `reference_000146`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.767841 |
| 错误参考 MaxSim | +0.773310 |
| 正确减错误的分差 | -0.005469 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 60 个。正向贡献 +0.017601，负向贡献 -0.023070，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 660 | -0.004871 | -0.005395 |
| nonvisual | 11 | +0.000009 | +0.000591 |
| subject_nontext | 21 | -0.000189 | -0.006583 |
| subject_text | 39 | -0.000418 | -0.007838 |

最主要负差额来自 `background`，贡献 -0.004871。只保留主体 query token 后，本对分差为 -0.007399。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 40 | 34.5% | True | +12.398812 | 73.3% |
| 错误 | 27 | 29.5% | True | +12.189106 | 73.3% |

主体中 39 个 token 与 OCR 行相交，26 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 501 | background | -0.240913 |  |  |  |
| 28 | background | -0.239687 |  |  |  |
| 26 | background | -0.232594 |  | 11 |  |
| 328 | background | -0.199324 |  |  |  |
| 273 | background | -0.176939 |  | treatment with Increxxa 25 Injectable Solution. |  |
| 586 | background | -0.176766 |  |  |  |
| 296 | background | -0.174591 |  |  |  |
| 82 | background | -0.145516 |  | more than 11.5 mL per injection site. |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 446 | background | +0.308695 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 453 | subject_text | +0.279405 | Net Contents: 250mL | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 462 | background | +0.276360 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 468 | background | +0.248161 |  | Approved by FDA under ANADA # 200-665 | Approved by FDA under ANADA # 200-665 |
| 650 | background | +0.229144 |  | Net Contents: 250 mL |  |
| 147 | background | +0.210019 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 535 | subject_nontext | +0.201440 |  |  |  |
| 107 | background | +0.196906 |  | this drug to use by or on the order of a |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 25 mg of tulathromycin/mL | 5 | -0.000307 |
| Net Contents: 250mL | 3 | +0.000402 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_72ad5f2e7435bc37` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005469 | +0.015164 |
| 只保留主体分差 | -0.007399 | +0.034226 |
| 主体 token 数 | 60 | 210 |

[成功对照 token 表](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
