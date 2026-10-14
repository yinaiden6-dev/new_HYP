# B03 · photo_fb6b4915947444a2

按当前评测真值分析，未新增身份人工复核。

包装 `package_036`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_004768`，**纯图像实际 Top-1 错误参考** `reference_004769`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.626190 |
| 错误参考 MaxSim | +0.627316 |
| 正确减错误的分差 | -0.001126 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 65 个。正向贡献 +0.020517，负向贡献 -0.021643，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 655 | -0.000764 | -0.000852 |
| nonvisual | 11 | -0.000120 | -0.007988 |
| subject_nontext | 41 | -0.000423 | -0.007544 |
| subject_text | 24 | +0.000181 | +0.005516 |

最主要负差额来自 `background`，贡献 -0.000764。只保留主体 query token 后，本对分差为 -0.002722。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 69 | 7.1% | False | 不可评估 | 不可评估 |
| 错误 | 35 | 6.5% | False | 不可评估 | 不可评估 |

主体中 24 个 token 与 OCR 行相交，15 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 54 | background | -0.358971 |  |  | Manufactured by: |
| 51 | background | -0.352469 |  | Swiftwater, PA 18370 USA | Manufactured by: |
| 254 | background | -0.337195 |  |  | Manufactured by: |
| 245 | background | -0.308323 |  |  | Manufactured by: |
| 248 | background | -0.253980 |  |  | Manufactured by: |
| 43 | background | -0.189081 |  |  |  |
| 257 | background | -0.187221 |  |  | Manufactured by: |
| 427 | background | -0.183298 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 336 | subject_nontext | +0.344221 |  |  |  |
| 326 | background | +0.316121 |  |  |  |
| 276 | background | +0.266298 |  |  |  |
| 26 | background | +0.224413 |  |  |  |
| 73 | background | +0.214924 |  | Swiftwater, PA 18370 USA | Co., USA. |
| 656 | background | +0.180712 |  |  |  |
| 240 | background | +0.171040 |  |  |  |
| 393 | subject_text | +0.170880 | Swiftwater, PA 18370 USA | Swiftwater, PA 18370 USA |  |

## 同包装成功对照

`photo_09c56f438b411fb1` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001126 | +0.019731 |
| 只保留主体分差 | -0.002722 | +0.024085 |
| 主体 token 数 | 65 | 70 |

[成功对照 token 表](../photo_09c56f438b411fb1/tokens.csv) · [对照摘要](../photo_09c56f438b411fb1/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
