# photo_079284acc905a73f

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001023`，F03 错误参考 `reference_001310`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001023`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001023 |
| BM25 真值排名 | 4 |
| 融合真值排名 | 3 |
| 图像正负分差 | +0.005185 |
| BM25 正负分差 | -19.243591 |
| 0.5 × 图像 z 分差 | +0.091195 |
| 0.5 × BM25 z 分差 | -1.625874 |
| 最终融合分差 | -1.534678 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.9469**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 120 个。正向贡献 +0.036332，负向贡献 -0.031147，相加等于图像分差 +0.005185。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 600 | +0.002469 | +0.003008 |
| nonvisual | 11 | +0.000231 | +0.015369 |
| subject_nontext | 50 | -0.000323 | -0.004725 |
| subject_text | 70 | +0.002808 | +0.029321 |

仅保留主体 query token 的两候选分差：+0.015135。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_nontext`，贡献 -0.000323。主体框内 token 占有效序列 16.4%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 204 | 46.7% | True | +6.281476 | 60.8% |
| 错误 | 116 | 10.1% | False | 不可评估 | 不可评估 |

主体中 70 个 token 与 OCR 相交，26 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 32 | background | -0.341612 |  |  |  |
| 38 | background | -0.311286 |  |  |  |
| 278 | subject_text | -0.294929 | Seltzerss | Alkze |  |
| 525 | background | -0.294078 |  |  |  |
| 486 | background | -0.292027 |  | Expiry Date. |  |
| 158 | background | -0.247155 |  |  |  |
| 476 | subject_text | -0.246645 | Leam about teen medicine abuse |  | Leam about teen medicine abuse |
| 98 | background | -0.239778 |  | DOES NOT CONTAIN ASPIRIN |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 379 | subject_text | +0.487665 | Cold&Flu | &Flu | Sinus Pressure |
| 184 | background | +0.454472 |  | &Flu |  |
| 79 | background | +0.441274 |  | &Flu |  |
| 259 | subject_nontext | +0.434463 |  | &Flu |  |
| 244 | background | +0.422005 |  | &Flu |  |
| 564 | background | +0.410797 |  | &Flu |  |
| 416 | subject_text | +0.391846 | DOES NOT CONTAIN ASPIRIN | DOES NOT CONTAIN ASPIRIN |  |
| 579 | background | +0.386858 |  | &Flu |  |

## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
