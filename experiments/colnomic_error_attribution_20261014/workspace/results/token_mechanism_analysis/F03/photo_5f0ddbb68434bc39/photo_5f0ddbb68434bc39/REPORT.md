# photo_5f0ddbb68434bc39

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000792`，F03 错误参考 `reference_003367`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_000792`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_000792 |
| BM25 真值排名 | 177 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.058574 |
| BM25 正负分差 | -88.221893 |
| 0.5 × 图像 z 分差 | +1.404552 |
| 0.5 × BM25 z 分差 | -2.807846 |
| 最终融合分差 | -1.403294 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.6666**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 143 个。正向贡献 +0.122331，负向贡献 -0.063757，相加等于图像分差 +0.058574。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 625 | +0.018786 | +0.023415 |
| nonvisual | 11 | +0.001718 | +0.121689 |
| subject_nontext | 55 | -0.001148 | -0.016257 |
| subject_text | 88 | +0.039217 | +0.347161 |

仅保留主体 query token 的两候选分差：+0.207384。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_nontext`，贡献 -0.001148。主体框内 token 占有效序列 18.4%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 72 | 35.8% | True | +5.119068 | 60.0% |
| 错误 | 7 | 9.3% | False | 不可评估 | 不可评估 |

主体中 88 个 token 与 OCR 相交，22 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 126 | background | -0.610386 | NASAL | Contact wearers: consult physician prior to | NASAL MIST |
| 96 | background | -0.546051 | No Drip | Active ingredients ....Purpose | No Drip |
| 158 | background | -0.542336 | NASAL | Contact wearers: consult physician prior to | NASAL MIST |
| 191 | background | -0.540099 | NASAL |  | ASAL L |
| 159 | background | -0.538052 | NASAL | containing 0.000002% alkaloids | NASAL |
| 125 | background | -0.526958 | NASAL | walgreens.com ©2012 Walgreen Co. |  |
| 223 | background | -0.494747 | NASAL |  |  |
| 99 | background | -0.493567 | CAS | Walgreens Pharmacist Survey | CVS |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 337 | subject_text | +0.666507 | Walgreens | Walgreens | Inc., distributor of Afrin® No Drip |
| 305 | subject_text | +0.652017 | Walgreens | Walgreens | Inc., distributor of Afrin® No Drip |
| 275 | subject_text | +0.606998 | Well | Well |  |
| 276 | subject_text | +0.606690 | Well | Well | Warnings |
| 89 | background | +0.598172 | HiL | Well |  |
| 308 | subject_text | +0.598083 | Well | at | OXYMETAZOLINE HCI 0.05% |
| 336 | subject_text | +0.595255 | Walgreens | Walgreens |  |
| 533 | subject_text | +0.591972 | Eye Drops | Eye Drops |  |

## 成功对照

`photo_53000d6acda74399` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.058574 | +0.042773 |
| 仅主体分差 | +0.207384 | +0.227551 |
| 主体 token 数 | 143 | 176 |

[对照 token 数据](../photo_53000d6acda74399/tokens.csv) · [对照摘要](../photo_53000d6acda74399/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
