# photo_65fa10852c465cb1

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001644`，F03 错误参考 `reference_001643`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001644`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001644 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.011496 |
| BM25 正负分差 | -6.466408 |
| 0.5 × 图像 z 分差 | +0.157735 |
| 0.5 × BM25 z 分差 | -0.725551 |
| 最终融合分差 | -0.567817 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.8214**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 182 个。正向贡献 +0.034477，负向贡献 -0.022980，相加等于图像分差 +0.011496。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 586 | +0.006874 | +0.009139 |
| nonvisual | 11 | +0.000671 | +0.047537 |
| subject_nontext | 104 | +0.001566 | +0.011727 |
| subject_text | 78 | +0.002385 | +0.023819 |

仅保留主体 query token 的两候选分差：+0.016909。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 +0.000671。主体框内 token 占有效序列 23.4%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 282 | 46.5% | True | +6.459123 | 57.5% |
| 错误 | 25 | 2.5% | False | 不可评估 | 不可评估 |

主体中 78 个 token 与 OCR 相交，16 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 19 | background | -0.226471 |  |  |  |
| 568 | background | -0.226395 |  |  |  |
| 297 | subject_nontext | -0.218275 |  |  |  |
| 148 | background | -0.215579 |  |  | NDC: 68001-477-48 |
| 330 | subject_text | -0.194031 | NDO:69001-477-46 | Bacitracin Ointment, USP | NDC: 68001-477-48 |
| 695 | background | -0.188044 |  |  |  |
| 694 | background | -0.181054 |  |  |  |
| 50 | background | -0.177647 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 489 | subject_text | +0.463956 | LABCIATORIES | LARCEATORIES |  |
| 394 | subject_text | +0.424093 | Bacitracin Ointment, USP | LARCEATORIES |  |
| 488 | subject_text | +0.422158 | LABCIATORIES | For BluePoint Laboratories |  |
| 74 | background | +0.385137 |  | LARCEATORIES |  |
| 603 | background | +0.384986 |  |  |  |
| 490 | subject_text | +0.380015 | LABCIATORIES | LARCEATORIES |  |
| 264 | background | +0.350019 |  | For BluePoint Laboratories |  |
| 748 | background | +0.344312 |  |  |  |

## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
