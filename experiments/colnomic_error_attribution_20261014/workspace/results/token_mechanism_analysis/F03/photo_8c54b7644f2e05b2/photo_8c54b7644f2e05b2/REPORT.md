# photo_8c54b7644f2e05b2

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_004975`，F03 错误参考 `reference_004976`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 10 |
| 图像 Top-1 参考 | reference_004976 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.014849 |
| BM25 正负分差 | +1.073883 |
| 0.5 × 图像 z 分差 | -0.358738 |
| 0.5 × BM25 z 分差 | +0.065616 |
| 最终融合分差 | -0.293123 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.1546**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 165 个。正向贡献 +0.014539，负向贡献 -0.029389，相加等于图像分差 -0.014849。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 603 | -0.009063 | -0.011708 |
| nonvisual | 11 | -0.000175 | -0.012374 |
| subject_nontext | 91 | -0.001992 | -0.017053 |
| subject_text | 74 | -0.003620 | -0.038104 |

仅保留主体 query token 的两候选分差：-0.026494。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.009063。主体框内 token 占有效序列 21.2%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 192 | 44.3% | True | +6.553124 | 66.7% |
| 错误 | 152 | 42.0% | True | +6.187395 | 62.8% |

主体中 74 个 token 与 OCR 相交，44 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 35 | background | -0.536095 |  |  |  |
| 367 | subject_text | -0.415421 | Telangana-502281,INDIA |  |  |
| 273 | subject_text | -0.358975 | Memphis, TN 38141. |  |  |
| 368 | subject_text | -0.357787 | Telangana-502281,INDIA |  |  |
| 306 | subject_text | -0.324521 | Manufactured by: |  |  |
| 275 | subject_text | -0.310771 | Memphis, TN 38141. | Memphis, TN 38141. |  |
| 528 | subject_text | -0.298871 | WZ | NDC 72603-327-10 |  |
| 304 | subject_text | -0.276797 | Manufactured by: |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 567 | subject_text | +0.350080 | 2005 Northstar Healthcare Holdings Ltd. |  | © 2005 Northstar Healthcare Holdings Ltd. |
| 566 | subject_text | +0.327269 | 726031132710111 | © 2005 Northstar Healthcare Holdings Ltd. | © 2005 Northstar Healthcare Holdings Ltd. |
| 366 | subject_nontext | +0.222618 |  |  |  |
| 18 | background | +0.217190 | Perficore |  |  |
| 564 | subject_text | +0.185740 | 726031132710111 |  | © 2005 Northstar Healthcare Holdings Ltd. |
| 374 | subject_text | +0.183111 | Mfg. Lic. No.: 36/MD/AP/2013/F/G |  |  |
| 335 | subject_text | +0.175914 | Aspiro Pharma Limited | Aspiro Pharma Limited | Manufactured by: |
| 763 | background | +0.150188 |  |  |  |

## 成功对照

`photo_709e8a3ee42539bc` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.014849 | +0.008916 |
| 仅主体分差 | -0.026494 | +0.026366 |
| 主体 token 数 | 165 | 153 |

[对照 token 数据](../photo_709e8a3ee42539bc/tokens.csv) · [对照摘要](../photo_709e8a3ee42539bc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
