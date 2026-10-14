# photo_8bc3a7a3577518f0

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001182`，F03 错误参考 `reference_002727`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003251`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1004 |
| 图像 Top-1 参考 | reference_003251 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 16 |
| 图像正负分差 | -0.157122 |
| BM25 正负分差 | +10.092270 |
| 0.5 × 图像 z 分差 | -1.473304 |
| 0.5 × BM25 z 分差 | +0.114070 |
| 最终融合分差 | -1.359234 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.0719**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 190 个。正向贡献 +0.006758，负向贡献 -0.163880，相加等于图像分差 -0.157122。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 530 | -0.103219 | -0.142364 |
| nonvisual | 11 | -0.000368 | -0.024431 |
| subject_nontext | 40 | -0.005927 | -0.108309 |
| subject_text | 150 | -0.047609 | -0.232015 |

仅保留主体 query token 的两候选分差：-0.205972。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.103219。主体框内 token 占有效序列 26.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 491 | 76.1% | True | +13.662335 | 79.1% |
| 错误 | 9 | 19.9% | True | +7.734418 | 89.7% |

主体中 150 个 token 与 OCR 相交，121 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 460 | background | -0.565941 |  |  | Overdose warning |
| 230 | subject_text | -0.538957 | Ask a doctor before use If you hava |  | Liver disease  a breathing problem such |
| 464 | background | -0.528674 |  |  | Overdose warning |
| 278 | subject_text | -0.519747 | Ask a doctor or pharmacist before use If you are | minor aches and pains with accompanying | prostate gland = glaucoma |
| 465 | background | -0.505349 |  |  | Overdose warning |
| 712 | background | -0.502628 |  | This product Is not affillated with the makers/owners of | get inedical help or contact a Polson Control |
| 478 | subject_text | -0.497058 | recommended dose (overdose) may cause liver |  |  |
| 138 | background | -0.493752 |  | shellac glaze, stearic acid, titanium dioxide, | prostate gland = glaucoma |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 214 | subject_text | +0.212007 | Drug Facts (continued) | Drug Facts (continued) | Drug Facts(continned) |
| 86 | background | +0.193872 |  | Drug Facts | Drug Facts(continund) |
| 159 | background | +0.178987 |  |  |  |
| 195 | background | +0.177783 |  | Drug Facts (continued) | Drug Facts(continned) |
| 215 | subject_text | +0.174541 | Drug Facts (continued) | Drug Facts (continued) | Drug Facts(continned) |
| 145 | background | +0.161587 |  |  |  |
| 177 | background | +0.159923 |  |  |  |
| 116 | background | +0.159723 |  | sleeplessness persists continuously for more | sleeplessness persists continuously for |

## 成功对照

`photo_441c254874dcdd8d` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.157122 | +0.197560 |
| 仅主体分差 | -0.205972 | +0.211878 |
| 主体 token 数 | 190 | 190 |

[对照 token 数据](../photo_441c254874dcdd8d/tokens.csv) · [对照摘要](../photo_441c254874dcdd8d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
