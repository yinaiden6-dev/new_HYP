# photo_ee67048e97d12344

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_003207`，F03 错误参考 `reference_003238`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003123`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_003123 |
| BM25 真值排名 | 66 |
| 融合真值排名 | 21 |
| 图像正负分差 | +0.034440 |
| BM25 正负分差 | -5.882217 |
| 0.5 × 图像 z 分差 | +0.595444 |
| 0.5 × BM25 z 分差 | -2.012887 |
| 最终融合分差 | -1.417443 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.7717**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 210 个。正向贡献 +0.050896，负向贡献 -0.016456，相加等于图像分差 +0.034440。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 510 | +0.019995 | +0.028659 |
| nonvisual | 11 | +0.000395 | +0.026267 |
| subject_nontext | 136 | +0.007639 | +0.041058 |
| subject_text | 74 | +0.006412 | +0.063335 |

仅保留主体 query token 的两候选分差：+0.048908。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 +0.000395。主体框内 token 占有效序列 28.7%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 303 | 64.5% | True | +8.159546 | 73.7% |
| 错误 | 20 | 60.2% | True | +10.387807 | 93.3% |

主体中 74 个 token 与 OCR 相交，30 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 217 | subject_nontext | -0.248562 |  |  |  |
| 247 | subject_nontext | -0.209797 |  |  |  |
| 302 | background | -0.199973 |  | 1. |  |
| 31 | background | -0.194436 |  |  |  |
| 249 | subject_nontext | -0.194142 |  |  |  |
| 250 | subject_nontext | -0.181334 |  |  | 1. |
| 77 | background | -0.178791 |  | ，，， |  |
| 248 | subject_nontext | -0.177341 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 468 | subject_text | +0.318188 | 0-13,0. | 0,00\|-13,0\|\|0, |  |
| 400 | subject_nontext | +0.311659 |  | Polyisobutene Propylparaben Sodium Polyacrylate Sodium Hyaluronate Trisodium EDTA | 景0 |
| 163 | background | +0.302355 |  | 1. |  |
| 68 | background | +0.301068 |  |  |  |
| 285 | background | +0.292391 |  | 1. | Discontinue use if signs of irritation or rashes appear. |
| 166 | background | +0.287796 |  |  |  |
| 307 | subject_nontext | +0.284942 |  | worse and need to consult a dermatologist. 1) In case of having problems such |  |
| 373 | subject_nontext | +0.272045 |  |  |  |

## 成功对照

`photo_43eca55aded800ef` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.034440 | +0.092892 |
| 仅主体分差 | +0.048908 | +0.134425 |
| 主体 token 数 | 210 | 224 |

[对照 token 数据](../photo_43eca55aded800ef/tokens.csv) · [对照摘要](../photo_43eca55aded800ef/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
