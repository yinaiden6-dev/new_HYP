# photo_3ff837bd4edbded5

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_004359`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003474`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_003474 |
| BM25 真值排名 | 30 |
| 融合真值排名 | 4 |
| 图像正负分差 | +0.005321 |
| BM25 正负分差 | -22.349510 |
| 0.5 × 图像 z 分差 | +0.194561 |
| 0.5 × BM25 z 分差 | -0.861856 |
| 最终融合分差 | -0.667294 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.8158**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 160 个。正向贡献 +0.038210，负向贡献 -0.032889，相加等于图像分差 +0.005321。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 608 | -0.002294 | -0.002939 |
| nonvisual | 11 | +0.000284 | +0.020142 |
| subject_nontext | 123 | +0.009182 | +0.058152 |
| subject_text | 37 | -0.001851 | -0.038970 |

仅保留主体 query token 的两候选分差：+0.035693。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.002294。主体框内 token 占有效序列 20.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 83 | 34.2% | True | +7.646179 | 76.3% |
| 错误 | 29 | 4.2% | False | 不可评估 | 不可评估 |

主体中 37 个 token 与 OCR 相交，17 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 225 | background | -0.404345 | Pharmacy |  |  |
| 226 | background | -0.378578 | Pharmacy | If you are pregnant or breast-feeding, only use this medicine |  |
| 257 | background | -0.376310 | Pharmacy |  |  |
| 352 | background | -0.344628 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |
| 321 | background | -0.343466 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |
| 320 | background | -0.322226 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |
| 289 | background | -0.319576 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |
| 258 | background | -0.318834 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 432 | subject_nontext | +0.443621 |  |  |  |
| 441 | subject_nontext | +0.420988 |  |  | STOP SMOKING AID |
| 437 | subject_nontext | +0.406757 |  |  | STOP SMOKING AID |
| 380 | background | +0.357818 |  |  | Compare to the active ingredient |
| 427 | subject_nontext | +0.352429 |  | artificial mint flavor, potassium bicarbonate, sodium alginate, | Compare to the active ingredient |
| 377 | subject_nontext | +0.339044 |  |  | STOP SMOKING AID |
| 497 | subject_nontext | +0.334123 |  |  |  |
| 715 | background | +0.330808 |  |  | Compare to the active ingredient |

## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.005321 | +0.003739 |
| 仅主体分差 | +0.035693 | +0.023522 |
| 主体 token 数 | 160 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
