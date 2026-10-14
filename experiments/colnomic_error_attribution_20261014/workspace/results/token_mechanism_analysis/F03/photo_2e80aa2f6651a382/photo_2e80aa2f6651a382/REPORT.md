# photo_2e80aa2f6651a382

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000098`，F03 错误参考 `reference_005118`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_000098`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_000098 |
| BM25 真值排名 | 17 |
| 融合真值排名 | 3 |
| 图像正负分差 | +0.082449 |
| BM25 正负分差 | -83.602982 |
| 0.5 × 图像 z 分差 | +1.508911 |
| 0.5 × BM25 z 分差 | -1.602926 |
| 最终融合分差 | -0.094015 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5151**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 308 个。正向贡献 +0.167140，负向贡献 -0.084692，相加等于图像分差 +0.082449。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 460 | -0.010490 | -0.017764 |
| nonvisual | 11 | +0.001017 | +0.072055 |
| subject_nontext | 171 | +0.035870 | +0.163407 |
| subject_text | 137 | +0.056051 | +0.318714 |

仅保留主体 query token 的两候选分差：+0.232488。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.010490。主体框内 token 占有效序列 39.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 166 | 36.4% | True | +10.648450 | 74.3% |
| 错误 | 8 | 2.1% | False | 不可评估 | 不可评估 |

主体中 137 个 token 与 OCR 相交，42 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 315 | background | -0.602642 | 40mg per 0.4 mL |  | r0.4mL |
| 757 | subject_nontext | -0.595683 |  | (nitenpyram) |  |
| 249 | background | -0.595129 | Enoxaparin | (nitenpyram) | Enoxaparin |
| 349 | subject_text | -0.594842 | actre gederte |  |  |
| 250 | background | -0.587147 | Enoxaparin | (nitenpyram) | Enoxaparin |
| 251 | background | -0.586211 | Enoxaparin | and older. Weigh your pet prior to administration to ensure proper dosage. It is also safe for use in | Enoxaparin |
| 778 | nonvisual | -0.585378 |  | (nitenpyram) |  |
| 473 | subject_nontext | -0.583841 |  |  | 10 x 0.4 mL Single-Dose Prefilled |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 400 | subject_text | +0.716782 | CAPSTAR | CAPSTR |  |
| 399 | subject_text | +0.714816 | CAPSTAR | CAPSTR | Sterile, Nonpyrogenic, Preservative-free, PVC-free, DEHP-free. |
| 398 | subject_text | +0.696653 | CAPSTAR | CAPSTR | Sterile, Nonpyrogenic, Preservative-free, PVC-free, DEHP-free. |
| 745 | background | +0.689794 |  |  |  |
| 672 | subject_nontext | +0.677993 |  |  |  |
| 713 | background | +0.669790 |  |  |  |
| 624 | subject_text | +0.668975 | infestations on dogs |  |  |
| 569 | subject_nontext | +0.661331 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 11.4mg | 5 | +0.001957 |
| For dogs 2-25 lbs \|6 tablets (11.4 mg each) | 21 | +0.011137 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_b881c4736ab7cef0` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.082449 | +0.308738 |
| 仅主体分差 | +0.232488 | +0.408509 |
| 主体 token 数 | 308 | 264 |

[对照 token 数据](../photo_b881c4736ab7cef0/tokens.csv) · [对照摘要](../photo_b881c4736ab7cef0/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
