# photo_e806395db591b23e

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001966`，F03 错误参考 `reference_005118`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001966`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001966 |
| BM25 真值排名 | 20 |
| 融合真值排名 | 7 |
| 图像正负分差 | +0.037090 |
| BM25 正负分差 | -137.454102 |
| 0.5 × 图像 z 分差 | +0.607221 |
| 0.5 × BM25 z 分差 | -2.128935 |
| 最终融合分差 | -1.521713 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.7781**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 144 个。正向贡献 +0.124717，负向贡献 -0.087627，相加等于图像分差 +0.037090。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 624 | -0.015607 | -0.019484 |
| nonvisual | 11 | -0.000011 | -0.000744 |
| subject_nontext | 43 | +0.007598 | +0.137652 |
| subject_text | 101 | +0.045109 | +0.347923 |

仅保留主体 query token 的两候选分差：+0.285134。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.015607。主体框内 token 占有效序列 18.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 127 | 54.6% | True | +2.408641 | 44.6% |
| 错误 | 9 | 23.4% | True | +12.224940 | 94.9% |

主体中 101 个 token 与 OCR 相交，39 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 462 | background | -0.659549 | For Subcutaneous |  | For Subcutaneous Use |
| 419 | background | -0.605922 |  |    d t  |  |
| 184 | background | -0.600918 | Enoxaparin | IBUPROFEN |  |
| 217 | background | -0.577387 | Enoxaparin | inw the ri hot a, hetla and en | Enoxaparin |
| 676 | background | -0.574134 |  |  |  |
| 367 | background | -0.573660 | Enoxap |  | Enoxaparin |
| 778 | nonvisual | -0.571588 |  |    d t  |  |
| 187 | background | -0.567457 | Enoxaparin | If pctae emar tad an dnc wtn s boune cray | Enoxaparin |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 338 | subject_text | +0.683109 | TWIN PACK | TWIN PACK |  |
| 370 | subject_text | +0.661777 | TWIN PACK | TWIN PACK | TEAR OFF 1O OPEN BOCHERE |
| 561 | subject_text | +0.631696 | NSAID | NSAID* |  |
| 404 | subject_text | +0.606979 | CAREOne | CAREOne | TEAR OFF 1O OPEN BOCHERE |
| 346 | subject_text | +0.606701 | 2 × 100 COUNT | 2 × 100 COUNT |  |
| 665 | subject_text | +0.603840 | COATED TABLETS | COATED TABLETS |  |
| 528 | subject_text | +0.600139 | Tablets, USP | Tablets, USP |  |
| 403 | subject_text | +0.597023 | CAREOne | CAREOne |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 200 mg | 11 | +0.004857 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_eb65c801ffc82c76` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.037090 | +0.202070 |
| 仅主体分差 | +0.285134 | +0.344067 |
| 主体 token 数 | 144 | 256 |

[对照 token 数据](../photo_eb65c801ffc82c76/tokens.csv) · [对照摘要](../photo_eb65c801ffc82c76/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
