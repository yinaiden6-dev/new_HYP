# photo_fd3bb2cdcb3bd3a7

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000265`，F03 错误参考 `reference_003310`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 24 |
| 图像 Top-1 参考 | reference_003310 |
| BM25 真值排名 | 247 |
| 融合真值排名 | 34 |
| 图像正负分差 | -0.040275 |
| BM25 正负分差 | -46.631111 |
| 0.5 × 图像 z 分差 | -1.314761 |
| 0.5 × BM25 z 分差 | -1.259015 |
| 最终融合分差 | -2.573776 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 116 个。正向贡献 +0.049185，负向贡献 -0.089460，相加等于图像分差 -0.040275。缓存复现误差 3e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 652 | -0.059246 | -0.070786 |
| nonvisual | 11 | -0.000276 | -0.019549 |
| subject_nontext | 37 | +0.000472 | +0.009929 |
| subject_text | 79 | +0.018775 | +0.185136 |

仅保留主体 query token 的两候选分差：+0.129251。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.059246。主体框内 token 占有效序列 14.9%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 10 | 5.1% | False | 不可评估 | 不可评估 |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 79 个 token 与 OCR 相交，14 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 530 | background | -0.695353 | BEES |  | BEES |
| 536 | background | -0.680526 | TOOTHPASTE |  | Multi-Care |
| 763 | background | -0.668646 |  | JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC |  |
| 562 | background | -0.656629 |  | JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC |  |
| 532 | background | -0.654691 | BEES | ONCE A MONTH | Complete |
| 498 | background | -0.650360 | BURT'S |  | BURT'S |
| 597 | background | -0.648898 |  | ONCE A MONTH |  |
| 313 | subject_nontext | -0.647880 |  |  | Multi-Care |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 337 | subject_text | +0.559950 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 305 | subject_text | +0.547630 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 304 | subject_text | +0.527265 | Comfortis (spinosad) | Comfortis | Complete |
| 338 | subject_text | +0.522637 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 273 | subject_text | +0.517800 | Comfortis (spinosad) | Comfortis™(spinosad) | Complete |
| 458 | background | +0.511342 |  | Comfortis™(spinosad) |  |
| 334 | subject_text | +0.502134 | Comfortis (spinosad) | Comfortis |  |
| 302 | subject_text | +0.498188 | Comfortis (spinosad) | Comfortis™(spinosad) |  |

## 成功对照

`photo_d68d74e6a16e45bf` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.040275 | +0.179158 |
| 仅主体分差 | +0.129251 | +0.387085 |
| 主体 token 数 | 116 | 247 |

[对照 token 数据](../photo_d68d74e6a16e45bf/tokens.csv) · [对照摘要](../photo_d68d74e6a16e45bf/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
