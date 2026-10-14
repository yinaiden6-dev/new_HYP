# photo_aabee66f53cb53e8

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_004359`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003367`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 3 |
| 图像 Top-1 参考 | reference_003367 |
| BM25 真值排名 | 5 |
| 融合真值排名 | 4 |
| 图像正负分差 | -0.010339 |
| BM25 正负分差 | -38.258041 |
| 0.5 × 图像 z 分差 | -0.269002 |
| 0.5 × BM25 z 分差 | -0.902297 |
| 最终融合分差 | -1.171299 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 121 个。正向贡献 +0.023533，负向贡献 -0.033872，相加等于图像分差 -0.010339。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 647 | -0.016708 | -0.020117 |
| nonvisual | 11 | -0.000022 | -0.001540 |
| subject_nontext | 9 | +0.000538 | +0.046566 |
| subject_text | 112 | +0.005853 | +0.040707 |

仅保留主体 query token 的两候选分差：+0.041143。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.016708。主体框内 token 占有效序列 15.5%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 225 | 52.9% | True | +2.695987 | 46.3% |
| 错误 | 9 | 10.6% | False | 不可评估 | 不可评估 |

主体中 112 个 token 与 OCR 相交，76 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 294 | background | -0.614934 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | CVSHealth |
| 263 | background | -0.593457 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | CVSHealth |
| 262 | background | -0.557249 |  | 73460 UW C3 | CVSHealth |
| 326 | background | -0.434213 | pharmacy | *This product is not manufactured or distrbuted by GlaxoSmithKline | © 2022 CVS/pharmacy |
| 27 | background | -0.399811 |  |  |  |
| 295 | background | -0.371275 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | © 2022 CVS/pharmacy |
| 325 | background | -0.346790 |  | *This product is not manufactured or distrbuted by GlaxoSmithKline |  |
| 679 | background | -0.319832 | M·CYVM | *This product is not manufactured or distrbuted by GlaxoSmithKline |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 502 | subject_text | +0.534752 | Love it or your money back. | Love it or your money back. |  |
| 213 | subject_text | +0.438434 | Love it or your money back. | the enclosed User's Guide. |  |
| 342 | subject_text | +0.399289 | Love it or your money back. |  | back |
| 476 | background | +0.359084 | dsuel |  | CVS.com/returnpolicy |
| 309 | subject_text | +0.330444 | Satisfaction guaranteed- | Satisfaction guaranteed – | CVS.com/returnpolicy |
| 341 | subject_text | +0.328989 | Satisfaction guaranteed- | Satisfaction guaranteed – | CVS.com/returnpolicy |
| 373 | subject_text | +0.314034 | Satisfaction guaranteed- | Satisfaction guaranteed – | CVS.com/returnpolicy |
| 277 | subject_text | +0.301483 | Satisfaction guaranteed- | Satisfaction guaranteed – | CVS.com/returnpolicy |

## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010339 | +0.003739 |
| 仅主体分差 | +0.041143 | +0.023522 |
| 主体 token 数 | 121 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
