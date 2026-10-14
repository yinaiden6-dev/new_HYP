# photo_37c9ab8958ab73bc

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001182`，F03 错误参考 `reference_002727`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 355 |
| 图像 Top-1 参考 | reference_002727 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.059276 |
| BM25 正负分差 | +29.174332 |
| 0.5 × 图像 z 分差 | -0.675174 |
| 0.5 × BM25 z 分差 | +0.320529 |
| 最终融合分差 | -0.354644 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.3219**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 72 个。正向贡献 +0.022189，负向贡献 -0.081465，相加等于图像分差 -0.059276。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 648 | -0.051687 | -0.058307 |
| nonvisual | 11 | -0.000201 | -0.013385 |
| subject_nontext | 19 | -0.000549 | -0.021112 |
| subject_text | 53 | -0.006839 | -0.094326 |

仅保留主体 query token 的两候选分差：-0.075006。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.051687。主体框内 token 占有效序列 9.8%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 199 | 57.6% | True | +11.168757 | 83.3% |
| 错误 | 3 | 0.0% | False | 不可评估 | 不可评估 |

主体中 53 个 token 与 OCR 相交，46 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 453 | subject_text | -0.512498 | contact a Polson Control Center right away. | DISTRIBUTED BY: fred's, Inc. | Quick medical attention is critical for adults |
| 447 | background | -0.489042 |  |  |  |
| 443 | background | -0.488518 |  |  |  |
| 333 | subject_text | -0.477900 | trouble urtnating due to an enkarged prostate gland |  | trouble urinating due to an enlarged |
| 413 | subject_text | -0.471641 | fever get worse or lasts more than 3 days |  |  |
| 632 | background | -0.461038 |  |  | may canse liver danage. In case of overdose, |
| 543 | background | -0.441017 |  |  | prostate gland = glaucoma |
| 446 | background | -0.439855 |  |  | Overdose warning |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 532 | subject_text | +0.418084 | 531000003 |  |  |
| 542 | background | +0.392319 |  |  |  |
| 533 | subject_text | +0.349958 | 531000003 | FREE AREA | Contains No Aspirin |
| 541 | background | +0.315037 |  | FREE AREA | Contains No Aspirin |
| 722 | background | +0.314135 |  |  | Contains No Aspirin |
| 127 | background | +0.255434 |  |  |  |
| 314 | subject_nontext | +0.251430 |  |  |  |
| 712 | background | +0.231837 |  | 300000186 | ACETAMINOPHEN 500mg |

## 成功对照

`photo_375b67daeeeeff5f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.059276 | +0.111672 |
| 仅主体分差 | -0.075006 | +0.133162 |
| 主体 token 数 | 72 | 66 |

[对照 token 数据](../photo_375b67daeeeeff5f/tokens.csv) · [对照摘要](../photo_375b67daeeeeff5f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
