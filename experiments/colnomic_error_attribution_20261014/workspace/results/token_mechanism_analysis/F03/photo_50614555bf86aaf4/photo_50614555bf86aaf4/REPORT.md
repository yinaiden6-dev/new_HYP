# photo_50614555bf86aaf4

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_003087`，F03 错误参考 `reference_001096`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003265`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 64 |
| 图像 Top-1 参考 | reference_003265 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 12 |
| 图像正负分差 | -0.052627 |
| BM25 正负分差 | +27.058472 |
| 0.5 × 图像 z 分差 | -0.519986 |
| 0.5 × BM25 z 分差 | +0.135710 |
| 最终融合分差 | -0.384277 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.2070**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 224 个。正向贡献 +0.024737，负向贡献 -0.077364，相加等于图像分差 -0.052627。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 496 | -0.030372 | -0.044761 |
| nonvisual | 11 | +0.000117 | +0.007796 |
| subject_nontext | 54 | +0.000117 | +0.001580 |
| subject_text | 170 | -0.022489 | -0.096704 |

仅保留主体 query token 的两候选分差：-0.073011。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.030372。主体框内 token 占有效序列 30.6%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 321 | 66.1% | True | +9.925350 | 73.8% |
| 错误 | 20 | 14.0% | False | 不可评估 | 不可评估 |

主体中 170 个 token 与 OCR 相交，131 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 414 | subject_text | -0.462577 | more than 2 days, is accompanied or followed by fever. | trouble urinating due to an enlarged prostate gland |  |
| 270 | subject_text | -0.460295 | temporarily reduces fever | sore throatheadache | temporarily restores freer breathing through the nose |
| 601 | background | -0.457752 |  | skin reddeningblistersrash |  |
| 269 | subject_nontext | -0.412045 |  | PENTREXCILINA | temporarily restores freer breathing through the nose |
| 573 | subject_text | -0.406838 | smoking, asthma chronic bronchitis, or emphysema |  |  |
| 571 | subject_text | -0.400367 | smoking, asthma chronic bronchitis, or emphysema | diabetes |  |
| 219 | background | -0.400261 |  |  | temporarily restores freer breathing through the nose |
| 394 | subject_text | -0.396113 | If a skin reaction occurs, stop use and seek medical |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 171 | subject_text | +0.528385 | Active ingredients | Drug Facts | 3 or more alcoholic drinks every day while using this |
| 170 | subject_text | +0.450856 | (in each 20 mL) | Drug Facts |  |
| 679 | background | +0.358434 |  | Drug Facts |  |
| 622 | background | +0.324332 |  | If pregnant or breast-feeding, ask a health professional |  |
| 123 | background | +0.302380 |  |  | If a skin reaction occurs, stop use and seek medical help |
| 657 | background | +0.291608 |  | Drug Facts |  |
| 633 | subject_text | +0.269117 | do not exceed recommended dose | PENTREXCILINA |  |
| 678 | background | +0.265504 |  | DOLOR DE CABEZA |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| (in each 20 mL) | 3 | +0.001151 |
| Acetaminophen 500 mg | 6 | +0.001649 |
| Dextromethorphan HBr 26.66 mg | 5 | +0.000179 |
| Guaifenesin USP 400 mg | 7 | +0.000607 |
| Phenylephrine HCl 10 mg | 4 | +0.000221 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_1089688627444cf6` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.052627 | +0.190676 |
| 仅主体分差 | -0.073011 | +0.288235 |
| 主体 token 数 | 224 | 112 |

[对照 token 数据](../photo_1089688627444cf6/tokens.csv) · [对照摘要](../photo_1089688627444cf6/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
