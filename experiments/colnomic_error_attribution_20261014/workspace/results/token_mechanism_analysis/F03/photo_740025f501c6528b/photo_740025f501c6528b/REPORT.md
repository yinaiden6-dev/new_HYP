# photo_740025f501c6528b

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001023`，F03 错误参考 `reference_001338`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_004316`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 160 |
| 图像 Top-1 参考 | reference_004316 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 21 |
| 图像正负分差 | -0.064999 |
| BM25 正负分差 | +15.765472 |
| 0.5 × 图像 z 分差 | -0.626406 |
| 0.5 × BM25 z 分差 | +0.202043 |
| 最终融合分差 | -0.424364 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.2439**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 187 个。正向贡献 +0.022811，负向贡献 -0.087809，相加等于图像分差 -0.064999。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 533 | -0.049018 | -0.067227 |
| nonvisual | 11 | +0.000018 | +0.001224 |
| subject_nontext | 81 | -0.005106 | -0.046076 |
| subject_text | 106 | -0.010894 | -0.075127 |

仅保留主体 query token 的两候选分差：-0.062543。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.049018。主体框内 token 占有效序列 25.6%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 695 | 59.0% | True | +8.882153 | 83.6% |
| 错误 | 31 | 19.5% | True | +9.762745 | 70.9% |

主体中 106 个 token 与 OCR 相交，87 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 110 | background | -0.526394 |  |  | Severe liver damage may occur if you take |
| 111 | background | -0.511961 |  | Cold | Severe liver damage may occur if you take |
| 105 | background | -0.501307 |  |  | Severe liver damage may occur if you take |
| 657 | background | -0.485869 |  | If a skin or general allergic reaction occurs, stop use and seek |  |
| 656 | background | -0.483306 |  | If a skin or general allergic reaction occurs, stop use and seek |  |
| 420 | background | -0.466872 |  |  | Severe liver damage may occur if you take |
| 530 | subject_nontext | -0.461277 |  | If a skin or general allergic reaction occurs, stop use and seek |  |
| 73 | background | -0.441998 |  | &Flu |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 291 | subject_text | +0.397087 | (in each tablet) | Active ingredients | Active ingredients |
| 281 | background | +0.373428 |  | Active ingredients | 24 LIQUID GELS (UIUID FLD LAPSULES) |
| 38 | background | +0.344674 |  | &Flu | •Sinus Pressure |
| 285 | background | +0.329542 |  |  |  |
| 292 | subject_text | +0.286190 | (in each tablet) |  |  |
| 250 | subject_nontext | +0.249394 |  |  | PLUS |
| 37 | background | +0.236766 |  | Cold | temporarily relieves these symptoms due to a cold or |
| 418 | subject_text | +0.236002 | Liver waming: This product contains acetaminophen. Severe | Liver waming: This product contains acetaminophen. Severe |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acetaminophen 250 mg | 6 | +0.001119 |
| Dextromethorphan hydrobromide 10 mg.Cough suppressant | 12 | -0.001232 |
| Chlorphentamina maleato 2 mg | 5 | -0.000165 |
| Phenyfephrine hydrochloride 5 mg. Nasal decongestant | 13 | -0.001367 |
| more than 4,000 mg of acataminophen in 24 hours | 9 | -0.002236 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
