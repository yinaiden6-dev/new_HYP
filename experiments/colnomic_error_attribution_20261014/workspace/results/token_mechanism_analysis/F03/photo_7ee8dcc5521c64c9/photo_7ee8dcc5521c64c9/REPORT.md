# photo_7ee8dcc5521c64c9

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001182`，F03 错误参考 `reference_003249`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 166 |
| 图像 Top-1 参考 | reference_003249 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 9 |
| 图像正负分差 | -0.085926 |
| BM25 正负分差 | +30.457260 |
| 0.5 × 图像 z 分差 | -0.786756 |
| 0.5 × BM25 z 分差 | +0.567605 |
| 最终融合分差 | -0.219151 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4191**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 220 个。正向贡献 +0.014356，负向贡献 -0.100283，相加等于图像分差 -0.085926。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 500 | -0.058985 | -0.086237 |
| nonvisual | 11 | -0.000253 | -0.016821 |
| subject_nontext | 67 | -0.005615 | -0.061261 |
| subject_text | 153 | -0.021073 | -0.100683 |

仅保留主体 query token 的两候选分差：-0.088677。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.058985。主体框内 token 占有效序列 30.1%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 188 | 73.2% | True | +8.037643 | 71.3% |
| 错误 | 7 | 8.8% | False | 不可评估 | 不可评估 |

主体中 153 个 token 与 OCR 相交，124 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 322 | background | -0.542474 |  | Keep out of the reach of children. |  |
| 369 | subject_text | -0.537095 | Liver marminge This product contains |  |  |
| 657 | background | -0.526965 |  |  | If a skin reaction occurs, stop use and seek medical |
| 477 | subject_text | -0.518851 | U |  | If a skin reaction occurs, stop use and seek medical |
| 399 | background | -0.515159 |  | Pain Reliever/Nighttime Sleep Aid |  |
| 409 | subject_text | -0.505224 | nours | more than 4,000 mg of acetaminophen In 24 |  |
| 459 | background | -0.505000 |  | skin reactlons. Symptoms may Include: | If a skin reaction occurs, stop use and seek medical |
| 326 | background | -0.501522 |  | Keep out of the reach of children. |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 552 | subject_text | +0.301049 | In chlidren under 12 years of age | if you have ever had an allergic reaction to this |  |
| 56 | background | +0.243961 |  | if you have ever had an allergic reaction to this | NIGHTTIME SLEEP-AID |
| 277 | subject_text | +0.226077 | Purpose |  | Pain reliever |
| 129 | background | +0.209826 |  | skin reactlons. Symptoms may Include: | if you are allergic to acetaminophen or any of the |
| 438 | subject_nontext | +0.182590 |  | skin reactlons. Symptoms may Include: | if you are allergic to acetaminophen or any of the |
| 132 | background | +0.182415 |  | product or any of Its Ingredlents | inactive ingredients in this product |
| 9 | background | +0.173452 |  |  |  |
| 43 | background | +0.169592 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Acatamlinophen, Usp 500 mg | 6 | -0.000325 |
| Diphenhydramina HC/ 25 mg.. Miphitime sleap-ald | 17 | -0.001439 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_441c254874dcdd8d` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.085926 | +0.070027 |
| 仅主体分差 | -0.088677 | +0.089461 |
| 主体 token 数 | 220 | 190 |

[对照 token 数据](../photo_441c254874dcdd8d/tokens.csv) · [对照摘要](../photo_441c254874dcdd8d/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
