# photo_59d5e8c638df29f0

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001401`，F03 错误参考 `reference_001308`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003429`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 114 |
| 图像 Top-1 参考 | reference_003429 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 23 |
| 图像正负分差 | -0.067503 |
| BM25 正负分差 | -18.888092 |
| 0.5 × 图像 z 分差 | -0.648218 |
| 0.5 × BM25 z 分差 | -0.118425 |
| 最终融合分差 | -0.766642 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 221 个。正向贡献 +0.019564，负向贡献 -0.087067，相加等于图像分差 -0.067503。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 499 | -0.043432 | -0.063624 |
| nonvisual | 11 | -0.000199 | -0.013229 |
| subject_nontext | 81 | -0.006661 | -0.060117 |
| subject_text | 140 | -0.017211 | -0.089866 |

仅保留主体 query token 的两候选分差：-0.078963。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.043432。主体框内 token 占有效序列 30.2%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1159 | 77.3% | True | +8.526282 | 71.8% |
| 错误 | 18 | 24.0% | True | +8.495927 | 88.6% |

主体中 140 个 token 与 OCR 相交，98 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 156 | background | -0.560298 |  |  |  |
| 130 | background | -0.551797 |  |  |  |
| 303 | background | -0.506789 |  |  | in behavior with nausea and vomiting occur |
| 176 | background | -0.452721 |  |  |  |
| 660 | background | -0.451021 |  | you experience any of the following signs of stomach bleeding |  |
| 294 | subject_text | -0.449200 | Reye's syndrome: Chidren and fenagers who have or are recoverng rum chicken pox or fu-lis symptoms should not | if you are allergic to aspirin or any other pain reliever/fever reducer | Reye's syndrome: Children and teenagers |
| 553 | background | -0.441969 |  | Seltzer | If pregnant or breast-feeding, ask a health professional |
| 314 | subject_text | -0.418142 | Allergy alert: Asprin may cause a severe aliergc meacton wtich may include: |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 500 | subject_nontext | +0.306166 |  | problems in the unborn child or complications during delivery. |  |
| 395 | subject_text | +0.280552 | if you have ever had an alergc reaction to this product or any of ts angradents | if you have ever had an allergic reaction to this product or any of its ingredients |  |
| 574 | background | +0.273886 |  | if you have ever had an allergic reaction to this product or any of its ingredients |  |
| 510 | subject_text | +0.271541 | during the last 3 months of pregnancy uniess definitely directed to do so by a doctor bocause it may cause | problems in the unborn child or complications during delivery. |  |
| 412 | subject_text | +0.261730 | you have high blood pressure, heart discase, Iver ornhosis, or kidney disease |  you have a history of stomach problems, such as heartburn | a diureticyou have asthmayou have a |
| 504 | background | +0.247429 |  | problems in the unborn child or complications during delivery. |  |
| 45 | background | +0.240890 |  |  |  |
| 92 | background | +0.232242 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Anthydrous citric acid 1000 mg | 4 | -0.000898 |
| Aspirin 325 mg (NSAID) | 5 | -0.000191 |
| Sodium bicartborate (heat-treated) 1916 mg | 5 | +0.000252 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_4f2addadc5937695` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.067503 | +0.048921 |
| 仅主体分差 | -0.078963 | +0.079648 |
| 主体 token 数 | 221 | 221 |

[对照 token 数据](../photo_4f2addadc5937695/tokens.csv) · [对照摘要](../photo_4f2addadc5937695/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
