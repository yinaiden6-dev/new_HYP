# photo_a2020c21aa08ba7b

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001321`，F03 错误参考 `reference_001310`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_001326`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 9 |
| 图像 Top-1 参考 | reference_001326 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 3 |
| 图像正负分差 | -0.021175 |
| BM25 正负分差 | +2.739380 |
| 0.5 × 图像 z 分差 | -0.529988 |
| 0.5 × BM25 z 分差 | +0.122279 |
| 最终融合分差 | -0.407710 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.1875**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 108 个。正向贡献 +0.027232，负向贡献 -0.048407，相加等于图像分差 -0.021175。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 612 | -0.016298 | -0.019467 |
| nonvisual | 11 | +0.000283 | +0.018821 |
| subject_nontext | 59 | -0.005002 | -0.061972 |
| subject_text | 49 | -0.000159 | -0.002365 |

仅保留主体 query token 的两候选分差：-0.034928。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.016298。主体框内 token 占有效序列 14.8%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 86 | 14.6% | False | 不可评估 | 不可评估 |
| 错误 | 15 | 0.6% | False | 不可评估 | 不可评估 |

主体中 49 个 token 与 OCR 相交，26 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 155 | background | -0.528397 |  | Slze |  |
| 95 | background | -0.526108 |  |  |  |
| 257 | background | -0.483308 |  | Dist. by: Bayer HealthCare LLC, Whippany, NJ 07981 | Dist. by: Bayer HealthCare LLC, Whippany, NJ 07981 |
| 294 | subject_text | -0.461285 | Pat.: patents.livewell.bayer.com | Bayer, the Bayer Cross, Alka-Seltzer Plus, and PowerMax are registered trademarks of Bayer. | Dist. by: Bayer HealthCare LLC, Whippany, NJ 07981 |
| 72 | background | -0.443750 |  |  |  |
| 35 | background | -0.433146 |  |  |  |
| 154 | background | -0.431583 |  |  | Dist. by: Bayer HealthCare LLC, Whippany, NJ 07981 |
| 177 | background | -0.421147 |  | Dist. by: Bayer HealthCare LLC, Whippany, NJ 07981 | Dist. by: Bayer HealthCare LLC, Whippany, NJ 07981 |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 168 | background | +0.453910 |  |  |  |
| 359 | subject_text | +0.429964 | PAPER | PAPER | PARENTS: |
| 307 | subject_nontext | +0.405617 |  | LOT/EXP |  |
| 232 | background | +0.385603 |  | PAPER | PARENTS: |
| 120 | background | +0.349324 |  | PAPER |  |
| 233 | background | +0.342502 |  | PAPER |  |
| 288 | subject_text | +0.333797 | LOT/EXP | LOT/EXP |  |
| 400 | background | +0.330808 |  | TRAY |  |

## 成功对照

`photo_47430c79b3238e4a` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.021175 | +0.098700 |
| 仅主体分差 | -0.034928 | +0.120145 |
| 主体 token 数 | 108 | 108 |

[对照 token 数据](../photo_47430c79b3238e4a/tokens.csv) · [对照摘要](../photo_47430c79b3238e4a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
