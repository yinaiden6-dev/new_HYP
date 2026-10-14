# photo_fde389ec0d74a099

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000976`，F03 错误参考 `reference_003816`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_002732`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2181 |
| 图像 Top-1 参考 | reference_002732 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.035787 |
| BM25 正负分差 | +3.471163 |
| 0.5 × 图像 z 分差 | -1.005880 |
| 0.5 × BM25 z 分差 | +0.875133 |
| 最终融合分差 | -0.130748 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.4652**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 120 个。正向贡献 +0.023248，负向贡献 -0.059036，相加等于图像分差 -0.035787。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 600 | -0.031127 | -0.037924 |
| nonvisual | 11 | -0.000033 | -0.002199 |
| subject_nontext | 99 | +0.000016 | +0.000115 |
| subject_text | 21 | -0.004642 | -0.161597 |

仅保留主体 query token 的两候选分差：-0.028185。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.031127。主体框内 token 占有效序列 16.4%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 70 | 8.3% | False | 不可评估 | 不可评估 |
| 错误 | 13 | 11.4% | False | 不可评估 | 不可评估 |

主体中 21 个 token 与 OCR 相交，8 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 355 | subject_text | -0.543842 | Thia product Is not manufacturad or diatributed by McNell Con aumar Haalthcare, Divtslon of |  |  |
| 296 | background | -0.520402 |  |  |  |
| 236 | background | -0.512555 |  |  |  |
| 357 | subject_text | -0.454614 | Thia product Is not manufacturad or diatributed by McNell Con aumar Haalthcare, Divtslon of | Qtons 7 cal 1-877-770-3183: Weakdays 0:00 A t0o 4:30 PM EST. |  |
| 358 | subject_text | -0.449704 | Thia product Is not manufacturad or diatributed by McNell Con aumar Haalthcare, Divtslon of |  |  |
| 338 | background | -0.425412 |  |  |  |
| 555 | background | -0.416351 |  | * Thia product la not manufacturad or datrlbuted by Mchell Conaumar Healthcare, Divtalon of |  |
| 356 | subject_text | -0.412069 | Thia product Is not manufacturad or diatributed by McNell Con aumar Haalthcare, Divtslon of |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 389 | subject_nontext | +0.460120 |  |  |  |
| 398 | subject_nontext | +0.339127 |  |  |  |
| 419 | subject_nontext | +0.327968 |  |  |  |
| 411 | subject_nontext | +0.321257 |  |  |  |
| 376 | subject_nontext | +0.282220 |  |  | P1054647 |
| 378 | subject_nontext | +0.269540 |  |  | Poiso n Con trol Cen ter righit away. [1-8 00-2 22-1 222] |
| 396 | subject_nontext | +0.269513 |  | Qtons 7 cal 1-877-770-3183: Weakdays 0:00 A t0o 4:30 PM EST. | Poiso n Con trol Cen ter righit away. [1-8 00-2 22-1 222] |
| 691 | background | +0.264210 |  |  | Poiso n Con trol Cen ter righit away. [1-8 00-2 22-1 222] |

## 成功对照

`photo_4dfba78901c38dc4` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.035787 | +0.091082 |
| 仅主体分差 | -0.028185 | +0.153987 |
| 主体 token 数 | 120 | 108 |

[对照 token 数据](../photo_4dfba78901c38dc4/tokens.csv) · [对照摘要](../photo_4dfba78901c38dc4/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
