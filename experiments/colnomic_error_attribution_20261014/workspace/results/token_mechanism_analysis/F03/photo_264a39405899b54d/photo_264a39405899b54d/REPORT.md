# photo_264a39405899b54d

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000221`，F03 错误参考 `reference_000220`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 4 |
| 图像 Top-1 参考 | reference_000220 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.002339 |
| BM25 正负分差 | -1.906372 |
| 0.5 × 图像 z 分差 | -0.040134 |
| 0.5 × BM25 z 分差 | -0.017506 |
| 最终融合分差 | -0.057640 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 336 个。正向贡献 +0.025912，负向贡献 -0.028252，相加等于图像分差 -0.002339。缓存复现误差 1.8e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 384 | -0.001493 | -0.002842 |
| nonvisual | 11 | +0.000036 | +0.002420 |
| subject_nontext | 79 | -0.000514 | -0.004753 |
| subject_text | 257 | -0.000369 | -0.001050 |

仅保留主体 query token 的两候选分差：-0.001920。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.001493。主体框内 token 占有效序列 46.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 365 | 71.9% | True | +4.867102 | 56.9% |
| 错误 | 210 | 70.4% | True | +4.775294 | 56.2% |

主体中 257 个 token 与 OCR 相交，225 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 315 | subject_text | -0.529831 | Fluralaner is a member of the isoxazoline class. This |  | Fluralaner is a member of the isoxazoline class. This |
| 311 | subject_text | -0.474120 | Fluralaner is a member of the isoxazoline class. This |  | Fluralaner is a member of the isoxazoline class. This |
| 570 | subject_nontext | -0.401893 |  |  |  |
| 401 | background | -0.389032 |  |  | Human Warnings: Not for human use. Keep this and all |
| 466 | background | -0.383006 |  | Human Warnings: Not for human use. Keep this and all |  |
| 400 | background | -0.379488 |  | BRAVECTO | Human Warnings: Not for human use. Keep this and all |
| 406 | background | -0.370194 |  |  |  |
| 404 | background | -0.363075 |  | BRAVECTO | Human Warnings: Not for human use. Keep this and all |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 240 | background | +0.505201 |  |  | to provide a minimum dose of 11.4 mg/lb (25 mg/kg) |
| 241 | background | +0.502293 |  |  | to provide a minimum dose of 11.4 mg/lb (25 mg/kg) |
| 322 | background | +0.501862 |  | class has been associated with neurologic adverse | of seizures. Use with caution in dogs with a history of |
| 262 | background | +0.495181 |  | to provide a minimum dose of 11.4 mg/lb (25 mg/kg) |  |
| 332 | subject_text | +0.493191 | reactions including tremors, ataxia, and seizures. | class has been associated with neurologic adverse |  |
| 353 | subject_text | +0.381320 | of seizures. Use with caution in dogs with a history of | of seizures. Use with caution in dogs with a history of |  |
| 261 | background | +0.358188 |  | to provide a minimum dose of 11.4 mg/lb (25 mg/kg) |  |
| 410 | subject_text | +0.334434 | Human Warnings: Not for human use. Keep this and all | Human Warnings: Not for human use. Keep this and all |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| to provide a minimum dose of 11.4 mg/lb (25 mg/kg) | 19 | +0.000624 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_3bf9d27254c310cf` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002339 | +0.020583 |
| 仅主体分差 | -0.001920 | +0.043400 |
| 主体 token 数 | 336 | 308 |

[对照 token 数据](../photo_3bf9d27254c310cf/tokens.csv) · [对照摘要](../photo_3bf9d27254c310cf/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
