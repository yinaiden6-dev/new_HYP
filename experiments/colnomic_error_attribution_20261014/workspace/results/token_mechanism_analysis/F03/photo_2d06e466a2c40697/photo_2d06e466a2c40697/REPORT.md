# photo_2d06e466a2c40697

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000221`，F03 错误参考 `reference_000220`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000219`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 3 |
| 图像 Top-1 参考 | reference_000219 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 3 |
| 图像正负分差 | +0.002445 |
| BM25 正负分差 | -5.583618 |
| 0.5 × 图像 z 分差 | +0.041090 |
| 0.5 × BM25 z 分差 | -0.050936 |
| 最终融合分差 | -0.009846 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5535**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 128 个。正向贡献 +0.025468，负向贡献 -0.023023，相加等于图像分差 +0.002445。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 592 | +0.001409 | +0.001740 |
| nonvisual | 11 | -0.000014 | -0.000940 |
| subject_nontext | 42 | -0.000425 | -0.007389 |
| subject_text | 86 | +0.001475 | +0.012535 |

仅保留主体 query token 的两候选分差：+0.005997。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_nontext`，贡献 -0.000425。主体框内 token 占有效序列 17.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1193 | 60.1% | True | +7.022996 | 64.8% |
| 错误 | 574 | 58.5% | True | +6.759309 | 64.6% |

主体中 86 个 token 与 OCR 相交，72 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 355 | subject_text | -0.398428 | Fluralaner is a member of the isoxazoline class. This |  | Fluralaner is a member of the isoxazoline class. This |
| 392 | subject_text | -0.397347 | americanum ticks beyond 8 weeks after dosing. |  | Human Warnings: Not for human use. Keep this and all |
| 398 | background | -0.385604 |  |  |  |
| 439 | background | -0.371098 |  | moisture. The pouch should only be opened immediately |  |
| 609 | background | -0.368161 |  |  |  |
| 722 | background | -0.366983 |  |  |  |
| 438 | background | -0.362261 |  |  |  |
| 393 | subject_text | -0.302781 | americanum ticks beyond 8 weeks after dosing. | Human Warnings: Not for human use. Keep this and all |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 45 | background | +0.369729 |  | class has been associated with neurologic adverse | Rev. 04/19 |
| 332 | subject_text | +0.361292 | americanum ticks. Treatment with Bravecto may begin |  |  |
| 12 | background | +0.343281 |  |  |  |
| 351 | subject_text | +0.309823 | class has been associated with neurologic adverse | class has been associated with neurologic adverse |  |
| 352 | subject_text | +0.298143 | Fluralaner is a member of the isoxazoline class. This | class has been associated with neurologic adverse | of seizures. Use with caution in dogs with a history of |
| 272 | subject_text | +0.274854 | Dermacentor vaniabilis (American dog tick) |  | of Amblyomma americanum (lone star tick) infestations |
| 711 | background | +0.235965 |  |  | administered topically as a single dose every 12 weeks |
| 518 | background | +0.220320 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| to provide a minimum dose of 11.4 mg/ib (25 mg/kg) | 7 | +0.000591 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_613ba5f27d0c5cfe` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.002445 | +0.018619 |
| 仅主体分差 | +0.005997 | +0.043043 |
| 主体 token 数 | 128 | 112 |

[对照 token 数据](../photo_613ba5f27d0c5cfe/tokens.csv) · [对照摘要](../photo_613ba5f27d0c5cfe/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
