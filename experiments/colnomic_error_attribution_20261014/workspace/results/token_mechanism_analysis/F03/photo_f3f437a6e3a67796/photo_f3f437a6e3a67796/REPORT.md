# photo_f3f437a6e3a67796

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000934`，F03 错误参考 `reference_001965`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_004350`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 7 |
| 图像 Top-1 参考 | reference_004350 |
| BM25 真值排名 | 7 |
| 融合真值排名 | 6 |
| 图像正负分差 | -0.012294 |
| BM25 正负分差 | -4.773014 |
| 0.5 × 图像 z 分差 | -0.142527 |
| 0.5 × BM25 z 分差 | -0.409496 |
| 最终融合分差 | -0.552023 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 45 个。正向贡献 +0.021283，负向贡献 -0.033577，相加等于图像分差 -0.012294。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 675 | -0.012402 | -0.013431 |
| nonvisual | 11 | +0.000103 | +0.006822 |
| subject_nontext | 26 | +0.000079 | +0.002226 |
| subject_text | 19 | -0.000074 | -0.002829 |

仅保留主体 query token 的两候选分差：+0.000092。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.012402。主体框内 token 占有效序列 6.2%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 49 | 10.6% | False | 不可评估 | 不可评估 |
| 错误 | 6 | 11.0% | False | 不可评估 | 不可评估 |

主体中 19 个 token 与 OCR 相交，11 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 159 | background | -0.348288 |  |  | Extended-release tablets USP, 650 mg |
| 421 | background | -0.333453 |  |  |  |
| 392 | subject_text | -0.323462 | Tablets USP, 650 mg |  | Extended-release tablets USP, 650 mg |
| 432 | subject_nontext | -0.316348 |  |  |  |
| 526 | background | -0.301364 |  |  | Extended-release tablets USP, 650 mg |
| 102 | background | -0.294526 |  | V6001648 TEMH200000002710 | Extended-release tablets USP, 650 mg |
| 292 | background | -0.290291 |  |  | For The Temporary Relief Of Minor Arthritis Pain |
| 404 | background | -0.281686 |  | Actiro Ingrvedet of Tyle | COMPARE TO THE ACTIVE INGREDOENT |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 22 | background | +0.251517 |  |  |  |
| 542 | background | +0.206700 |  |  | 8 |
| 369 | background | +0.204062 |  |  | 8 |
| 556 | background | +0.173112 |  |  |  |
| 31 | background | +0.172861 |  |  | 8 |
| 377 | background | +0.170883 |  |  |  |
| 489 | background | +0.166085 |  |  |  |
| 533 | background | +0.153094 |  | 8 Hour | 8 |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Tablets USP, 650 mg | 3 | -0.000526 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_4917ad5f3fac9f55` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.012294 | +0.053036 |
| 仅主体分差 | +0.000092 | +0.069648 |
| 主体 token 数 | 45 | 50 |

[对照 token 数据](../photo_4917ad5f3fac9f55/tokens.csv) · [对照摘要](../photo_4917ad5f3fac9f55/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
