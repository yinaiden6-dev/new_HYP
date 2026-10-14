# photo_7f0add741b1b243c

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_004975`，F03 错误参考 `reference_000395`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_004974`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_004974 |
| BM25 真值排名 | 69 |
| 融合真值排名 | 4 |
| 图像正负分差 | +0.013277 |
| BM25 正负分差 | -36.390884 |
| 0.5 × 图像 z 分差 | +0.357424 |
| 0.5 × BM25 z 分差 | -0.451214 |
| 最终融合分差 | -0.093790 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5580**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 144 个。正向贡献 +0.100257，负向贡献 -0.086981，相加等于图像分差 +0.013277。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 624 | -0.034475 | -0.043039 |
| nonvisual | 11 | +0.001251 | +0.088619 |
| subject_nontext | 62 | +0.013667 | +0.171723 |
| subject_text | 82 | +0.032833 | +0.311913 |

仅保留主体 query token 的两候选分差：+0.251554。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.034475。主体框内 token 占有效序列 18.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 168 | 51.6% | True | +15.626101 | 88.4% |
| 错误 | 14 | 6.4% | False | 不可评估 | 不可评估 |

主体中 82 个 token 与 OCR 相交，28 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 63 | background | -0.652494 | INTERCEPTOR | 0.25% 25 mg/10 mL (2.5 mg/mL) | INTERCEPTOR |
| 319 | background | -0.637918 |  | © 2005 Northstar Healthcare Holdings Ltd. | For dogs and puppies 50.1-100 Ibs |
| 610 | background | -0.633695 | FOR DOGS |  |  |
| 408 | background | -0.619494 | Purpos | Aspiro Pharma Limited |  |
| 736 | background | -0.617931 |  | USP controlled room temperature.] |  |
| 97 | background | -0.611785 |  | NaOH or HCl. |  |
| 349 | background | -0.607868 |  | USP controlled room temperature.] |  |
| 254 | background | -0.603994 |  | USP controlled room temperature.] |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 723 | background | +0.631708 | be discar |  |  |
| 750 | background | +0.628188 |  |  |  |
| 659 | subject_nontext | +0.626619 |  |  |  |
| 753 | background | +0.594083 |  |  |  |
| 627 | subject_nontext | +0.593387 |  |  |  |
| 628 | subject_nontext | +0.581380 |  |  |  |
| 308 | subject_text | +0.579624 | For INFILTRATION, NERVE BLOCK, CAUDAL, |  | the last exposure to mosquitoes may not provide complete |
| 337 | subject_text | +0.572837 | Bupivacaine Hydrochloride |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 0.25% 25 mg/10 mL (2.5 mg/mL) | 21 | +0.009245 |
| 10 x 10 mL Single-Dose Vials | 7 | +0.002920 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_709e8a3ee42539bc` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.013277 | +0.022158 |
| 仅主体分差 | +0.251554 | +0.224435 |
| 主体 token 数 | 144 | 153 |

[对照 token 数据](../photo_709e8a3ee42539bc/tokens.csv) · [对照摘要](../photo_709e8a3ee42539bc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
