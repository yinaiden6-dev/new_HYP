# photo_9f799473a3f5aacc

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_004768`，F03 错误参考 `reference_004777`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_004786`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_004786 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.032642 |
| BM25 正负分差 | -4.100815 |
| 0.5 × 图像 z 分差 | +0.600106 |
| 0.5 × BM25 z 分差 | -0.787901 |
| 最终融合分差 | -0.187796 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5676**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 136 个。正向贡献 +0.050398，负向贡献 -0.017757，相加等于图像分差 +0.032642。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 584 | +0.025870 | +0.032382 |
| nonvisual | 11 | -0.000013 | -0.000871 |
| subject_nontext | 108 | +0.004557 | +0.030847 |
| subject_text | 28 | +0.002227 | +0.058141 |

仅保留主体 query token 的两候选分差：+0.036467。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 -0.000013。主体框内 token 占有效序列 18.6%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 77 | 12.3% | False | 不可评估 | 不可评估 |
| 错误 | 37 | 10.4% | False | 不可评估 | 不可评估 |

主体中 28 个 token 与 OCR 相交，14 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 26 | background | -0.296887 |  | *Triton® X-100 – Registered trademark of Union Carbide, Co., USA. |  |
| 7 | background | -0.244887 |  |  |  |
| 155 | background | -0.239807 |  | *Triton® X-100 – Registered trademark of Union Carbide, Co., USA. |  |
| 15 | background | -0.234449 |  | US Govt License #1725 |  |
| 146 | background | -0.232519 |  | US Govt License #1725 |  |
| 156 | background | -0.208783 |  | *Triton® X-100 – Registered trademark of Union Carbide, Co., USA. |  |
| 16 | background | -0.207007 |  | US Govt License #1725 |  |
| 722 | background | -0.188965 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 507 | background | +0.423147 |  |  |  |
| 387 | background | +0.421955 |  |  |  |
| 327 | background | +0.412146 |  |  | Swiftwater PA 18370 USA |
| 523 | background | +0.394658 |  |  |  |
| 691 | background | +0.393443 |  |  |  |
| 694 | background | +0.390584 |  |  |  |
| 340 | background | +0.388310 |  |  | Swiftwater PA 18370 USA |
| 313 | subject_text | +0.383934 | Sanofi Pasteur Inc. | Swiftwater, PA 18370 USA | Swiftwater PA 18370 USA |

## 成功对照

`photo_9fbc291d90193703` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.032642 | +0.138748 |
| 仅主体分差 | +0.036467 | +0.183536 |
| 主体 token 数 | 136 | 128 |

[对照 token 数据](../photo_9fbc291d90193703/tokens.csv) · [对照摘要](../photo_9fbc291d90193703/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
