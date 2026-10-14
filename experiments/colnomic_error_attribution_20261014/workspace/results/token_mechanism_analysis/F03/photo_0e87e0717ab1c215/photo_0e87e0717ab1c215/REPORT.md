# photo_0e87e0717ab1c215

按当前真值分析，未新增身份复核。 用户指定主题：细小文字差异。

正确参考 `reference_000147`，F03 错误参考 `reference_000146`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000146 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.005469 |
| BM25 正负分差 | -3.961304 |
| 0.5 × 图像 z 分差 | -0.079247 |
| 0.5 × BM25 z 分差 | -0.136362 |
| 最终融合分差 | -0.215609 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 60 个。正向贡献 +0.017601，负向贡献 -0.023070，相加等于图像分差 -0.005469。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 660 | -0.004871 | -0.005395 |
| nonvisual | 11 | +0.000009 | +0.000591 |
| subject_nontext | 21 | -0.000189 | -0.006583 |
| subject_text | 39 | -0.000418 | -0.007838 |

仅保留主体 query token 的两候选分差：-0.007399。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.004871。主体框内 token 占有效序列 8.2%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 40 | 34.5% | True | +12.398812 | 73.3% |
| 错误 | 27 | 29.5% | True | +12.189106 | 73.3% |

主体中 39 个 token 与 OCR 相交，26 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 501 | background | -0.240913 |  |  |  |
| 28 | background | -0.239687 |  |  |  |
| 26 | background | -0.232594 |  | 11 |  |
| 328 | background | -0.199324 |  |  |  |
| 273 | background | -0.176939 |  | treatment with Increxxa 25 Injectable Solution. |  |
| 586 | background | -0.176766 |  |  |  |
| 296 | background | -0.174591 |  |  |  |
| 82 | background | -0.145516 |  | more than 11.5 mL per injection site. |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 446 | background | +0.308695 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 453 | subject_text | +0.279405 | Net Contents: 250mL | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 462 | background | +0.276360 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 468 | background | +0.248161 |  | Approved by FDA under ANADA # 200-665 | Approved by FDA under ANADA # 200-665 |
| 650 | background | +0.229144 |  | Net Contents: 250 mL |  |
| 147 | background | +0.210019 |  | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 535 | subject_nontext | +0.201440 |  |  |  |
| 107 | background | +0.196906 |  | this drug to use by or on the order of a |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 25 mg of tulathromycin/mL | 5 | -0.000307 |
| Net Contents: 250mL | 3 | +0.000402 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_72ad5f2e7435bc37` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.005469 | +0.015164 |
| 仅主体分差 | -0.007399 | +0.034226 |
| 主体 token 数 | 60 | 210 |

[对照 token 数据](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
