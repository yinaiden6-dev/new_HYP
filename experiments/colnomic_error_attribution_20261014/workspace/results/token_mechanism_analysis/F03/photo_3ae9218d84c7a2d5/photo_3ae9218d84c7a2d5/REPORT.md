# photo_3ae9218d84c7a2d5

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000147`，F03 错误参考 `reference_000146`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000146 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.006099 |
| BM25 正负分差 | +1.440506 |
| 0.5 × 图像 z 分差 | -0.099102 |
| 0.5 × BM25 z 分差 | +0.047845 |
| 最终融合分差 | -0.051256 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.3256**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 210 个。正向贡献 +0.020245，负向贡献 -0.026344，相加等于图像分差 -0.006099。缓存复现误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 510 | -0.007033 | -0.010081 |
| nonvisual | 11 | -0.000076 | -0.005048 |
| subject_nontext | 85 | -0.000796 | -0.006844 |
| subject_text | 125 | +0.001806 | +0.010559 |

仅保留主体 query token 的两候选分差：+0.003515。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.007033。主体框内 token 占有效序列 28.7%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 73 | 53.4% | True | +12.567017 | 75.1% |
| 错误 | 41 | 45.1% | True | +12.337908 | 75.6% |

主体中 125 个 token 与 OCR 相交，34 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 179 | background | -0.345173 |  | Product of China. |  |
| 287 | background | -0.261978 |  | more than 11.5 mL per injection site. |  |
| 525 | background | -0.240353 |  | Elanco |  |
| 114 | background | -0.228247 |  | Increxxa25 |  |
| 221 | background | -0.209026 |  |  |  |
| 14 | background | -0.200741 |  | Increxxa 25 |  |
| 271 | subject_text | -0.182867 | (tulathromycin injection) | 25 mg of tulathromycin/mL | (tulathromycin injection) |
| 290 | subject_nontext | -0.181924 |  |  | Injectable Solution |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 557 | subject_text | +0.281704 | Approved by FDA under ANADA # 200-665 | Approved by FDA under ANADA # 200-665 | Approved by FDA under ANADA # 200-665 |
| 533 | subject_text | +0.279350 | Net Contents: 250 mL | Net Contents: 250 mL | 25 mg of tulathromycin/mL |
| 714 | background | +0.266340 |  | Net Contents: 250 mL |  |
| 708 | background | +0.247826 |  |  |  |
| 410 | subject_text | +0.204744 | ruminating cattle. | ruminating cattle. |  |
| 475 | subject_nontext | +0.199805 |  |  |  |
| 38 | background | +0.192308 |  |  |  |
| 411 | subject_text | +0.184419 | ruminating cattle. | treatment with Increxxa 25 Injectable Solution. |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 25 mg of tulathromycin/mL | 9 | +0.000036 |
| Net Contents: 250 mL | 6 | +0.000669 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_72ad5f2e7435bc37` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.006099 | +0.015164 |
| 仅主体分差 | +0.003515 | +0.034226 |
| 主体 token 数 | 210 | 210 |

[对照 token 数据](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
