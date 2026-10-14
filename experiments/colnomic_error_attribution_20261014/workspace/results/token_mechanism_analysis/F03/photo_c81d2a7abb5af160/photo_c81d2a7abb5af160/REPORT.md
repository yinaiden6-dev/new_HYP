# photo_c81d2a7abb5af160

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_004816`，F03 错误参考 `reference_005118`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_005119`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 5 |
| 图像 Top-1 参考 | reference_005119 |
| BM25 真值排名 | 7 |
| 融合真值排名 | 7 |
| 图像正负分差 | -0.000335 |
| BM25 正负分差 | -95.077530 |
| 0.5 × 图像 z 分差 | -0.005750 |
| 0.5 × BM25 z 分差 | -1.716685 |
| 最终融合分差 | -1.722435 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 72 个。正向贡献 +0.106350，负向贡献 -0.106685，相加等于图像分差 -0.000335。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 696 | +0.000083 | +0.000093 |
| nonvisual | 11 | +0.000010 | +0.000724 |
| subject_nontext | 17 | -0.000303 | -0.013886 |
| subject_text | 55 | -0.000125 | -0.001770 |

仅保留主体 query token 的两候选分差：-0.004631。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_nontext`，贡献 -0.000303。主体框内 token 占有效序列 9.2%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 5 | 3.7% | False | 不可评估 | 不可评估 |
| 错误 | 6 | 7.8% | False | 不可评估 | 不可评估 |

主体中 55 个 token 与 OCR 相交，21 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 184 | background | -0.687966 | Enoxaparin |  |  |
| 736 | background | -0.632423 |  |  |  |
| 463 | background | -0.627688 |  |  |  |
| 731 | background | -0.626412 |  |  |  |
| 272 | background | -0.625500 |  |  |  |
| 152 | background | -0.624638 | Enoxaparin |  |  |
| 443 | background | -0.623699 |  |  |  |
| 188 | background | -0.622534 | Enoxaparin |  | Enoxaparin |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 371 | background | +0.614712 | (levothyroxine sodium) tablets | (levothyroxine sodium) tablets | 40 mg per 0.4 mL |
| 637 | background | +0.598250 | PROVELL | PROVELL |  |
| 364 | background | +0.593302 | Euthyrox® |  | Enoxaparin |
| 402 | background | +0.593189 | 150 mcg per tablet, USP | (levothyroxine sodium) tablets | 40 mg per 0.4 mL |
| 397 | background | +0.591374 | (levothyroxine sodium) tablets |  | Sodium Injection, USP |
| 332 | background | +0.585285 | Euthyrox® |  |  |
| 534 | background | +0.584379 |  |  |  |
| 735 | background | +0.580888 |  |  | Enoxaparin |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 140 mg | 8 | +0.000454 |
| 5 Capsules | 6 | +0.000720 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_0eb729e9f7b895d9` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.000335 | +0.183787 |
| 仅主体分差 | -0.004631 | +0.255442 |
| 主体 token 数 | 72 | 242 |

[对照 token 数据](../photo_0eb729e9f7b895d9/tokens.csv) · [对照摘要](../photo_0eb729e9f7b895d9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
