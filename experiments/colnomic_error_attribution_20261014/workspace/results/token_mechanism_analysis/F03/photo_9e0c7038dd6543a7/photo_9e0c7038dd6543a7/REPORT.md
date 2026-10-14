# photo_9e0c7038dd6543a7

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001321`，F03 错误参考 `reference_001310`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000867`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 13 |
| 图像 Top-1 参考 | reference_000867 |
| BM25 真值排名 | 7 |
| 融合真值排名 | 9 |
| 图像正负分差 | -0.004728 |
| BM25 正负分差 | -11.937332 |
| 0.5 × 图像 z 分差 | -0.087023 |
| 0.5 × BM25 z 分差 | -0.771253 |
| 最终融合分差 | -0.858276 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 95 个。正向贡献 +0.028427，负向贡献 -0.033155，相加等于图像分差 -0.004728。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 625 | -0.004511 | -0.005276 |
| nonvisual | 11 | -0.000044 | -0.002938 |
| subject_nontext | 66 | -0.001029 | -0.011397 |
| subject_text | 29 | +0.000856 | +0.021585 |

仅保留主体 query token 的两候选分差：-0.001329。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.004511。主体框内 token 占有效序列 13.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 82 | 29.8% | True | +6.979804 | 73.4% |
| 错误 | 35 | 1.2% | False | 不可评估 | 不可评估 |

主体中 29 个 token 与 OCR 相交，14 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 508 | subject_text | -0.516991 | Bayer |  | www.StopMedicineAbuse.org |
| 675 | background | -0.343514 |  |  |  |
| 683 | background | -0.338084 |  |  |  |
| 602 | background | -0.327709 |  |  |  |
| 180 | background | -0.323159 |  | Drug Facts |  |
| 668 | background | -0.310189 |  |  |  |
| 184 | background | -0.307568 |  | PARENTS: |  |
| 511 | subject_nontext | -0.306784 |  |  | www.StopMedicineAbuse.org |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | subject_text | +0.504990 | ACTUAL SIZE |  | POWERFASTFIZZ™ |
| 349 | subject_text | +0.481546 | ACTUAL SIZE |  |  |
| 210 | background | +0.444647 |  |  |  |
| 389 | subject_text | +0.416109 | CONTAIN ASPIRIN |  | Do not use if pouch |
| 533 | background | +0.413677 |  |  | Seltzer |
| 419 | background | +0.408893 |  | Do not use if blister | Seltzer |
| 717 | background | +0.401351 |  | Do not use if blister |  |
| 31 | background | +0.401239 |  |  |  |

## 成功对照

`photo_1ba61c7d4c7edb4b` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004728 | +0.101586 |
| 仅主体分差 | -0.001329 | +0.120744 |
| 主体 token 数 | 95 | 156 |

[对照 token 数据](../photo_1ba61c7d4c7edb4b/tokens.csv) · [对照摘要](../photo_1ba61c7d4c7edb4b/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
