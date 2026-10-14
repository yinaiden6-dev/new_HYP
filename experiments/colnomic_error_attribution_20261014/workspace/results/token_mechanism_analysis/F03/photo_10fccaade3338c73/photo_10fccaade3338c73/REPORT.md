# photo_10fccaade3338c73

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_004359`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_004359 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 3 |
| 图像正负分差 | -0.003373 |
| BM25 正负分差 | -30.817886 |
| 0.5 × 图像 z 分差 | -0.118598 |
| 0.5 × BM25 z 分差 | -1.007858 |
| 最终融合分差 | -1.126456 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 121 个。正向贡献 +0.028665，负向贡献 -0.032037，相加等于图像分差 -0.003373。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 647 | -0.005915 | -0.007121 |
| nonvisual | 11 | +0.000177 | +0.012531 |
| subject_nontext | 33 | -0.000097 | -0.002299 |
| subject_text | 88 | +0.002462 | +0.021796 |

仅保留主体 query token 的两候选分差：+0.015225。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.005915。主体框内 token 占有效序列 15.5%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 133 | 48.4% | True | +5.118594 | 61.2% |
| 错误 | 9 | 7.4% | False | 不可评估 | 不可评估 |

主体中 88 个 token 与 OCR 相交，68 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 378 | background | -0.342448 | sseplatz 1, 1020 Vienna |  |  |
| 56 | background | -0.338290 | rmacy | non-nicotine stop smoking drugtaking prescription medicine for |  |
| 19 | background | -0.335950 |  | lozenge for a longer perlod to keep from smoking, talk to your | during the first six weeks. |
| 119 | background | -0.330386 | rmacy |  | © 2022 CVS/pharmacy |
| 62 | background | -0.309714 |  | If you are pregnant or breast-feeding, only use this medicine |  |
| 237 | background | -0.304514 |  | 1. You must be motivated to quit |  |
| 318 | background | -0.301598 | COIE |  |  |
| 257 | background | -0.283913 | erview Plan |  | © 2022 CVS/pharmacy |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 533 | subject_text | +0.485806 | Satisfaction guaranteed- | Love it or your money back. |  |
| 501 | subject_text | +0.461394 | Satisfaction guaranteed- | Love it or your money back. |  |
| 262 | background | +0.408490 |  | the enclosed User's Guide. |  |
| 207 | background | +0.348653 |  | Love it or your money back. |  |
| 182 | background | +0.325347 | MetroU2 |  |  |
| 405 | subject_text | +0.323162 | Satisfaction guaranteed- | Satisfaction guaranteed – | CVS.com/returnpolicy |
| 47 | background | +0.318507 |  | Love it or your money back. |  |
| 57 | background | +0.311386 |  |  | during the first six weeks. |

## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.003373 | +0.003739 |
| 仅主体分差 | +0.015225 | +0.023522 |
| 主体 token 数 | 121 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
