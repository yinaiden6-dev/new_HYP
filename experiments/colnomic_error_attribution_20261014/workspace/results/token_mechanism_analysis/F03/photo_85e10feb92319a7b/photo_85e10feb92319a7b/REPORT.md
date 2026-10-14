# photo_85e10feb92319a7b

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_000512`，F03 错误参考 `reference_000510`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000337`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 16 |
| 图像 Top-1 参考 | reference_000337 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.021370 |
| BM25 正负分差 | -40.056824 |
| 0.5 × 图像 z 分差 | -0.410419 |
| 0.5 × BM25 z 分差 | -0.923613 |
| 最终融合分差 | -1.334030 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 84 个。正向贡献 +0.012121，负向贡献 -0.033491，相加等于图像分差 -0.021370。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 636 | -0.020753 | -0.023853 |
| nonvisual | 11 | -0.000202 | -0.013430 |
| subject_nontext | 17 | -0.000348 | -0.014963 |
| subject_text | 67 | -0.000066 | -0.000724 |

仅保留主体 query token 的两候选分差：-0.003606。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.020753。主体框内 token 占有效序列 11.5%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 898 | 56.1% | True | +12.341897 | 88.1% |
| 错误 | 767 | 56.4% | True | +11.638073 | 84.5% |

主体中 67 个 token 与 OCR 相交，53 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 73 | background | -0.298589 |  |  | and early pregnancy loss. |
| 184 | background | -0.243440 |  |  |  |
| 287 | background | -0.232251 |  | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. | HUMAN WARNINGS: Keepoutofreachofctildren. Notforhuman use. |
| 348 | background | -0.229988 |  | CAUTION: Federal (USA) law restricts this drug to use by or on the | HUMAN WARNINGS: Keepoutofreachofctildren. Notforhuman use. |
| 107 | background | -0.226411 |  | HUMAN WARNINGS: Keep outofreach ofchildren. Not forhuman use. | HUMAN WARNINGS: Keepoutofreachofctildren. Notforhuman use. |
| 167 | background | -0.212983 |  | CAUTION: Federal (USA) law restricts this drug to use by or on the |  |
| 466 | background | -0.204629 |  | Do not use inpregnant dogs. Studies conducted with trilostane inlaboratory animals fave shown teratogenic efects |  |
| 573 | background | -0.204198 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 395 | subject_text | +0.154282 | caps I es if pr egnant or if ty ing to c onc eiv e Tri los tane is assoc iated with terabgeric effects and early pre grancy loss | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide tfe capsules. Do not hardle the | CONTRA NDICA TIDNS: The use is c ontraindicated in dogs fhat have demonstrated hyper sens itivity b ti los tane. |
| 488 | subject_text | +0.152421 | -11 | Overland Park, KS 66211 USA | 17033-160-305 |
| 163 | background | +0.126492 |  |  | bloodvolume ard renal per fusi on. Potas sium sparing d ur etics(e g. spi ronolactore) stould rot be us ed with VETORYL |
| 614 | background | +0.113093 |  |  |  |
| 185 | background | +0.107347 |  | VARNISH |  |
| 445 | background | +0.105846 |  | bloodvolume ard renal per fusi on. Potas sium sparing d ur etics(e g. spi ronolactore) stould rot be us ed with VETORYL | Wash handsafter use. Do ndt emptycaps ule contents ard do not attempt to divide the capsules. Do not hardle the |
| 220 | background | +0.103188 |  |  |  |
| 313 | background | +0.098387 |  | bloodvolume ard renal per fusi on. Potas sium sparing d ur etics(e g. spi ronolactore) stould rot be us ed with VETORYL | bloodvolume ard renal per fusi on. Potas sium sparing d ur etics(e g. spi ronolactore) stould rot be us ed with VETORYL |

## 成功对照

`photo_1b6a28535da238b9` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.021370 | +0.015219 |
| 仅主体分差 | -0.003606 | +0.022365 |
| 主体 token 数 | 84 | 66 |

[对照 token 数据](../photo_1b6a28535da238b9/tokens.csv) · [对照摘要](../photo_1b6a28535da238b9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
