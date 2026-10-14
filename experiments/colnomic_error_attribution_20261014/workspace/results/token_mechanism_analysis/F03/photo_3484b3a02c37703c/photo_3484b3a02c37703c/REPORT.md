# photo_3484b3a02c37703c

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000518`，F03 错误参考 `reference_004741`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_004064`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 71 |
| 图像 Top-1 参考 | reference_004064 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.004534 |
| BM25 正负分差 | -1.164358 |
| 0.5 × 图像 z 分差 | -0.097342 |
| 0.5 × BM25 z 分差 | -0.412614 |
| 最终融合分差 | -0.509956 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 105 个。正向贡献 +0.036299，负向贡献 -0.040833，相加等于图像分差 -0.004534。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 615 | -0.006542 | -0.007776 |
| nonvisual | 11 | -0.000137 | -0.009121 |
| subject_nontext | 81 | +0.002706 | +0.024418 |
| subject_text | 24 | -0.000561 | -0.017074 |

仅保留主体 query token 的两候选分差：+0.014934。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.006542。主体框内 token 占有效序列 14.4%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 75 | 15.0% | True | +3.526589 | 54.5% |
| 错误 | 19 | 3.6% | False | 不可评估 | 不可评估 |

主体中 24 个 token 与 OCR 相交，1 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | subject_nontext | -0.431972 |  |  |  |
| 236 | background | -0.429420 |  |  |  |
| 230 | background | -0.424310 |  | TOTAL NET WT 1 OZ (28.4 g) |  |
| 234 | background | -0.424278 |  |  |  |
| 231 | background | -0.417857 |  | ORG code is an internal Periscope code, |  |
| 225 | background | -0.398162 |  |  |  |
| 352 | subject_nontext | -0.383870 |  |  |  |
| 58 | background | -0.359711 |  | ORG code is an internal Periscope code, |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 319 | subject_nontext | +0.398879 |  |  | Walgreens |
| 320 | subject_nontext | +0.391563 |  |  | Walgreens |
| 367 | subject_nontext | +0.379169 |  |  | Walgreens |
| 388 | subject_nontext | +0.377332 |  |  | Walgreens |
| 380 | subject_nontext | +0.363169 |  |  | Other informationstore at 20-25°C (68-77°F) store in a cool dry |
| 110 | background | +0.362947 |  |  | Other informationstore at 20-25°C (68-77°F) store in a cool dry |
| 415 | subject_nontext | +0.349802 |  |  | Walgreens |
| 378 | subject_nontext | +0.335243 |  |  | Walgreens |

## 成功对照

`photo_12f7c257ae1f8f3c` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.004534 | +0.087609 |
| 仅主体分差 | +0.014934 | +0.088559 |
| 主体 token 数 | 105 | 196 |

[对照 token 数据](../photo_12f7c257ae1f8f3c/tokens.csv) · [对照摘要](../photo_12f7c257ae1f8f3c/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
