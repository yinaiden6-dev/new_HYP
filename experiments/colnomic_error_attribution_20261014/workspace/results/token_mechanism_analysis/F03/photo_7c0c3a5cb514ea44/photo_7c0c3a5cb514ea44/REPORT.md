# photo_7c0c3a5cb514ea44

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_000512`，F03 错误参考 `reference_000511`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_000507`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000507 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.001054 |
| BM25 正负分差 | -1.885185 |
| 0.5 × 图像 z 分差 | +0.017753 |
| 0.5 × BM25 z 分差 | -0.138368 |
| 最终融合分差 | -0.120615 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.8863**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 114 个。正向贡献 +0.022895，负向贡献 -0.021841，相加等于图像分差 +0.001054。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 606 | -0.000589 | -0.000710 |
| nonvisual | 11 | +0.000004 | +0.000272 |
| subject_nontext | 75 | +0.002020 | +0.019685 |
| subject_text | 39 | -0.000381 | -0.007137 |

仅保留主体 query token 的两候选分差：+0.010509。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.000589。主体框内 token 占有效序列 15.6%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 201 | 65.6% | True | +2.517043 | 46.0% |
| 错误 | 110 | 60.1% | True | +2.540238 | 48.7% |

主体中 39 个 token 与 OCR 相交，19 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 387 | subject_text | -0.499976 | 7015 College Boulevard, Sulte 525 |  | 7015 College Boulevard, Suite 525 |
| 388 | subject_text | -0.366504 | 7015 College Boulevard, Sulte 525 | Overland Park, KS 66211 USA |  |
| 78 | background | -0.358364 |  | Approved by FDA under NADA # 141-291 |  |
| 386 | subject_text | -0.319423 | 7015 College Boulevard, Sulte 525 | Overland Park, KS 66211 USA | 7015 College Boulevard, Suite 525 |
| 588 | background | -0.259734 |  | Overland Park, KS 66211 USA |  |
| 586 | background | -0.185218 |  | Overland Park, KS 66211 USA | 7015 College Boulevard, Suite 525 |
| 267 | background | -0.180156 |  | 7015 College Boulevard, Suite 525 | 7015 College Boulevard, Suite 525 |
| 371 | subject_nontext | -0.168040 |  |  | 7015 College Boulevard, Suite 525 |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 395 | subject_nontext | +0.343362 |  | Overland Park, KS 66211 USA |  |
| 482 | subject_nontext | +0.339779 |  |  |  |
| 216 | background | +0.339087 |  | Overland Park, KS 66211 USA |  |
| 462 | subject_text | +0.314689 | 4 |  |  |
| 407 | subject_text | +0.306608 | (866)933-2472 | Overland Park, KS 66211 USA |  |
| 663 | background | +0.304122 |  |  |  |
| 405 | subject_text | +0.288886 | (866)933-2472 | Overland Park, KS 66211 USA |  |
| 275 | background | +0.288133 |  | Overland Park, KS 66211 USA |  |

## 成功对照

`photo_1b6a28535da238b9` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.001054 | +0.007873 |
| 仅主体分差 | +0.010509 | +0.011756 |
| 主体 token 数 | 114 | 66 |

[对照 token 数据](../photo_1b6a28535da238b9/tokens.csv) · [对照摘要](../photo_1b6a28535da238b9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
