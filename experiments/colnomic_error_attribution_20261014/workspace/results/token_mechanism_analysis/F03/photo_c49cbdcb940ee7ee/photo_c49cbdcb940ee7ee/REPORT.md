# photo_c49cbdcb940ee7ee

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001112`，F03 错误参考 `reference_004677`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003822`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 74 |
| 图像 Top-1 参考 | reference_003822 |
| BM25 真值排名 | 111 |
| 融合真值排名 | 49 |
| 图像正负分差 | +0.019054 |
| BM25 正负分差 | -776.528259 |
| 0.5 × 图像 z 分差 | +0.797816 |
| 0.5 × BM25 z 分差 | -7.614665 |
| 最终融合分差 | -6.816849 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.9052**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 84 个。正向贡献 +0.054311，负向贡献 -0.035257，相加等于图像分差 +0.019054。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 684 | -0.008684 | -0.009890 |
| nonvisual | 11 | +0.001487 | +0.105274 |
| subject_nontext | 17 | +0.003762 | +0.172401 |
| subject_text | 67 | +0.022490 | +0.261483 |

仅保留主体 query token 的两候选分差：+0.243454。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.008684。主体框内 token 占有效序列 10.8%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 111 | 66.6% | True | +8.306452 | 78.0% |
| 错误 | 0 | 0.0% | False | 不可评估 | 不可评估 |

主体中 67 个 token 与 OCR 相交，36 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 322 | background | -0.399679 | Austria | 321130788980 | WWW.SUSANNEKAUFMANN.COM |
| 353 | background | -0.387120 | Austria | 321130788980 | WWW.SUSANNEKAUFMANN.COM |
| 354 | background | -0.373645 | Austria | 321130788980 | WWW.SUSANNEKAUFMANN.COM |
| 571 | background | -0.242149 | G3\| Rotenturmstr. 21 | one 10 mg tablet once daily; do | or higher and other sun protection measures including |
| 602 | background | -0.230985 | baerenland.at | one 10 mg tablet once daily; do | or higher and other sun protection measures including |
| 382 | background | -0.226068 | Der Nordturm ist mit | Varnish |  |
| 191 | background | -0.213259 |  |  | Retains SPF after 40 minutes of activity in the water onsweating |
| 246 | background | -0.199292 | Tagsüber Frühstück und | 321130788980 |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 506 | subject_text | +0.632087 | 45 TABLETS | 45 TABLETS |  |
| 439 | subject_text | +0.619864 | TABLETS USP, 10 mg - Antihistamine | ALLERGY RELIEF | Susanne Kaufmann Kosmetik GmbH |
| 483 | background | +0.605146 |  | 45 TABLETS |  |
| 435 | subject_text | +0.599560 | TABLETS USP, 10 mg - Antihistamine | TABLETS USP, 10 mg - Antihistamine |  |
| 671 | background | +0.587776 |  |  | Retains SPF after 40 minutes of activity in the water onsweating |
| 768 | background | +0.585136 |  |  | SUSANNE KAUFMANN |
| 404 | subject_text | +0.575256 | ALLERGY RELIEF |  | Stop use and ask a doctor if rash occurs |
| 417 | background | +0.573071 |  | TABLETS USP, 10 mg - Antihistamine | Susanne Kaufmann Kosmetik GmbH |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| TABLETS USP, 10 mg - Antihistamine | 6 | +0.003650 |
| 45 TABLETS | 3 | +0.001844 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_2df687d8a2f89f40` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.019054 | +0.183806 |
| 仅主体分差 | +0.243454 | +0.244333 |
| 主体 token 数 | 84 | 312 |

[对照 token 数据](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
