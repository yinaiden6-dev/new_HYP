# photo_39034fd7db4f883a

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001112`，F03 错误参考 `reference_004677`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_001130`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 4399 |
| 图像 Top-1 参考 | reference_001130 |
| BM25 真值排名 | 58 |
| 融合真值排名 | 360 |
| 图像正负分差 | -0.023352 |
| BM25 正负分差 | -661.428833 |
| 0.5 × 图像 z 分差 | -1.015910 |
| 0.5 × BM25 z 分差 | -6.121917 |
| 最终融合分差 | -7.137826 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 91 个。正向贡献 +0.013020，负向贡献 -0.036372，相加等于图像分差 -0.023352。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 677 | -0.023723 | -0.027297 |
| nonvisual | 11 | -0.000091 | -0.006419 |
| subject_nontext | 11 | +0.000029 | +0.002054 |
| subject_text | 80 | +0.000432 | +0.004211 |

仅保留主体 query token 的两候选分差：+0.003950。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.023723。主体框内 token 占有效序列 11.7%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 644 | 64.9% | True | +11.140132 | 88.6% |
| 错误 | 9 | 30.0% | True | +25.818626 | 100.0% |

主体中 80 个 token 与 OCR 相交，64 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 322 | background | -0.412522 | Austria | 321130788980 | WWW.SUSANNEKAUFMANN.COM |
| 339 | background | -0.319543 | ElGauchoimDesigntower |  | WWW.SUSANNEKAUFMANN.COM |
| 323 | background | -0.274636 | Austria | drowsiness may occur avoid alcoholic drinks | WWW.SUSANNEKAUFMANN.COM |
| 519 | background | -0.265616 | Performances an 34 | 45 TABLETS | WWW.SUSANNEKAUFMANN.COM |
| 381 | background | -0.253686 | Lift zuganglich | drowsiness may occur avoid alcoholic drinks | WWW.SUSANNEKAUFMANN.COM |
| 321 | background | -0.240292 | Austria |  | WWW.SUSANNEKAUFMANN.COM |
| 745 | background | -0.197991 |  | EXP.: | WWW.SUSANNEKAUFMANN.COM |
| 571 | subject_text | -0.196410 | 10:30-18 Uhr |  | or higher and other sun protection measures including |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 353 | background | +0.568065 |  |  | Retains SPF after 40 minutes of activity in the water onsweating |
| 354 | background | +0.562519 | 安 |  | ist wasserabweisend und enthält aus- |
| 684 | background | +0.514915 |  |  | Suncare |
| 443 | subject_text | +0.396413 | Drug Facts (continued) | Drug Facts (continued) | repair mechanism and helps the skin |
| 506 | subject_text | +0.202275 | pregnant: ask a health protessional betore use. | Drug Facts (continued) |  |
| 409 | subject_text | +0.171092 | away (1-800-222-1222). |  | If swallow ed get medical help or contact a Poison Control |
| 466 | subject_text | +0.148798 | Questions or comments? 1-877-290-4008 | Questions or comments? 1-877-290-4008 | Brugg 35, 6870 Bezau, Austria |
| 102 | background | +0.138482 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| tablet in 24 hours. A 5 mg | 3 | -0.000087 |
| not take more than one 10 mg | 5 | -0.000096 |
| one 10 mg tablet once daily; do | 4 | -0.000085 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_2df687d8a2f89f40` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.023352 | +0.183806 |
| 仅主体分差 | +0.003950 | +0.244333 |
| 主体 token 数 | 91 | 312 |

[对照 token 数据](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
