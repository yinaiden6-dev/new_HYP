# photo_d2377339b5c178d4

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_004359`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001129`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001129 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.006482 |
| BM25 正负分差 | -12.401886 |
| 0.5 × 图像 z 分差 | +0.198578 |
| 0.5 × BM25 z 分差 | -0.293409 |
| 最终融合分差 | -0.094831 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5964**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 135 个。正向贡献 +0.031435，负向贡献 -0.024953，相加等于图像分差 +0.006482。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 633 | +0.001327 | +0.001634 |
| nonvisual | 11 | -0.000121 | -0.008581 |
| subject_nontext | 33 | +0.000346 | +0.008160 |
| subject_text | 102 | +0.004930 | +0.037650 |

仅保留主体 query token 的两候选分差：+0.030441。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 -0.000121。主体框内 token 占有效序列 17.3%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 302 | 59.6% | True | +2.965808 | 49.2% |
| 错误 | 32 | 8.3% | False | 不可评估 | 不可评估 |

主体中 102 个 token 与 OCR 相交，53 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 384 | background | -0.392526 | Pharmacy | If you are pregnant or breast-feeding, only use this medicine |  |
| 416 | background | -0.357658 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for |  |
| 719 | background | -0.317778 |  |  | © 2022 CVS/pharmacy |
| 770 | background | -0.305483 | CENTER |  | © 2022 CVS/pharmacy |
| 385 | background | -0.305087 | Pharmacy |  |  |
| 675 | background | -0.303280 | 1020 Vienna | non-nicotine stop smoking drugtaking prescription medicine for |  |
| 715 | background | -0.296790 |  |  |  |
| 479 | background | -0.285955 | Pharmacy | non-nicotine stop smoking drugtaking prescription medicine for | © 2022 CVS/pharmacy |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.486634 | R | R | DIE |
| 507 | subject_text | +0.479717 | up&up | up&up |  |
| 475 | subject_text | +0.453733 | up&up | up&up |  |
| 443 | subject_text | +0.404103 | up&up | up&up |  |
| 332 | background | +0.385944 | MAGNES | Inactive ingredients acesulfame potassium, calcium | PLASTIC |
| 707 | background | +0.342209 |  | R |  |
| 506 | subject_text | +0.316960 | R | R | Actual Size |
| 67 | background | +0.296705 | 1-5 June |  | Opening Directions: |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 2 mg (Nicotine) | 8 | +0.000151 |
| Polacrilex Lozenge, 4 mg | 4 | +0.000003 |
| 2mg | 3 | +0.000049 |
| 20 LOZENGES, 2 mg EACH | 7 | +0.000303 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.006482 | +0.003739 |
| 仅主体分差 | +0.030441 | +0.023522 |
| 主体 token 数 | 135 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
