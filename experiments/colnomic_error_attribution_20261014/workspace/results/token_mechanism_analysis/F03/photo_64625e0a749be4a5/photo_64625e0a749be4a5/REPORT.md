# photo_64625e0a749be4a5

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_003973`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 19 |
| 图像 Top-1 参考 | reference_003973 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 4 |
| 图像正负分差 | -0.010822 |
| BM25 正负分差 | +4.375122 |
| 0.5 × 图像 z 分差 | -0.363654 |
| 0.5 × BM25 z 分差 | +0.046830 |
| 最终融合分差 | -0.316824 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.1141**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 120 个。正向贡献 +0.016841，负向贡献 -0.027663，相加等于图像分差 -0.010822。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 648 | -0.008776 | -0.010551 |
| nonvisual | 11 | -0.000232 | -0.016415 |
| subject_nontext | 5 | +0.000102 | +0.015955 |
| subject_text | 115 | -0.001916 | -0.012981 |

仅保留主体 query token 的两候选分差：-0.011775。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.008776。主体框内 token 占有效序列 15.4%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 652 | 71.9% | True | +8.200064 | 71.7% |
| 错误 | 8 | 23.9% | True | +18.110134 | 100.0% |

主体中 115 个 token 与 OCR 相交，107 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 368 | subject_text | -0.332436 | 1 lozenge every |  |  |
| 402 | subject_text | -0.275799 | and allow the lozenge to slowly dissolve. Minimize swallowing. |  | Nicotine Polacrilex Lozenge, |
| 448 | background | -0.247284 | Pharmacy | If you are pregnant or breast-feeding, only use this medicine | Ask a doctor or pharmacist before use if you are |
| 336 | subject_text | -0.228717 | 1 lozenge every |  |  |
| 646 | background | -0.218336 |  |  | the other until completely dissolved |
| 661 | background | -0.205463 |  |  | the other until completely dissolved |
| 343 | subject_text | -0.199602 | Other information store at 20-25℃ (68-77°F) | Questions or comments?call tll-free 1-866-751-9303 | the lozenge for a longer period to keep from smoking, talk to your |
| 764 | background | -0.190304 | Ent |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 491 | background | +0.258681 |  | if you smoke your first cigarette within 30 minutes of |  |
| 396 | background | +0.252782 | MAGNES | Inactive ingredients acesulfame potassium, calcium | Inactive ingredients acacia, acesulfame potassium, calcium |
| 129 | background | +0.237924 | 1-5 June 20 |  |  |
| 98 | background | +0.237665 | 1-5 June 20 |  |  |
| 97 | background | +0.220352 | 1-5 June 20 |  |  |
| 427 | background | +0.208233 | CALCIU | Inactive ingredients acesulfame potassium, calcium | Inactive ingredients acacia, acesulfame potassium, calcium |
| 428 | background | +0.192429 | MAGNES | Inactive ingredients acesulfame potassium, calcium | Inactive ingredients acacia, acesulfame potassium, calcium |
| 314 | subject_text | +0.185353 | This product is not manufactured or distributed by GlapSrithKine |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| waking up, use 4 mg nicotine lozenge if you smoke your | 9 | +0.000216 |
| 2 mg nicotine lozenge aocording to the following 12 week | 12 | +0.000261 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010822 | +0.014850 |
| 仅主体分差 | -0.011775 | +0.023667 |
| 主体 token 数 | 120 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
