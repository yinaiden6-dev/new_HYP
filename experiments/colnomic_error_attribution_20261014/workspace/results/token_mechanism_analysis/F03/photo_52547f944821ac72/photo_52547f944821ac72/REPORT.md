# photo_52547f944821ac72

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001112`，F03 错误参考 `reference_003821`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 15 |
| 图像 Top-1 参考 | reference_003821 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 3 |
| 图像正负分差 | -0.037965 |
| BM25 正负分差 | +20.111816 |
| 0.5 × 图像 z 分差 | -0.485195 |
| 0.5 × BM25 z 分差 | +0.240569 |
| 最终融合分差 | -0.244626 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.3315**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 299 个。正向贡献 +0.017318，负向贡献 -0.055282，相加等于图像分差 -0.037965。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 469 | -0.018045 | -0.029972 |
| nonvisual | 11 | -0.000114 | -0.008064 |
| subject_nontext | 68 | -0.003494 | -0.040022 |
| subject_text | 231 | -0.016312 | -0.055009 |

仅保留主体 query token 的两候选分差：-0.051601。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.018045。主体框内 token 占有效序列 38.4%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 544 | 66.7% | True | +4.903694 | 59.3% |
| 错误 | 51 | 22.8% | True | +4.401301 | 58.6% |

主体中 231 个 token 与 OCR 相交，138 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 464 | subject_text | -0.472733 | temporarily relieves these symptoms due to hay fever or | Keep out of reach of children. In case of overdose, get |  |
| 259 | background | -0.446590 |  |  |  |
| 440 | subject_text | -0.423346 | tranquilizers or sedatives. |  | drowsiness may occur |
| 279 | subject_text | -0.413771 | disease. Your doctor should determine if you need a | READ AND KEEP CARTON FOR COMPLETE | disease. Your doctor should determine if you need a |
| 474 | subject_text | -0.383227 | alcohol, sedatives, and tranquilizers may increase |  | drowsiness may occur |
| 139 | background | -0.380600 |  | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |
| 311 | subject_text | -0.373129 | disease. Your doctor should determine if you need a | Ask a doctor or pharmacist before use if you are taking | disease. Your doctor should determine if you need a |
| 378 | subject_text | -0.367356 | drowsiness may occur avoid alcoholic drinks | When using this product |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 235 | subject_text | +0.489189 | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |  |
| 203 | subject_text | +0.447989 | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |  |
| 200 | subject_nontext | +0.418958 |  | READ AND KEEP CARTON FOR COMPLETE |  |
| 331 | subject_text | +0.312570 | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |  |
| 555 | subject_nontext | +0.271898 |  | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |
| 241 | subject_text | +0.271552 | temporarily relieves these symptoms due to hay fever or | Antihistamine | Temporarily relieves these symptoms due to hay fever or |
| 459 | subject_text | +0.267375 | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  | SAFETY SEAL UNDER CAP IS BROKEN OR MISSING. |
| 392 | subject_text | +0.248473 | READ AND KEEP CARTON FOR COMPLETE | READ AND KEEP CARTON FOR COMPLETE |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Cetirizine HCl 10 mg. | 7 | -0.000566 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_2df687d8a2f89f40` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.037965 | +0.023698 |
| 仅主体分差 | -0.051601 | +0.041633 |
| 主体 token 数 | 299 | 312 |

[对照 token 数据](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
