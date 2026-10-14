# photo_1f48d45127db4129

按当前真值分析，未新增身份复核。 用户指定主题：文字排版结构。

正确参考 `reference_001139`，F03 错误参考 `reference_004489`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 148 |
| 图像 Top-1 参考 | reference_004489 |
| BM25 真值排名 | 1 |
| 融合真值排名 | 10 |
| 图像正负分差 | -0.119856 |
| BM25 正负分差 | +39.514648 |
| 0.5 × 图像 z 分差 | -1.239275 |
| 0.5 × BM25 z 分差 | +0.460947 |
| 最终融合分差 | -0.778328 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.2711**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 252 个。正向贡献 +0.012130，负向贡献 -0.131986，相加等于图像分差 -0.119856。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 468 | -0.068335 | -0.106737 |
| nonvisual | 11 | -0.000199 | -0.013207 |
| subject_nontext | 70 | -0.010582 | -0.110504 |
| subject_text | 182 | -0.040740 | -0.163633 |

仅保留主体 query token 的两候选分差：-0.148875。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.068335。主体框内 token 占有效序列 34.5%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 847 | 77.8% | True | +8.620227 | 78.2% |
| 错误 | 24 | 8.4% | False | 不可评估 | 不可评估 |

主体中 182 个 token 与 OCR 相交，145 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 329 | subject_text | -0.555177 | Warnings | Itcontains important information. |  |
| 439 | background | -0.524504 |  |  | bronchitis, or emphysema |
| 438 | subject_nontext | -0.509344 |  |  | bronchitis, or emphysema |
| 310 | subject_text | -0.509183 | the intensity of coughing | 719283151100 |  |
| 330 | subject_text | -0.508319 | Warnings | Itcontains important information. |  |
| 450 | subject_text | -0.502422 | cough accomparied by too much phlegm(mucus) |  | occurs with smoking, asthma, chronic |
| 495 | subject_text | -0.502070 | cough lasts more than 7 days, comes back, or occurs | Itcontains important information. |  |
| 521 | background | -0.500871 |  | meijer. | health professional before use. |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 293 | subject_text | +0.400216 | cough due to minor throat and bronchial rritation as | the impulse to cough to help you get to sleep | cough due to minor throat and |
| 303 | background | +0.307176 |  |  | Amneal Pharmaceuticals LLC |
| 183 | background | +0.264926 |  |  |  |
| 107 | background | +0.264553 |  |  |  |
| 307 | subject_nontext | +0.219715 |  |  |  |
| 180 | background | +0.190448 |  | Drug Facts | †Compare to the active ingredients of Mucinex® DM |
| 69 | background | +0.187719 |  | Dextromethorphan HBr 60 mg..... Cough suppressant | EXPECTORANT |
| 165 | background | +0.185065 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| De tromethorphan HBr 60 mg.....Cough suppreesant | 10 | -0.001056 |
| Gualeneein 1200 mg | 8 | -0.000407 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_02c11dd14bf4116f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.119856 | +0.082564 |
| 仅主体分差 | -0.148875 | +0.119655 |
| 主体 token 数 | 252 | 220 |

[对照 token 数据](../photo_02c11dd14bf4116f/tokens.csv) · [对照摘要](../photo_02c11dd14bf4116f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
