# photo_4b09c283ed00faca

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_004360`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003367`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 290 |
| 图像 Top-1 参考 | reference_003367 |
| BM25 真值排名 | 8 |
| 融合真值排名 | 6 |
| 图像正负分差 | -0.014228 |
| BM25 正负分差 | -45.737793 |
| 0.5 × 图像 z 分差 | -0.286057 |
| 0.5 × BM25 z 分差 | -0.365015 |
| 最终融合分差 | -0.651071 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 180 个。正向贡献 +0.019269，负向贡献 -0.033497，相加等于图像分差 -0.014228。缓存复现误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 588 | -0.007336 | -0.009719 |
| nonvisual | 11 | +0.000116 | +0.008190 |
| subject_nontext | 20 | -0.000603 | -0.023488 |
| subject_text | 160 | -0.006405 | -0.031184 |

仅保留主体 query token 的两候选分差：-0.030329。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.007336。主体框内 token 占有效序列 23.1%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 479 | 76.2% | True | +7.576145 | 78.7% |
| 错误 | 15 | 25.1% | True | +10.215504 | 81.6% |

主体中 160 个 token 与 OCR 相交，145 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 263 | background | -0.583856 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | See inside for more details or |
| 262 | background | -0.528612 | CVS | 73460 UW C3 | CVS.com/StartToStop |
| 475 | subject_text | -0.469060 | before use. No studles have been done to show if this product | nicotine replacement medicine. This medicine is believed to be | if you are under 18 years of age, ask a doctor before use. No studies have been done to |
| 495 | subject_text | -0.400736 | on the advice of your health care prov Ider. Smoking can |  | If you are pregnant or breast-feeding, only use this medicine on the advice of your health |
| 435 | subject_text | -0.399616 | dlabetes history of selzures | heart attack, or iregular heartbeat. Nicotine can increase your | high blood pressure not controlled with medication. Nicotine can increase your blood pressure. |
| 505 | subject_text | -0.372188 | to remove the lozenge, wrap it in paper and throw away in the |  |  |
| 403 | subject_text | -0.358505 | Nicotine can Increase your blood pressure. stomach ulcer or | heart attack, or iregular heartbeat. Nicotine can increase your | high blood pressure not controlled with medication. Nicotine can increase your blood pressure. |
| 270 | subject_text | -0.326632 | If you are pregnant or breast-feeding, only use this medicine | printed seal under cap is broken or missing. |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 297 | background | +0.488240 | Compare |  |  |
| 292 | background | +0.360981 |  |  |  |
| 222 | background | +0.312692 | NDC 69043-047-02 | NDC 82442-734-60 |  |
| 201 | background | +0.241498 |  | 094 09 0036 R00 C-002262-01-008-0000 |  |
| 406 | subject_text | +0.211995 | heartbeat or palpitations occur you get symptoms of nicotine |  | you get symptoms of nicotine overdose such as nausea, vomiting, dizziness, diarrhea, weakness |
| 441 | subject_text | +0.193939 | trash. In case of overdose, get medical help or contact a Poison | to remove the lozenge, wrap it in paper and throw away in the | Keep out of reach of children and pets. Nicotine lozenges may have enough nicotine to make |
| 202 | background | +0.191815 | 011948412551817 | 094 09 0036 R00 C-002262-01-008-0000 |  |
| 277 | subject_text | +0.182274 | persistent indigestion or severe sore throat occurs irregular | persistent indigestion or severe sore throat occursirregular |  |

## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.014228 | +0.045373 |
| 仅主体分差 | -0.030329 | +0.094385 |
| 主体 token 数 | 180 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
