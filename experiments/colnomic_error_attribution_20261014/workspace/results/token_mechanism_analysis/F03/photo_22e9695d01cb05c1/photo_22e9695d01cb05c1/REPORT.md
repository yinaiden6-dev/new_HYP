# photo_22e9695d01cb05c1

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001112`，F03 错误参考 `reference_004677`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003832`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 20 |
| 图像 Top-1 参考 | reference_003832 |
| BM25 真值排名 | 26 |
| 融合真值排名 | 14 |
| 图像正负分差 | +0.039225 |
| BM25 正负分差 | -309.824829 |
| 0.5 × 图像 z 分差 | +1.130321 |
| 0.5 × BM25 z 分差 | -3.538962 |
| 最终融合分差 | -2.408641 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.7579**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 170 个。正向贡献 +0.066248，负向贡献 -0.027023，相加等于图像分差 +0.039225。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 598 | +0.000192 | +0.000251 |
| nonvisual | 11 | +0.001191 | +0.084312 |
| subject_nontext | 22 | +0.000930 | +0.032946 |
| subject_text | 148 | +0.036911 | +0.194283 |

仅保留主体 query token 的两候选分差：+0.173404。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 +0.000192。主体框内 token 占有效序列 21.8%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 327 | 66.6% | True | +6.644394 | 59.6% |
| 错误 | 25 | 22.6% | True | +11.695843 | 89.7% |

主体中 148 个 token 与 OCR 相交，96 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 367 | background | -0.308312 | FiveGuysGraben30 | 321130788980 | WWW.SUSANNEKAUFMANN.COM |
| 291 | background | -0.270589 |  | Signature | 08 |
| 219 | background | -0.213097 |  | Drug Facts (continued) |  |
| 546 | background | -0.186827 |  | 321130788980 | WWW.SUSANNEKAUFMANN.COM |
| 494 | background | -0.178378 |  |  | WWW.SUSANNEKAUFMANN.COM |
| 382 | background | -0.176523 | Mo-Fr: 10-18 Uhr |  | or higher and other sun protection measures including |
| 498 | subject_text | -0.167472 | tranquilizers or sedatives. |  | When using this product keep out of eyes. Rinse with water |
| 284 | background | -0.156479 | Optionen, Lakritz und |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 246 | subject_text | +0.586669 | Active ingredient (in each tablet) Purpose | Active ingredient (in each tablet) Purpose |  |
| 564 | subject_text | +0.583021 | be careful when drivinga motorvehicle or operating | be careful when driving a motor vehicle or operating |  |
| 177 | subject_text | +0.576730 | TAMPER EVIDENT: DO NOT USE THIS PRODUCT IF | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF | SUSANNE KAUFMANN |
| 568 | subject_text | +0.549244 | be careful when drivinga motorvehicle or operating | be careful when driving a motor vehicle or operating | DE Die innovative Sonnencreme mit |
| 401 | subject_text | +0.544864 | this product or any of its ingredients, or to an | ·Sneezing ·Runny Nose - Itchy, Watery Eyes · Itchy Throat or Nose 45 TABLETS | ric acid, cetearyl alcohol, cetearyl olivate, sorbitan olivate, water, |
| 591 | background | +0.530663 | 月 |  | to all skin exposed to the sun |
| 147 | subject_text | +0.524677 | READ AND KEEP CARTON FOR COMPLETE | READ AND KEEP CARTON FOR COMPLETE | When using this product keep out of eyes. Rinse with water |
| 565 | subject_text | +0.497620 | be careful when drivinga motorvehicle or operating | be careful when driving a motor vehicle or operating |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Cetrizine HCI 10 mg. | 3 | +0.000772 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_2df687d8a2f89f40` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.039225 | +0.183806 |
| 仅主体分差 | +0.173404 | +0.244333 |
| 主体 token 数 | 170 | 312 |

[对照 token 数据](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
