# COST1：完整 128 候选决策数据

[全部 1536 行 CSV](data/all_12_cases_1536_candidates.csv) · [数据 ZIP](../COST1_medicine_C128_scores.zip)

每个案例包含自然 C128 的完整 128 行：127 个 challenger 直接使用原封存 COST1 logit；RAW 首选对应 HOLD，其 **policy_score 固定为 0**，head_logit、六项头特征和贡献均为 null。HOLD 不是把零向量输入头后得到的偏置。

CSV/JSON 提供原始 RAW 分数与名次、原 C128 位置、图库物理行、参考图路径、目标/RAW首选/最终选择标记、六项特征、六项加权贡献、偏置、封存 logit、policy_score 和各自十六进制 FP64 表示。决策分数不是校准身份概率。

特征顺序为 RAW / S / M / L / Q / R；完整名称和头参数保存在每例 JSON。candidate_position 从 0 开始，raw_rank 和 policy_rank 从 1 开始。HOLD 与 challenger 同为 0 时保持 HOLD；正分并列采用原 challenger 顺序。

为方便自行绘图，数据另附 plot_raw_component 与 plot_other_component，分别为 RAW 项加权贡献和封存分数减去该贡献，二者之和等于原决策分数。HOLD 对应的 (0,0) 只是决策基准，其真实头特征仍为 null。此包仅提供数据，不含新生成点图。

仍为同一批 H593 分组留出案例，参数来源记录在 JSON。无重训、无新编码器或 RoMa 前向。

| 案例 | 候选行数 | CSV | JSON |
|---|---:|---|---|
| OUTCOME-0438 | 128 | [CSV](data/01_COST1_OUTCOME-0438_C128.csv) | [JSON](data/01_COST1_OUTCOME-0438_C128.json) |
| DIFFICULT-0013 | 128 | [CSV](data/02_COST1_DIFFICULT-0013_C128.csv) | [JSON](data/02_COST1_DIFFICULT-0013_C128.json) |
| DIFFICULT-0023 | 128 | [CSV](data/03_COST1_DIFFICULT-0023_C128.csv) | [JSON](data/03_COST1_DIFFICULT-0023_C128.json) |
| DIFFICULT-0043 | 128 | [CSV](data/04_COST1_DIFFICULT-0043_C128.csv) | [JSON](data/04_COST1_DIFFICULT-0043_C128.json) |
| DIFFICULT-0061 | 128 | [CSV](data/05_COST1_DIFFICULT-0061_C128.csv) | [JSON](data/05_COST1_DIFFICULT-0061_C128.json) |
| DIFFICULT-0083 | 128 | [CSV](data/06_COST1_DIFFICULT-0083_C128.csv) | [JSON](data/06_COST1_DIFFICULT-0083_C128.json) |
| DIFFICULT-0102 | 128 | [CSV](data/07_COST1_DIFFICULT-0102_C128.csv) | [JSON](data/07_COST1_DIFFICULT-0102_C128.json) |
| NDV2-011-P04 | 128 | [CSV](data/08_COST1_NDV2-011-P04_C128.csv) | [JSON](data/08_COST1_NDV2-011-P04_C128.json) |
| OUTCOME-0015 | 128 | [CSV](data/09_COST1_OUTCOME-0015_C128.csv) | [JSON](data/09_COST1_OUTCOME-0015_C128.json) |
| OUTCOME-0087 | 128 | [CSV](data/10_COST1_OUTCOME-0087_C128.csv) | [JSON](data/10_COST1_OUTCOME-0087_C128.json) |
| OUTCOME-0108 | 128 | [CSV](data/11_COST1_OUTCOME-0108_C128.csv) | [JSON](data/11_COST1_OUTCOME-0108_C128.json) |
| OUTCOME-0131 | 128 | [CSV](data/12_COST1_OUTCOME-0131_C128.csv) | [JSON](data/12_COST1_OUTCOME-0131_C128.json) |
