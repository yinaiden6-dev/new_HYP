# H593：全部M与481／492答案对照

日期：2026-09-27。数据：原593 query、ColNomic自然C128、原组件五折OOF。481指原NATIVE7 COST1，492指GAP_BIAS2；不是两次同模型训练。已打开开发集的描述性审计，零新增训练、零编码器或RoMa前向。

## 全量对照

M高低以正确reference在本query的完整C128内的排名定义；所有绝对M仍保存，未设置全局高低阈值。没有第一名并列。

|目标M排名|样本数|481正确|492正确|492救回481|492误伤481|两者都错|
|---|---:|---:|---:|---:|---:|---:|
|严格第1|379|339|350|12|1|28|
|第2–5|74|60|61|2|1|12|
|第6–10|22|15|16|1|0|6|
|第11–128|95|67|65|1|3|27|
|目标不在C128|23|0|0|0|0|23|
|合计|593|481|492|16|5|96|

两者共同正确476张。直接用argmax(M)只有379/593；相对481救40损142，相对492救29损142。答案指导的“旧头或M最大”联合覆盖为521，是诊断oracle，不能记为模型准确率或可学习上限。两种头各有142张在target的M不是第一时仍答对，所以M不第一并不意味着无用或必错。

## 492还错的101张

- 23张：目标缺失，固定C128无法救回。
- 30张：目标已是最高挑战者，仍HOLD。
- 40张：目标不是最高挑战者；仅改HOLD阈值不足以修正排序。
- 8张：RAW原本正确，492切换错。

上述四类互斥。GAP_BIAS2只在原头HOLD时尝试放行原最高挑战者，原SWITCH不变；它不是完整候选重排序。

## M严格第一却仍错的29张

18张最高挑战者正确但HOLD，10张挑战者排序受阻，1张RAW正确被改错。不能把“原481的40例中492还错28例”当作492全部M-first失败：新增误伤OUTCOME-0721必须计入。

|query|target M|481答案|492答案|492失败类型|
|---|---:|---|---|---|
|DIFFICULT-0050|0.071203|bef4f237-ccb8-4d28-bfc2-41c511a17d7f-01 (HOLD)|bef4f237-ccb8-4d28-bfc2-41c511a17d7f-01 (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0090|0.253064|5-75BlisterCarton (HOLD)|5-75BlisterCarton (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0093|0.068354|5-75BlisterCarton (SWITCH)|5-75BlisterCarton (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0097|0.136877|1q3-c2-childrens-24-hour-allergy-nasal-spray (HOLD)|1q3-c2-childrens-24-hour-allergy-nasal-spray (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0105|0.217462|1q3-c2-childrens-24-hour-allergy-nasal-spray (HOLD)|1q3-c2-childrens-24-hour-allergy-nasal-spray (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0106|0.295084|cvs-health-childrens-fluticasone-propionate-nasal-spray-carton-image (HOLD)|cvs-health-childrens-fluticasone-propionate-nasal-spray-carton-image (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0107|0.215254|1q3-c2-childrens-24-hour-allergy-nasal-spray (HOLD)|1q3-c2-childrens-24-hour-allergy-nasal-spray (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0132|0.039972|Bronze Glow (HOLD)|Bronze Glow (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0134|0.113883|CDER 100ct C (SWITCH)|CDER 100ct C (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0138|0.120494|CDER 100ct C (SWITCH)|CDER 100ct C (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0150|0.071087|cvshealth-mucus-extended-release-carton-image (HOLD)|cvshealth-mucus-extended-release-carton-image (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0331|0.120749|careone-infants-acetaminophen-160mg-per-5ml-grape-carton-image (HOLD)|careone-infants-acetaminophen-160mg-per-5ml-grape-carton-image (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0341|0.175450|careone-infants-acetaminophen-160mg-per-5ml-grape-carton-image (HOLD)|careone-infants-acetaminophen-160mg-per-5ml-grape-carton-image (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0407|0.089394|CDER portal (SWITCH)|CDER portal (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0435|0.195088|ASP Sinus Congestion Pain PMG 24ct (HOLD)|ASP Sinus Congestion Pain PMG 24ct (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0441|0.056595|ASP Sinus Congestion Pain PMG 24ct (SWITCH)|ASP Sinus Congestion Pain PMG 24ct (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0445|0.070366|ASP Cold Cough LG 10ct (HOLD)|ASP Sinus Congestion Pain PMG 24ct (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0468|0.164319|Bayer_50 (SWITCH)|Bayer_50 (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0477|0.169141|Bayer_50 (SWITCH)|Bayer_50 (SWITCH)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0534|0.159752|SH230414B (100 ml Carton) (HOLD)|SH230414B (100 ml Carton) (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0536|0.160733|SH230414B (100 ml Carton) (HOLD)|SH230414B (100 ml Carton) (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0538|0.078712|SH230414B (100 ml Carton) (HOLD)|SH230414B (100 ml Carton) (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0557|0.069034|-Pentrexcilina AD 4 fl oz (RTE) EN-ES Rev 11.2024 (AKRON)_001 (HOLD)|-Pentrexcilina AD 4 fl oz (RTE) EN-ES Rev 11.2024 (AKRON)_001 (HOLD)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0676|0.135391|bravecto-05 (HOLD)|bravecto-05 (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0721|0.130837|cuvitru-19 (HOLD)|cuvitru-13 (SWITCH)|RAW_CORRECT_BROKEN|
|OUTCOME-0733|0.112413|dro0b-0001-03 (HOLD)|dro0b-0001-03 (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0818|0.155743|vetoryl-08 (HOLD)|vetoryl-08 (HOLD)|TARGET_TOP_BUT_HOLD|
|OUTCOME-0819|0.079727|vetoryl-04 (HOLD)|vetoryl-04 (HOLD)|CHALLENGER_RANKING_BLOCKED|
|OUTCOME-0821|0.080493|vetoryl-04 (HOLD)|vetoryl-04 (HOLD)|TARGET_TOP_BUT_HOLD|

## 分数路径：哪些项抵消了有利M

逐例按原冻结COST1分数做精确代数分解，并保存target相对最高原头错误答案（包含HOLD=0）的RAW/S/M/L/Q/R/bias贡献。这个分解说明程序如何得出分数，不把有关联的输入项当独立物理因果。

在18个M-first且目标最高挑战者但HOLD的样本里：M项18/18有利，RAW项18/18不利，S项18/18不利，bias18/18不利。平均贡献依次为M +0.3470、RAW −0.2795、S −0.2182、bias −1.2621；L项16/18有利。说明并非“模型没有读到M”，而是有利M未抵消其他项与接受偏置。

10个M-first但目标不是最高挑战者的样本里，M项10/10有利、RAW项10/10不利、S项9/10不利。进一步训练必须同时记录对原正确样本的损失，不能据此逐例手工调参数。

OUTCOME-0721在481时正确，492放行了错误挑战者；这说明仅放宽接受规则会出现新的M-first误伤。

## 两个用户指定案例与历史去重

- DIFFICULT-0011：target M第19，仅比最终错选答案高不等于胜过全部候选。
- OUTCOME-0477：target M第1，但原挑战者分数第2，原头已错误SWITCH；GAP492不改既有SWITCH，因而仍错。历史DIAG_CE13（493）已经救回该例，不能再次算新发现。
- 历史CONTENT_BOX_CE_POLISH18为496/593，原28个M-first未修复案例中已有6例被它修复。后续新模型须同时对照481、492、496，不能只与较弱头比较。

## 交付与核验

- [可筛选网页](../results/rc_h593_m_481_492_comparison_20260927_v1/index.html)：593张逐例并列真实identity、M值／排名、481与492答案和HOLD/SWITCH。
- [全部593 CSV](../results/rc_h593_m_481_492_comparison_20260927_v1/cases593.csv)。
- [全部128候选分数](../results/rc_h593_m_481_492_comparison_20260927_v1/all_candidates.jsonl)：75,904个M、自由内容、两个头的分数，RAW/HOLD=0。
- [29个M-first失败](../results/rc_h593_m_481_492_comparison_20260927_v1/M_first_but_492_wrong_29.csv)、[21个改判](../results/rc_h593_m_481_492_comparison_20260927_v1/492_rescues_and_breaks_21.csv)。
- [分数分解](../results/rc_h593_m_481_492_comparison_20260927_v1/score_decomposition.json)。

全部593×127个非HOLD logits从FP64原特征与原头复算，最大绝对差1.7764e-15；492的全部127 logits按其原放行规则逐元素完全一致。独立validator已通过封存payload、结果、导出CSV和各类计数核验：[validation.json](../results/rc_h593_m_481_492_comparison_20260927_v1/validation.json)。

所有target标签仅用于分析，不能作为部署时的分流特征。当前结论不证明M唯一、充分或身份概率。
