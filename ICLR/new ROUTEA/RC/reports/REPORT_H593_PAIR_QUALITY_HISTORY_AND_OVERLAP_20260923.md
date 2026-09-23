# ColNomic 与 RoMa 质量来源：历史及重复性核查

2026-09-23。针对用户要求，先查历史与未释放任务，再决定区分实验。没有把旧NO-GO改写为成功，没有重提已有单图融合或统一RoMa采集。

## 历史已经提供的证据

|历史机制|实际信息/训练|发现|能否隔离当前分界|
|---|---|---|---|
|P-V2、早期ColNomic P/V|token相似性提出区域，再独立验证|exec089不同reference的query mask同为135/150，候选条件化不足|否，proposal/lock约束混入|
|ColNomic RGH V2–V9B|4357参数适配/assignment、dustbin、单端可靠性，再跨图循环/投影/atom聚合|六例1222方向重投影合法数0；OOF pilot27→26；atom权重近乎均匀|证明这些对应/空间资格失败，不是没有几何门的全图质量读出不可行|
|DINO-RCDE GX R0|DINO关系表示/consensus/空间证据，多个修复|窄资格NO-GO，formal1/16 vs11/16门|输入、支持、损失/门不同，不能归结ColNomic tokens无信息|
|V7/V8及C6d-F1C|reference去重、拥挤度，逐token有符号证据/可靠性|历史已实际实现/训练且失败|不能将去重、局部可靠性称作新原理；也不是当前同协议替换M|
|learned ColNomic-only|完整tokens先压成F/B/Gq/Gr/U等五统计，再训练七参数头|H593 COST1 427，CE431；联合481/486|支持这些固定统计不足，不隔离整个token表示|
|本次41张通路干预|固定原COST1，改质量头输入或RoMa图像|J置零、仅换低通所得M，均失去原3次纠错；粗质量保留|支持交互/细节经M影响原决策；不能证明只换ColNomic读出就可恢复|

源码重要区别：`src/rc_aslo_xf/cw0_connected_window_v1.py:1002` 的旧query/reference可靠性分别由各自适配token的线性层产生；随后参与双图soft assignment。旧方案已经有双图对应，不能概括为从未做交互。但它不等于让质量MLP直接读取本次完整双图soft-context、差/积通道、位置和循环诊断后，经全图M接回固定内容读出。

2026-09-22 `plan/RC_ROMA_FEATURE_COLNOMIC_FUSION_FEASIBILITY_20260922.md` 已提出纯ColNomic token级候选条件质量，明确尚未实施。2026-09-11 `reports/NOTE_NEW_HYP_MATCH_DIRECTIONS_HISTORY_CORRECTION_20260911.md` 已纠正“去重/可靠性从未探索”的说法。本轮不是新原理宣称，而是变更接口/对照后的受控复验。

## 当前已提交、held 与未释放分支

核对07:50 UTC左右scheduler及磁盘：

- 单图融合：四来源×FREE/ROMA_WEIGHTED×两损失×两seed×五折，20/200配置已有验证。5158421_91运行，5158958/60/61/67/69共28项依赖原5158421。该分支学习单图4096参数输出通道门控；其中COL_ONLY_FREE会检验单图重读出，故不重复另做单图token适配器。
- 统一RoMa中间量采集：5158898等待dev名额；原pool/M内部/M视觉重复生产任务held，统一资格后原控制器处理。共享采集输出供坐标、通路、视觉干预消费者；不训练新的ColNomic候选质量。
- 5158704原M路径CPU读出held，消费既有RoMa质量并重训标量头；不读取新ColNomic完整关系。
- 当前未发现已提交的“冻结内容L0，来源×交互2×2、逐token质量学习”作业。仅补该缺项，既有任务/源码/依赖未改变。

## 本轮结论与执行

不能说历史没有探索关系、去重或可靠性；也没有找到能将“ColNomic信息缺失”与“当前读出没利用信息”彻底分开的同协议完整结果。

新方案 `plan/RC_H593_PAIR_QUALITY_V1_20260923.md` 固定内容，比较两个现有表示来源和有/无双图关系，保留同预算NONE/ROMA对照。先全C128容量资格，合格后原分组五折。新GPU只用于缓存张量的小模块训练，无RoMa或ColNomic重新编码。另保留表示池化/固定投影、模型容量、训练目标仍可能导致负结果的解释限制。

核心历史来源：

- `reports/REPORT_RGH_ATOM_SELECTION_HISTORICAL_AUDIT_AND_AS1_DECISION_20260831.md`
- `plan/ROMAV2_CORRESPONDENCE_SOURCE_QUALIFICATION_CONTRACT_V1_20260831.md`
- `reports/REPORT_DINO_RCDE_GX_R0_THREE_REPAIR_ROUNDS_CLOSURE_20260826.md`
- `reports/NOTE_NEW_HYP_MATCH_DIRECTIONS_HISTORY_CORRECTION_20260911.md`
- `results/rc_h593_learned_colnomic_only_v1/report_zh.md`
- `reports/REPORT_H593_SUBSET41_FROZEN_PATH_INTERPRETATION_20260923.md`
- `plan/RC_H593_FEATURE_FUSION_TRAIN_V2_20260922.md`
- `plan/RC_H593_UNIFIED_ACQUISITION_V1_20260923.md`
