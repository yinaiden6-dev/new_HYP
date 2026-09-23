# H593 粗细特征融合实验：执行状态

## 最近核查与解释修正（2026-09-22 17:41 CEST）

- 全量缓存已通过：5324个唯一图片/几何条目、21片；完整C128容量资格也已通过。
- `5157532_0` 首片正常完成，但日志和检查点明确为联合训练 `step=0/2000`：完成的是FREE输入准备与初始头拟合，不能记作融合更新已经完成。
- `5157555_[1–46%46]` 均在accelerated因Priority排队，dispatcher共享心跳正常，完整配置仍为0/200。尚无融合准确率。
- 原模型直接读取RoMa匹配质量权重，不直接融合RoMa embeddings；上游粗细特征经由质量权重的间接贡献尚不能排除。现有四来源×两评分对照用于隔离这个问题。
- 评分算子五折已全部通过核验。整体M×自由MaxSim在同协议重训后为COST1 481/593，与完整头正确总数相同，但相对完整头4救回/4损失，不是相同决策。
- 完整定位说明：`REPORT_H593_FEATURE_VS_MATCH_QUALITY_LOCALIZATION_20260922.md`。下文为历史快照，最新状态以上述核查为准。

## 最近任务核查（2026-09-22 16:06 CEST）

- 特征数组5157341：9/21分片已出合格回执；已验证新增2304个唯一图片/几何条目，另有129个pilot复用。剩余任务排队/运行中，5157342完整缓存汇总仍等待依赖。
- 容量任务5157368尚未启动，dev_accelerated等待QOS运行名额；修复后的容量资格尚未验证，正式融合训练未启动。
- 算子支线75/75片、593/593 query准备完成；5157407五折已有3/5通过独立NumPy核验，5157408诊断排队，5157409等待两者完成。
- 自动调度状态心跳持续更新；本次未发现新增运行报错。没有完整的新准确率。
- 本轮核查回执：`H593_TASK_CHECK_20260922T140626Z.json`。


## 当前执行状态（2026-09-22，接续5157186）

- **索引完成**：5157273 COMPLETED 0:0，2分18秒，验证为FUSION_ALL593_CATALOG_PASS。全593个query、每图自然C128，去重后5324个图片/几何条目；复用pilot的129个，新增5195个，未读身份标签。
- **特征提取已启动**：5157341_[0-20]，21片，每片最多256个唯一条目、5分钟GPU；完整性汇总5157342等待afterok。首片256项已通过，Slurm墙钟1分57秒，实际编码约64秒。完整query/reference特征、中间投影、原tokens与几何来源均保留。
- **容量测试修复后重提**：5157314在第一臂的ADAPTER_GRADIENT断言处失败。FREE下M=1，S与L特征相同，Q/R差值为0；原临时测试头S、L系数为相反数，消掉内容梯度（浮点实现可能留下舍入残差）。这是临时测试头的退化设置，尚未开展正式训练。保留V1源码、失败日志和停止状态；V2仅改临时测试head为[-.2,.2,0,.1,.2,.3,.4]，另存结果，不改变正式初始化、损失、预算或划分。修复后的完整128候选容量任务为 **5157368**。
- **自动续接V3已重启**：`programs/dispatch_rc_h593_feature_fusion_v3.py`，记录在`results/rc_h593_feature_fusion_dispatch_v3/`。仅在全量特征与容量验证均合格后，先跑config0的正式训练；首个正常checkpoint回执出现后，按最多46项并行释放其余配置。200配置、每项2000次完整C128 query更新、原五折、两seed、COST1/CE和同预算NO_ADAPTER对照均保留。
- 修复后普通反向与两遍反向测试、优化器下一步恢复测试、三种真实shell启动参数检查通过；提交脚本已与Slurm spool逐字节核对。

**当前没有新的准确率。** 原RAW426/593、COST1481/593、CE486/593属于原H593分组五折结果，不能将工程完成写成模型增益。后续预测全量封存后才join身份标签。本轮是输出token融合，进入ColNomic LLM之前的融合尚不属于已运行内容。

入口：

- 正式训练冻结协议：`plan/RC_H593_FEATURE_FUSION_TRAIN_V2_20260922.md`
- 训练及逐候选中间结果：`results/rc_h593_feature_fusion_train_v2/`
- 实时自动续接状态：`results/rc_h593_feature_fusion_dispatch_v3/status.json`
- 全量特征：`results/rc_h593_feature_fusion_cache_v1/`

以下为此前各阶段执行记录，当前状态以上文及实际validation为准。

## 后续已接入自动执行（2026-09-22）

用户要求继续5157186后续。本轮已实现并提交全量特征准备与完整C128反向容量检查，并配置了从二者合格到正式训练、续跑、预测和汇总的自动调度。

- `5157273`：H593全部593 query与其自然C128联集的去重索引，已迁入dev_cpuonly。复用129个合格pilot缓存；每个新唯一图片/几何条目仅提取一次。索引完成后自动提交每片256图、5分钟的GPU特征数组及完整性join。
- `5157314`：八臂×COST1/CE的完整128候选反向容量检查，15分钟GPU；只用固定合成目标索引，不读取真实身份，工程参数丢弃。
- 自动续接：`programs/dispatch_rc_h593_feature_fusion_v2.py`，运行记录在`results/rc_h593_feature_fusion_dispatch_v2/`，已启动独立后台进程。逐项核验特征ready和容量检查后，先提交config0的真实训练分片；正常检查点回执出现后再最多46项并行。
- 正式配置：八融合臂＋两种NO_ADAPTER对照，COST1/CE、seed17/29、原五折，共200个fold配置。每项2000次完整C128 query更新；15分钟分片，安全保存参数与optimizer后按同一配置续跑。没有使用top-K、降低FP64或减少步骤来适配时限。
- 首次调度脚本V1在启动前发现语法错误，未运行任何调度；保留失败记录，由经语法核验的V2取代。V2另加首个真实训练分片门。源码、授权和动态shell检查均完成。
- 两遍梯度与普通完整反向的八种合成对照最大差1.734723475976807e-18；优化器恢复后下一步逐位一致。独立最终模型文件避免预测续跑时优化器检查点重写导致来源SHA改变。

这是已接通的执行链，**不是200项已完成，也没有新的准确率**。当前具体阶段以dispatcher/status.json、Slurm及各validation共同核查。正式预测全量封存前不读取held身份。调度最多持续48小时、每配置最多128分片，失败会记录并停止，不将缺项记为完成。共享dev名额内自动优先放置本分支的短前置、提取与join任务；可能短暂停止旧CPU补位helper补入新任务（最多20分钟），不暂停任何正在执行的Slurm计算，之后恢复。

新入口：

- 全量缓存计划：`plan/RC_H593_FEATURE_FUSION_EXECUTION_V1_20260922.md`
- 正式训练协议：`plan/RC_H593_FEATURE_FUSION_TRAIN_V1_20260922.md`
- 全量缓存：`results/rc_h593_feature_fusion_cache_v1/`
- 训练、模型、中间量及预测：`results/rc_h593_feature_fusion_train_v1/`
- 自动续接状态：`results/rc_h593_feature_fusion_dispatch_v2/status.json`

以下为试运行阶段历史记录。

2026-09-22 14:49 CEST 更新：`5157186` 已在dev_accelerated COMPLETED 0:0，总墙钟1分29秒。`validation.json` 为 `FEATURE_FUSION_PILOT_FULL128_PASS`；独立核验129个图片payload/validation SHA及最终封存文件。query/reference首个自然配对的16项池化特征hook比较逐位一致，原生可见性一致；完整C128、八臂初始输入与1024项独立评分回放通过，全部原生评分逐位恢复。主worker计时40.924秒（不含独立验证及启动等开销），A100 40GB峰值分配4.105GB。训练更新仍为0，没有新增准确率。下一阶段为全量去重特征提取与训练成本资格检查，之后冻结五折训练配方。

调度更新：用户要求将 `5157186` 移至 dev。已原地迁为 `dev_accelerated`，逐项核实仍为15分钟、8 CPU、64G、1 GPU，原脚本与依赖不变。迁移后首次读回为PENDING；回执：`results/rc_h593_feature_fusion_pilot_v1/dev_migration_5157186.json`。下文accelerated为首次提交记录。

2026-09-22 14:26 CEST。用户授权“开始”，第3组实验开始实施。坐标精度及评分算子任务继续独立执行，未改变它们的模型或程序。

## 已完成

- 四来源：COL_ONLY、COARSE、FINE、COARSE_FINE；各自 FREE / ROMA_WEIGHTED 两种评分，共八臂。
- 使用统一4096参数的低秩输出门控；固定投影统一输入维度，避免可训练参数量随来源增加。输出门控零初始化，仅在评分处归一化。
- 合成检验通过：FP64区域池化与独立直接均值、零初始化保留原评分、上下层有效梯度、inference tensor进入反向传播、可微六特征与旧公式逐位一致、候选轴重排等变性。
- 启动脚本真实shell的参数、含空格路径、离线环境动态检查通过；提交脚本与冻结文件逐字节一致。

## 当前任务

`5157186`：`accelerated`，1 GPU，8 CPU，64G请求，15分钟。提交时PENDING / Priority。

原H593 execution ordinal0的一张query、全部自然C128 reference，共129图像/几何条目；未按标签或成败选图。

任务先从独立单图编码提取DINO两个粗特征层、VGG两阶段三个细特征层，并对首个自然候选执行一次完整RoMa，用hook逐位核对所有池化特征及原可见性。其他图片不执行matcher/refiner，缓存对齐特征和固定投影。随后完整128候选重放八臂初始评分和127×6输入，检查无标签评分的梯度，独立子进程重新核算全部评分与匹配贡献。

**这是工程试运行，不是训练结果。** 当前训练更新数0、身份标签读取数0；不输出或解释准确率。全量唯一图片提取和五折训练尚未提交。通过后依据实际运行成本冻结训练批量/步数，沿用原分组及retrieval-only监督，再启动八臂训练；不得根据held标签选择方案。原RAW、COST1、CE、旧纯内容统计头均保留，另补相同训练预算的无适配器对照。

## 入口

- 实施计划：`plan/RC_H593_FEATURE_FUSION_V1_20260922.md`
- 执行授权：`registry/rc_h593_feature_fusion_pilot_authority_v1_20260922.json`
- 程序：`programs/run_rc_h593_feature_fusion_pilot_v1.py`
- 可微融合/评分：`programs/rc_feature_fusion_core_v1.py`
- 中间数据：`results/rc_h593_feature_fusion_pilot_v1/`
- 提交与验证：同目录 `submitted.json`、`submission_verified.json`、`submitted.sbatch`、`preflight.json`、`launcher_dynamic_check.json`
- 日志：`logs/h593-fusion-p-5157186.out`、`logs/h593-fusion-p-5157186.err`
- 试运行完成标志：同目录 `validation.json` 必须为 `FEATURE_FUSION_PILOT_FULL128_PASS`，不能以Slurm状态替代。
