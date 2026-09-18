# new HYP：重训证据充分性对照的独立来源复审

2026-09-09。复审范围为现有结果清单、五个固定训练头及其来源/封存代码；
不运行新模型预测、不提交作业、不读取新试验结果。当前研究保持 retrieval-only。

## 现有证据与确实缺失的比较

| 已完成比较 | 原模型/候选/群体 | 结果 | 证据范围 |
|---|---|---|---|
| 单项中和 delta M / delta L | FROZEN_C，RAW C128，opened EVAL32 | 27→25 / 24 | 固定头对两项的计算依赖 |
| 同上 | FROZEN_C，RAW C128，opened difficult90 | 69→61 / 61 | 八个旧救回均失去，不能替代重训比较 |
| 因子一致 M=1，保留原 L/Lq/Lr | 同旧头、difficult90 | 69→61，全部 HOLD | 完整标量 M 对该固定计算图的作用 |
| 原/学习后 M 与 normalized evidence 的 2×2交叉 | 冻结 NATIVE7，matched RAW C128，EVAL32 | 原/原28，其余三格26 | 学习后 prior 的总体负结果；不是单因素重训 |
| COMMON3 C | matched RAW C128，EVAL32 | 26/32 | 特征是 RAW gap 与 delta S，没有分解 M/L |
| COMMON3、NATIVE7 的 A_ALL | 同 matched bundle，EVAL32 | 均25/32，0救0损 | 已有 image-token 无权重内容/action基线，无需重跑 |

本轮检索 programs、src、reports、plan、registry 后，未找到同一原PAIR/FULL
监督协议下重训 RAW-only、RAW+M、RAW+L、RAW+M+L 的既有结果。因此五头
方案补的是边际特征经重新拟合之后是否足够解释当前28/32，未重复原项级
中和或原/学习后因素交叉。本结论限定于上述检索到的当前检索主线来源。

L 仍含 query/reference visibility，不是质量信息完全剥离后的纯ColNomic。
RAW+M 仍读取 RAW 内容先验，亦不等于独立RoMa分类。已有A_ALL只能表述为
原 image-token scorer/action对照，不可写成完整raw passage编码器基线。

## 预定五头与解释

固定原native列：RAW2=[0]；RAW_PLUS_M3=[0,2]；RAW_PLUS_L3=[0,3]；
JOINT4=[0,2,3]；ORIGINAL7=[0,1,2,3,4,5]。数字含bias。PAIR/FULL的
REAL及适用的C_BIND直接选列；原Q/R定义、tokens、地图、RAW/C128保持。

JOINT4保留原28个正确决策且分别胜过两单因素，可作为内部机制简化候选；
不强制超过28，也不能自动升级为普遍或外部理论GO。比较中训练容量不同，
必须披露。同协议下的联合使用收益不等于证明非线性交互或信息论协同。
各单因素与原ORIGINAL7的成败集合、成对净差和supergroup方向都应报告。

## 旧结果SHA绑定

- `results/rc_frozen_roma_action_terms_v1/result.json`：`9af91982d06630da09e84120dce9cdd673d06f0c11cb019750e818dfa59300a6`
- `results/rc_frozen_roma_difficult90_action_terms_v1/result.json`：`53bd8b545d2b51e97b9917c02e779b5c346aa613d4655413fe24fca17dcddd89`
- `results/rc_frozen_roma_visibility_mass_factor_v1/result.json`：`79c95466ed0edfcd0fa34024baae71e16362e5595570456a7c5c497745300fb1`
- `results/rc_shared_query_prior_mass_content_v1/result.json`：`bfd4fe4276edb363738cf6bf9b8ed2e0ab6effb4f8db973eef9e64a9be742636`
- `results/rc_shared_query_prior_mass_content_v1/validation_json_compatibility.json`：`9a481d2a685fa34e4598323654e06ef3eaebb13e85c85fabf1186f656efceda4`
- `results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json`：`591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3`
- 同目录 `independent_validation.json`：`1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df`

## 新源码复审

已读取计划、runner、launcher及其复用的输入、训练、角色屏障实现，
未发现阻止本轮预检/执行的源码问题。此结论是独立静态源码复审，不是
训练结果或动态预检通过声明。

- 五组原列与计划一致；PAIR只消费原REAL，FULL REAL/C_BIND都直接选列。
  每组数据同时用literal逐元素索引和逐列hash核验；ORIGINAL7保持原张量。
- 原PAIR64与FULL32样本顺序、原C标量/六特征经父输入closure精确核验。
  新头没有消费父工具顺带计算的ABS/REL扩展列；没有改变Q/R控制定义。
- 通过同一frozen pure train_head调用各头；该原函数每头重设seed17、FP64
  全零初始化、AdamW及2000次完整PAIR/FULL sign更新。ORIGINAL7参数与
  原NATIVE7 seal逐项binary64相同是强制断言。
- 输入helper的ReadBarrier阻止EVAL角色以及旧结果读取，允许的是明确的
  hash-only来源核验。五头参数、FULL64两条件的全部127 logits写入封存后，
  summarize才释放屏障并读取EVAL身份。屏障拒绝计数必须为零。
- EVAL ORIGINAL7的REAL与C_BIND完整动作列表必须和旧NATIVE7列表精确相同。
  TRAIN/EVAL query、identity、supergroup不相交再次断言。所有预封存预测
  在postjoin重新计算并比较。
- 两个预定主比较各自要求query净差与group等权准确率差均为正；保留原28
  使用对ORIGINAL7的break==0且net>=0，不能靠相同总数但替换正确集合过门。
  五头结果、所有预定比较、成败query、group方向、C_BIND保留均输出。
- validate在新进程重建全部源特征、重训五头、重新计算预测和结果，逐项
  对照；不把同程序fresh-process重放称作独立的新学习方法。
- launcher为dev_cpuonly、8CPU、4G、59分钟、NIL环境，明确PATH与TMPDIR
  fallback、禁GPU且离线；run和validate各自检查用户截止时间并有timeout。
  无requeue、无后台监控循环。动态NIL启动由主agent预检负责。

本轮C_BIND沿用原bundle的既有干预定义，仅验证和消费原特征，不把它
重新描述成删除了全部质量/内容的干预。尤其质量单因素臂可能按定义对
该控制不变，应按该原算子解释结果。

复审绑定的新源码SHA：

- `plan/RC_RETRAINED_EVIDENCE_SUFFICIENCY_V1_20260909.md`：`b577695ce9c19941e99e06a3b2ba7bf340b901af39ac19da823a8e095a12942e`
- `programs/run_rc_retrained_evidence_sufficiency_v1.py`：`1ef0043c59fc718754f257ea92d31035ecbe5cecc47ebdde66cbc162c4a5679e`
- `slurm/rc_retrained_evidence_sufficiency_v1_dev_cpuonly_59m.sbatch`：`f0f28d463c1bb610c731af135cfe2df26981695ca4df95d250aa2549f757c9e4`

## 首次动态预检后的单行修复复审

首次预检暴露原hash helper处理长度1、stride为6的PAIR单列视图时，
contiguous后仍可能无法view(uint8)的问题。该阶段尚未训练；主agent报告
在正式产物写出前中止。本复审不将首次预检失败解释为科学结果。

独立检查确认runner只改一行：逐列hash比较改为按原列顺序比较每个
FP64值的float.hex字符串。整矩阵与literal矩阵hash核验仍保留，ORIGINAL7
原矩阵hash核验仍保留。将该单行恢复后可精确得到先前复审的runner SHA，
证明没有顺带改变特征、模型、标签、loss、optimizer或比较门。

修复后有效runner SHA为：
`783c7f04ed269cbafd89c2de183418abdc665589d1d36eb4813aaf5d41be2027`。
前述plan及launcher SHA不变。单行来源修复复审通过；第二次动态预检
结果由主agent负责核验。本复审未另行执行测试或新模型预测。
