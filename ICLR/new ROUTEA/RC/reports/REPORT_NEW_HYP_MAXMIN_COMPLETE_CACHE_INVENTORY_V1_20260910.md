# new HYP：完整支持缓存的有界输入盘点

本次仅读取已有源代码和元数据；未读取数组、自然预测、EVAL标签、pilot输出或
scheduler状态，未执行自然LP、训练、缓存生产或提交。以下只为pilot合格后
可能的下一阶段节约重复工作，不授权该阶段。

## 现有输入与复用

| 输入 | query数 | 已存free profile/query | 原读出实际评分候选/query | query token长度 |
|---|---:|---:|---:|---|
| FULL TRAIN | 32 | 128 | 128 | 720或768 |
| FULL EVAL | 32 | 128 | 128 | 720或768 |
| PAIR TRAIN | 64 | 2 | 2 | 720或768 |

全部128条记录的query ID和query-token SHA均各不相同；不存在相同
query-token SHA + 同一有序C128轴的重复。因此当前FULL/PAIR之间没有
可直接合并的整条profile或LP问题。

原则上，相同query-token字节、相同有序C128轴、同一reference-image
bank及FP64 MaxSim算术允许复用完整free-profile矩阵。优化支持若只依赖
该矩阵，也可复用已合格的支持与界；原RoMa权重、RAW及旧C六特征仍须
保留每条记录自身的来源。候选集合相同但顺序不同，不保证HiGHS返回
相同支持字节，不能直接当成同一次确定性求解。

实际token长度不能照搬pilot的768常数：FULL为11条768、53条720；
PAIR为10条768、54条720。冻结pilot无需修改，未来独立实现须覆盖两类。

## all128 PAIR profiles的位置

当前已验证的共享缓存位于
`results/rc_shared_query_target_prior_cache_v1/episodes/*/arrays.npz`。
FULL条目的`a`为完整`[128,N]`；PAIR条目的`a`只有原winner/challenger
两行，例如`PAIR_0316/arrays.npz`为`[2,720]`，原candidate位置另有保存。

[specificity输入程序](../programs/prepare_rc_same_support_specificity_inputs_v1.py)
在PAIR上调用`ReferenceCache.all_maxsim`计算`[128,N]`，但仅在内存中使用。
其输出JSON保存`all128_free_maxsim_sha256`和全部reference来源，未把
PAIR完整profile矩阵写成数组文件。因此不能把该SHA当作可加载缓存。

在本次检查的合格谱系内，已有8,320条(query,reference) profile；完整
128条query各自C128共需16,384条，缺少8,064条PAIR profile。可以保留
已有128条PAIR profile，只补其余126×64条，再逐位验证已有两行和
完整矩阵SHA。全部来自原冻结token，无需新增encoder或RoMa forward。

FULL共享`a`的字节已与
`rc_full64_full_reference_cache_v1`的`a_RIMAGE_same_source`独立hash相符，
不需要再次计算FULL的8,192条free profiles。

## 实际LP数量

沿用当前训练/评价接口，只需：

- FULL TRAIN：32×128＝4,096个game；PAIR TRAIN：64×2＝128个game。
- FULL EVAL：32×128＝4,096个game。
- 合计8,320个game，即独立primal+dual共16,640次LP调用。

PAIR虽然只求原两个候选的支持，每次game仍必须比较全部127个其它
reference；不能把“只评分两个”误写成“只有一个对手”。没有必要额外
为PAIR未参与原训练的126个候选各求支持；那会把总量扩大到16,384个game。

若冻结pilot的512个game全部完成并合格，且未来求解合同完全相同，
其证书可作为其中的512个来源保留，新增7,808个game、15,616次LP调用。
这是条件计数，未检查或假定当前pilot已经完成。新进程证书核验本身不
需要重新调用LP。

## 新列接口与signed margin

现有[七参数特征模块](../src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py)
的`symmetric(a,b)=(a-b)/(|a|+|b|+1e-12)`支持负数；不能为了使用它而
把max-min margin截成非负数、取绝对值，或删除负margin候选。

候选级margin与challenger–winner附加列是两层不同接口。当前
[SPECIFIC8 runner](../programs/run_rc_same_support_specificity_v1.py)
保留原六列，追加一列`symmetric(J_c,J_w)`，得到七特征加bias。
原始signed max-min标量可以存入候选级证据；是否仍采用这一对比算子
必须由后续计划明确，不能把原始margin与其symmetric对比混称同一特征。

symmetric不是保留幅度的恒等映射：符号相反时，不同绝对幅度会被压到
接近±1。LP原始solver目标、选定支持的可达margin L，以及界[L,U]也
不是无条件相同的数值字段。未来必须先明确缓存哪个可执行标量，再
固定唯一附加列；本次没有选择或调整该公式。

若继续原C_BIND接口，新候选证据应沿原整份evidence donor映射转移；
旧六列及RAW来源不改。此处仅说明接口要求，不产生新head或新模型成绩。

元数据锚点：共享manifest SHA
`f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c`。
