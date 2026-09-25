# new HYP：TRAIN group jackknife 设计与源码复核

**复核结论：受审设计和当前源码具备提交条件，未发现训练子集择优、评价轴
缩减或提前读取 EVAL 标签的阻塞问题。此结论仅是设计/源码复核通过；52 次
实际拟合及新进程重训尚须由正式运行验证，不是科学 GO。**

## 受审版本

- 计划：`plan/RC_TRAIN_GROUP_JACKKNIFE_STABILITY_V1_20260909.md`
  SHA256 `b844ee287731af3c3787aa2ff1d122845ff7c4bc6be3c461fd23c8cf93b4960f`。
- runner：`programs/run_rc_train_group_jackknife_stability_v1.py`
  SHA256 `d19d96ec09538d3ddea49b52af820453c6d990e6b08b58e0191e66e32e9ea3a5`。
- launcher：`slurm/rc_train_group_jackknife_stability_v1_dev_cpuonly_59m.sbatch`
  SHA256 `8c934ecbec96d79bcea71363264c9538897a77171845315dab078d03aee6bc10`。
- 零更新预检：`results/rc_train_group_jackknife_stability_v1_preflight_v2/result.json`
  SHA256 `8dae66ec90bfefd4da247aa32ce6002b85a70cd19a6a92fa902d2dc659b0aecc`，
  状态 `TRAIN_GROUP_JACKKNIFE_PREFLIGHT_PASS`，训练更新 0，运行时 EVAL
  target role 读取 0。旧预检保留。

本复核读取上述文件及原冻结训练/action 函数；没有启动训练或查询调度器。

## 1. 设计是否挑选了有利训练子集

没有。`prepare()` 从既有 FULL TRAIN32 取得全部 12 个 supergroup，按名称
排序，建立 FULL_TRAIN 基线和全部 12 个逐组删除条件。删组计数加总必须为
32，每个删除条件均非空且保留非空训练集。由预检确定的删除规模依次为
2、1、3、1、4、1、3、6、1、3、2、5；剩余 FULL TRAIN 为 26–31 条。

全部 PAIR64 保持不变，原排序由父输入闭包绑定；每个保留 FULL TRAIN 子集
仍保持原 execution 顺序。原 EVAL32、C128 和全部 127 个 challenger 保留。
不存在按 EVAL 成绩挑选删除组、设置额外阈值或采用最高分删除模型的分支。
结果明确 `best_deletion_selected=false`、`model_adopted=false`。

## 2. 训练与基线是否可比

四头固定为 JOINT4、PRODUCT5、RESPONSE6、ORIGINAL7，直接调用原冻结
`train_head`，没有另写损失或优化器。每次均为原 FP64、全零初始化、seed17、
2000 步 AdamW；原 PAIR loss 和剩余 FULL TRAIN 的 mean sign loss 相加。
13×4=52 次拟合；新进程验证再做完整 52 次，共 104 次拟合执行。

FULL_TRAIN 的四头**完整参数记录**必须与父试验一致，包括 binary64 参数、
损失记录、有限性计数及原函数 hash。其 FULL64、REAL/C_BIND 的全部 logits
和预测也必须等于父 `eval_prejoin.json`；封存后还需验证 TRAIN/EVAL 全部
action 字段及指标与父结果一致。删除条件不错误地套用“参数应等于基线”的
断言。上述回归是正式运行的失败即停止条件，不表示预检已完成这些重训。

## 3. 标签封存与指标复核

全部 52 次拟合完成后，先生成所有条件、四头、FULL64、REAL/C_BIND 的
完整 127 logits，封存参数和预测 hash，再释放 EVAL 角色读取限制。父
factorial 结果/验证文件及本轮结果/验证文件均加入读取屏障；封存前只允许
参数和不含 outcome 的预测回放。

`literal_metric()` 从封存的十六进制 logits 独立选择最大 challenger，
复核 physical-row 平局规则、严格大于零才 SWITCH、最终排名及 top-1/MRR。
新增 `checked_paired()` 使用正确 query 集合差及分组交集，独立复核
rescue/break/net 和分组计数；等权 group 差以 `math.fsum` 交叉检查到 1e−15。
来源动作/指标是逐项精确回归，这一容差只用于不同求和顺序的组平均复核。

更新预检仅在 TRAIN 上用已有四头参数执行上述检查，没有新拟合或 EVAL
目标读取。正式新进程还会重建输入、重训 52 次、逐项复核参数/预测及整个
结果。12 条件的稳定性聚合使用受审统计代码重复计算，不应另称为第二套
独立统计实现。

## 4. 稳定性统计是否会造成伪重复

代码把 FULL_TRAIN 单独作为基线，所有删除稳定性分母严格为 12。全部
四条 factorial 边都输出 query net 和等权 EVAL group 差各自的正/零/负
次数及 min/median/max，不挑选有利边。每个 query 分别输出：

- 正确频率；
- HOLD/SWITCH 决策一致率；
- physical candidate 一致率；
- corrected reference identity 一致率。

这些量不能互换：两个条件都 SWITCH，不代表它们选择同一 reference。
删除条件的 TRAIN32 全集诊断与实际保留训练子集指标也分别记录。

统计范围恰当地限制为**当前 FULL TRAIN 组删除敏感性**。12 次删除共享
大量训练数据并使用同一 EVAL32，不能当作 12 个独立测试集，不能从方向
次数套用独立样本显著性检验。PAIR 的 20 个组始终未受扰动，因此该设计
没有证明对 PAIR 训练池变化稳健。

此外，删除大小不同，剩余 FULL loss 会重新平均。设原 FULL 平均损失为
F、删除 r 条的组平均为 F_g，则剩余平均为

    F_remaining=(32*F−r*F_g)/(32−r).

PAIR 平均损失的总系数不变。这是删除数据及重新分配剩余样本权重的联合
敏感性，不是无穷小影响函数，也不能将不同大小的删组视为等幅扰动。

## 5. 执行和结论边界

launcher 为单个 CPU 作业，8 CPU、4G、59 分钟，顺序 run/validate；NIL
环境显式设置 PATH 和临时目录，研究截止时间另外生效。约 30 分钟只是由
已知四头运行时间推算的预算，完成状态仍以实际输出及独立验证为准。

若删除训练组后结论方向变化，应报告机制解释的训练依赖；若方向保持，
可报告当前内部流程的敏感性结果。两者都不能被转写成已获得新的独立人群
确认，也不能将某个删除模型的高分采用为新成绩。当前结果的模型和候选
来源仍是原 RAW C128 / NATIVE7 系列，不能与旧 FROZEN_C difficult90 混称。

未引入新特征、模型 forward、空间监督或 ownership；本次复核无需改变
当前部署模型或已有科研结论。
