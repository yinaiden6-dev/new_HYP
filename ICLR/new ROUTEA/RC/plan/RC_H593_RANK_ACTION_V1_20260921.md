# H593：平方排序增益的嵌套行动验证

2026-09-21，用户授权继续优化。前轮5153850已完成：预定主臂FULL_QUAD27为104/144，没有通过；预定次臂DIAG12为107/144，相对COST1为4救1损，相对同条件CE的RANK_ONLY6为4救0损。该次臂信号用于提出本轮探索假设，不回写前轮主结论。现有最终成绩RAW426、COST1 481、CE486、GAP_BIAS2 492均以完整593为分母。

## 两条固定路径

LINEAR_ACTION3：冻结本外折RANK_ONLY6排序头 → 选最高挑战者 → 三参数行动头。

DIAG_ACTION3：冻结本外折DIAG12排序头 → 选最高挑战者 → 相同配方的三参数行动头。本轮探索主臂。

冻结排序头保留全部127个挑战者分数；禁止用target选择候选或修改分数。每个外层TRAIN在其余四个原fold中依次留一，用剩余三个fold新训练对应排序头，生成TRAIN内层OOF。两臂共40个新inner rank fits，独立新进程完整重训再复算40次。不能复用其它outer-fold头作为inner头，因为那可能读过本外折留出的标签。

inner排序保持原配方：原六维/六维加平方、FP64、无新增归一化、零初始化AdamW lr=.03、weight_decay=.001、2000步；条件127-way CE分母为inner fitting中全部target在场query数，RAW正确贡献0，target缺失排除。外层rank参数直接复用封存结果，不重训、不因本轮结果挑选rank头。

## 共同的三参数行动头

对每条inner OOF预测，top按冻结候选轴第一个最大值选择，计算d=top1−top2分数间隔、r=原六维X[top,0]（该挑战者相对RAW winner的标准化内容分差）。d≥0，r保持原符号。二者均为推理时可见证据。

每臂仅用本外折TRAIN内层OOF全部query学习两项RMS：s=sqrt(mean(feature²))，零尺度置1。没有中心化、截断或outer统计。g=((d/s_d)*a+(r/s_r)*c)+b，以显式FP64逐项运算执行；三个系数无符号约束。g>0时SWITCH到该rank头的最高挑战者，否则HOLD RAW。所有query均允许决策，旧正SWITCH不锁定，0分严格HOLD。

身份检索标签产生delta=I(top正确)−I(RAW正确)，取−1/0/+1。训练目标为Σ(delta≠0)logaddexp(0,−delta*g)/N_all + .001/2*(a²+c²+b²)。全部内层OOF记录包括neutral和target缺失，保留在N和RMS中；delta0对损失分子贡献0。无人工类型、区域、对应点监督。bias从该固定损失中学习，不叠加经验净增阈值扫描。

零初始化、一次无约束L-BFGS-B，maxiter2000、ftol1e-12、gtol1e-8；不选epoch、不扫超参。优化器失败即停止，不静默换配方。独立梯度最大绝对值须<1e-5，否则不接受该折。若全部delta为0，使用全零参数确定性HOLD；只有单一有信息类别仍正常拟合。

## 读出与解释

唯一主比较：DIAG_ACTION3相对共同门对照LINEAR_ACTION3。强性能参照为GAP_BIAS2=492；另列RAW/COST1/CE。所有比较保留完整593张及23个C128缺失。报告实际选中身份的准确数、救回/损失/净增、RAW原正确损失率、切换数、五折分解和原40排序障碍的最终救回数。不把107/144或533的oracle机会记为最终准确率。

内部性能正信号要求DIAG_ACTION3相对LINEAR_ACTION3与GAP_BIAS2均为正净增和正64组件等权方向；bootstrap区间单列（seed20260920，10000次）。不根据外层结果选模型或调bias，不手修4个救回样本。该区间条件于固定预测，不涵盖共享训练或多轮开发选择，不能当外部确认。

本轮若失败，只说明当前冻结rank＋三参数行动配方未兑现最终净增，不擦除DIAG的较窄排序信号；若成功，仍是已打开H593上的开发成果，不自动替换论文主模型，不新增ownership声明。

## 执行隔离与预算

fit仅允许当前外折TRAIN roles、冻结公共split/worker/gallery、无标签feature、当前折旧头/预测；屏蔽其它折roles、curator、容量证书、rc_opened、D1-MI、formal392、外部数据。inner fitting与inner held在身份/组件/原图SHA上隔离；outer标签只在全部五折预测与独立验证封存后join。

独立NumPy核验inner与outer的全部rank logits、action特征/RMS/梯度/分数/strict动作。每折新进程重训重放；汇总新进程重放。源码、计划、输入与两个launcher均SHA冻结。

dev_accelerated数组0–4、并行上限5，每项10分钟、8CPU/16G/1GPU，实际CPU计算。每折包含8个inner rank fits及重放。依赖afterok的汇总任务同分区10分钟。超时仅从已验证折恢复，不能改训练步数或优化配方。自然训练仅Slurm执行。
