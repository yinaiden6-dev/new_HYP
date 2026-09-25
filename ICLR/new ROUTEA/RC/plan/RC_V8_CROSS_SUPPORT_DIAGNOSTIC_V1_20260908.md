# V8 固定区域交叉解释诊断

2026-09-08。用户要求继续解决反复 NO-GO；本次只做 opened TRAIN32 原始失败的
机制诊断。V8 独立验证已完成，REAL 9/32 的 NO-GO 不变。这不是 V9、训练、
正式评估或新的识别结果。禁止读取正式 392 的 P/RoMa 结果，不进入 V、action、
HOLD/SWITCH、ownership，不读改或依赖 D1-MI、GroZi 或其它任务。

## 要回答的问题

保存的 V8 决策显示，20 个 target-H1 失败中，19 个 target 区域比 strongest-wrong
区域大；15 个 wrong 区域严格包含于 target 区域。大小相关性不能证明原因。
本次固定参数、对应、原 selected H 与原候选轴，测量：错误来自不同观测区域
之间的比较，还是冻结读出在相同 query 内容上也不能区分这两个 reference？

## 唯一预先指定的计算

读取 V8 最终 update 512 REAL head；不改 FP64、score、对应或 family。
对全部 32 query、每条原 C128 的每个原 selected H，分别用同一 query source
cell 集合计算全部 C128 reference 的原 V8 coherent energy 与 K=exp(-E/2)。
owner-H0 保留为 UNAVAILABLE。任何一个所需 query cell 缺少该 reference 的
有效原 REAL RoMa assignment，则整个 cross 项为 UNKNOWN，并保存缺失数量；
不取交集、不删点、不插值、不使用 appearance argmax，不把 UNKNOWN 填成分数。

交叉区域不一定是 scoring candidate 的合法 maximal component。因此 cross K
仅是受控读出，不是新的合法 proposal、候选 score 或 HYP 成功。逐个 owner 的
对角项必须逐位重放原候选分数、区域和 physical pooling witnesses。不能修改
原 ranking，也不能在交叉矩阵上重新选择 H 或 winner 作为新准确率。
共同 query 支持仍保留 reference 各自的 assignment 与分组测度；保存各 cross
项的 reference-group 数、mean/max 成本，便于解释，不能把同域读出失败等同于
图片或原始特征没有身份信息。

producer 不读 target。先写完 32 条完整 cross 矩阵、核验原输入闭包并封存，
然后才使用原已打开 TRAIN32 postjoin roles。每 query 保留原最佳 target T、
原 strongest-wrong W，包含全部成功、失败及 H0，不筛选案例。

## 固定配对分解和全 C128 检查

令 H_T/H_W 为原区域，m=K_T(H_T)-K_W(H_W)，
a=K_T(H_T)-K_W(H_T)，b=K_T(H_W)-K_W(H_W)，
s_W=K_W(H_W)-K_W(H_T)，s_T=K_T(H_T)-K_T(H_W)。
以保存 binary64 的精确有理数验证 m=a-s_W=b+s_T；另存可读浮点值。
s_W/s_T 可以为负，因为 cross 区域未必属于该 candidate 的合法 family。

原失败 m<=0 时，预先指定互斥诊断类别：

- 必需的任何 cross 项不可用：UNKNOWN，不计作任何正向证据。
- a>0 且 b>0：目标在两块共同观测上均胜，但原不同区域比较失败。
- a<=0 且 b<=0：冻结读出在两块共同观测上均未区分出目标。
- 其余：区分方向依赖观测区域。

另在固定 H_T 上统计全部 C128 的已知 blocker（wrong K>=target K）及 UNKNOWN。
存在 blocker 足以否定此区域上的全候选优势；没有 blocker 但存在 UNKNOWN
仍为 UNDETERMINED；完整可比且没有 blocker 才能称此 label-assisted 固定区域
具有条件性全候选优势。任何上述数量都不是新系统 accuracy/rescue 或 GO。

## 执行与验收

隔离 worker、authority、results 和单个 CPU 作业；4 CPU、64 GiB、59 分钟上限，
torch 单线程，无 GPU，无训练。提交前完成缺失 fail-closed、source-axis、
分解、共同观测反转和 artifact 篡改拒绝的合成检查。文件 SHA 绑定原 V8
authority、result、独立 validation、checkpoint、scores、adapter、core、runner、
本计划、worker 与 launcher。已存在产物只允许相同字节重放。

完成后保存每 query 的全部 cross scores/energies、缺失数、source H 和源绑定，
独立 artifact validator 检查哈希、全轴、对角项、postjoin 统计和精确分解。
该 validator 不冒称重新读取 raw residual 计算全部 cross energies。

按用户要求，只做提交后的单次激活核查，不循环监控。没有后续独立工作则退出。
本诊断不自动触发任何拟合。只有证据支持某个具体缺陷，才另写科学后继；
若同域读出仍失败，不能继续仅凭区域大小调阈值、尺度、温度或 top-K。
