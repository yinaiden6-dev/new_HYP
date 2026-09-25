# 当前分支：局部小头拟合与泛化隔离

## 最终完成：三头逐图答案全部不变

5142672四折COMPLETED/0:0/70秒，5142673 COMPLETED/0:0/41秒。MEAN109→109、CURVE111→111、JOINT110→110，全部128张图的selected reference、correct、target rank逐项不变；97,536 logits独立核算及精确损失界重放通过。11/12项达到1e-6预设近最优阈值，JOINT第1折剩余界3.219e-6，因此不能写成全部达标。全部旧头的盒内最优损失差距上界最大6.7053e-5。

本轮说明继续最小化当前CE没有新的决策收益，不能证明整个模型类的准确率上限。JOINT对同协议CURVE仍0救1损；两折JOINT训练CE更低而留出CE更高，没有稳定跨组优势。输入信息、读取方式、目标与组分布尚未被唯一归因。

全部任务完成，无需监控。完整报告reports/REPORT_PAIRED_CE_OPTIMIZATION_ISOLATION_V1_20260912.md；结果及result_validation.json、completion_analysis.json位于results/rc_paired_ce_optimization_isolation_v1/。结果SHA256为6b9af296ea69d07f6675cfb287715624fa644f7c4ab7bfd202e6ed5403a803be。没有新EVAL成绩或HYP GO。

---

最新启动复核：5142672_0、1、2、3已全部在accelerated/hkn0520同时RUNNING，首次日志检查无错误；5142673正常等待完整四折Dependency。尝试为第0折增加dev分区时调度器已启动该任务，因此未执行修改，所有任务仍沿用原提交资源和程序。记录fold0_partition_attempt.json。

2026-09-12用户jiu/继续，延续有界TRAIN研究。上一轮已经完成：TRAIN128原四折RAW86、BASE108、MEAN109、CURVE111、JOINT110、GLOBAL114；JOINT对CURVE0救1损，没有独立优势。旧EVAL从未因该110改变。

本轮保留相同输入、相同2/3参数函数类、原四折与原BASE7，分别对MEAN2/CURVE3/JOINT3的FULL-C128 CE数据损失做固定[-64,64]盒内优化，检查数据拟合空间及其跨组效果。旧参数不重训；新优化点从旧点启动。原七参数成本4凸诊断已做过，此处针对不同的新三参数CE，不冒称此前没做优化检验。

重要口径：这是不含额外正则的CE数据损失，不能说成继续原AdamW的decoupled decay过程。更低训练损失不自动证明训练算法更好。盒内证书也不证明整个参数域、统计最优或HYP成功。

程序以L-BFGS-B提供候选点；以分母2^52的精确概率、Fraction仿射系数、50位mpmath区间熵和logsumexp证明界。每个头保存旧点/新点目标区间、新点全盒下界和旧点差距范围。仅剩余界≤1e-6标记固定盒近最优；未达则记录未决，无自动换优化器或扩盒。新进程重放证书并独立Torch/NumPy核算全部损失/梯度和127 logits/动作。

已通过合成预检：完整128项CE梯度、精确概率总和、区间证书重放、全盒下界、合成近最优证书。冻结后已提交：

- 5142672_[0-3%4]：accelerated，4折并行，每折10分钟，分别优化三头并独立核算。
- 5142673：dev_accelerated,accelerated，10分钟，afterok:5142672_*；四折预测完整封存后join原留出标签，报告旧/新模型、每折TRAIN/HELD CE、5412身份排名、救/损/净及组不确定性。

提交时fit等待调度，join正常Dependency。两份Slurm实际spool与冻结launcher逐字节相同。无新准确率，禁止把提交或数值证书当成HYP GO。

计划：plan/RC_PAIRED_CE_OPTIMIZATION_ISOLATION_V1_20260912.md。

Authority：registry/rc_paired_ce_optimization_isolation_authority_v1_20260912.json；SHA256 42dc8ed1a516a5e1a5038bd126d25a3d4f6111f6fa075b70daf812c6502cc279。

程序：programs/run_rc_paired_ce_optimization_isolation_v1.py；汇总：programs/collect_rc_paired_ce_optimization_isolation_v1.py。

结果目录：results/rc_paired_ce_optimization_isolation_v1/。先读submission.json；四折fold00..03/{payload.json,validation.json}；完成看result.json/result_validation.json/report.md。该128始终是原TRAIN128分组OOF，不是旧99/128的EVAL。

判断固定：训练损失下降且留出净增，才支持本配方存在可利用拟合空间；训练下降但留出不改善，说明最小化此数据损失不足以实现跨组技术提升；差距证书很小，才限制该盒/目标内继续优化的空间。不从其中任何一项推出真实图没有身份线索或new HYP整体不可能。不自动追加新特征/新门/EVAL试验。
