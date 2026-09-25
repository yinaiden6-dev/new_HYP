# TRAIN128：围绕原七参数头的保护式投影训练

本计划不使用EVAL容量证书的系数、方向或样本选择。只保留“原线性类并非完全没有严格增量空间”这一已打开数据的诊断结论，独立测试一个TRAIN检索监督学习规则。所有rc_opened_*输入（包括hash）均在拟合进程中禁止；EVAL仍不开放。原ec7参数及旧28/新99结果保持不变。

## 固定数据与模型

使用已有TRAIN128、32正例身份/组、同一固定四折（各留出8组，图数35/27/32/34）。复用上一OOF已独立重训/封存的每折BASE7；每个基头原先只使用其他组的原FULL32/PAIR64。每折重新调整也只读取该折其余24组的全C128 TRAIN六维特征与身份标签。

输入仍是原六个native features，不增加F/A/D或新图像特征。共享参数theta=(六个系数,bias)共7个，原RAW winner策略分数0、自然C128、全部127 challengers及physical-row并列规则不变。该训练允许候选排序改变，不等同于已经排除的一维GAP。

四折已在开发中复用，不称未触碰独立测试；推理不读取query身份、正确标签或训练修复例ID。

## 训练候选构造

每折用固定BASE参数theta0在该折TRAIN全部query上计算原动作。令P为所有原BASE正确query，E为所有原BASE错误且target自然在C128的query，保持原execution顺序。缺席target不插入，仍计入训练报告。

从原binary64端点和theta0用Fraction构建完整action margins，取P的最小值delta。P为空、delta非正、或转成LP binary64后非有限/非正，则两臂均精确返回theta0，并记原因，不自行修改地板。

每个e只尝试一次LP，每臂恰好一轮完整E枚举。两臂共用delta、基头和候选顺序：

- PROTECTED7（primary）：要求P及e各自全部127个正确action约束在数值LP中达到delta。
- REPAIR_ONLY7（control）：仅要求e的全部127个正确action约束达到同一个delta，不加入P。

正确action的线性形式：target=RAW时，-phi_wrong·theta为margin；target!=RAW时，phi_target·theta及(phi_target-phi_wrong)·theta为margin。不是将全部wrong强制压负的额外目标。

每次LP目标为最小化||theta-theta0||_1，用Delta_plus/Delta_minus非负变量，theta=theta0+Delta_plus-Delta_minus。固定SciPy HiGHS dual simplex、行列顺序、presolve、30秒、primal/dual feasibility tolerance 1e-9；不用EVAL选择求解器或再试不同目标。

## 数值结果验收和TRAIN选择

LP只是数值搜索。delta是搜索的正margin目标，不宣称数值最优点按FP64保存后精确达到delta。保存的theta按固定FP64算序构造，记录实际L1和required rows的精确最小margin；从原端点Fraction核验required rows严格>0，再重放实际FP64全127动作。e必须修复；PROTECTED还必须完整保留P。失败则拒绝为数值/动作不合格，不称数学不可行，不调整delta或容差重试。

原theta0始终在候选池中。合格候选及原头在全部该折TRAIN上评价，按以下顺序唯一选择：
1. TRAIN各来源组准确率的Fraction等权均值最大；
2. TRAIN总正确数最大；
3. 最终保存FP64 theta相对theta0的Fraction精确L1最小（不使用solver.fun代替）；
4. 原execution ordinal最小；原theta0在完全相同指标/L1时优先。

无合格修复时原theta0逐bit返回。所有候选状态、数值失败、原错误未在C128、实际训练保持/新增及selection账本都保存。只选择预定有限候选池中的最好者，不声称在整个七参数类中最大化正确数，也不声称LP最优点唯一。

保护的是TRAIN正确性，不是每个原margin完整保持，更不是留出数据无损保证。delta低于许多原margin，训练调整可能降低这些margin；这正是OOF需要检验的泛化问题。

## 封存、复核与门

每折source/参数/全部heldout127 logits封存；fresh进程重做同一LP流程、候选选择与FP64参数/预测，要求bit一致。基头的heldout原127 logits须与旧封存BASE逐bit回归。四折全部合格后才读取heldout标签，完整汇合128query。

primary性能门：PROTECTED相对BASE总正确数严格增加，且完整保留OOF BASE正确集合。等权组平均增益须>0，作为前两项所推出的一致性检查与效应量，不当三份独立证据。REPAIR_ONLY并列报告训练保护的代价/作用，不因为它的准确率更高就自动更换primary；也不把战胜它设成额外的用户性能门。

报告三模型correct、RAW救回/损失、相对BASE/控制的paired变化、全32组与每折结果。固定seed20260910、10000次组bootstrap及精确双侧组sign-flip，p只描述。若OOF未过，停止本配方、不读EVAL、不改超参重试；若过，也需另行冻结全TRAIN拟合与同一旧/新EVAL验证，不能把OOF成绩写成超过99。

## 研究与执行范围

最小参数变动满足margin约束的思想可参考Crammer等的[Online Passive-Aggressive Algorithms（JMLR 2006）](https://www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf)。本试验改用L1、加入原正确集合保护并作有限候选选择，不是原PA算法复现，也不继承其理论保证。

所有学习仅用TRAIN query/reference身份关系，无任务框、mask、点、人工crop、SAM或新encoder/RoMa。基础模型预训练如实披露。CPU Slurm单次最多59分钟，所有程序与launcher遵守用户截止：北京时间2026-09-11 24:00（UTC16:00）。无D1-MI、GroZi、formal392推理或ownership；所有EVAL容量证书目录禁止作为训练输入。
