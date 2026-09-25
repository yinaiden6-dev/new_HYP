# 联合旧EVAL32/新EVAL128的严格线性保持容量诊断

这是已打开EVAL标签感知的数学容量诊断，不是训练/验证新模型。本轮不输出新准确率、可部署checkpoint或HYP GO。所有端点和证书只保存到 results/rc_opened_joint_eval_linear_retention_capacity_v1；现有/rc_opened_路径禁读保护继续适用于后续训练。任何witness/dual系数不得进入训练、阈值、样本选择或部署。

## 固定问题

固定原ORIGINAL7/NATIVE7 ec7的六个特征（RAW gap、dS、dM、dell、dQ、dR）定义与FP64输入端点，允许一个共享实数线性头的六个系数和bias自由改变。原RAW winner策略分数仍为0，候选是每图原自然C128，完整127 challenger；不加特征、不改变候选轴、不插入target。

同时保留旧EVAL32的原28正确与新EVAL128的原99正确，共127条不同query。对新128的原29错中目标自然在C128的22条，按原execution ordinal逐条加正确动作要求；每个系统包含上述127条加当前1条，共128条、16256个不等式。另7条召回缺席保留在人口账本中，不能被本次rerank变量修复。其它未选中的原错误不添加约束；22系统全部报告，不找到首个可行就停。

本问题不是旧32-only保28增一、旧32任意29子集，也不是固定原排序的一维GAP容量。不得复用那些旧证书直接声称覆盖新系统。

## 数学与边界

每个相关query必须在C128中有唯一正确候选位置，否则本轮停止为范围未决，不能静默把身份正确的析取条件收紧为任意单一target。

特征加bias记phi，参数theta共7维。只检验严格正action margin；有限样本共同缩放为单位margin：
- target为RAW winner：所有127 wrong满足 -phi_wrong·theta>=1。
- target为challenger：phi_target·theta>=1；其余126 challenger满足 (phi_target-phi_wrong)·theta>=1。

先核验原ec7的精确有理theta对联合127保护query具有严格正margin，确保原基线和构造一致。所有差向量都从原binary64端点的Fraction精确差计算，不把已经舍入的差作为原始证据。

浮点LP仅搜索；可行结论需要七维有理witness对该系统全部约束精确>=1。不可行结论需要非负有理Farkas权重、权重和为1、七列加权和精确为0。数值超时、数值可行但精确margin不正、或无精确dual都记UNRESOLVED，继续完成其余系统，不用solver状态当证明。

严格不可行仅排除严格正margin实数线性类，不排除实际HOLD=0、physical-row ties或FP64特殊舍入解，也不排除非线性头、其它表示、允许部分原正确损失的总体改善。可行只表明这种严格决策在有限输入上存在，不保证TRAIN可以学到或泛化。

## 输入与独立验证

使用已独立资格化的统一readout结果获得原正确集合/target/recall；旧32用其authority绑定的shared EVAL metadata/arrays及原四标量重建X，新128使用其prejoin records六维端点，并从原四标量重建核验。所有160条原ec7动作和20320个原logit逐bit重放；原28/99及RAW25/88都必须回归。

输入模块显式列全部source hashes。首次run前冻结producer/validator/loader/plan/旧helper与源资格文件的authority。writer只写隔离输出目录；不得读D1-MI、GroZi、formal392或任何新图像/encoder/RoMa。基线来源只读，原主模型和此前结果不修改。

新fresh validator不调用solver；独立从端点逐条构建Fraction约束和顺序，核验全部22个证书及原127严格margin，保存源绑定、结果SHA与进程nonce。原数据重建可复用已验证的输入方法；原代数约束重建与证书验证必须独立于浮点solver。

## 执行预算与收口

CPU Slurm运行，不占GPU。最多22次完整certify调用（每次既有primal30秒、必要时dual30秒上限），遇到非数学工程异常留原产物并报错；数值未决单独记录。Slurm最多30分钟，用户总研究截止北京时间2026-09-11 24:00（UTC16:00）；启动器和程序均检查截止。合成预检只用合成不等式，不读取自然EVAL端点/标签。

任一精确可行说明原线性类尚有这项严格保持目标的表达空间；不得从witness抄参数。若22条全部精确不可行，才可排除这个严格增一目标；若尚有UNRESOLVED则结论未决。完整结论进入报告与当前状态，不把capacity结果或EVAL标签拟合系数写成超过99的检索成绩。
