# new HYP：reference支持最大最小竞争margin的TRAIN工程pilot

继续retrieval-only授权至北京时间2026-09-11 24:00。用户目标是保留当前
28正确并新增正确；本pilot不计为该目标达成，不生成新的EVAL成绩。

## 直接问题

固定RoMa soft support上的SPECIFIC8/FREE8已验证为27，原头28。
在相同free-MaxSim token表示中，固定支持是否错过了可区分证据？
本pilot只检验允许任意非负query支持时的可分性和求解工程可行性，
不是又一个固定头权重/阈值搜索，也不是encoder普遍信息上界。

## 固定自然输入

取原FULL TRAIN按execution升序的前4条：56、66、72、75。每条完整
C128，全部128个candidate都求解，不能按正确label筛候选。原query
image tokens与完整reference-image free-MaxSim a_h(i)沿用已验证
cache，均FP64，768个query token。共512个候选问题。

不读取EVAL图、EVAL角色/结果、容量oracle系数或当前模型checkpoint
以外的新权重；不新增encoder/RoMa forward、SAM/空间标注、superregion、
ownership、D1-MI、GroZi或正式392。原C128轴无714/715等价重复对。

## 固定优化问题

对每个g，h遍历全部127个其它reference，i遍历全部query tokens：

    D[h,i]=a_g(i)−a_h(i)
    primal: maximize t over p≥0, sum_i p_i=1,
            D[h,:]·p ≥ t for every h≠g
    dual: minimize u over alpha≥0, sum_h alpha_h=1,
          (D^T alpha)_i ≤ u for every i

这是query-token关系空间中的支持权重，不要求连通，也不称物理mask。
多个g可能有正margin，不能由此推断唯一真实物体或用户意图。

原/对偶独立调用SciPy1.16.3 highs-ds、threads1、parallelFalse、seed0，
原/对偶可行容差1e-9，每次solver时间上限5秒，presolve开启。数值设置
固定，不以自然数据结果调节。求解器状态不是科学证书。

## 独立可行分布与界

将solver返回的p和alpha中负的数值系数裁为0，按其精确binary64有理数
总和归一化，得到数学上非负且和为1的分布。原始返回值保留。若向量
无正质量或存在非有限值，失败，不把它当作无可分证据。

由原a的精确dyadic端点差（不是先舍入的D）独立重算：

    L=min_h p·(a_g−a_h)
    U=max_i sum_h alpha_h*(a_g(i)−a_h(i))

必须L≤U；宽度U−L≤1e-8才算该候选最优值工程资格闭合。L>0说明
存在非负支持能严格区分该g；U≤0说明这个非负池化类没有正分离margin。
跨0的区间是未决，不按solver近零数值强行分正/负。

若原wq_g有正和，将其也按精确端点归一化，计算固定支持真实margin。
理论最优不低于这个可行分布的margin。保存原J的FP64位值并单独回放，
不能无条件把clamped/浮点池化值等同于精确simplex基线。若原sumwq=0，
候选保留，但不称原clamped J=0为可行simplex分布。

## 封存、预算与解释

先封存全部4×128支持/对偶、精确界、源hash及耗时，再只join这些已用于
训练的身份label作描述性汇总。必须报告全部candidate的资格/正负/未决
数量和耗时，不能只保留正确reference或正margin者。TRAIN目标的界和
固定支持差仅为诊断，不称识别模型得分或群体泛化结果。

总体工作软上限600秒；未完成时保存部分记录并明确INCOMPLETE，不把
超时、数值失败或未决当作不可分。全部512完成且资格闭合后，才评估
是否有理由构建完整label-free支持cache并训练共享读出。独立验证器
重新读取源a、原/对偶返回值、精确分布/界与固定支持margin；不依赖
solver的成功标志或再次选择支持。

CPU1、4G、15分钟Slurm job，run+独立validate顺序执行，一次激活核对后
不持续监控。提交后继续独立工作。所有输出append-only，原模型不修改。
