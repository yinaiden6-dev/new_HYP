# new HYP：同一query支持上的reference竞争证据

目标保持不变：保留当前NATIVE7的28个正确，并新增正确；既有结果不
被新候选覆盖。继续授权至北京时间2026-09-11 24:00，提交后继续工作。
任务训练只用原检索身份/配对，无SAM、bbox/mask/point/crop teacher、
superregion、ownership、D1-MI、GroZi或未授权正式392。

## 依据与实际差别

5139021已独立验证：MOMENT8保留原28但无新增；CURVE8为27。离散度
没有在这个训练/评价中带来新增正确。下一项不再追加普通分布矩，
而检验更直接的reference区分线索：reference g提出的同一份soft query
支持上，g是否比其它reference解释得更好。

旧V6已经使用“先聚合、再与最强竞争者比较”的思想，不能声称这一步
新颖。它使用legal四格seed、RoMa硬对应centered cosine和P-only选择。
本轮使用原soft wq、自由完整reference-image MaxSim，并保留原C六特征
及RAW action，属于不同输入/职责的受控桥接，旧失败结论不改写。

## 固定新统计

C是每条原始自然RAW C128，q_i为原query-image token。对每个h∈C：

    a_h(i)=max_j cosine_FP64(q_i, r_hj)

j遍历原reference完整image slice [4:-7]，不加入prefix/suffix上下文，
不使用RoMa指定的单token硬对应。对每个被评分候选g，原wq_g固定，
所有h使用同一份g的query权重：

    A[g,h] = torch.sum(wq_g * a_h) / clamp_min(torch.sum(wq_g),1e-12)
    F_g = A[g,g]
    J_g = F_g - max_{h in C, h != g} A[g,h]

规范数值次序为逐support g、逐reference h的原FP64乘法及1D torch.sum，
再除同一g的分母；FREE与J必须来自同一个A的diagonal，不能各自重算。
这是新统计的直接加权均值定义，不假称逐bit等于旧B.score/B.mass或D/M。
旧C四标量及六特征依然必须逐bit保持。

先按完整支持聚合、再在127个其它reference间取最大，不可先逐token
取各自最强其它reference后合成一个并不存在的“组合错误reference”。
没有top-K截断、温度、可调阈值。原C128全轴不变；当前全部轴均没有
同时包含冻结的唯一等价physical pair714/715，需断言，故self排除与
排除同身份reference无冲突。最强其它reference若平局按最小physical
row记录，max值不依赖该诊断平局选择。

J>0只表示g在g的soft支持上胜其它reference；不同g的支持不同，多物体
query可同时支持多个g。因此J不单独决定用户意图、不声称空间定位或
ownership，继续保留RAW先验和原C校准。

## 两个新头与旧头

ORIGINAL7：原native6+bias，原参数与REAL/C_BIND动作须精确回放。
SPECIFIC8：原六列不动，附加symmetric(J_c,J_w)。
FREE8：原六列不动，附加symmetric(F_c,F_w)，作同参数数的控制。

symmetric为原(a−b)/(|a|+|b|+1e-12)。两头仅区别是否减去同支持下
最强竞争reference。相同参数数不保证完全相同函数类或有效容量。
FREE8不是旧D_IMAGE的重命名：它追加独立身份读出，不替换C的质量、
内容和Q/R响应列。

## 完整输入与必要桥接

FULL64已具有完整128条a_h向量；PAIR64原cache只有winner/challenger
两条，必须用原q与冻结raw-gallery的reference-image tokens补齐全部
128条a_h。不得在PAIR用2候选均值/max而在FULL用128。无需为缺少的
reference执行RoMa，只计算已有embedding的FP64相似度。

旧C标量/六特征从原输入回放。PAIR原q/r混合shift保留。新F/J的C_BIND
按与原C四标量相同的candidate donor映射整体转移，RAW不动。追加项
单独EXTRA_BIND诊断取REAL旧六列+CBIND的新列；ORIGINAL7此条件应与
REAL逐bit相同。没有新训练head、模型选择或像素干预。

标签自由E0：support权重(1,1)，g的a=(.75,.75)，两个其它reference
分别为(1,0)、(0,1)，其它列0。F_g=.75，J_g=.25；错误的逐token竞争
合成会给−.25。检查self排除、完整127其它列、共享g权重和确定性平局。
这只检验算子，不证明自然识别收益。

## 原训练与结果判据

原PAIR64+FULL TRAIN32、原顺序、原FP64全零初始化/seed17、AdamW
lr=.03、weight_decay=.001、betas=.9/.999、eps=1e-8、2000步、原PAIR
加权BCE+FULL sign loss不变。推理仍全部127 challenger、原零阈值
HOLD/SWITCH。所有参数与FULL64全部条件预测封存后才能join EVAL。
新进程完整重训、重建输入并验证所有输出；任何EVAL容量证书目录
均不允许进入模型输入/训练。

主比较SPECIFIC8对ORIGINAL7；预定对照SPECIFIC8对FREE8、FREE8对
ORIGINAL7。报告TRAIN/EVAL、paired救/损/净增、MRR、等权group方向、
RAW损失以及两个控制的新增救回保留。满足用户目标必须：相对原28
净增>0且break==0，同时group平均差>0。正净增伴随损失另列，不能
称满足本轮目标。若只有FREE8提高，不宣称竞争项有效；若两者同样
提高，也不能独占归因为J。旧空间门、外部确认和部署不自动放行。

CPU8/4G/59分钟，输入与代码先验证冻结，单作业训练及新进程独立
重训；一次激活核对后不持续监控，提交后继续必要的独立工作。
