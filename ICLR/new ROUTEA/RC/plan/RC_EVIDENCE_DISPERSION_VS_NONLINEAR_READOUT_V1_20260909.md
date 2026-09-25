# new HYP：匹配分布信息与非线性读取的固定对照

授权延续至北京时间2026-09-11 24:00。用户要求克服已定位瓶颈，提交后
继续推进。只用原检索身份/配对监督；不用SAM、空间标注、superregion、
ownership、D1-MI、GroZi或未授权正式392，不新增编码器/匹配器forward。

## 依据与两个不同方向

已打开EVAL的精确容量证书表明：原六特征实数线性类在严格保持RAW25
正确时不能达到29；允许样本取舍则存在29的数学容量。它不是全局28
上限，任何EVAL证书系数不得用于本试验的输入、初始化、学习或筛选。
本轮改变的是可表达的读出，而非继续调同一线性类的阈值。

两个候选原因分别检验：旧均值统计未保留匹配分布信息；或旧信息
仍需更灵活的非线性读取。它们只是待验证解释，不预言识别提升。

## 三头固定定义

ORIGINAL7：原六特征+bias，必须精确回放原NATIVE7参数/动作，EVAL32为28。
MOMENT8：原六特征不动，附加一个candidate-conditioned匹配离散度对比。
CURVE8：原六特征不动，附加原dL*abs(dL)，作为同参数数的非线性读取对照。
CURVE变换已在旧REL12使用过，这里只取L单列作8参数控制，不包装新方法。
同参数数不等于函数类或有效容量完全相同，二者不能直接互相归因。

对于原完整C scorer的每个candidate g，使用原缓存

    b_g(i)=max_j [w_r^g(j) cos(q_i,r_gj)]
    L_g=original_S_g/max(original_M_g,1e-12)
    V_g=sum_i w_q^g(i)*(b_g(i)-L_g)^2 / clamp_min(sum_i w_q^g(i),1e-12)
    extra_M=(V_c-V_w)/(|V_c|+|V_w|+1e-12)
    extra_C=old_native_feature[3]*abs(old_native_feature[3])

全部原FP64操作顺序固定，V的中心直接取旧S/max(M)，不得另重算mean。
V描述visibility-weighted匹配值的离散度，不叫纯身份、纯内容或概率。
不添加top-K、量化阈值、温度、crop或新空间轴。原q/ref tokens、完整
legal source/candidate轴及原所有S/M/SQ/SR和六特征保持逐bit相同。

输入producer及新进程独立算术validator先回放FULL8192+PAIR128原C
四标量、REAL/C_BIND原六特征，再封存V与附加特征。PAIR原roll1/half
历史控制位移不变。C_BIND使用原完整evidence donor映射，V也从同一
candidate donor转移；RAW prior不转移。CURVE的C_BIND从旧C_BIND dL
生成，不能误用REAL dL。

标签自由E0展示同均值、不同离散度：uniform权重的(.5,.5)与(1,0)
均值均.5，离散度分别0与.25；原S/M/Q/R可相同。这只说明新统计量
可能保留旧均值缺失的信息，不证明自然身份识别变好。

## 原训练不改

原PAIR64和FULL TRAIN32，原顺序和全部C128/127 challenger均不变。
每头FP64全零初始化，seed17，原AdamW lr=.03、weight_decay=.001、
betas=.9/.999、eps=1e-8、2000更新。原PAIR加权BCE+FULL sign loss完整
复用。不用EVAL反馈选择额外维度、seed、epoch、损失或阈值。
原始ORIGINAL7回放负责识别任何输入、优化器或算术漂移。

## 预封存与评价

三头参数及FULL64预测先封存，然后join EVAL角色/旧outcome。
REAL、C_BIND均报告。另加EXTRA_BIND固定头诊断：只把附加第七列换为
同一query原C_BIND对应附加列，原六列保持REAL；ORIGINAL7无附加列，
该诊断应与REAL完全相同。它检验新增项的candidate对应关系，不叫
像素/空间干预。没有新增训练head，也不择优选择控制结果。

新进程重建输入并完整重训三头，参数、全部预测、动作与指标逐项复核。
所有模型输入程序禁止读取EVAL容量证书目录；证书不进入模型链。

## 固定主次比较及推进条件

主比较MOMENT8对ORIGINAL7，报告query配对救/损/净增、MRR、等权
supergroup准确率差、RAW原正确损失。内部总体改进候选要求净增>0且
group平均差>0，不要求保留原28每一条；损失数量和样本始终单列。
无损改进作为更强的独立标记，不能用它覆盖总体正增益。

MOMENT8对CURVE8为预定归因比较；CURVE8对ORIGINAL7为预定控制结果。
若只有CURVE8提高，记录非线性读取方向的结果，不宣称离散度有效。
若MOMENT8提高但不胜CURVE8，也不能据此独占归因为新分布信息。
EXTRA_BIND下新增救回保留情况和全体配对变化均报告，不凭一个case
宣布普遍机制。全部结果均为已打开内部EVAL32（11 groups），不是新
外部确认、旧空间HYP GO或模型自动部署。

## 执行

代码/输入/预检/授权先冻结，单作业dev_cpuonly、8CPU、4G、59分钟，
训练加新进程独立重训。一次激活核对后不持续监控；提交后继续必要
的独立工作，并在完整验证后按实际结果推进。原始/冻结产物不改写。
