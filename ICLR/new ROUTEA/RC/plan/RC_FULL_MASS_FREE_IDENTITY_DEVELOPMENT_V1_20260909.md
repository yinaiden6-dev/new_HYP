# 保留完整质量、解除reference内容门控的唯一新臂

2026-09-09。用户要求在原最好系统上定位并修复HYP瓶颈；本分支接续已完成
的无损P/V修复与两例旧错误的精确字段恢复。原部署、参数、历史结果不改。

## 依据与未测单元

旧difficult90固定两例25/64恢复恰好4次RoMa match，8个map SHA及16个原
scalar hex全等，独立验证通过。reference visibility乘进MaxSim的a→b步骤，
把25例全部362个原目标优势格翻负，b层wrong强于target的格数为768/768；
64例把370个目标优势格中的348个翻负。全局M仍是旧系统实际救回所依赖的
信号；不能为解除内容压制又将M一起丢掉。

既有B_QUERY同时把reference从质量与identity读出中去掉；它在既有RAW
EVAL32的COMMON3/NATIVE7均为26/32。C_PAIRED是软双侧visibility与完整
reference MaxSim，COMMON3为26、NATIVE7为28；旧FROZEN_C为27。这些已做
结果不能称为新实验，也不能据两例声称全局去除wr必然提高准确率。

本次只补：完整C质量M保留，identity使用B的query-weighted完整reference
读出。没有新编码器、ROI、阈值、温度、top-K或更大action网络。

## 唯一数值定义

对既有同一query、candidate的B与C封存证据，明确采用以下FP64顺序：

    rho = C.visibility_mass / B.visibility_mass
    D.visibility_mass = C.visibility_mass
    D.real_score = B.real_score * rho
    D.query_control_score = B.query_control_score * rho
    D.reference_control_score = B.reference_control_score * rho

D名为D_FULL_MASS_FREE_IDENTITY。该定义保留完整质量M与B的归一化内容
至浮点舍入。四个分数使用同一rho；不冒称与另行重算各roll后C质量的
浮点次序相同。B质量零时，只接受与C质量及B四证据一致的零输入，否则
拒绝；已核对8320个实际candidate occurrence，B质量均大于零。

B、C原臂原样作为回归与比较，禁止用新D覆盖旧B或C。D不是原P-only head
或旧冻结head的无损重命名，而是有一个已明确读出改动的内部development。

## 数据与拟合

仅用已有RAW_FEATURE_LEDGER_ONLY的FULL64（TRAIN32＋EVAL32，每query
原RAW C128）和既有PAIR64训练特征，合计8320个candidate occurrence。
不访问D1-MI、GroZi、正式392或增加difficult90训练样本。两例difficult90
只用于先前机制定位，不作为本次拟合或独立验证数据。

复用原COMMON3/NATIVE7两种head及纯训练函数，不增设头。PAIR64按原payload
顺序，TRAIN32按execution排序；每个head固定seed17、FP64 Linear全零初始化，
AdamW lr=.03、weight_decay=1e-3、2000步，无earlystop或参数扫描。每步使用
完整PAIR64与TRAIN32×127；原loss权重和maxwrong定义不变。

仅读取TRAIN32相应role及PAIR64训练标签。全部6个head（两family×B/C/D）
拟合后冻结参数，再计算完整EVAL32×127无标签REAL/C_BIND及既有Q/R标量
诊断预测并封存；封存后才读取EVAL role与旧最终结果。不能直接调用旧
load_inputs/main，因为它们会提前读取EVAL标签与旧结果。

COMMON3是严格共同的两个输入加bias；NATIVE7是六输入加bias的名义七参数，
保留并报告各臂特征退化，不能声称相同有效容量。Q/R沿既有标量替换规则，
不是新增像素干预或完整S11因果实验。

## 验收与解释

先要求同软件/顺序下B/C的原head参数及REAL/C_BIND动作重放旧结果。失败
记BASELINE_REGRESSION_ABORT，不解释D、不调参数追逐EVAL表现。

成功后分别报告D对相同family的旧B/C、旧FROZEN_C与原RAW base的逐query
rescue/break、R1/MRR、C_BIND保留及Q/R描述性结果。两family都报告，不挑
更有利的family宣称成功。新D的参考位置不再控制identity读出，不能用它
声明reference坐标ownership；HYP科学门、旧P NO-GO与外部确认均不自动晋级。

没有预设D必胜或新的GO阈值。若D无增益或破坏旧能力，必须如实记录，
不得以两例局部优势替代完整EVAL结论。部署默认仍为原FROZEN_C。

实现：reference_visibility_full_mass_free_identity_v1.py及
run_rc_full_mass_free_identity_development_v1.py。CPU-only、8核、32GiB、59分钟；
训练与独立结果验证分进程执行，不申请GPU或轮询队列。
