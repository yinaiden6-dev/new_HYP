# new HYP：优化支持与固定支持的同容量识别检验

用户已授权retrieval-only研究至北京时间2026-09-11 24:00，并要求提交后
继续推进。当前目标为在原RAW C128、已打开matched EVAL32上，保留
NATIVE7的28个正确并至少新增1个；正净增但有损失另报。现有模型不替换。

## 为什么进入这一步

固定前4条TRAIN、全部512个候选的pilot（5139121）完成且通过独立精确
验证。DIFFICULT-0101的正确reference原支持margin=-0.00125697，优化
支持的可达margin=0.17824540，确认固定支持遗漏可分证据。另一方面，
508个错误candidate中502个也能取得正分离支持，因此支持存在性本身
不等于身份判定。这里检验优化支持值是否为原共享校准提供有用信息。
没有选择EVAL样本、使用EVAL容量证书或改变原TRAIN/PAIR监督。

## 冻结的算子与输入

沿用pilot的a_h(i)：原query image tokens对完整reference-image tokens
的FP64 free MaxSim。原gallery、C128顺序、图像token切片、规范化及
已有a的hash不变；不用模板扩展token，不新增encoder或RoMa forward。

对candidate g，p为任意非负、和为1的query-token权重：

    v_g = max_p min_(h!=g) sum_i p_i (a_g(i)-a_h(i))。

复用pilot已冻结的原/对偶求解选项、负数自由margin变量、精确binary64
端点证书和gap<=1e-8。新程序支持720与768 query tokens。每项保留
原始solver返回、精确归一化p/alpha、[L,U]及固定支持精确margin；不
裁去负margin。用于模型的唯一新标量T_g为精确可达下界L按binary64
就近舍入；不用solver目标、上界或选取最好的多种公式。

完整缓存覆盖FULL64×128及PAIR64×2，共8,320个game；PAIR的每个game
仍比较原全部127其它reference。全部FULL query、candidate保留，不能
依target选样。满足输入hash、候选轴、算子合同完全一致时，可复用
pilot的512项，其余7,808项新求解。复用项仍需独立证书核验。

FULL的全部a复用双重资格化缓存；PAIR补全all128 a并核对先前已封存
的完整矩阵SHA和原两条a。持久化补全矩阵，避免独立验证重新做矩阵乘。
缓存阶段不读任何角色target、EVAL结果或容量oracle。任何候选失败或
未达到数值资格，整个读出阶段不开放；记工程不完整，不称科学失败。

## 唯一读出比较

全部使用原PAIR64+FULL TRAIN32、原2000step/seed17/AdamW与原损失：

| 名称 | 输入 | 可训练参数 |
| --- | --- | --- |
| ORIGINAL7 | 原native6 | 7 |
| FIXED8 | 原native6 + symmetric(J_c,J_w) | 8 |
| MAXMIN8 | 原native6 + symmetric(T_c,T_w) | 8 |

J为5139024中固定RoMa query支持的原FP64同支持最强其它reference margin。
symmetric(x,y)=(x-y)/(|x|+|y|+1e-12)，不改尺度、温度或阈值。FIXED8
必须完整复现已冻结SPECIFIC8的训练参数/预测，ORIGINAL7必须复现原
NATIVE7参数与预测；只有MAXMIN8列替换固定支持为优化支持。

原六列、RAW gap、原PAIR候选顺序与混合控制定义保持逐bit相同。
REAL、完整C_BIND、仅新增列EXTRA_BIND三模式沿用旧donor映射：C_BIND
转移整份候选证据，EXTRA_BIND保持原六列只转移第七列。原ORIGINAL7
的EXTRA_BIND与REAL相同。不训练控制专属head，不搜索控制强度。

原全部127-challenger、最大logit、>0 SWITCH否则HOLD、原physical-row
tie-break保持不变。所有FULL64预测先封存，再读取EVAL角色并计算结果。
独立新进程重新拟合三头并逐项核对参数、logits、动作、指标及源绑定。

## 成果口径和结束条件

同时报告对RAW25、ORIGINAL28及FIXED8的rescue/break、MRR、原group
统计和candidate-binding/EXTRA_BIND对照。主要用户目标：REAL至少29
正确，且对ORIGINAL28为>=1 rescue、0 break。若只是margin提升、TRAIN
提升或有取舍的净增，按其实际证据报告。控制变化单独解释，不能用
准确率达标直接声称严格空间Reference HYP/ownership成立。

这是已打开内部EVAL上的固定开发比较，不是未触碰外部确认。不得用
EVAL调参、阈值、seed、checkpoint、支持约束或学习子集。无SAM、本任务
空间标注、D1-MI、GroZi、旧V9或正式392。算子数学属于标准有限矩阵
博弈/线性规划，不将其代数重命名为新理论。

缓存与读出采用独立新文件和append-only输出；先冻结源码、launcher、
source pins及本计划，完成Slurm启动核对后继续有用工作，不持续监控。
缓存预算单job8CPU/12G/30min，8个独立单线程worker；后续三头与独立
重训单job8CPU/4G/15min。两阶段在用户截止前执行；工程失败先明确定位，
不能将不完整输出或待调度状态当作本方法的识别结果。
