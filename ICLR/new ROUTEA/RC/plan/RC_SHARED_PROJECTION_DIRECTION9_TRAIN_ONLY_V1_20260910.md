# new HYP：用检索监督学习一个共享投影方向

用户提出坐标变换不必局限y=x，并举绕原点旋转为例。本计划在读取
5139151的新EVAL结果前定义；不修改正在运行的四头比较。只增加一个
共享方向参数，检验能否从TRAIN身份配对学到比固定方向更有用的投影。
先前TRAIN0101的反例限制该点的单独投影排名，不否决群体识别目标。

## 两个头与唯一差异

固定输入仍为已独立合格的8,320项缓存、原native6及原RAW C128。
对candidate g，S_g=FP64(J_g+T_g)，D_g=FP64(T_g-J_g)，定义：

    d_theta(g) = abs(cos(theta)*S_g + sin(theta)*D_g) / math.sqrt(2.0)
    phi_theta(c,w) = (d_theta(c)-d_theta(w)) /
                     (d_theta(c)+d_theta(w)+1e-12)。

实数意义的投影轴方向为pi/4+theta，单位向量尺度固定。
所有query、reference、REAL/C_BIND/EXTRA_BIND共享同一theta。

- FIXED_PROJECTION8：theta固定0，原六列+phi_0，共8参数；必须逐bit
  复现5139151的PROJECTION8训练参数、完整预测与动作。
- DIRECTION9：theta初值0，与原8个共享参数一起学习，共9参数。

theta是未wrap的实数，训练中不取模、不夹范围、不试多个初值。
展示时可按模pi解释，不声称方向唯一。两头构造同样七列后调用原
nn.Linear(7,1)算术；不将七列dot拆成两次加和。全程FP64。

## 原监督、优化与验证保持

只用原PAIR64+FULL TRAIN32检索监督。seed17、全零线性参数、2000步、
AdamW(lr=.03,weight_decay=.001)、PAIR负例4倍BCE及原FULL sign loss、
样本顺序不变。DIRECTION9的theta也使用该优化器设置，衰减到0的偏好
明确保留。原训练函数只替换模型构造/可微特征，不改损失、遍历和更新
次序。固定theta头须先证明初始七列及完整训练结果复现。

使用tensor cos/sin/abs构造角度路径，不将可微值转换为Python float。
第一个更新前投影系数为0，theta梯度为0是预期；theta.grad不能为None。
记录首步及后续梯度是否非零、最终theta、投影系数、全部有限性。
没有学到角度也原样报告，不以EVAL重启或手动旋转补救。零点沿用
torch.abs的零次梯度，不新增平滑项。

自然执行前以合成输入验证方向梯度可连通、第一次为0之后可非零，
theta固定0逐bit恢复PROJECTION8列，负和距离公式正确。原六列与RAW
不变，donor仍为原整份证据转移；EXTRA_BIND只变投影列。训练后才按
冻结theta生成全部FULL64的三模式七列及127个logits并封存参数/预测，
再开放EVAL角色和5139151结果。独立新进程完整重训两个头并复算。

## 比较和边界

主比较DIRECTION9 vs ORIGINAL7，用户目标仍是保留原28并至少新增1个；
同时报告vs FIXED_PROJECTION8的rescue/break、MRR、组统计与控制变化。
明确新头多1个参数，不将函数类扩展自动称为新原始信息或普遍理论。
两个头、所有候选、所有失败均报告，不以theta或checkpoint择优。

TRAIN0101及其它标签感知方向容量诊断的系数/角度不能读取或馈入模型，
不做按query、identity或target选择角度。禁止EVAL容量oracle、角度
网格搜索、SAM、本任务空间标注、D1-MI、GroZi、旧V9和正式392。
这是已打开内部开发比较，未触碰外部确认及部署均不由本计划授权。

独立新程序、authority、预检和Slurm输出；8CPU/4G/30分钟，训练加
独立重训。执行前绑定本计划、原主计划、投影距离V2、所有输入与源码；
仍遵守北京时间2026-09-11 24:00截止及提交后继续、不持续监控调度器。
