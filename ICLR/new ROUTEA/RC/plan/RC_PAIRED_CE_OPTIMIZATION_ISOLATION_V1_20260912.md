# 局部统计小头：拟合空间与跨组泛化隔离

用户继续授权2026-09-12。上一轮JOINT3=110、CURVE3=111、MEAN2=109，原TRAIN四折BASE108、GLOBAL114；主统计没有显示独立优势。原七参数成本4的凸优化诊断已做过，本次仅针对新标准化特征上的2/3参数FULL-C128 CE，不重做旧试验。

本轮是优化诊断，不是新特征、新机制或EVAL晋级。复用原TRAIN128/32身份/32组四折、各折冻结BASE7、m1/m2/标量曲率、TRAIN标准化、完整C128和127物理顺序challenger。无backbone/visibility变化，无人工类型标签，无其他593/旧EVAL/正式prerecall/D1-MI/ownership数据。

每折分别处理MEAN2、CURVE3、JOINT3。旧AdamW末步参数直接读取。保持相同函数类，仅在预先固定盒[-64,64]^k内最小化原记录的无额外正则FULL CE数据损失，RAW logit0。旧参数必须在盒内。L-BFGS-B从旧参数启动，maxiter2000、maxls50、ftol1e-14、gtol1e-10；无留出驱动选择、重启或参数扫描。旧AdamW采用decoupled weight decay，不能把这里的无正则目标说成继续同一个优化过程。

不信任solver.success作为最优性证据。对于F(theta)=mean(logsumexp(b+Dtheta)-b_target-D_target theta)，任何逐query概率向量p给出全局仿射下界：

    F(theta) >= mean[p·b - b_target + H(p)]
                + mean[p·D - D_target]·theta。

把浮点softmax量化成分母2^52的非负整数概率，总和严格为1；差额补给最大项。系数和offset以Fraction精确重建，熵下界及目标值上下界用50位mpmath区间向外取界。盒内下界为max(0, offset_lower -64*sum(abs(slope)))，目标值上界由所有128项的区间logsumexp获得。保存概率整数和有理数界用于新进程重放。仅界间隙≤1e-6才记录该固定盒、该实数端点目标的近最优证书；否则如实未决，不扩大盒或换solver。没有参数域外全局最优或统计最优声明。

保留旧头和新优化头的全部127 logits。独立Torch交叉熵/梯度与NumPy实现核对；新进程从封存输入重建上下界、验证所有预测/动作和旧头逐bit重放。四折封存后再join原留出标签，报告RAW/BASE/GLOBAL及三头旧/新结果、每折训练损失区间与留出CE、完整5412去重身份排名、救/损/净与组不确定性。

解读预先固定：有训练损失改善且留出改善，支持本配方存在可利用的拟合空间；有训练改善但留出无改善，只说明进一步最小化该数据损失不足以换取跨组收益；盒内剩余间隙很小，才可在本盒/目标下限定进一步优化空间。不能由任何一种结果直接推出真实图片无身份信息、全部HYP不可能或新机制GO。比较JOINT与CURVE也须在同优化协议下进行。旧EVAL分开保留。

任务为四折10分钟数组加10分钟汇总，全部新命名空间；保留已完成的所有程序、模型、缓存和authorities，不自动追加其他实验。
