# 固定NATIVE7的训练目标2×2对照

研究继续至北京时间2026-09-11 24:00。使用原RAW C128与完整127-challenger
HOLD/SWITCH，retrieval-only任务监督。原当前基线NATIVE7为已打开EVAL32
28/32（RAW25）；旧FROZEN_C 27/32与69/90另属独立头，不混用。

本实验不增加特征、参数、编码器或空间监督，不使用SAM/superregion，
不要求ownership，不读取D1-MI、GroZi或未授权正式392。原模型不替换。

## 依据及其限度

ABS12/REL12试验已完成并独立验证，两模型EVAL32均26，未超过NATIVE7。
TRAIN-only精确有理数证书证明：在已有十二参数特征上，FULL TRAIN32
严格约束可以满足，加入PAIR64约束后不能同时满足。NATIVE7在FULL32
自身也不能满足所有严格线性决策约束。这不是错误标签证明、EVAL成绩、
不可提升上界或32/32资格门，也不证明去掉PAIR就有收益。

原base错误的FULL损失要求目标logit>0且所有错误logit<0；实际action
只要求目标logit>0且高于其余错误。两种条件不同，严格前者可能增加
拟合负担。本实验固定同一个七参数头，分别检验监督混合和FULL损失
形式，避免同时改变表示、参数量与优化目标后无法解释结果。

## 四格、唯一主比较

所有头使用原六维特征+bias，共七参数。四格如下：

| 名称 | PAIR损失 | FULL中base错误时的损失 |
|---|---|---|
| PAIR_SIGN | 保留原PAIR64 BCE | 原绝对压负 |
| PAIR_RANK | 保留原PAIR64 BCE | 目标相对错误排序 |
| FULL_SIGN | 不参与优化 | 原绝对压负 |
| FULL_RANK | 不参与优化 | 目标相对错误排序 |

主比较预设PAIR_RANK对PAIR_SIGN；另两格用于分离PAIR项的影响，不
择优替换主比较或宣布主要假设通过。若次要格改善，只如实报告并另行
确定后续确认，不继承外部/旧H资格。

令t为正确challenger的logit，u为完整其余错误challengers的最大logit。
RAW winner的action证据为0。FULL目标为RAW winner时，两类损失均为
4*softplus(max_challenger_logit)，保持原样。base错误时：

    SIGN = softplus(-t) + 4*softplus(u)
    RANK = softplus(-t) + 4*softplus(u-t)

PAIR仍是原标签0权4、标签1权1的BCE均值。PAIR格总loss=PAIR+FULL均值；
FULL格总loss=FULL均值，不额外乘系数或归一化。PAIR loss仍可作为训练
池描述值计算，但不进入FULL格梯度。移除该loss项同时改变总优化目标，
不能冒称是其它条件均不变的独立数据价值定理。

相同FP64、seed17、Linear全零初始化、AdamW lr=.03、weight_decay=.001、
betas=.9/.999、eps=1e-8、2000固定更新。PAIR原顺序、TRAIN按execution
升序、每步完整batch。原正/负权重4不变，不扫描任何超参/seed/温度，
不早停或挑checkpoint。直接从冻结原训练函数做上述两处受控替换。

## 数据、验收和边界

输入复用已验证的原六维C_PAIRED特征，不使用ABS12/REL12新增列。
PAIR64与FULL TRAIN32是唯一任务监督。EVAL32在全部四头参数、完整
REAL/C_BIND预测封存后再join标签；历史EVAL已打开，只作内部开发。
C_BIND继续原完整evidence shift64、RAW轴不变，不只shuffle最终分数。

PAIR_SIGN必须逐bit复现原NATIVE7参数，并回归原EVAL REAL/C_BIND
所有actions；否则工程ABORT不解释新格。验证进程重新构造源输入，
重训四头并重放全部预测/统计。所有格和TRAIN/EVAL结果完整报告。

报告相对基线、两个因素的主比较/交互、逐query rescue/break/net、
MRR、supergroup方向、PAIR训练池描述以及C_BIND救回保留。主格内部
继续候选要求PAIR_RANK相对PAIR_SIGN严格正净增且不破坏原正确样本；
净增但有破坏另列，不能称没有增益。所有结果都不自动成为部署、新H、
空间P-only或外部确认GO。

CPU8、4G、59分钟，run与独立validate顺序执行，启动时检查截止时间。
只做一次提交激活核对，不监控队列。所有源码、参数、结果append-only。
