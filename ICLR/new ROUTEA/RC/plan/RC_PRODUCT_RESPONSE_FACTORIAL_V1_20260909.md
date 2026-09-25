# new HYP：联合评分S与Q/R响应的固定2×2对照

继承retrieval-only授权至北京时间2026-09-11 24:00。用户现在要求提交
任务后继续有用工作，不因提交就退出；不持续监控调度器的要求保留。
无SAM、人工空间标签、superregion、ownership、旧V9、D1-MI、GroZi或
未授权正式392。旧Reference HYP与当前new HYP不混称。

## 直接依据

5138847已完成独立验证。同一已打开RAW C128 EVAL32：RAW2为25，
RAW+M3、RAW+L3、JOINT4均26，原NATIVE7为28。原NATIVE7分别比两个
重新训练单通道多对2条且0损失、group均值为正。JOINT4没有复现原28，
且正确集合与M3相同。因此当前完整模型优势不被已测单通道解释；
简单M+L加法也不足以解释完整模型。

JOINT4与完整原模型差的仅为原三列：S比较列1及Q/R响应列4、5。
本轮分别添加这两组，固定2×2，不扩展其它特征或更改原scorer。

## 四个固定头

| head | 原native列 | 含bias参数数 | S列 | Q/R两列 |
| --- | --- | --- | --- | --- |
| JOINT4 | [0,2,3] | 4 | 无 | 无 |
| PRODUCT5 | [0,1,2,3] | 5 | 有 | 无 |
| RESPONSE6 | [0,2,3,4,5] | 6 | 无 | 有 |
| ORIGINAL7 | [0,1,2,3,4,5] | 7 | 有 | 有 |

列0=标准化RAW gap；1=symmetric(S)；2=symmetric(M)；3=symmetric(L)；
4/5=symmetric(S−SQ)、symmetric(S−SR)。L=S/max(M,1e-12)。四头都保留
原M、L列。PRODUCT5名字只标识加入原联合S比较，不引入新乘法实现。
JOINT4和ORIGINAL7须同时精确回放原特征、参数、REAL/C_BIND动作。

在正M、正L且无epsilon/floor的理想实数条件，S=M L，

    dS = (dM+dL)/(1+dM*dL).

因此原S比较可为线性head提供M/L对比的非线性函数；它不是在这些
理想条件下独立的新原始信息。实际保留原FP64的S、M、L和epsilon，
不以这个恒等式重写模型。公式本身是已有代数性质，不包装成新定律；
在有floor或非正L时不能无条件应用。

## 原训练完整保留

原PAIR64与FULL TRAIN32、已打开EVAL32、自然RAW C128、全部127
challenger及零阈值HOLD/SWITCH固定。PAIR原顺序，TRAIN execution
升序。原PAIR V1 roll1和V2/FULL half定义保留。任务监督仅检索身份/
配对。原C的REAL/C_BIND六特征先精确重建，再按表取列。

所有头FP64全零初始化、seed17、原AdamW lr=.03、weight_decay=.001、
betas=.9/.999、eps=1e-8，原PAIR加权BCE+FULL sign loss、2000更新。
除固定特征列及相应参数数外无其它变化，不扫阈值/seed/训练步数。

四头参数和完整FULL64的REAL/C_BIND全部logits先封存，再join EVAL
角色及读取旧结果。新进程重建输入、重训四头并复现完整结果。
原JOINT4和ORIGINAL7的参数与EVAL动作必须分别逐bit/逐字段回放。

## 全部预定比较与结论

S无Q/R时的贡献：PRODUCT5对JOINT4。
S有Q/R时的贡献：ORIGINAL7对RESPONSE6。
Q/R无S时的贡献：RESPONSE6对JOINT4。
Q/R有S时的贡献：ORIGINAL7对PRODUCT5。

四项都报告，不能在结果出来后挑有利的条件称唯一主比较。再报告
PRODUCT5及RESPONSE6分别对ORIGINAL7的保留、救/损/净增与group方向。
各模型容量不同是明确的嵌套特征设计，不能宣称相同容量的模型比较。

PRODUCT5内部机制简化候选：保留ORIGINAL7全部正确且数量不低于原，
同时对JOINT4有正配对净增及正等权supergroup准确率差。RESPONSE6的
同样条件作为预定的另一种解释，两个结果都保留。若仅完整ORIGINAL7
更好，说明本次消融未找到一个更小、等效的读出，不能宣称其它组无用。

所有结果限于固定优化协议和已打开EVAL32（11 groups）。即使S组有益，
也不单凭此证明普遍非线性交互、空间所有权或独立外部泛化。可得出的
是对当前完整模型优势更具体、可复现的内部机制解释。任何模型若超过
原28且无break则另列内部模型改进候选，不自动部署或宣布普遍HYP GO。

## 执行与提交后工作

预检后冻结源代码和授权。单作业8CPU/4G/59分钟，训练+新进程独立
重训验证，一次激活配置核对后不持续轮询。提交后继续完成代数说明、
旧结果独立收口、论文证据边界及下一阶段必要准备；有完成产物时按
阶段依赖读取验证，不因一次提交停止推进。所有历史产物append-only。
