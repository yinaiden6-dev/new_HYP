# 593图独立分支：保留同支持竞争端点符号

用户批准继续下一步。原593五折实验、全部输入/模型/结果保持冻结；新增CPU分支复用同一593图、68身份、64组、5折。原RAW43片已经完成，RoMa尚在排队。新分支不新增encoder/RoMa调用，不读取rc_opened_诊断产物或原593汇总成绩。

## 唯一机制问题与固定对照

同支持竞争J_g = mean_{wq_g}(free_g) - max_{h!=g} mean_{wq_g}(free_h)，完整128 reference、先pool后max。新缓存直接从已资格化tokens/maps重新计算，来源无标签；不会把隔离诊断缓存解封给训练。

对challenger c与RAW winner w，令d=|J_c|+|J_w|+1e-12，r=(J_c-J_w)/d，s=(J_c+J_w)/d。

| 模型 | 固定基头 | 新增输入 | 新训练参数 |
| --- | --- | --- | ---: |
| BASE7 | 对应原593折内BASE7_ALL | 无 | 0 |
| RELATIVE1 | 同上 | r | 1 |
| RELATIVE2 | 同上 | r、常数1 | 2 |
| ENDPOINT2（primary） | 同上 | r、s | 2 |

logit_new=logit_base+gamma·input，gamma全零初始化。RELATIVE2允许全局偏置，ENDPOINT2以端点符号/相对大小状态替代常数；两者参数数目相同。r/s等价于保留两个归一化端点，不声称保留J的绝对幅度。这不是旧SPECIFIC8全八参数重训的逐bit复制，而是固定基头后的输入信息对照。

所有新残差参数使用同一fold的完整训练组、原FULL身份损失（负向权4、标量max平局均分梯度）、FP64、seed17、AdamW lr=.03/decay=.001、2000固定步，无早停/选checkpoint/超参扫描。target不在自然C128时不伪造正确challenger；这些query全部保留在593分母。原七参数不更新。

## 输入与标签边界

BASE7_ALL仅取对应fold已通过新进程重训/预测重放的参数；不看其OOF成绩选基头。每折只读原role/foldN.json的训练身份，禁止其它fold的参数和训练标签（它们可能见过当前heldout）。全部新参数与heldout预测封存、独立重放后再统一join标签。当前593已是历史来源开发交叉验证，不能与旧99/128直接比较或称未触碰外部集。

J缓存复用256图分32片、新337图分43片，CPU8、每片10分钟、数组上限46，独立NumPy选点重算全部B/F/J。与原特征的自由内容dF逐位回归，候选axis与RAW winner不变。

## 输出与判断

固定REAL、J_BIND（只把J在完整candidate键上half-roll，原六特征不动）、CBIND（原六特征整包roll并同步roll J）。BASE7在J_BIND须逐位等于REAL；BASE7的REAL/CBIND须逐位复现原fold输出。

主比较ENDPOINT2对BASE7与RELATIVE2；RELATIVE1帮助判断额外自由度的影响。完整报告正确数、rescue/loss、MRR、各fold与各组差、组bootstrap区间以及J_BIND对增量救回的保留。内部支持标记要求对两个主对照均总正确和等权组均值提高；可靠性另看预定组区间，不以正净增单独宣布GO。

沿用用户允许少量损失且要求可靠净增的目标，不恢复旧零损门；593分母不自动套用旧EVAL128的2张预算，也不凭探索结果改门。没有自动部署或new HYP普遍证明。

科研截止UTC2026-09-11 16:00（北京时间9月11日24:00），全链保留运行时timeout。原GPU时限RAW10/RoMa15及并行46不改变。
