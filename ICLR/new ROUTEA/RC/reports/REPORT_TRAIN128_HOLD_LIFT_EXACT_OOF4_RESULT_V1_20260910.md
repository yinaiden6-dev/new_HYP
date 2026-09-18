# TRAIN留组结果：内容校准未通过，GAP对照有探索性增益

**这些是TRAIN128四折按组留出结果，不是EVAL128。冻结原ec7在EVAL128仍为99/128，旧EVAL32仍28/32。**

同一固定4折、32正例身份/组、自然RAW C128、完整127 challengers；每折BASE使用此前已验证的排除留出组基头。两个一参数臂锁住原SWITCH及原最高challenger身份，只在原HOLD时提升原top。系数只由各fold TRAIN检索身份标签精确求得，保证TRAIN原BASE正确全部保留；四折全部封存并由独立FP64 solver复算后才join。

| TRAIN OOF模型 | 正确 | 对RAW救回/损失 | 对BASE新增/损失 | 等权组准确率 |
| --- | ---: | ---: | ---: | ---: |
| RAW | 86/128 | 0/0 | — | — |
| 每折BASE7 | 108/128 | 23/1 | 0/0 | 83.125% |
| GAP_HOLD1控制 | 110/128 | 25/1 | 2/0 | 85.469% |
| CONTENT_HOLD1主臂 | 108/128 | 25/3 | 2/2 | 83.021% |

CONTENT主臂未通过任何一个固定门，父试验维持NO-GO，不以GAP更高改写门或宣称EVAL晋级。CONTENT救回OUTCOME-0150/DIFFICULT-0100，却损失DIFFICULT-0101/0106；前次0087/0102已由锁原SWITCH结构保住，但原HOLD正确仍会损失。TRAIN严格保护不能外推为OOF保护。

GAP救回OUTCOME-0555/DIFFICULT-0057，完整保留BASE108正确。其等权组增益+2.344个百分点，固定组bootstrap95%[0,+6.25]个百分点。这个信号来自原联合评分的HOLD校准，不能归于自由内容新增证据。它也是看过对照结果后的探索性发现，不是预注册主臂确认。

四折GAP系数约0.49365/0.15866/0.36637/0.10962；CONTENT约1494.41/0/4.77796/4.26866。CONTENT第0折的最大可救TRAIN样本需要极大的系数，尽管该折TRAIN所有原正确仍保住，留出样本却出现新的误切换。第1折则没有满足保护约束的TRAIN可救样本而精确回退alpha=0。这里已求出该一维受限类的全局经验最优，不能将结果归为迭代未收敛；也不能据此排除全部内容特征或所有条件模型。

5139348运行1分06秒、COMPLETED0:0；4折参数及48768新heldout logits独立重放通过，原16256 BASE logits回归通过，0新编码器/匹配器、0 optimizer、0基头重训、0 EVAL读取。

下一步独立探索计划：仅用全TRAIN128、固定ec7原参数求一个GAP系数，再一次完整报告同一旧EVAL32和EVAL128。此计划独立授权，承认对照选择与已打开EVAL的适应性偏差，父CONTENT NO-GO保持不变。若新EVAL不能超过原99且保留原99、或旧EVAL不能保留原28，就不采用；不从EVAL回调系数。

- [机器结果](../results/rc_train128_hold_lift_exact_oof4_v1/result.json)
- [独立复核](../results/rc_train128_hold_lift_exact_oof4_v1/validation.json)
- [原冻结计划](../plan/RC_TRAIN128_HOLD_LIFT_EXACT_OOF4_V1_20260910.md)
- [独立探索计划](../plan/RC_ORIGINAL7_GAP_HOLD_EXPLORATORY_FOLLOWUP_V1_20260910.md)
