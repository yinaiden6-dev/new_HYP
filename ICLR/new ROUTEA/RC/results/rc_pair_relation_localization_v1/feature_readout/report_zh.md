# 固定单图特征读出：已有71张自然C128面板

所有9,088个候选均参与。固定8×8有效patch池化，无训练、无模型前向；先封存新分数再做标签join。
ROMA_COARSE使用保存的单图coarse_11/17各1024维；两个块分别归一化后等权合并。COL使用原128维。全局均值读出使用原patch均值。
局部内容与结构统计使用完全相同池化token；除构建池化单元外不读坐标。纯argmax是读出诊断，不是COST1的HOLD/SWITCH策略。

RAW=51/71，原七参数COST1=57/71，原救回=6；groups=37。

|固定读出|正确/71|救/损 vs RAW|原COST1救回中 target > RAW|
|---|---:|---:|---:|
|RAW|51|0/0|0/6|
|NATIVE_M|43|8/16|6/6|
|COL/forward_maxsim|37|1/15|2/6|
|COL/reverse_maxsim|22|1/30|1/6|
|COL/bidirectional_maxsim|29|1/23|1/6|
|COL/global_mean_cosine|27|2/26|1/6|
|COL/mutual_top1_fraction|10|0/41|1/6|
|COL/unique_hit_coverage|15|0/36|1/6|
|COL/soft_cycle|22|0/29|2/6|
|ROMA_COARSE/forward_maxsim|2|0/49|4/6|
|ROMA_COARSE/reverse_maxsim|4|0/47|4/6|
|ROMA_COARSE/bidirectional_maxsim|1|0/50|4/6|
|ROMA_COARSE/global_mean_cosine|0|0/51|2/6|
|ROMA_COARSE/mutual_top1_fraction|1|0/50|4/6|
|ROMA_COARSE/unique_hit_coverage|0|0/51|3/6|
|ROMA_COARSE/soft_cycle|2|1/50|5/6|

结构差值与M差值的相关性仅为描述，不能证明M来自该结构，也不能由某个固定读出失败推断冻结特征缺信息。
71张此前已打开；本轮冻结前为查schema读取过既有结果首行，不宣称全盲预注册。
原六臂SINGLE对query/reference分别出标量再相乘，其同query相对M对比消去query公共因子；它不覆盖这里的单图向量配对比较。

原full-token Col自由分数仅在67张原缓存存在，仍保留于逐query封存文件；为统一71张比较，表中不纳入该辅助列，未删除任何query。

此次结果没有把native M还原成某个简单内容分数或无位置结构统计。按37组等权汇聚，query内coarse双向MaxSim与M的相关为0.07783；互近邻、命中覆盖及soft往返支持分别为0.05159、0.05630、0.00387。相邻且MaxSim差≤0.01的8,997个候选对中，结构差与M差的对应组等权相关为0.03295、0.03611、−0.05248。该配对仅约束一个标量，不能视作全部外观匹配或因果控制。

原6次COST1救回里，native M全部给target高于RAW错误者的分数，coarse双向MaxSim为4/6、soft往返支持为5/6；这是按既有纠错选出的描述组。native M纯排序43/71也说明M本身不是足以替代身份内容和接受规则的身份判别器。

RoMa粗特征原始余弦的纯排序很弱；这只否定本次固定池化与读出组合能直接替代matcher输出，未排除其他归一化、保留完整token、其他读出或训练能恢复区分信息。不能据此认定RoMa单图表示缺信息，更不能宣布全局几何一致性已经得到证明。

独立验收：16条完整读出、145,408个候选分数的选择及身份计数一致；6个矩阵统计检查在FP64 NumPy下通过。池化构建未由另一实现单独复算，因此验收范围见validation.json。
