# 同一query支持、同一平均测度的剩余原因诊断

2026-09-09。用户要求验收5137873并继续。本次仍仅opened TRAIN32；不拟合V9，
不产生新ranking、rescue或自然GO。原V8 NO-GO、正式392封闭及P-only边界不变。
不读改或依赖D1-MI、GroZi，不进入V、action、HOLD/SWITCH或ownership。

## 已核实的事实与本次隔离的聚合问题

完整2691个合法regions的诊断已通过独立复核：target有certificate20/32，
严格唯一11/32，wrong有certificate14/32。扩展完整family没有新增任何target
certificate query。当前证书机制的容量限制不是所有识别机制的准确率上限。

43个CERTIFIED regions中，22个属于target、21个属于wrong；21个wrong区域
在target上的对应全部完整，16个在全部C128上完整，不能再用缺失解释它们。
其中14个wrong区域的reference group数多于同域target。原能量
`mean_reference_group(max_atom a) + max_atom b`仍以各reference自己的分组
和分母比较。相同query cells没有保证相同观测权重。

本次只回答：错误证书有多少由这个分组测度制造；又有多少在逐query-atom
的共同测度上仍有错误身份优势？后一种结果意味着应追查身份读出及区分内容
是否被观测，而不能继续仅靠区域/分组修正期待突破。

## 先验数学缺口

四格query上，设target的a=b为[.1,.1,.1,.9]，分组[0,0,0,1]；wrong的a=b
为[.12,.12,.12,.95]，分组[0,1,2,3]。每个atom的两个target成本均更低。
原分组能量却为target1.4、wrong1.2775，反向选择wrong。
共同query平均下为target1.2、wrong1.2775，保留逐atom优势。

这组成本在uniform V8 weights下可由单位向量实现（a=b=1−cos）；它说明原
分组比较不保证逐atom优势保持。它不是TRAIN32中的因果发生率，也不是新方法
有效性证明。共同query平均和max分别保持逐atom非严格支配，因此组合也保持。
仅有非max位置的b严格降低时总能量仍可相等；不能将“至少一个atom严格
更好”自动视为严格证书。

## 唯一冻结公式与完整范围

原32×C128、原全部2691合法REAL regions、原V8最终参数、RoMa对应、FP64
atom costs、source坐标全部保持。只比较一个新解析表达式：

`E_Q*(c,H) = sum_{i in H} a_ci / N_H + max_{i in H} b_ci`。

这里N_H为原H的query atom数量，对全部reference相同。没有面积奖励、
裁剪、删点、重配对应、top-K、温度、参数扫描；保留global-max项。
原分组能量和其完整证书逐项重放，不能改写旧产物。

缺失时整个观测score仍为UNKNOWN，仅记录假想补全下界：
`L_Q* = sum_observed a_ci / N_H + max_observed b_ci`，全缺失为0。
这个0是下界中的放宽，未被填成实际匹配。任一非负成本补全不能降低该式。
无效assignment的nearest-token占位残差不得参与；合成NaN占位必须被排除。

各FP64 atom成本视为精确二进制有理数，Fraction进行求和、除法及符号判断。
该分析不覆盖所有PyTorch reduction/exp的浮点路径，也不是替换原FP64模型。
完整保存每个cell的observed source IDs、reference IDs、a/b成本及缺失轴，
使artifact检查能独立重建两种聚合，而不冒称重做raw特征前向。

## 预先指定的输出

producer不读target：全部2691×128原分组能量、证书及新共同query测度封存
之后，才关联原TRAIN32 roles。原344448项分组成本/能量/缺失与原全部证书
必须重放一致，任一原始漂移为工程ABORT。

按原full-family同一规则记录CERTIFIED、REFUTED、UNRESOLVED。竞争者始终
包含全C128，不能过滤自身family为H0的reference。target有证书、wrong有
证书和严格唯一target分别记录；严格唯一要求所有wrong regions均REFUTED
或属于structural-H0，wrong有UNRESOLVED时不能称唯一。

固定报告全部原21个wrong证书及22个target证书的状态迁移，不挑有利案例。
对原wrong证书与其实际target，保存同域逐atom的成本差、mean_a及max_b，
统计target是否在每个atom的a/b都不更差、是否至少一项严格更好。仅当
逐atom支配实际发生且原分组排名逆转，才称实际测度支配反例。

原wrong certificate在新测度消失只是固定区域证据变化，不是识别救回。
即便新证书覆盖达到26，也不自动训练或宣称P GO；多个wrong可各有局部优势。
若原wrong仍在共同测度上胜target，应明确保留wrong-positive证据，下一机制
需要解释身份区别所需的可见内容，不能继续以局部存在性作为完整身份充分条件。

## 执行与退出

隔离worker、authority、结果，单个4CPU/64GiB/59分钟CPU作业，torch单线程。
内部3000秒guard和逐query原子封存，无自动续跑、不重复提交。完成后同job
另起artifact validator；明确不包含独立raw残差重算。
提交前完成共同测度的逐atom支配性质、原分组反例、固定N_H缺失补全界、
完整候选与源轴、平局/UNKNOWN/H0、NaN占位排除及产物篡改的合成检查。
实际空环境启动通过脚本文件验证。只做一次激活核查，随后退出、不监控。
