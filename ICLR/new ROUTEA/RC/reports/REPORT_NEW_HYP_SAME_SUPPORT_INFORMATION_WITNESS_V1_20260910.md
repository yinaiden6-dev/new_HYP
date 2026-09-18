# new HYP：同支持竞争信息不能由候选自身摘要普遍恢复

状态：精确的无标签 token 接口反例及新进程独立算术验证均 PASS。
这不是自然识别成绩，也不声称下述 token 必然由某张图像经当前编码器生成。

## 已证明的有限命题

即使同时给出全部 C128 候选各自的旧 C 四标量、自由内容均值 F、
离散度 V 和固定 RAW 分数，也不能在所有有效 token 接口输入上确定
“在 g 的同一 query 支持上，g 相对其它 reference 的优势”J。

证明方式是构造两种输入 A、B：以上全部自身摘要逐 bit 相同，
而 J_g 从 +0.25 变成 −0.25。因此不存在只读取这些摘要、却对
所有此类接口输入都精确恢复 J 的确定函数。该命题不涉及正确身份标签。

## 可复核的构造

query 为四个正交单位向量 e1、e2、e3、e4，补零至128维。
g 的权重为 (1,0,0,0)，h 的权重为 (0,1,0,1)。所有 reference
可见性权重均为1，其余126个 reference 与 query 正交。

| 量 | 输入 A | 输入 B |
|---|---|---|
| g 的完整自由 MaxSim profile | (.5,.5,.5,.5) | (.5,.5,.5,.5) |
| h 的完整自由 MaxSim profile | (.25,.5,.75,.5) | (.75,.5,.25,.5) |
| g 自身的 F、V | .5、0 | .5、0 |
| h 自身的 F、V | .5、0 | .5、0 |
| h 在 g 支持上的内容均值 | .25 | .75 |
| J_g | +.25 | −.25 |
| J_h | 0 | 0 |

h 在其自身支持上只读取第二和第四个 query token，所以自身摘要
看不到第一、第三位置的交换；在 g 的支持上，两种 h 的解释能力不同。
全部128个候选的旧 S/M/SQ/SR、F、V，以及固定 RAW 分数和基线 winner
均保持不变。原127条 challenger 的六特征也全部相同。FREE 新增列为0，
SPECIFICITY 新增列则改变符号。

所有输入 token 都以 FP16 精确表示且已经是单位向量。reference token
的主分量 .25/.5/.75 分别配15个 .25、3个 .5、7个 .25 的 query 正交
分量，使平方范数精确等于1；FP64 normalize 不改变任何分量。
程序使用冻结的旧 scorer、旧特征函数和本轮同支持竞争接口完成回放。
独立子进程另写 scorer、特征及逐 reference 聚合公式复核全部输出。

## 对主线的意义与边界

这个反例说明：同支持跨 reference 的竞争表包含旧自身摘要可能丢失的信息；
它不只是对旧六特征作另一种非线性变换。它也不证明这些额外信息能改善
自然识别、能决定用户意图，或能通过当前开发集的晋级条件。
识别收益仍由单独的真实数据训练与固定对照结果决定；这里未读取任何
自然输入缓存、标签、成绩或任务状态，也未改动任何模型。

## 证据

- 程序：[verify_rc_same_support_information_witness_v1.py](../programs/verify_rc_same_support_information_witness_v1.py)
- 输出：[result.json](../results/rc_same_support_information_witness_v1/result.json)
- 独立验证：[validation.json](../results/rc_same_support_information_witness_v1/validation.json)
- 输出 SHA256：`963eafe154cd8b9f5eaa53f3fb2a278fbfd642505fe47712e78f2ce415057cf8`
- 验证 SHA256：`7988354049ccb0589f515bbf9d49c123aac1778fa68fbb44f27e9d9312b236a7`
