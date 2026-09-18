# 旧V9保持暂停，先闭合RAW与局部V的输入差异

2026-09-09。用户询问现在是否仍需启动V9。当前结论：不启动旧V9。

## 当前已验证的新结果

Job5138508完成且产物复算通过：固定0006已选自动H147，原V targetrank6，
KEEP_H仍6，裁剪后在映回同一物理H的608个新token上为7；原图token、
完整a与H147读出全部逐bit复现。这次裁剪重编码没有修复局部V错误。
source result SHA：
cc68ecbeac2de1395b9187fdec709de6c8bf434b39716f7498a0c6384ba5d8de。

同一query的原RAW base却已经将target599排第1，原NATIVE7/C保持正确。
因此不能把局部V失败直接称为原系统失败、样本必然不可识别，或编码器
根本不含有效信息；应先对齐两个读出的实际输入和算术。

## 输入区别已经闭合到字节

当前C128全部128个reference均为同一VALIDATED4976源：原完整passage
是前4 template + N个image + 后7 template。视觉切片、原image indices
和完整passage source SHA均128/128一致，排除了reference源被替换。
query的三条件也保存了同次image、11个template和原RAW拼接顺序。

旧RAW是完整序列、GPU FP32、不再次L2归一化、SUM-MaxSim；当前局部V
是image-only、FP64重归一化，并读取全image或H子集。固定query集合时
SUM和MEAN只差共同正比例，不能单独解释排名变化；模板与算术的实际
贡献仍须通过桥接量化，不能提前都归因于模板。

来源：`results/rc_difficult0006_RAW_full_vs_image_reference_bridge_v1/`。

## V9与当前已发现问题不相符

旧runner草稿当前SHA仍为
87912a354a56142b35acebd94e4fc217d96645dcb346d33ffe6515b2c3af337c，
与原暂停凭据一致。其机制仍为rank-one变换、RoMa硬指派cosine减均值、
四格连通区域及P-only拟合。尚无对应core、独立core E0或执行authority。

没有证据说明该度量变换能修复本次完整/局部序列差异。故保持原暂停，
不把恢复训练或增加参数作为下一步。见
`registry/rc_v9_keep_paused_after_RAW_interface_check_v1_20260909.json`。
这不等于证明所有度量学习不可能有效，也不更改任何旧结果。

## 已提交的唯一下一计算

Job **5138523** 已提交，唯一启动核验为PENDING（Priority），不持续
监控。只对已保存向量重算评分：零encoder/RoMa加载或forward、零训练。
原5413 gallery物理batch16只为重放旧算术，所有输出和比较仍限原C128。
ORIGINAL原128分数须先逐bit回归，随后才执行固定Q/R token分组和
数值路径分解。原全图H+template包含全图上下文，不能称region-only。

authority：`registry/rc_difficult0006_raw_interface_bridge_authority_v1_20260909.json`，
SHA fd5db1cc064e3c9a6d51bc1cef6fe9185f77278ab281ac9d19483173f21c932a。
程序E0及独立源码审查通过。新进程validator重算全部矩阵并字节重放，
另有旧完整RAW AST和synthetic独立算术对照；不冒称全部自然分解已有
另一套独立算法。接手读取同prefix结果目录的result/independent_validation。

终点仍为reference HYP；不追加ownership要求，不更换标签、候选轴或
统计分母，不将桥接回归成功称HYP GO。
