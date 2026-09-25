# HYP：生成通过后的身份信息研究决定

2026-09-08。依据用户继续研究、检索原始文献与创新的指令。
这是研究定义和下一诊断的依据，不是selector训练authority或科学GO。

**新事实。** Job5137251产出generation coverage 29/32，冻结门26/32。
独立JSON产物审计已确认12,288条candidate/control families的结构、轴、SHA
和覆盖归约。全部23个旧V6成功query仍有目标region；旧9失败中6个仍有。
P_COORD为0/4096非空family，故后续其margin下降不能单独证明身份空间关系。
来源：`results/rc_cycle_closed_region_generation_artifact_audit_v1/result.json`。
该审计没有加载原banks，完整maximality/resume raw replay仍需补验。

**本项目尚未解决的是身份可辨性。** 原九失败的case03/04中，target和
wrong旧支持均能通过native-cell cycle。它们在新region内仍可存在。
因此可靠局部对应不是正确规格的充分条件；这不否定几何的生成价值。
下一步需确认现有观测中哪些信息可区分candidate，然后定义能保留该信息的P。

**原始文献提供的设计约束与创新边界。**

- [Doppelgangers, ICCV 2023](https://arxiv.org/html/2309.02420v1)，§5.1—5.3：
  在粗对齐后联合使用图像以及keypoint/match分布判别欺骗性相似图像。
  可借鉴“匹配以外的信息也要保留”；其额外学习网络和训练标签并不是
  当前已批准输入。该论文没有授权我们将任意未匹配位置直接判为身份冲突。
- [Finer-CAM, CVPR 2025](https://arxiv.org/html/2501.11309v2)，§3.2：
  对类别logit做比较，以突出有区分力的特征；这是解释方法，不能代替本项目
  target-free区域生成和自然验证。共有特征抑制、竞争差分本身已有先例。
- [Jégou and Chum, ECCV 2012](https://cmp.felk.cvut.cz/~chum/papers/Jegou-ECCV12.pdf)，
  §3—4讨论BoW中共同缺失词与共现计数及whitening。这里的negative evidence
  与本项目“可见规格冲突”含义不同，不能借用名称混淆UNKNOWN和已观测反证。

候选贡献应落在：在已有RoMa/ColNomic观测上，构造可重放的连通P假设，
同时保留它的身份支持、竞争性不利证据以及未知观测，并验证这些信息确实
带来严格匹配对照之外的增量。当前尚未确立该机制的新颖性或有效性。

**先检验一个可以直接计算、尚未实证的具体来源。**

原atom代码定义 `c_k(a)=s_k(a)-b_k(a)`，其中s是assigned cosine，
b是该query token对该reference全部有效token的平均cosine。
b随reference而变，并不是所有candidate共享的零点。
对同一个固定region H与同一个固定竞争candidate l：

`mean_H(c_k-c_l) = mean_H(s_k-s_l) - mean_H(b_k-b_l)`。

因此正的centered差值可能来自更好的实际对应，也可能来自更低的reference
基线；两者的占比必须在固定pair上分解。这个恒等式本身不说明centering
有害，不是切换成raw cosine的新模型，也不说明数据中已经发生了方向反转。
原assigned/baseline/centred在atom生产时以FP32计算，baseline未独立存储。
诊断从已保存s与c反推的量必须标为implied baseline；精确分解针对这两个
已保存数值，不冒称对原FP32 mean-reference中间值的逐位重放。

下一单次诊断遍历全部32×C128 REAL最大连通region，继承V6的“同region
先平均，再对完整127竞争者取最大”定义。保留无效atom的保守界及UNKNOWN，
不在查看target后重选region或对手，不扫描温度、top-K或多个公式。
全部target-free数值封存后才开原opaque roles，报告正向证据存在性、
wrong-positive存在性及上述固定pair分解。无训练readout仅作描述性诊断，
不报告新模型GO或用它替代matched controls。
该诊断同时从原banks重放全部REAL family及独立union-find检查，补验生成
产物的完整最大分量；不把仅检查JSON中的布尔字段称为raw replay。

**为何继续调已有正乘子不够。** 若两个输入经当前摘要映射得到完全相同
的观测z，则任何仅以z为输入的确定性共享readout都给出相同输出。
这是函数定义的直接结果，不是“本批失败样本已经观测相同”的实证结论。
若关键区别在从128D token关系压成标量时消失，后续geometry乘子无法恢复。
是否需要保留对齐后的多通道交互，应由信息保留诊断决定，不能再凭愿望选式子。

拟合前仍须冻结唯一selector/loss/family reducer、真query-only区域的独立
生成定义以及原强共同轴竞争对照的门。原V6净差+1 NO-GO保留；其JSON
兼容验证修复已作为Job5137263单独提交，无训练、无门变化。正式392、V、
HOLD/SWITCH、ownership和受保护D1-MI/GroZi均未开启。


**文献收束后的单一机制方向，尚未冻结为模型。** 在完整connected H内
固定观测与对应的质量预算，联合保留已解释支持和未解释观测。同一reference
cell被重复指派不能自动获得额外质量；不能通过删除低匹配部分或重新归一化
将规格差异排除。缺失与无法确认共同可比性的观测仍是UNKNOWN。
若要使用完整通道信息，接入已有封存的query/reference 128D tokens，保持
backbone冻结；九个标量只保留若干投影，不能恢复向量方向。这不是PCA压缩。
这个方向仍缺可比性判定与具体shared readout，尚未达到训练就绪状态。

[SuperGlue §3.2—3.3](https://arxiv.org/html/1911.11763v2#S3.S2)
提供部分assignment/容量约束的先例，但其dustbin有真实对应监督，不能
当作当前的identity null。
[DeepEMD](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_DeepEMD_Few-Shot_Image_Classification_With_Differentiable_Earth_Movers_Distance_and_CVPR_2020_paper.pdf)
提供运输预算的先例；其cross-reference权重降低不共现部件贡献的做法
不能直接搬来当规格冲突读出。该方向的有效性与新颖性尚待证明，不扫描
多个度量或运输参数，不在当前无结果时先将一个公式写成已获验证的理论。
