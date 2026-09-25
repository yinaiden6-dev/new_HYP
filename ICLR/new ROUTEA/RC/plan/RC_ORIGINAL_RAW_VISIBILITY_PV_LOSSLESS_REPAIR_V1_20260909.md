# 原RAW+RoMa系统的可见性P/V接口修复

2026-09-09。依据用户要求，回到27/32与69/90所属的冻结七参数系统，
修复已确认的输入信息损失。此工作不恢复V9训练，不替换旧结果或P NO-GO。

## 已确认的机制依赖

原RAW C128与完整127 challenger回放逐条复现EVAL32的25→27、difficult90的
61→69。单独中和显式visibility-mass项，两组原救回分别全部消失；中和
normalized-local-similarity项也丢失全部原救回。对difficult90进一步保持
L=S/M、Lq=Sq/M、Lr=Sr/M，统一M=1后，经原六特征、原参数、原零阈值及
完整127对手重算，得到61/90、零SWITCH。全部34560个内容比值位级不变。

源证据：

- `results/rc_frozen_roma_action_terms_v1/result.json`
- `results/rc_frozen_roma_difficult90_action_terms_v1/result.json`
- `results/rc_frozen_roma_visibility_mass_factor_v1/result.json`

这些结论是固定原计算图中的依赖，不是像素空间因果，也不是新模型评估。
零显式mass项并非移除全部mass通道；最后一个因子一致干预才移除了全局
标量M在该特征构造中的作用，L内部仍含原visibility加权内容。

旧M来自完整双侧cell均值。V3/GISC/V6的atom接口只保留query中心及warp(query)
处的有限采样，V8评分只使用指定对应残差与合法component。一般无法从这些
信息重建旧M。未采样reference overlap区域可以改变旧M而不改变保留的atom；
这证明接口一般有信息损失，但不能冒称它解释了所有自然失败。

## 本次具体修复

P保存完整FP64 query_visibility、reference_visibility、canonical轴、来源
绑定及完整双侧均值与M。P不承担候选身份排名。

V从原RAW tokens与这份P输出，按原操作顺序重算：

    normalized query/reference tokens
      → complete reference weighted MaxSim
      → query visibility weighted mean
      → original mass multiplication

同时保留原half-axis Q/R诊断分数，输出原七参数action使用的相同字段。
不把local reciprocal均值、component面积或只在warp(query)采样的量冒充M。
原RAW候选、RAW score、七参数、阈值、tie和HOLD/SWITCH规则保持原定义。

## 输入与验收

使用原current-runtime TRAIN32+EVAL32完整64×C128。可从普通matched-three-arm
历史包提取RAW候选的visibility maps；只使用RAW字段，不访问D1-MI。提取的
全部map字节SHA必须等于原RAW+RoMa JSON记录。原tokens仍来自原RAW bridge
payload，独立输出RAW-only封存包，不重跑RoMa或编码器。

先做独立旧score函数oracle的合成算术和来源约束测试；再对64×128完整
候选验证新P/V输出与原REAL/M/Q/R标量，重放完整127 action。原EVAL32
必须保持27/32和原2救回0破坏；TRAIN与EVAL分别报告。

若数值或来源不一致，记录具体差异并修复接口，不能换候选、忽略样本、
调整阈值或直接复制旧分数当作新V计算。验收只是旧机制的无损分解，
不能命名为HYP科学成功。

## 科学边界

原27/32和69/90属于同一冻结head的不同已打开数据集；不混用本轮fresh-D1
TRAIN32全图A的27/32。此修复不使旧空间对照自动通过，也不证明连通区域
身份充分性、ownership或独立泛化。后续任何区域约束必须从已恢复且验证的
原机制出发，明确测量它新增或破坏的证据。正式392、D1-MI、GroZi不进入。
