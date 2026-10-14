# 新运行时干预候选对照

全部12个独立身份、60个Query arms与预先冻结的28个Reference重放。40张新编码中12个Query base与原干预base全部逐位相等；运行时版本、处理后端、模型配置与数值encoder源码一致。历史/新Reference分别保留145个完整矩阵，17个候选对共170个pair-arm统计。

28个新Reference中0个与历史cache逐位相等。fresh_vs_frozen.csv保留差值、edit-minus-sham效应及同索引轴上的argmax切换诊断；token trace与软ROI分区保留完整Delta、top2、切换、interaction与守恒分解。旧/新manifest worker源码SHA不同，旧worker源码快照缺失，不能确定差异原因；兼容性由数值源码/参数相等及12个Query原图逐位复现桥接。这些是预选候选的条件响应，不能视为全图库准确率、编辑后身份标签或因果机制完整证明。

旧字段 comparable_positive_lattice / comparable_negative_lattice 只表示有效 Token 的序号轴相同。旧图库处理器未知，因此不证明物理 patch 格点相同；新旧 argmax 变化是序号诊断。同一 bank 的 edit 与 sham 比较仍保留完整对应。
