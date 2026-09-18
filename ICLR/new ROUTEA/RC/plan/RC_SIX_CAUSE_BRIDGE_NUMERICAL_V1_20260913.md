# 六项桥接数值定义（首个自然缓存前冻结）

总方案所说16箱落实为[-1,1]的16等长区间、17个连续线性帽函数。每个值向两端点分配质量，避免硬分箱的浮点边界跳变。POOL按query visibility对MaxSim值计质量；MATCH按query visibility和reference token均匀质量对完整余弦矩阵计质量。均保留原六统计及FREE标量。

CHANNEL的128维为Σ_i normalized_wq(i) qnorm_i ⊙ rnorm_argmax(i)。128分量求和重建FREE；不声称保留全部tokens。输入取candidate−RAW winner差。CHANNEL_PERM只在每个query完整128候选内部按query_id哈希决定非零循环偏移，再作差；原六统计及真实FREE保留。该对照不是随机类别标签训练。

两读取方式为线性、以及线性加逐输入分量平方；没有隐式交叉项或新网络。两损失为全128 CE、等成本UNIT1最大错误softplus：RAW为真时softplus(max challenger z)，challenger为真时softplus(-z_target)+softplus(max other challenger z)。BASE冻结，所有残差从0开始，全训练query、2000步末点、FP64、AdamW .03/.001。不按留出选步数。各臂参数量不同，必须报告。

主比较固定CHANNEL_LINEAR_CE对FREE_LINEAR_CE、CHANNEL_PERM_LINEAR_CE、原BASE7、原GLOBAL7，同时检查图像净增和等组均值。其他23个新模型为预先声明的有限因素诊断，不把最好一项回称为预注册主机制成功。内部screen不等于new HYP外部确认。
