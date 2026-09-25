# 同位置候选内容差：TRAIN128 原四折有界检验

用户 2026-09-12 连续明确要求继续推进。本轮延续该授权，只执行此处冻结的一组 TRAIN 开发对照；旧 9 月 11 日截止不用于阻止后续明确授权。原模型和所有历史产物保留。

已确认的问题：逐候选的边际均值、方差甚至直方图不能普遍恢复候选与 RAW winner 在同一 query token 上的联合比较。该数学信息缺口不等于自然身份增益，也不解释所有历史失败。此前同支持 specificity、逐候选 variance、标量曲率、CONDITIONAL4 和早期局部门控均已做过；本轮只检验下面精确定义的统计量及接口，不声称首次使用局部比较。

数据为原 TRAIN128（128 图、32 身份、32 来源组），原四折不变。不是原 EVAL128，也不读取 EVAL32、EVAL128、其他 H593、D1-MI、正式 prerecall 或 ownership 数据。训练仅使用本折 query→reference 检索标签；无 OCR、SAM、人工照片类型或区域标注。

所有 C128 候选、所有 query/reference image tokens 均保留。由缓存 FP16 tokens 转 FP64 后 normalize，计算 a_g(i)=max_j cosine(q_i,r_gj)，允许搜索完整 reference。旧六特征及其 reference visibility 完整保留，原折 BASE7 不重训。

对 candidate c 与 RAW winner w，u_i=wq_c(i)+wq_w(i)，rho_i=u_i/max(sum(u),1e-12)，d_i=a_c(i)-a_w(i)。定义 m1=sum(rho*d)，m2=sum(rho*d*abs(d))。使用共同 query 位置，未使用硬 RoMa reference 对应。零支持时两值均为零。原 F=(wq*a).sum()/clamp(wq.sum()) 必须与已验证缓存数值重放一致。

四个残差臂均以冻结 BASE7 的 logits z0 为起点：

| 模型 | 新参数 | 输入 |
|---|---:|---|
| BIAS1 | 1 | 1 |
| MEAN2 | 2 | 1, m1 |
| CURVE3 | 3 | 1, m1, m1*abs(m1) |
| JOINT3（唯一主臂） | 3 | 1, m1, m2 |

每一非恒定特征用本折 TRAIN 的所有 127 challenger 值计算均值及总体标准差，scale 下限 1e-12；bias 不标准化。CURVE3 与 JOINT3 同参数量同协议，仅最后统计量不同。m2 是有符号二阶量，不是方差；它既可能由局部幅度/分布产生收益，也可能来自对应关系，单独胜出不能证明空间因果。

统一 FULL C128 CE：RAW logit 0，全部 127 challenger 参与分母；AdamW lr=.03、weight_decay=.001、FP64、零残差初始化、2000 步、取末步，无温度/阈值/epoch/seed 扫描。物理 candidate 顺序及 max>0 SWITCH 否则 HOLD 保留。比较 RAW、原折 BASE7、已有同协议 GLOBAL7（复用冻结预测，不重训）和四个新臂。

缓存按原 16 分片并行，每片 8 query。每片在独立新进程用 NumPy 重算全部 cosine/max/联合统计及旧 F；拟合后 fresh 重训位级复现参数/预测，再独立 NumPy 核算全部 logits 与动作。四折预测均封存后才 join heldout labels，检查 query/image/identity/group 无交集。报告完整排名、逐图 rescue/break/net、每折结果和组不确定性。

主结果预先固定：JOINT3 对 BASE7、BIAS1、MEAN2、CURVE3、GLOBAL7 分别报告 paired net。允许损失。仅观察超过 BASE7 不称技术突破；超过 CURVE3 是这项额外统计有用的开发信号，超过所有对照才标记本轮 screen_positive。不据此直接宣称 new HYP GO、部署替换或旧 EVAL 成绩提高。无收益只否定本统计与本训练配方，不能否定全部局部信息机制。

作业每项 10 分钟上限，缓存 16 分片，拟合四折，依赖完成后汇总。原始量一次性封存，不改正在运行程序；工程失败只修复明确错误并留下新版本/记录，不读取结果选新模型。
