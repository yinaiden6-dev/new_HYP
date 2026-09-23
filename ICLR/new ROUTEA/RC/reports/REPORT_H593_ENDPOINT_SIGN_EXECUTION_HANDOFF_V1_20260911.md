# J端点符号分支已提交

固定同一593图/68身份/64组/五折，保留原七参数基头；原593方案不改。只训练新增参数，比较BASE7、RELATIVE1(r)、RELATIVE2(r,1)、ENDPOINT2(r,s)。主比较ENDPOINT2对BASE7和同为两参数的RELATIVE2，检验端点状态作为条件偏置是否优于全局偏置。r/s保留归一化端点信息，未声称保留绝对幅度。

模型/缓存实现和实际序列化检查通过；共享相对差的正/负端点镜像在REL2相同、END2可区分；零残差、零J、候选置换、检索损失梯度检查通过。诊断缓存、原593汇总和其它fold训练标签/基头的读取均被实际冻结audit函数拒绝。

| Job | 阶段 | 依赖 |
| --- | --- | --- |
| 5141116 | 256张J缓存及NumPy重放，32片CPU | 无 |
| 5141117 | 337张J缓存及NumPy重放，43片CPU | 5140925 RoMa与5141116 |
| 5141118 | 端点符号五折拟合及新进程重训/预测重放 | 原基头5140747及两个J缓存数组 |
| 5141119 | 五折预测封存后join | 5141118 |
| 5141121 | 独立动作/排名/分组计数复核及报告 | 5141119 |

缓存数组每片10分钟、上限46并行；拟合每折30分钟，5折并行；join/review各10分钟。全部使用CPU，不增加RoMa/encoder计算。用户RAW10/RoMa15与截止UTC9月11日16:00保持不变。

原593主链最近检查：RAW43/43片完成；RoMa9片完成、10片运行、24片排队，未见失败；这是当次快照，不代替后续实时状态。新分支没有新模型成绩，不能与旧99/128混比。

数据来自正常资格化tokens/maps，重新计算J；不读取rc_opened_诊断产物训练。原每fold的BASE7_ALL参数冻结，BASE的REAL/CBIND预测须与原fold逐位一致。新模型的REAL、J_BIND（只打乱J）和CBIND（整包证据与J同打乱）固定报告。所有方法2000步、同损失、同FP64/seed17，不按结果换主臂或checkpoint。

计划：plan/RC_H593_ENDPOINT_COMPETITION_SIGN_V1_20260911.md。
结果预期：results/rc_h593_endpoint_competition_v1/result.json、result_validation.json；自动报告reports/REPORT_H593_ENDPOINT_SIGN_OOF5_RESULT_V1_20260911.md。尚未宣称新性能或new HYP GO。
