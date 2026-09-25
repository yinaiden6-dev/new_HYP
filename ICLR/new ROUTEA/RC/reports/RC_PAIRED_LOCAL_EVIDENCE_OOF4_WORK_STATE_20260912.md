# 同位置候选内容差：恢复入口

## 最终完成：主臂110/128，未超过CURVE111和GLOBAL114

5142654汇总修复已COMPLETED/0:0/41秒；result_validation.json、join_repair_receipt.json、completion_readout.json均已封存。RAW86、BASE108、BIAS109、MEAN109、CURVE111、JOINT110、GLOBAL114。主臂对BASE6救4损，对同容量CURVE0救1损，对GLOBAL1救5损，screen_positive=False。旧EVAL未重测。

全部16缓存和4折训练完成；81,280 logits独立核算及三个旧基线1152项逐图结果核对通过。当前全部任务完成，无需监控。完整结论见reports/REPORT_PAIRED_LOCAL_EVIDENCE_OOF4_V1_20260912.md；结果SHA256为8993ca3b81f4baee871964eca2e061c9fb6abc6b653dcd4c9607d78fc157adf1。

---

## 继续后的状态：训练全部完成，仅补汇总

16个缓存分片与5142612四折均COMPLETED/0:0。四折fresh拟合及81,280个logit独立核算全部通过，最大误差2.13e-14。5142613汇总失败于错误的5413长度断言：上游原输入资格程序第96行已规定完整排序为5412个去重身份（图库有5413张物理图）。这是汇总错误，不能记模型NO-GO。

已新增join-only修复程序programs/validate_rc_paired_local_evidence_oof4_join_v2.py、repair authority registry/rc_paired_local_evidence_join_repair_authority_v1_20260912.json和作业5142654（dev_accelerated,accelerated/10分钟）。保持原程序、训练、预测、prelabel seal及authority不动，只修正身份排序断言并检查5412身份全覆盖。实际spool与新launcher逐字节一致。修复成功后以result_validation.json及join_repair_receipt.json认定完成。

---

2026-09-12，依据用户“继续啊哥”继续有界 TRAIN 检验。

最新提交复核：5142610_0 已 COMPLETED/0:0，Slurm 实际运行 1分23秒；cache00独立验证最大误差9.99e-16，原F重放误差3.33e-16。其余5142611_1至15已全部并行RUNNING，四折及汇总为正常Dependency等待。这是阶段状态，最终指标仍以result_validation.json为准。

纠正上一段接口误判：RAW payload 顶层 references 包含完整 reference tokens；RoMa candidates 包含实际 query/reference visibility，并非只有哈希。CONDITIONAL4 已有历史实验，不能当作新方向再试。本轮未新增 backbone、分割或人工照片标签。

精确问题：相同 query token 上，candidate 与 RAW winner 的自由内容匹配差，是否包含旧平均值接口未保留、且能由 retrieval-only 训练利用的证据。唯一主臂 JOINT3，m2=sum(rho*d*abs(d))；与同参数 CURVE3 的 mean(d)*abs(mean(d)) 比较，并保留 MEAN2、BIAS1。所有残差围绕原各折 BASE7；另直接复用既有 GLOBAL7 强对照。数学信息缺口是已知事实，自然检索增益待检验。

数据：原 TRAIN128/32身份/32组、原四折，完整 RAW C128/127 challenger、FP64、threshold0、2000步末步。只读本折 TRAIN query→reference 标签；四折预测封存后才能 join 留出标签。本轮 128 不是原 EVAL128。

任务链全部已提交，每项 10 分钟，Slurm 实际 spool 均逐字节匹配冻结 launcher：

- 5142610_0：首缓存片，dev_accelerated；8 query/1024 candidate。已产出 cache-complete，含独立新进程全量 NumPy 重算。该阶段计时约 47 秒（不是整个排队加运行时间）。
- 5142611_[1-15%16]：其余15片，accelerated；afterok:5142610_*。
- 5142612_[0-3%4]：四折拟合、fresh参数位级重放、独立 NumPy 全logit/动作核算；accelerated；afterok 两组完整缓存数组。
- 5142613：dev_accelerated,accelerated，afterok:5142612_*；四折预测全封存后 join、完整图库排名、rescue/loss/net、组不确定性及报告。

当前提交结果不构成科学通过。预先固定主比较：JOINT3 对 BASE7、BIAS1、MEAN2、CURVE3、GLOBAL7 的 paired net；允许损失。超过全部对照才记本轮 screen_positive，仍不自动称 new HYP GO 或替换旧模型。

原模型 EVAL32 28/32、EVAL128 99/128；LISTWISE_UNIT1 26/32、101/128；上一轮 GLOBAL7_T128 27/32、98/128。原 TRAIN四折另为 RAW86、BASE108、GLOBAL114，禁止拼入旧 EVAL 账本。

工程记录：初版 authority V1 已冻结但未运行自然任务，合成断言全部通过后写入 preflight 收据时，原子临时文件被自身路径限制挡住。仅补临时收据路径许可；V2 独立冻结并记录 supersedes，完整合成预检通过。所有自然任务绑定 V2，无科学配方变动，不修改历史 authority。

计划：plan/RC_PAIRED_LOCAL_EVIDENCE_OOF4_V1_20260912.md。

有效 authority：registry/rc_paired_local_evidence_oof4_authority_v2_20260912.json，SHA256 ef38ab76d10f9fa2e85bb01c8431b2390b1e1889bad662dd0661fb3a4fad9191。

程序：programs/run_rc_paired_local_evidence_oof4_v1.py；独立核算：programs/validate_rc_paired_local_evidence_oof4_v1.py。

产物目录：results/rc_paired_local_evidence_oof4_v1/。submission.json 记录全 DAG；cache00..15/validation.json、fold00..03/independent_validation.json 为阶段验收；最后查看 result.json、result_validation.json、report.md。仅首片日志不是全结果。没有提交额外 EVAL 或后续新配方任务。
