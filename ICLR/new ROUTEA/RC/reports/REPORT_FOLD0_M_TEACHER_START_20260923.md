# 第一折质量教师诊断已启动

用户确认继续区分表示与读出问题。仅启动一个COL_ONLY_PAIR学生、第一折，后四折继续暂停。首片5160039：dev_cpuonly、8CPU、32GB、10分钟；控制器自动保存和续跑，拟合及预测完成后自动执行冻结头汇总。

训练仅使用旧第一折457个有效TRAIN query的原生RoMa M，每query完整128候选，原8353参数质量网络、seed17、FP64、2000次query更新。119张held不参与训练，held输入清单不含教师M或身份。复用全部现有token缓存，无新编码器/RoMa前向、无GPU申请、无新决策头训练。

这是教师监督机制诊断，不替换原retrieval-only主模型，不以超过原105/119为目标。检验是否能恢复原生候选质量区分，再通过冻结原七参数M_ONLY回放及冻结五参数自由内容回放检验原纠错。第一条路径仍保留原生局部权重，不能称完全绕过RoMa。

准备检查通过：全C128逐对累积梯度与整批图梯度的误差2.22e-16；log损失等于中心化误差加均值偏差；M_ONLY独立NumPy特征、原M恒等替换和零质量情形通过。TRAIN/held清单互斥，源码SHA固定，提交spool逐字节一致。以上是工程检查，不是科学成功结论。

入口：
- `plan/RC_FOLD0_ROMA_M_TEACHER_DIAGNOSTIC_V1_20260923.md`
- `registry/rc_fold0_mass_teacher_authority_v1_20260923.json`
- `results/rc_fold0_mass_teacher_v1/status.json`
- `results/rc_fold0_mass_teacher_v1/checkpoint.pt`（产生后）
- `results/rc_fold0_mass_teacher_v1/result.json`（全部封存及汇总完成后）
