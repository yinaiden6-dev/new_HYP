# new HYP product / response factorial V1：独立源码审查

状态：`PRODUCT_RESPONSE_FACTORIAL_SOURCE_REVIEW_PASS`。本结论只表示下述冻结源码的静态检查通过，不表示模型训练完成、运行时回归通过或科学增益成立。本审查未运行模型、提交作业或查询调度器。

## 审查绑定

- `programs/run_rc_product_response_factorial_v1.py`
  SHA256 `3d8190a481de2dc4bf76a5be7a1d9c795d1faec8471271c309fe7bff54bf23f3`
- `plan/RC_PRODUCT_RESPONSE_FACTORIAL_V1_20260909.md`
  SHA256 `1220b46a48c1f9e83d5b0d3bf440e2bd7da95081cb389f5843a3360450c78a0e`
- `slurm/rc_product_response_factorial_v1_dev_cpuonly_59m.sbatch`
  SHA256 `1bd7d771d888f891214c80e3a7dcf7b5e2f4bd5030ba7d54d605d35f19e384c3`

## 实际完成的检查

1. **四头与列映射。** JOINT4 `[0,2,3]`、PRODUCT5 `[0,1,2,3]`、RESPONSE6 `[0,2,3,4,5]`、ORIGINAL7 全六列，含 bias 分别 4/5/6/7 参数，与计划完全一致。新头只从原 REAL / C_BIND 六列中选择原 FP64 元素；完整头沿用原 native key。未发现复制替换造成的错列、错 family 或旧状态名误用。
2. **旧输入与新选择的回归。** 原 helper 先按冻结 candidate_feature 重建全部 native 六列并核对 tensor SHA。每个子集另经 literal 元素索引和每列 float.hex 核对。PAIR64 与 FULL64 的 JOINT4 / ORIGINAL7 还核对上一轮 128 条 feature seal。PAIR 只用于 REAL 训练，FULL64 含 REAL / C_BIND；原 PAIR V1 / V2 混合控制定义未重算。
3. **训练代码无替换。** 调用 helper 从原 runner AST 中提取的 train_head，只向同一 FAMILIES 字典注册子集字段及列名。原 FP64、零初始化、seed17、2000 次更新、PAIR 加权 BCE、FULL sign loss、AdamW lr=.03 / weight_decay=.001 均保留；没有新优化器、阈值或 checkpoint 选择。每头训练都会重新设 seed 和初始化，头顺序不复用参数状态。
4. **双基线回放。** ORIGINAL7 旧参数先与独立 NATIVE7 参数 seal 比较；本轮训练所得 JOINT4 和 ORIGINAL7 权重/bias 逐十六进制回归。EVAL 两头分别与上一轮 JOINT4、原 NATIVE7 的 REAL / C_BIND 动作完整字段比较。上一轮 parameters、result、input_closure、independent_validation 的四个源码内固定 SHA 已独立读取文件并验证匹配；旧结果与计划中的 25 / 26 / 26 / 26 / 28 一致。新运行时回归仍待作业执行。
5. **EVAL 时序。** U.prepare 建立角色读取 barrier，只 join TRAIN32；追加阻止上轮 result 和 validation 的语义读取。允许旧参数及无标签 feature seals 用于回归，没有把旧 EVAL 正确性送进 fit。四头参数和 FULL64 REAL / C_BIND 全部 127 logits 写入并 seal 后，summarize 才 release barrier、join EVAL 角色并读取旧结果。预测在 join 后逐条重放；自写 predict 与原 actions 使用相同的物理行号 tie break 和 `max_logit > 0` SWITCH 规则。已打开 EVAL32 不因此成为 untouched 数据。
6. **四项 factorial 与门。** 预先列出并统一输出 PRODUCT5−JOINT4、ORIGINAL7−RESPONSE6、RESPONSE6−JOINT4、ORIGINAL7−PRODUCT5；没有从两组中择优后省略另一项。PRODUCT5 和 RESPONSE6 都分别报告相对完整头和 JOINT4 的救/损/净增及等权 supergroup 方向。简化候选要求相对完整头 break=0、net≥0，并相对 JOINT4 query net>0、group 平均差>0；该布尔条件确实要求保留完整头全部正确集合。严格新模型改进另要求相对完整头 net>0 且 break=0。
7. **独立重执行。** launcher 顺序 run / validate；validate 为新进程重建输入、重训四头、重放所有预测和完整结果，而不是只复用产出参数计算指标。两阶段均有北京时间 9 月 11 日 24:00 对应的 UTC 截止检查，且 wrapper 的 timeout 覆盖每阶段。输出与授权均 append-only。
8. **Launcher 与语法。** Python AST 解析及 `bash -n` 均通过。dev_cpuonly、8 CPU、4G、59 分钟、export NIL、PATH、SLURM_TMPDIR fallback、禁 GPU、空 LD_LIBRARY_PATH、固定线程设置已静态确认。实际 NIL 环境启动预检由主执行进程另行完成，本报告不冒充已经运行该检查。

## 解释限制

- 这是一组固定优化协议下的嵌套输入子集消融，参数数不同，不能称等容量比较。
- 四头都保留 M 与 L；RAW gap 本身来自内容检索，L 仍是 visibility-conditioned 内容量。这里没有“无质量”或“无内容”的强因果臂。
- 正 M、正 L 且无 epsilon / floor 时，计划的 `dS=(dM+dL)/(1+dM*dL)` 是已有代数恒等式；它解释 S 对线性读出的函数基作用，不证明新独立原始信息或普遍统计交互。运行没有用恒等式替换原 FP64 scorer。
- 结果仍限于已打开的 EVAL32、原 RAW C128、原 all127 HOLD/SWITCH。source PASS、内部简化候选、外部可靠性及普遍 new HYP 结论是不同证据等级。

结论：未发现妨碍冻结与执行的源码缺陷。科学方向由这四个预定比较共同判断；本报告不预测哪一头会成功。
