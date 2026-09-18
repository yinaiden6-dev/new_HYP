# 593图实验：RAW落盘错误修复与依赖重接

本次是工程失败，没有得到新模型NO-GO。旧RAW数组5140736的43片全部退出1，耗时2分08秒至5分03秒，均未触及时限。日志共337条不同query的TOKEN_RAW_QUERY_READY；随后统一在torch.save(payload, dot-prefixed temporary path)处收到invalid file name异常。raw目录没有payload、receipt或validation，不能恢复仅保存在进程内的计算结果。

256张复用特征的32片（5140738）已经全部完成。本轮重新核对全部payload与receipt/validation绑定及SHA，全部通过，保留不重算。保全记录为results/rc_new_hyp593_oof5_v1/reuse_features_preservation_v2.json。

## 唯一代码修复

新增programs/materialize_rc_new_hyp593_inputs_v2_serialization.py，仅替换原adapter的savepayload函数：用显式打开的文件流交给torch.save，flush/fsync完成后硬链接发布最终payload。旧源码、旧authority、训练程序和所有RAW/RoMa数值循环及验证函数均不改动；输出增加serialization_repair来源绑定。

两套真实运行环境都已复现旧临时路径错误，并完成新接口实际写盘、weights_only+mmap加载、FP16/FP64及负零逐位保持、禁止覆盖及临时文件清理检查。结果分别在serialization_repair_e0/raw/validation.json与serialization_repair_e0/roma/validation.json。这只证明写盘修复，不声称重跑已完成。

## 当前执行链

- RAW修复数组5140924：43片，上限46并行，10分钟/片。
- RoMa修复数组5140925：依赖全部5140924成功，上限46并行，15分钟/片。
- 旧死依赖RoMa 5140737已取消。
- 保留新特征job5140739，依赖已改为全部5140925成功。
- 原拟合5140747、join5140748、结果复核5140750继续使用原Job ID和原程序。

本轮已逐项核对调度依赖及时限。提交激活时修复数组为PENDING；未继续监控。北京时间9月11日24:00截止不变。

完整提交记录：[serialization_repair_submission_v2.json](../results/rc_new_hyp593_oof5_v1/serialization_repair_submission_v2.json)。

原EVAL128 99/128依旧是旧模型历史结果，593图实验尚未进入拟合。不得将工程失败描述成扩大训练或new HYP机制失败。
