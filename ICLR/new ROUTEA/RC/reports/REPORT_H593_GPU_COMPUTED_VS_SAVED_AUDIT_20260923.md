# GPU 中间量：已计算、已保存与跨任务重复

结论：存在用户指出的“同一次前向本可采集，早先没有保存，后来另一个任务重算后才保存”。另有“已经保存，但另一任务未读取”的重复。两者均有源码证据；不能把这些重复一概归为干预必需。

## 已确认的具体链路

1. 原训练128 RoMa 采集器 `materialize_rc_original7_train128_roma_v1.py:323` 执行完整匹配；324–342 行只把最终 token 级 u/v、评分与来源写入候选记录，343 行删除完整 prediction。粗匹配和细化阶段已被执行，但各阶段输出与完整单图特征未在这一记录中存档。历史缓存能复用终端评分，不能倒推出未保存的阶段量。
2. 内部通路通过 `run_rc_h593_m_inside_v1.py:40` 复用视觉 runner，执行原图 NATIVE 并保存七阶段；原视觉 runner `run_rc_h593_m_visual_origin_v1.py:131` 又为同一候选执行 NATIVE。自然 query000 的全部128候选、七阶段、双方向、全部已保存字段已经跨两份历史文件逐位相同。证据 `cache/rc_h593_shared_stages_v1/existing_native_parity.json`。这是“已存仍重算”，新复用版本的实际 GPU 跳过检验尚未结束，不能声称全量已经消重。
3. 视觉/内部 observer 在 `rc_roma_visual_origin_v1.py:83` 收到原分辨率 confidence 与 warp，但持久化的是 token 权重、16x16 overlap/warp 摘要及 shape。坐标实验 `run_rc_h593_roma_coordinate_precision_v2.py:149` 重新挂 matcher hook，保留完整 coarse warp_AB/BA 与 confidence_AB/BA，再于196行写入 `coarse_matcher`。相同原图、模型、预处理下，前一轮本可在该 hook 同时保存这些粗输出；只保留16网格不能无损还原它们。
4. 单图粗细描述子在普通完整 RoMa 前向中本来就会产生；融合采集器 `run_rc_h593_feature_fusion_cache_v1.py:164` 单独运行 f 与 refiner_features，保存 token 对齐 components/projected_inputs。对相同模型和输入，早先完整前向可同时生成并保存这些融合输入。现在5324份融合特征已完成，后续训练不再需要重新提取这份特征银行。

## 当前仍有限制的保存内容

- `rc_roma_m_inside_v1.py:45` 的 appearance / joint context / matching position 完整张量只在首个自然 pair 保存，其余 pair 仅 shape、RMS、16网格摘要；不能称全部 A/J/P embedding 已缓存。
- 每层 refiner confidence delta 在内存里完整产生，但保存的是均值、RMS、16网格。内部通路各分支的完整 token 权重只保留粗阶段和 HR1，其余阶段保留均值。它们足够当前已注册的 M 读出，但不足以任意重放新增内部干预。
- 当前坐标缓存已在同一 worker 内复用 query/reference 描述子和当前 pair matcher；视觉缓存也复用不变侧的单图特征。因此不是每条干预都从零重算。灰度、低通、打乱和坐标变更后受影响的匹配/细化确实需要新结果。
- 完整高分辨率激活不应因为“可能以后用”无差别落盘。应按现有消费者需求，合并同一前向能导出的必要字段；输入改变或可训练参数改变的输出单独计算。

本次核查只定位重复和保存缺口，没有修改冻结科学脚本、覆写已有中间结果，亦未增加科学实验臂。重复开销尚未完整计时，不能把重复前向次数直接等同于整轮可节省比例。
