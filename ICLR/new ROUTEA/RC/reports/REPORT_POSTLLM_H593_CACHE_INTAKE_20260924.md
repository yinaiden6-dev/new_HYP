# 后 LLM 内部改造：H593 缓存与划分盘点

本次扩展采用原 H593 分组五折和原 ColNomic 自然 C128。只复用冻结编码结果；不复用旧 TRAIN16 的适配器、小头或 M 归一化。

| 输入 | 已有 | 需补 |
|---|---:|---:|
| Query 原始检索 tokens | 593 | 0 |
| 原 C128 与 RoMa M | 593 × 128 | 0 |
| Reference 检索 tokens | 全部所需候选 | 0 |
| Query 后 LLM hidden | 24 | 569 |

共核对 5324 个去重图像 token 文件的 SHA；核对全部 593 张 query 原图 SHA。旧24张 hidden 的封存、payload 和 parity 均逐一核对。旧 V1 只有失败的 parity，未当作可复用缓存。

| 折 | 原 TRAIN | 可训练（target 在 C128） | TRAIN target 缺失 | 全部 held |
|---|---:|---:|---:|---:|
| 0 | 474 | 457 | 17 | 119 |
| 1 | 475 | 454 | 21 | 118 |
| 2 | 474 | 451 | 23 | 119 |
| 3 | 474 | 461 | 13 | 119 |
| 4 | 475 | 457 | 18 | 118 |

缺失 target 的 TRAIN 行仍记录但不进入该 COST1 身份训练；held 一张不删，最终每张仅接收所属折的 OOF 预测。M 的均值、标准差在每折实际可训练行上分别计算。

共享 manifest 不含 query 身份标签。每折 TRAIN 标签独立存储；curator 仅保留封存路径，准备阶段没有打开。零适配器校验、实际 fresh_L0 和 projection 一致性仍需 GPU 缓存采集/校验完成；本盘点不意味着训练已完成。

后 LLM hidden 不能由最终 128 维检索 tokens 无损倒推，故仅569张 query 需要补冻结编码前向。RoMa 与 reference 不重算；原图/token/hidden 均采用只读指针，不复制大文件。

[输入 manifest](../results/rc_postllm_h593_v1/input_manifest.json) · [盘点 JSON](../results/rc_postllm_h593_v1/intake_audit.json) · [准备程序](../programs/prepare_rc_postllm_h593_v1.py)
