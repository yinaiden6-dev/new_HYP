# 既有自动mask的完整区域读出：有限正信号与覆盖失败

2026-09-09。固定既有SAM3缓存可用性：TRAIN2/EVAL7，9条query、36个
文件、246个自动mask。全部prompt与mask进入同一固定bank。先用冻结
alpha或alpha*wq定位seed，再选择最高SAM confidence的合法seed连通
component（至少4个native cells），V均匀读取完整component的原
full-reference MaxSim。零新encoder/SAM/RoMa forward、零训练。

P、support、完整C128分数与预测封存后才join标签；新进程复算通过。
result：`results/rc_cached_sam3_region_closure_v1/result.json`，SHA
bf7970d5df75809f55b095a6d9d28d70ca821b195dfdf1f30cd0a3c5abb5a1ad。

## 固定结果

| Arm | TRAIN（2条） | EVAL（7条） | EVAL对ALL新增/丢失 |
|---|---:|---:|---:|
| ALL，全query平均 | 2/2 | 4/7 | — |
| QUERY_REGION | 0/2 | 6/7 | 2/0 |
| REF_REGION | 0/2 | 6/7 | 2/0 |
| QUERY_REGION_PRIOR_HALF_ROLL | 0/2 | 3/7 | 0/1 |
| REF_GEOMETRY_SHIFT | 0/2 | 6/7 | 2/0 |

QUERY_REGION两条TRAIN都为H0（没有合法seed区域）。REF_REGION的两个
target也均H0；不是把H0平局算成错误身份预测。EVAL的QUERY_REGION为
7×128全部候选可读，REF_REGION有39个candidate H0，target7/7可读。

原完整NATIVE7/C系统在同一EVAL7为6/7。QUERY_REGION相对它救回0050，
同时丢失0006，因此没有整体净提升。不能把V-only的4→6说成原完整系统
4→6，也不能在缺缓存的25条EVAL上暗加fallback拼成新32-query成绩。

## 自动P的具体正例

DIFFICULT-0050：原ALL与原NATIVE7/C均错选背景CVS Nasal Mist（3368）。
冻结query alpha的seed为原query index338；固定规则从旧box prompt
的mask21选择出120个native cells的完整连通component。选择阶段没有
读取target标签或V得分。均匀full-reference V将target1130排C128第1。

DIFFICULT-0044同样从ALL错误变为区域V正确；原完整NATIVE7/C原本也
正确。QUERY_REGION相对其half-roll在EVAL多3个正确，支持本次读出对
seed位置有依赖，但不能从7条内部样本推成原空间门通过。

REF_REGION与QUERY_REGION没有正确数增量；REF_GEOMETRY_SHIFT也保留
6/7。当前证据不支持reference-conditioned seed比query-only seed更强。

## 保留边界

这次结果证明一个已运行的自动区域选择规则可以在两个已打开样本上
改善纯内容读出，其中包含原完整系统的一条失败；也暴露了两条TRAIN
无法生成有效区域，以及0006仍然识别错误。没有HYP GO或部署替换。

旧SAM PT只存image_index/prompt，没有历史input image SHA。本次绑定
原CSV/per_sample索引和完整路径、当前图像SHA与实际mask bytes，不能
冒称历史输入字节已严格重放。9query可用性子集也不提供外部确认。

这不撤销旧系统27/32、69/90，也不撤销独立NATIVE7/C的28/32。旧成绩
仍是完整系统基线；新P/V需要证明自己的增量与机制，不能用旧成绩替代。
