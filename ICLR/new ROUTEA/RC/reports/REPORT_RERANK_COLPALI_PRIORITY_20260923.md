> 更正：用户要求ColPali自己的自然C128。本文293→388/593只属于固定ColNomic候选的辅助对照，不能作为ColPali检索管线迁移结果。已另立rc_colpali_native_c128_v1，5160680首片、5160681其余49片、5160682汇总。新基线和新召回尚无结果。

# Rerank与ColPali优先执行

用户指令：优先跑rerank和ColPali。其他融合/统一采集的排队元素已hold；两个融合续提controller已停止，GPFS锁核验空闲；当时运行中的融合短片保留，结束后不再自动补交。

## Rerank

5160471首图评分已经通过官方yes/no输出一致性；保留全部逐候选分数和断点。5160484为50片全量，等待首图完整128候选验收；5160485五折CPU校准；5160486最终汇总。

早期实测每候选45–63秒，5160603单独测量tokenize/传输/前向，要求固定首对输出逐bit一致。不能据“通过模型加载”宣称速度正常或全量完成。

## ColPali

编码缓存593/593已经完整验收。新实验固定原ColNomic C128：ColPali内容基线、同监督内容七参数头、原RoMa整体M与自由图像内容L组成的五参数头、候选M绑定打乱控制。各自COST1/CE、原五折与2000步预算。此为简化机制迁移，不是完整局部u/v七参数模型，亦不宣称ColPali自己召回候选的端到端结果。

5160599首片12张评分验收通过；5160600补齐剩余49片；5160601五折头训练；5160602汇总。新编码器/RoMa前向为0，GPU只执行缓存tokens的矩阵评分。逐候选中间统计与argmax全部保存。

原图SHA和轴源、token缓存SHA、训练标签边界、模型及任务记录见registry/rc_colpali_mass_transfer_authority_v1_20260923.json与results/rc_colpali_mass_transfer_v1/。

## 最新执行验收

ColPali的50个评分分片、593张query及五折训练均已完成并有逐项validation。5160602正在最终汇总（允许dev_cpuonly和cpuonly调度）。

Rerank执行profile已排除预处理和传输为主耗时：首对三次复测tokenize 0.17–0.34秒，GPU transfer 0.015–0.016秒，forward 64.86–67.33秒，logit均为-1.265625且与官方一致。5160659继续定位模块耗时。全量5160484暂hold，避免现有速度超过每片16次续跑上限；首图5160471继续保存断点。解除全量hold前须确认吞吐量和续跑预算。

## ColPali汇总已验收

固定原ColNomic C128、H593五折：ColPali原内容293/593，CONTENT7_COST1 294/593，CONTENT7_CE 301/593；整体M校准MASS5_COST1 388/593、MASS5_CE 413/593。MASS5_COST1相对自身ColPali原内容95救回、0损失；组平衡净增95% bootstrap区间[0.11785,0.22563]。M绑定打乱后COST1降至233/593。分数、所有593条预测、五折训练隔离与汇总SHA通过检查。说明该整体质量校准在这一跨检索器固定候选协议中有迁移收益，不能声称完整局部七参数版本已验证、不能算ColPali全图库召回率，也不能把计数替代原ColNomic主模型结果。

Rerank瓶颈进一步定位：visual.patch_embed 62.855秒/总63.760秒，language_model 0.137秒。5160661验证patch卷积等价执行方式，中间输出与最终logit同时核对。原科学模型与已完成分数未改。全量5160484等待加速验收后解除hold。
