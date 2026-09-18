# new HYP：593图、68身份、64组的检索监督交叉验证

用户明确选择593张分组交叉验证并要求开始。使用全部非formal392图片，595记录按图片SHA去重成593张；相同身份、相同来源组作为连通分量，五折按分量大小与固定SHA排序分配，禁止相关图片跨折。当前冻结为64分量，折大小119/118/119/119/118。旧EVAL来源进入本次开发池，因此本次是新开发协议，不是未触碰外部测试，不能与旧99/128直接相减。

## 已授权输入执行

复用256张已独立验证的当前runtime TRAIN128/EVAL128 RAW/RoMa输入，按图片SHA映射；逐payload校验绑定和完整C128。另337张分成43片、每片最多8张。原ColNomic查询编码、全5413物理reference RAW评分、身份去重5412、自然C128与原RoMa/FP64 weighted full-reference MaxSim运算保持不变。adapter直接编译冻结原程序的数值循环体。先补RAW，再补RoMa；GPU数组并行上限16。不得按target是否进入C128丢弃图片或插入target。

仅检索身份监督；不使用SAM、box、mask、point或ownership监督，不读取formal392私有标签/P/RoMa结果、D1-MI、GroZi或opened容量诊断。原部署、旧99/128账本不改。所有任务在北京时间9月11日24:00前停止。

## 固定模型比较

每折从零重训BASE7_ALL（原六统计量加bias）和BASE7_SMALL128。后者从该折训练池按组轮询、固定SHA顺序取128张，比较图片数量，不将其称为新增身份隔离对照。ALL使用该折全部474/475张。二者共享完整C128身份损失、seed17、FP64、2000步AdamW(lr=.03, weight_decay=.001)、零初始化和固定顺序，不用旧PAIR额外重复加权。它们沿用原七参数结构，不声称逐bit复制旧PAIR+FULL训练协议。

在BASE7_ALL冻结后，分别训练原CONSTANT1和CONDITIONAL4自由内容补偿；复用已定义的F/A/D与四项条件特征，不重新发明机制。补偿模型及优化规则沿用TRAIN128 disagreement版本。主比较CONDITIONAL4对BASE7_ALL，CONSTANT1用于判断条件化是否优于统一补偿；BASE7_ALL对BASE7_SMALL128检验更多图片的效果。

RAW本身正确时惩罚最大wrong正logit；正确candidate在C128且非RAW时提升target并抑制strongest wrong。target缺席C128的训练样本不能构造正确challenger，单列并不参与该身份损失；所有593张仍完整参与OOF评价，缺席计检索失败。此规则在读取新候选成绩前冻结。

训练实现及输入特征复核须通过后再提交CPU拟合；输入任务现在即可运行。每折仅打开本折训练标签，完成参数及全部held-out预测封存后，由另一步join该折评价标签。不在OOF结果后选seed、阈值、checkpoint或主臂。

## 读出与结论

报告所有模型总正确、自然C128召回、MRR、RAW救回/损失以及相对同折BASE7_ALL的救回/损失；报告64个分量上的等权差、按分量bootstrap、差异集中度。用户允许少量损失且要求可靠净增；不把旧EVAL128的一两张预算直接扩大或自动套到593张。不以内部正净增自动声明普遍new HYP成立或替换原部署。

本文件冻结实验与输入阶段，不表示输入齐备、训练完成或获得科学GO。
