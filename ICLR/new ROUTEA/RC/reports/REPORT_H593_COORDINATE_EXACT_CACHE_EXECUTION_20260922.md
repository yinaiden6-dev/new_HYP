# H593坐标精度缓存改造：执行状态

2026-09-22用户授权“改”。已实现缓存代码并接通验证、单个真实query、队列替换及后续全593批次。当前GPU逐位资格仍待运行；不得将实现完成写成加速已验证。

## 已完成

- `programs/rc_roma_exact_feature_cache_v2.py`：冻结f、refiner_features、matcher前向缓存；query跨128候选复用，reference/pair只保留当前对象。原match/forward/refiner/量化/评分代码未改。输入每次逐位核查、输出递归clone，隔离confidence等原地写。
- `programs/run_rc_h593_roma_coordinate_cache_v2.py`：完整128候选四精度的逐位资格；缓存生产调用原worker与原独立CPU验证器。原科学authority和评测入口保持，新增part/payload明确绑定缓存执行authority及qualification。中间数据及旧part断点保留。
- 本地自检通过：输出原地写隔离、query跨reference复用、输入漂移拒绝。V1测试在inference_mode外模拟原地修改而报错，未提交任务；V2让测试与真实推理模式一致，保留失败版本。
- 自动切换模拟四场景通过：成功切换、首个worker失败恢复、后继提交失败恢复、正在运行的旧任务不被取消。三种真实shell参数/工作目录动态检查通过，冻结源码SHA已核对。

## 已提交

- **5157565**：dev_accelerated，15分钟GPU，固定既有ordinal0/完整C128/四臂；比较新鲜未缓存与缓存的所有稠密输出字节、粗匹配、hook记录，以及原封存权重、坐标摘要、MaxSim命中/贡献、C4、127×6特征。固定合成head回放127分数，不读取真实标签。分别记录前向耗时和显存。
- **5157566**：cpuonly，5分钟，afterok:5157565。资格通过后才暂挂指定5157033_59–92中仍等待的项及其旧回调5157034，先提交一个真实缓存query到dev_accelerated；后继用afterany检查执行结果。
- 首个真实缓存query通过原CPU验证并有缓存执行回执后，取消被本程序暂挂的旧待运行项，释放新缓存dispatcher。新dispatcher等待任何仍在运行的旧任务结束，扫描全部合格query，随后最多46项一批直至593，再接原固定头/重训/诊断/汇总。
- 首个真实缓存query失败则释放旧队列；资格失败则5157566依赖取消，旧队列从未被暂挂。

原始C128、四种精度、FP64评分、五折训练和中间数据范围均保持。已合格59张结果复用。当前尚未部署到旧34项，也没有实测墙钟加速比。

## 入口

- 缓存资格及前向计时：`results/rc_h593_roma_coordinate_cache_v2/`
- 队列切换及恢复回执：`results/rc_h593_roma_coordinate_cache_cutover_v1/`
- 新批次调度：`results/rc_h593_roma_coordinate_cache_dispatch_v1/`
- 原科学结果及新增执行来源：`results/rc_h593_roma_coordinate_precision_v2/queryNNN/`，缓存生产额外写`exact_cache_execution.json`。

第一次提交因自动审批超时未执行，按系统允许重试一次后成功得到5157565；并非科学验证失败。
