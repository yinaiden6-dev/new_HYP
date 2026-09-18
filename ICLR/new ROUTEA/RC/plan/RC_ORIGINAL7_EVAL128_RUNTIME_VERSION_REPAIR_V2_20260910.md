# 128图执行的追加版本校验兼容修复

父执行权威SHA：95a15a6d83a53e70ca2e2750344f42c4c2483f0e9d454260835a2961f1996b6c。
主科学计划RC_ORIGINAL7_EVAL128_FULL_EVIDENCE_V1_20260910.md保持冻结。

## 已确认故障

5139176已完成，原TRAIN56的query/ref tokens、RAW候选与分数逐bit回放
通过；保留原payload、receipt、validation，不重跑。5139177在RoMa模型
构造前退出：profile记录torch.__version__为2.9.1+cu128，校验却拿
importlib.metadata.version('torch')的2.9.1比较。torchvision同样是
模块0.24.1+cu128、安装元数据0.24.1。实际模块版本、Python、venv路径
均与冻结profile相同；不是库漂移，也没有新128图或RoMa结果。
依赖数组5139186仍为DependencyNeverSatisfied，已取消，未运行。

## 修复范围

原V1程序、profile、authority、日志和已完成RAW输出全部保留。增加V2
compat程序、common、launcher及执行权威：RoMa统一用模块__version__
与原profile比对，同时记录安装元数据，CPU预检调用同一个运行时检查。
保留完整CUDA构建后缀；不关闭版本校验，不安装/升级库，不改模型。

TOKEN/RAW和CPU仅追加authority来源与脚本入口兼容，不改数学；RoMa
仅修复运行时版本校验及复用上游的provenance接口。原128张名单、
21来源组、24身份、模型权重、processor、RAW batch16与去重规则、
FP64 C4、原六维特征、127-challenger动作、置零/C_BIND和统计均不变。

## 精确复用与后续依赖

新权威显式绑定父权威、旧TOKEN桥接三个产物及其原authority；绝不把
旧产物重标为新authority。固定validation SHA为
0f9bc42fecce900bf845b06ba1e21f17f09515217143c70d416da48d4459e557。
common逐项验证此PASS、原payload/receipt、全部既有parity字段以及
所有非执行包装相关的父source binding与新权威完全相同。

仅执行包装、入口、校验和对应预检/launcher binding允许版本更新；
所有数学函数的原/新AST或固定原算子SHA另作回放证明。新权威不再
授权重跑TOKEN桥接，只重提修复后的RoMa桥接；随后为TOKEN16分片、
RAW汇总、RoMa16分片、CPU预测封存及新进程role join。

新128像素仍须在RoMa桥接与原数据逐bit一致后才能开放。此前或后续的
工程失败不是科学NO-GO，不丢弃query、不插入target、不调整分数和
阈值。研究截止仍为北京时间2026-09-11 24:00；提交后继续推进。
