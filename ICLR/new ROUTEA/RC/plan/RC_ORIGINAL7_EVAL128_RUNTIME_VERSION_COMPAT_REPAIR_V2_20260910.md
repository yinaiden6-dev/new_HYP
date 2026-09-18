# EVAL128 V2：仅修复版本表示校验，复用已合格RAW桥接

5139176已COMPLETED 0:0，原TRAIN56的query image tokens、grid、RAW C128
候选/排序/分数及全部reference tokens/grid逐bit通过，独立验证SHA
0f9bc42fecce900bf845b06ba1e21f17f09515217143c70d416da48d4459e557。

5139177在RoMa构造前FAILED 1:0。源码把importlib.metadata.version返回的
发行包版本与profile中从模块__version__取得的版本直接比较：

| 项目 | 已冻结profile/实际模块 | 安装包metadata |
| --- | --- | --- |
| torch | 2.9.1+cu128 | 2.9.1 |
| torchvision | 0.24.1+cu128 | 0.24.1 |

Python、venv路径、Pillow、NumPy及实际模块版本与profile一致。这是版本
表示API混用，不是已证实的环境漂移、模型失败或新128结果。失败发生在
模型构造之前，没有RoMa bridge输出；依赖失败的RAW数组5139186已取消，
新128图尚未开始推理。

## 唯一运行修复

RoMa V2使用模块__version__与原模块版本profile逐字比较，包括+cu128；
metadata发行包版本另行记录。相同runtime guard在CPU预检和自然GPU
构造之前执行，不截掉CUDA后缀、不取消版本约束。源码和launcher用
append-only兼容版本，原V1源码、profile、authority及产物全部保留。

encoder、RAW排序、reference resolver、RoMa权重/数值运算、cell means、
C4、FP64六特征、ORIGINAL7参数和全部动作规则不改。数据128张/24身份/
21组、来源排除、原主要比较和后续统计均按原计划保留，不重新选图。

## 已完成数据的明确继承

新authority声明parent authority SHA
95a15a6d83a53e70ca2e2750344f42c4c2483f0e9d454260835a2961f1996b6c，
并绑定上述已合格TOKEN/RAW桥接的payload、receipt、validation。其原
authority仍记V1，不重标来源、不覆盖旧输出，也不重跑其GPU编码。

common V2验证所有非执行包装类source binding与父authority相同，只
允许兼容程序、launcher、common及预检凭据等明确列出的执行来源变化。
TOKEN V2只继续新分片，拒绝重跑旧桥接；RoMa V2可读这个精确继承的
旧TOKEN桥接。新RoMa输出及所有新128分片、汇总和join均绑定V2 authority。
两个程序的数值函数保持一致，CPU版本全部函数AST保持一致。

V2实际NIL预检须覆盖真正的版本guard，而不只检查合成C4算术。CPU
检查torchvision导入不改变Torch RNG；seed17的原设置保留。合格后只
重提交RoMa bridge，再按原依赖链推进128图。截止时间仍为北京时间
2026-09-11 24:00。此次修复不改变任何科学判定门或数据边界。
