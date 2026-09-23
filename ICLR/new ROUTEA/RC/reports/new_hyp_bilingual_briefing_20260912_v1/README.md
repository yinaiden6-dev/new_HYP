# Route A / new HYP：中英文组会汇报

## 文件与使用

- `main_zh.tex` / `main_zh.pdf`：中文版，13页，16:9。
- `main_en.tex` / `main_en.pdf`：英文版，13页，16:9。
- `preamble.tex`：两版共用主题、字体与原生矢量绘图函数，必须与入口文件一起上传。
- `build.sh`：本地双遍XeLaTeX编译；格式缓存限制在本文件夹，不修改home配置。
- `data_source_audit.json`：结果源文件哈希、关键数值与两版一致性核对。
- `bilingual_latex_and_pdf.zip`：源码、说明和两套PDF完整包。

Overleaf：上传ZIP，编译器选择 **XeLaTeX**，主文件选 `main_zh.tex` 或 `main_en.tex`。
图表由LaTeX/TikZ直接绘制，无外部图片依赖；英文版不嵌入中文图表。
可修改标题页的 `\author{...}`、`\date{...}`。内容以2026-09-11已完成实验为准。

本地TeX安装缺少xeCJK的`ctexhook.sty`依赖，已从官方CTAN包下载到本目录并原样使用。
`ctex.tds.zip`保留完整上游源码和版权信息（LPPL）；未改动系统/home配置。
在完整Overleaf环境中通常可直接使用平台自带ctexhook；本地TeX Live 2020会提示该上游包
声明较新的LaTeX版本，但本稿以实际编译、缺字与溢出检查为准。

## 建议的口头开场（中文）

我目前研究的是参考图像定义的精确身份识别。以药盒为例，系统不是只判断照片里有药盒，
而是要区分具体的包装身份。我们的目标是提供一张reference后，共享模型就能识别该物品，
不需要为每个身份重新训练；这项可靠性还需要专门验证。

当前先解决原始检索可能选错近邻的问题。保留ColNomic的RAW检索，用RoMa提供候选相关的
匹配和可见性权重，再用ColNomic读取完整reference的内容证据。共享小头比较自然C128中的
全部127个challenger，只在其评分超过HOLD时替换原答案。任务内训练只用检索身份关系，
不增加人工mask或框；但基础模型的预训练监督必须另行说明。

目前有明确的内部正结果：冻结原头在新增128图上从88到99，12次救回、1次破坏。
593图的五折开发评估中，条件版为19救0损，组均衡七参数版为22救1损。
把候选证据错绑后，原来的救回大量消失，说明收益与specific-reference证据有关。
但这些结果不等于已经找到完整目标区域，也不等于外部可靠性已证明。
下一步是固定模型，核实原训练未见身份的识别能力，并把新身份登记与旧身份保持单独验证。

## Suggested opening (English)

My project studies reference-defined exact-instance recognition. For example, the system must distinguish
a specific drug-package identity rather than merely detect a medicine box. The long-term goal is to enroll
an identity through a visual reference without identity-specific retraining. Its reliability still needs
a dedicated evaluation.

The current milestone is to correct near-duplicate errors made by the original retriever. We retain
ColNomic RAW retrieval, obtain candidate-conditioned matching and visibility weights from RoMa, and use
ColNomic to read full-reference content evidence. A shared head considers all 127 challengers in the natural
C128 and either switches to the highest positive-scoring challenger or holds the original prediction.
Task training uses retrieval identity relationships, without additional masks or boxes; backbone
pretraining supervision must be disclosed separately.

The internal evidence is positive. A frozen original head improves from 88 to 99 correct on 128 additional
queries, with 12 rescues and one break. In 593-query grouped development cross-validation, the conditional
method obtains 19 rescues and no breaks, while the group-weighted seven-parameter method obtains 22 rescues
and one break. Rebinding candidate evidence removes the original rescues, supporting its specific-reference
role. These findings do not establish complete target localization or independent external reliability.
Next, we will freeze the model, audit which identities were absent from its training, and evaluate transfer
separately from dynamic enrollment and old-identity retention.

## 科研口径 / Scientific scope

- RAW = unadapted ColNomic full-gallery retrieval. All reported positive rows use natural RAW-C128, not D1.
- FROZEN_C and ORIGINAL7/NATIVE7 are different parameter heads. 593-query OOF refits within each fold;
  it is not a single deployed checkpoint and cannot be pooled with the old32/128 panels.
- New HYP concerns candidate identity explanations. It does not rename a failed spatial RGH experiment
  as a successful connected-region hypothesis.
- Candidate-bundle and increment-binding interventions support computational dependence, not token-level
  correspondence necessity or pixel-level spatial causality.
- GROUP_BASE440→447 comparison refers to ALL_BASE440 versus GROUP_BASE447. The predefined
  GROUP_COND-versus-ALL_COND robust primary advantage was not established; the slides retain that boundary.
- Shared functions permit evaluation of new references without new identity-specific parameters.
  This is a structural property, not a guarantee of recall, generalization, or error-free recognition.
- New-identity experiments on slide11 are **planned**, not executed by this document task.
- No new model training, Slurm submissions, source-model changes, or threshold selection were performed.

## 主要证据入口（相对RC根目录）

1. `results/romav2_colnomic_difficult90_frozen_regression_v1/result.json`
2. `results/rc_original7_eval128_full_evidence_v1/result.json`
3. `results/rc_original_mixed96_group_risk_v1/result.json`（复用冻结旧EVAL32预测）
4. `results/rc_h593_group_risk_strong_base_v1/result.json`
5. `results/rc_h593_increment_attribution_v1/result.json`
6. `results/roma_rgh_reference_first_natural_s0_v1/aggregate/postjoin_result.json`
7. `reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md`
8. `reports/NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md`
9. `reports/REPORT_H593_UNIFIED_HYPOTHESIS_PREDICTION_READOUT_V1_20260911.md`
10. `reports/REPORT_ROMA_RGH_S0_JOB5139368_FINAL_DECISION_AND_MULTIPLICITY_BOTTLENECK_20260910.md`

公式为报告用的条件化数学表达，不授权替换生产路径的FP64归约顺序或数值保护。
这份稿件汇报项目内部实验，不构成对新数学定理、首创风险理论或已有文献最优性的声明。
