"""Finalize audited evidence, publication plots, provenance and a bounded research report."""
import json,sys,shutil,platform,importlib.metadata,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,rows,csvout,sha,data
from scripts.colnomic_mechanism.geometry import metrics,rank_rows
from scripts.colnomic_mechanism.probes import position_data,pooled,feature_pair,logistic_predict
from evaluation.scoring import fuse


def manual_audit(d):
 audit=read(OUT/'stage1_manual_audit.json')
 case=audit['cases']['photo_1f48d45127db4129']
 case['same_face_success_control']=['photo_321b1f9c3a519900','photo_c0ed1211516e6abd']
 case['control_limitations']='Both show additional faces; photo_3487dd8994f438f3 is fusion-only success (image rank 5), not an image-success control.'
 audit['cases']['photo_41de3eb383934ec3']['same_face_control_search']='All seven package_085 photos inspected; the two contains/dosage/contraindications photos both rank 2. Sole successful photo shows a different face.'
 audit['cases']['photo_21bd7df9ed6795e7']=dict(face='CAPSTAR cat front',same_visible_panel=True,identity_visible=True,
  confounds='Pink cat vs blue dog packaging; animal graphic, dose and design also differ. Not a color-only natural pair.',
  same_face_success_control=None,scoring_stage='image top1 correct; BM25/image fusion top1 wrong')
 audit['cases']['photo_0e87e0717ab1c215']=dict(face='Increxxa 25 front',same_visible_panel=True,identity_visible=True,
  confounds='Net contents 250 mL vs 100 mL; reference line breaks/size also differ. Quantity is visible in the photograph.',same_face_success_control=None)
 audit['reviewed_focus_cases']=5;audit['reviewed_family_photos']=23
 audit['controlled_layout_previews_inspected']=['reference_000047','reference_000050','reference_005348','reference_000229','reference_000233']
 audit['all_error_faces_manually_verified']=False
 write(OUT/'stage1_manual_audit.json',audit)
 ledger=read(OUT/'stage1_candidate_ledger.json');full=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json');byid={r['photo_id']:r for r in full}
 csvout(OUT/'stage1_face_audit.csv',[dict(photo_id=r['photo_id'],package_id=r['package_id'],split=r['split'],
  manual_review_status='focus case reviewed' if r['photo_id'] in audit['cases'] else 'not individually face-verified',
  **audit['cases'].get(r['photo_id'],{})) for r in ledger])
 controls=[]
 for qid in ['photo_1f48d45127db4129','photo_321b1f9c3a519900','photo_c0ed1211516e6abd','photo_3487dd8994f438f3','photo_41de3eb383934ec3','photo_665b96b3fb0b01c7']:
  r=byid[qid];controls.append(dict(photo_id=qid,package_id=r['package_id'],face='Drug Facts active ingredients/uses/warnings' if r['package_id']=='package_047' else 'contains/dosage/contraindications',
   image_rank=r['image_positive_rank'],fusion_rank=r['fusion_positive_rank'],image_success=r['image_top1_correct'],fusion_success=r['fusion_top1_correct'],
   image_strongest_wrong=r['image_strongest_wrong'],fusion_strongest_wrong=r['fusion_strongest_wrong'],
   additional_faces_visible=r['photo_id'] in ['photo_321b1f9c3a519900','photo_c0ed1211516e6abd','photo_3487dd8994f438f3']))
 csvout(OUT/'stage1_same_face_controls.csv',controls)
 return audit,ledger,full


def supplement_margins_and_controls(d):
 folder=OUT/'stage2_geometry';selection=read(folder/'selection.json')['selections'];scores=np.load(folder/'full_gallery_scores.npz')
 has_text=[bool(d['qo'][qid]['text'].strip()) for qid in d['qb']['ids']]
 f03=fuse(torch.from_numpy(scores['geo_bonus'][selection['F03']['index']]),d['bm25']['scores'],has_text,.5).numpy()
 qi={qid:i for i,qid in enumerate(d['qb']['ids'])};ri={rid:i for i,rid in enumerate(d['rb']['ids'])}
 pairs=rows(folder/'same_candidate_comparisons.csv')
 for r in pairs:
  q,p,n=qi[r['photo_id']],ri[r['positive']],ri[r['negative']]
  r['gap_columns_before_supplement']='original_gap / geo_gap / penalty_gap are image scores; configuration denotes source cohort'
  r['original_image_gap']=r['original_gap'];r['geo_image_gap']=r['geo_gap']
  r['original_fusion_gap']=float(d['fusion']['scores'][q,p]-d['fusion']['scores'][q,n])
  r['geo_fusion_gap']=float(f03[q,p]-f03[q,n])
 csvout(folder/'same_candidate_comparisons.csv',pairs)
 fits={(r['photo_id'],r['reference_id']) for r in read(folder/'alignments.json')}
 full={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
 gate={k:{r['photo_id']:r for r in v} for k,v in read(folder/'alignment_gate_predictions.json').items()}
 primary=read(folder/'predictions.json');support=[]
 for mode in ['B03','F03']:
  baseline={r['photo_id']:r for r in primary[mode+'_original']}
  for r in primary[mode+'_geo_matching']:
   qid=r['photo_id'];positive=full[qid]['positive_reference_ids'][0];negative=baseline[qid]['top1']
   support.append(dict(configuration=mode,photo_id=qid,split=r['split'],baseline_correct=baseline[qid]['correct'],
    geo_correct=r['correct'],gate_correct=gate[mode][qid]['correct'],positive=positive,baseline_top1=negative,
    positive_aligned=(qid,positive) in fits,baseline_top1_aligned=(qid,negative) in fits))
 csvout(folder/'alignment_support_and_gate_outcomes.csv',support)
 return scores,primary,gate


def majority_controls(d):
 result=[]
 for kind in ['absolute','subject_relative','ocr_row']:
  _,y,p,split,w,_,_=position_data(d,kind);train=split=='train';test=split=='test'
  for dim in range(y.shape[1]):
   labels=np.minimum((y[:,dim]*4).astype(int),3);counts=np.bincount(labels[train],minlength=4);label=int(counts.argmax());correct=labels[test]==label
   result.append(dict(probe=kind,dimension=dim,split='test',training_majority_bin=label,
    accuracy=float(correct.mean()),package_macro_accuracy=float(np.mean([correct[p[test]==k].mean() for k in sorted(set(p[test]))])),
    selection='training counts only; no fit to test labels'))
 csvout(OUT/'stage3_probes/position_majority_controls.csv',result)
 return result


def identity_pair_ranking(d):
 folder=OUT/'stage3_probes';samples=rows(folder/'identity_sample_index.csv');byphoto={}
 for r in samples:byphoto.setdefault(r['photo_id'],{})[int(r['label'])]=r
 stored=torch.load(folder/'feature_only_identity_head.pt',weights_only=True,map_location='cpu')
 model={k:stored[k].numpy() for k in ['mean','scale','coefficient']}
 qindex={v:i for i,v in enumerate(d['qb']['ids'])};rindex={v:i for i,v in enumerate(d['rb']['ids'])};pools={};records=[]
 for qid,pair in byphoto.items():
  qi=qindex[qid];pi=rindex[pair[1]['reference_id']];ni=rindex[pair[0]['reference_id']]
  qpool=pooled(d['qb']['embeddings'][qi],d['metadata']['photos'][qi])
  for ri in [pi,ni]:
   if ri not in pools:pools[ri]=pooled(d['rb']['embeddings'][ri],d['metadata']['references'][ri])
  ps=float(d['image']['scores'][qi,pi]);ns=float(d['image']['scores'][qi,ni])
  logits=logistic_predict(model,np.stack([feature_pair(qpool,pools[pi],ps),feature_pair(qpool,pools[ni],ns)]));gap=float(logits[0]-logits[1])
  records.append(dict(photo_id=qid,package_id=pair[1]['package_id'],split=pair[1]['split'],positive=pair[1]['reference_id'],negative=pair[0]['reference_id'],
   raw_gap=ps-ns,probe_gap=gap,raw_correct=ps>ns or (ps==ns and pi<ni),probe_correct=gap>0 or (gap==0 and pi<ni)))
 csvout(folder/'identity_pair_ranking.csv',records);summary=[]
 for split in ['train','val','test']:
  selected=[r for r in records if r['split']==split];packages=sorted({r['package_id'] for r in selected})
  for method,key in [('MaxSim','raw_correct'),('feature_probe','probe_correct')]:
   summary.append(dict(split=split,method=method,photos=len(selected),packages=len(packages),
    pair_accuracy=float(np.mean([r[key] for r in selected])),package_macro_pair_accuracy=float(np.mean([np.mean([r[key] for r in selected if r['package_id']==p]) for p in packages])),
    scope='same-split positive-design catalog negative; NOT full-gallery Recall'))
 csvout(folder/'identity_pair_ranking_metrics.csv',summary)
 return summary


def provenance():
 import cv2
 files=['cache/image_embeddings/colnomic7b/photos.pt','cache/image_embeddings/colnomic7b/references.pt',
  'cache/scores/colnomic7b/image.pt','cache/scores/bm25/bm25.pt','cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt',
  'cache/ocr/photos.json','cache/ocr/references.json','dataset/annotations/subject_boxes.json',
  'configs/baselines.json','results/token_mechanism_analysis/F03/reproduction/token_metadata.json',
  'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json']
 files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'dataset/lists').glob('*.csv'))]
 original={r['path'].removeprefix('reproduction/'):r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/source_files.json')}
 manifest=[]
 for relative in files:
  path=ROOT/relative;digest=sha(path);prior=original.get(relative)
  if prior:assert digest==prior['sha256'],relative
  manifest.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest,unchanged_from_supplemented_archive=bool(prior)))
 write(OUT/'input_manifest.json',manifest)
 codefiles=list((ROOT/'scripts/colnomic_mechanism').glob('*.py'))+[ROOT/'scripts/analyze_colnomic_tokens.py',ROOT/'scripts/encode_colnomic_images.py',ROOT/'evaluation/scoring.py',ROOT/'tests/test_colnomic_mechanism_experiments.py',ROOT/'drugpack/method/token_layout.py',ROOT/'configs/baselines.json']
 codes=[]
 for path in codefiles:
  relative=path.relative_to(ROOT);target=OUT/'source_code'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
  codes.append(dict(path=str(relative),sha256=sha(path)))
 write(OUT/'code_manifest.json',codes)
 versions={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','Pillow','colpali-engine','peft','accelerate','matplotlib']};versions['opencv']=cv2.__version__
 write(OUT/'runtime.json',dict(python=platform.python_version(),packages=versions,
  causal_runtime='stage4_interventions/validated_embeddings/complete.json',
  artifact_timestamp_convention='UTC; date label fixed to experiment run 2026-10-13',baseline_data_modified=False))
 return manifest


def plots(analysis):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 folder=OUT/'figures';folder.mkdir(exist_ok=True)
 ms=rows(OUT/'stage2_geometry/metrics.csv');gate=rows(OUT/'stage2_geometry/alignment_gate_control.csv')
 fig,ax=plt.subplots(figsize=(8,4.2));x=np.arange(2);width=.2
 for i,(label,method,table) in enumerate([('Original','original',ms),('Geometry bonus','geo_matching',ms),('Pure penalty','geo_penalty',ms),('OCR alignment gate','alignment_availability_bonus_without_token_geometry',gate)]):
  values=[100*float(next(r['r1'] for r in table if r['configuration']==cfg and r['split']=='test' and r['method']==method)) for cfg in ['B03','F03']]
  bars=ax.bar(x+(i-1.5)*width,values,width,label=label)
  ax.bar_label(bars,fmt='%.1f',fontsize=8)
 ax.set(xticks=x,xticklabels=['Image (B03)','Fusion (F03)'],ylabel='Recall@1 (%)',ylim=(65,101),title='Held-out identities: 161 photos, complete gallery');ax.legend(loc='upper left',fontsize=8);fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(folder/('retrieval_ablation.'+ext),dpi=180)
 plt.close(fig)
 focal=rows(analysis/'focal_causal_scores.csv');fresh=[r for r in focal if r['query']=='fresh_query']
 fig,ax=plt.subplots(figsize=(7,4));ax.plot(range(6),[float(r['gap']) for r in fresh],marker='o');ax.axhline(0,color='black',linestyle='--',linewidth=1)
 ax.set(xticks=range(6),xticklabels=['Base','Sham','Shift 1','Shift 2','Shift 3','Shift 4'],ylabel='Positive minus negative mean MaxSim',title='details. relocation: within-runtime pair response');fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(folder/('controlled_details_margin.'+ext),dpi=180)
 plt.close(fig)


def report(audit,ledger,analysis):
 ms=rows(OUT/'stage2_geometry/metrics.csv');gate=rows(OUT/'stage2_geometry/alignment_gate_control.csv')
 lines=['# ColNomic 7B 四阶段竞争假设实验','',
  '已执行冻结特征的完整图库评分、按包装身份隔离的 Probe，以及确定性词位置干预。结果支持“额外的 OCR 对齐信息可改善检索”，尚不支持“原特征完整保留身份布局、只是 MaxSim 没有使用”的强机制结论。',
  '', '本轮没有训练编码器，也没有替换原始缓存或原有两个归档。第四阶段已执行，但原图回放未达到缓存一致性要求，因此其因果结论仅限于本轮运行时。', '',
  '## 1. 比较对象与身份核验','',
  f'- 两个历史错误队列共 288 张独立失败照片；加入既有成功对照后，精确候选清单覆盖 {len(ledger)} 张照片。保留正确候选、实际图像第一名、实际融合第一名和各自最强错误候选，防止把“错误第一名”与成功时的实际第一名混为一谈。',
  '- 完整评分覆盖 1117 张照片 × 5405 条图库记录；按图像 SHA 去重后评估 5330 个图库设计。没有插入真值候选或限制到 top-K。',
  '- 783/173/161 张训练/验证/测试照片；91/21/18 个包装身份。包装、设计、身份标签、来源 SHA 和来源组跨集合均不重叠。历史测试错误已被看过，本轮不构成历史上完全盲测。',
  '- 人工核验 5 个重点案例及两个家族的 23 张照片；其余个案未标记为“包装面已核验”。精确候选清单是全量的，人工面审核是有限范围的。',
  '- `photo_41de3eb383934ec3` 与 20/50 mL 候选为同一 contains/dosage/contraindications 面，照片看不到容量；其包装家族不存在该面检索成功的照片，不能用正面的成功照充当同面成功对照。',
  '- `photo_1f48d45127db4129` 已加入 `photo_321b1f9c3a519900`、`photo_c0ed1211516e6abd` 同面图像/融合成功对照。它们包含更多包装面，仍有视角和信息量差异；`photo_3487dd8994f438f3` 仅融合成功，图像排名为 5。',
  '- 用户指出的颜色案例确实有粉红猫盒/蓝色狗盒差异，但还同时存在动物图案和剂量差异；图像原本正确、融合变错。小字案例是可见的 250/100 mL 差异。自然案例不能当成仅改变单一因素的实验。',
  '- 保留既有正负 SHA 配对的身份歧义标记。另有固定剔除 19 张已标记 F03 照片的敏感性分析，不据新模型结果重新定义真值。', '',
  '证据：[候选清单](stage1_candidate_ledger.csv)、[人工审核](stage1_manual_audit.json)、[同面控制](stage1_same_face_controls.csv)、[划分核验](split_verification.json)。','',
  '## 2. 冻结特征：几何匹配与完整图库','',
  '原始分数为有效 Query Token 的 **mean MaxSim**，包含非零模板 Token；不是未除 Token 数的总和。使用原始 FP16 缓存、FP32 点积，不重新归一化向量。', '',
  '对齐采用主体框内唯一的相同 OCR 行文本（至少 8 字符、至少 4 行），在规范化坐标上做 RANSAC 单应变换。内点至少 4 行、覆盖率至少 0.15、禁止反射、局部条件数不超过 20；全图库候选提议、变换拟合和接受规则均不看身份真值。598 张照片获得至少一个对齐，共 20326 个接受的配对。OCR 行的顶点起点/顺逆序不一致，规范化修正另记为 implementation amendment；旧输出未作为最终结果。', '',
  '注册的主方法在主体视觉 Token 上使用 `dot + λ(1 − clipped_distance)`，无对齐时回退原始分数；其余 Token 不变。它含有“对齐可用性奖励”，**不能称为纯几何惩罚**。另保存 `dot − λ distance` 的纯惩罚消融和不使用 ColNomic 特征的几何支持分。几何支持分用 OCR 文本建立锚点，不是完全不含文字信息的坐标方法。', '',
  'λ 在验证集按包装宏平均 R@1、MRR 和较小参数择优：B03=0.1，F03=0.2。测试身份标签不选择参数；融合继续使用全图库 population z-score、BM25、α=0.5。对齐还使用人工确认的 Query 主体框；这是额外定位信息，不宣称完全不使用任何测试标注的部署系统。', '',
  '| 方法 | B03 测试 R@1 | F03 测试 R@1 | B03 全 1117 R@1 | F03 全 1117 R@1 |',
  '|---|---:|---:|---:|---:|']
 for label,method,table in [('原始','original',ms),('几何奖励匹配','geo_matching',ms),('相同 λ 的纯几何惩罚','geo_penalty',ms),('仅几何支持（含 OCR 锚点）','geometry_only',ms),('仅坐标网格距离','coordinate_only',ms),('仅 OCR 对齐可用性奖励＊','alignment_availability_bonus_without_token_geometry',gate)]:
  cells=[float(next(r['r1'] for r in table if r['configuration']==cfg and r['split']==split and r['method']==method))*100 for split in ['test','all'] for cfg in ['B03','F03']]
  lines.append('| '+label+' | '+' | '.join(f'{v:.2f}%' for v in cells)+' |')
 lines += ['', '＊对齐可用性控制在看过主实验结果后增加，沿用主实验 λ、不重新调参，是事后诊断，不是新的盲测主方法。它去掉了 Token 距离项，却取得非常接近的结果：B03 测试修复 8 例，F03 修复 8 例，均损害 0 例。B03 与几何匹配共享 7 个修复案例，另外几何匹配独自修复 3 个、门控独自修复 1 个；F03 几何匹配修复的 7 个案例全部被门控修复。仍可能存在额外空间贡献，不能把两者概括为完全等价。', '',
  '主方法在测试集 B03 修复 10/损害 0，F03 修复 7/损害 0；全 1117 张 B03 修复 45/损害 1，F03 修复 19/损害 1。两张被损害的训练照片分别为 `photo_b111a943691a2aa7` 和 `photo_24aba52659f86e92`。成功样本已纳入完整评估，不能把局部正负翻转当作全库改善。', '',
  '按 18 个测试来源组进行配对 bootstrap（5000 次）：B03 宏平均 R@1 增量 7.49 个百分点，95% 区间 [1.63, 15.87]；F03 增量 5.80 个百分点，区间 [0.00, 14.77]。少量来源组限制估计精度。', '',
  '**机制判断：相近的改善可在移除 Token 距离项后保留，OCR 对齐可用性所带来的额外判别信息是一个有力的竞争解释。纯几何惩罚没有稳定改善；尚不能证明 Token 的空间对应本身是主要瓶颈。** 规则网格距离独立评分有退化，不能用它的低分证明空间结构无用。', '',
  '| 用户重点案例 | 原始 B03 → 几何奖励 → 门控 | 原始 F03 → 几何奖励 → 门控 |',
  '|---|---:|---:|',
  '| `41de3eb383934ec3` 末尾词布局 | 2 → 1 → 1 | 2 → 1 → 1 |',
  '| `21bd7df9ed6795e7` 颜色/猫狗 | 1 → 1 → 1 | 2 → 2 → 2 |',
  '| `1f48d45127db4129` Drug Facts | 148 → 94 → 79 | 10 → 8 → 8 |',
  '| `0e87e0717ab1c215` 250/100 mL | 2 → 2 → 2 | 2 → 2 → 2 |',
  '| `2bdbf879ee1d7dd8` 数量差异 | 2 → 1 → 1 | 2 → 1 → 1 |', '',
  '数字为完整图库中正确设计的排名。末尾词案例虽然被几何奖励修复，但门控同样修复，不能归因于精细 Token 空间匹配；其纯几何惩罚排名仍为 2。颜色和小字案例在本方法下仍然失败。', '',
  '![完整图库消融](figures/retrieval_ablation.png)', '',
  '证据：[完整指标](stage2_geometry/metrics.csv)、[同候选正负差额](stage2_geometry/same_candidate_comparisons.csv)、[损益逐案表](stage2_geometry/per_photo_changes.csv)、[对齐门控消融](stage2_geometry/alignment_gate_control.csv)、[完整分数数组](stage2_geometry/full_gallery_scores.npz)。', '',
  '## 3. 冻结特征 Probe','',
  '每个位置 Probe 只输入 128 维冻结 Token 特征；坐标、行号、Token 索引、图像 ID 仅作为监督标签或样本索引，不输入预测头。标准化和拟合只用训练身份，正则化用验证身份选择，保留 3 个训练标签打乱对照。', '',
  '| 预测任务 | 测试包装宏平均 MAE | 训练均值对照 |',
  '|---|---:|---:|',
  '| 原始图像绝对位置 | 0.2098 | 0.2499 |',
  '| 主体相对位置 | 0.2337 | 0.2493 |',
  '| OCR 垂直行序 | 0.2534 | 0.2984 |', '',
  '这些结果说明冻结特征有一定位置相关可预测信号，但不能排除语义与位置的相关性，也不等同于身份布局可分。OCR 行标签来自轴对齐包围框与仅覆盖一行的网格，属于近似垂直序号；四分箱准确率 28.22%，低于训练多数类对照 33.14%，不能只选 MAE 描述成全面成功。', '',
  '身份配对 Probe 输入视觉特征的均值/标准差组合和同一特征计算的原始 MaxSim，共 513 维。每个划分的负例只在该划分有正标注的设计集合内选择最高图像分的错误候选；测试负例不是从全部 5330 个设计采样。训练负例因此不使用验证/测试设计，测试为 18 个未见身份的 322 个二分类样本。测试包装宏平均准确率：特征头 80.91%，文字头 88.73%，坐标头 51.39%；特征头标签打乱为 44.13%–51.09%。这是限定候选的配对分类，不是完整图库 R@1，也不是保持文字完全相同的天然布局分类。另保存同一正负对的 MaxSim/Probe 排序差额，防止把阈值分类率当作检索改善。文字头的 BM25 使用全图库无标签词频，属于公开图库的传导设定。', '',
  '在完全相同的正负对上比较排序，测试包装宏平均配对准确率为 MaxSim 98.06%、特征 Probe 91.66%。因此二分类 Probe 成功并没有转化为优于 MaxSim 的配对排序；这个测试负例集合较容易，不能用它说明真实困难布局已被编码。', '',
  '固定头也用于实际全图库最高错误候选构成的天然 hard-negative 对照。所选测试子集中有 60 张照片、81 个配对：MaxSim 正负差额为正的配对为 35，Probe 为 47，翻正 21、翻负 9。这是精选困难配对的净改善，不能充当完整图库 Recall，也未将它们自动标成“同文字纯布局差异”。参考文字 Jaccard 和逐对差额均保留。真正同文字的受控布局 Probe 见第四阶段。', '',
  '证据：[位置指标](stage3_probes/position_metrics.csv)、[多数类对照](stage3_probes/position_majority_controls.csv)、[身份指标](stage3_probes/identity_metrics.csv)、[天然困难配对](stage3_probes/natural_hard_negative_probe.csv)、各 `*_head.pt` 与 `*_selection.json`。', '',
  '## 4. 确定性布局干预与原图回放','',
  '按照用户确认采用 Pillow/OpenCV 搬移已有字形像素。5 个来源各有 sham 与 4 个水平位移（1–4 个原始视觉网格宽度），另编码原图；其他文字、颜色、容量不变。每组 sham/位移的 RGB 直方图相同、字形 RGB 完全相同、公共编辑带之外像素完全相同。sham 与原图的近白背景规范化差异单独报告。已逐组检查放大图，未合成新字体或替换文字。', '',
  '自然选择规则只根据 OCR、空白位置和划分：训练 2 个身份、验证 1 个身份、测试 1 个身份；此外加入用户指定 `details.` 的独立重点干预。选择前不看编码响应。测试身份移动的是另一包装面上的 `children.`，不是照片所见的 `details.`，不将它冒充该照片的干预。重点案例固定正确候选，只移动错误候选 `reference_000233` 的 `details.`。', '',
  '仅特征布局分类将位移 1/2 与 3/4 分成两个平衡类，所有类别均做像素编辑；包装身份严格隔离，标签在来源内打乱。全视觉池化头与词 ROI 特征头均为 **50% 测试准确率（仅 4 个样本、1 个身份）**。ROI 选择使用空间监督，但不输入坐标/索引，Token 数在同一来源的全部类别中一致。这个小型试验未证明身份相关布局可分；失败也不能证明布局信息完全不存在。', '',
  '**原缓存回放未通过。** 本轮使用同一固定权重，但受 GPU 可用资源限制采用 CPU 权重卸载、RTX 2080 Ti 计算。当前 PIL、默认 torchvision 以及历史 Transformers 4.57 环境均未复现原缓存：原图平均 Token 余弦约 0.95–0.97，最大分量误差约 0.35–0.42。最终运行时和代码哈希已逐样本记录。没有把新旧编码差异归因于布局。', '',
  '因此，以下是本轮运行时内的配对干预响应：重新编码 Query 与两个原图后，正负差额约 −0.00493；sham 后约 −0.00465，移动 `details.` 后约 −0.00375 至 −0.00451，仍未翻转。最高编辑效应约降低错误候选分数 0.00090；部分位移效应与 sham 本身的效应同量级。其他缓存图库保持固定的替换排名均为 2，但它是新旧特征混用的条件排名，**不作为完整新运行时的准确率或原缓存错误的因果证据**。', '',
  '20 个搬移配对中，MaxSim 的归一化余弦损失小于同索引对角匹配损失（比值约 0.34–0.54）。说明在这些同画布编辑上自由匹配更容忍位移；对角匹配额外使用索引对齐，不能据此声称原 Token 特征已编码身份布局。', '',
  '![末尾词搬移响应](figures/controlled_details_margin.png)', '',
  '证据：[像素守恒](stage4_interventions/pixel_validation.json)、[搬移示例](stage4_interventions/reference_000233_controlled_layouts.png)、[运行时](stage4_interventions/validated_embeddings/complete.json)、[原图一致性](stage4_interventions/analysis_validated_runtime/encoder_parity.json)、[正负差额](stage4_interventions/analysis_validated_runtime/focal_causal_scores.csv)、[布局 Probe](stage4_interventions/analysis_validated_runtime/layout_probe_metrics.csv)、[相似度矩阵](stage4_interventions/analysis_validated_runtime/similarity_matrices)、所有 `validated_embeddings/*.pt`。', '',
  '## 可成立的结论与仍待解决的问题','',
  '| 竞争假设 | 本轮证据 | 判断 |',
  '|---|---|---|',
  '| 额外 OCR 对齐信息能改善身份检索 | 完整图库、成功样本损害检查、门控消融 | 支持；收益有来源组不确定性 |',
  '| 纯粹的几何 Token 匹配是收益的主因 | 门控达到相近收益，纯惩罚不改善 | 尚不支持 |',
  '| 编码器完全不保留空间相关信息 | 位置 MAE 优于打乱标签/训练均值 | 过强；可预测信号存在，但可能含语义先验 |',
  '| 特征能区分同文字的身份布局 | 天然配对有语义混杂；受控头只有 1 个测试身份且为 50% | 未证实，不能从失败推出绝对不存在 |',
  '| 仅移动 details. 即可修复重点案例 | 配对编辑未翻转；旧缓存回放失败 | 本轮运行时未修复；原缓存因果归因未建立 |', '',
  '下一步最具体的缺口是找到可复现原缓存的真实编码路径，以及增加具有干净同文字布局差异和同面成功照片的身份集合。当前不能直接选取几何奖励增益作为“MaxSim 机制已证实”的论文证据。', '',
  '## 复现与验证','',
  '19 项既有/新科学不变量测试通过（最终测试日志以 `test_results.txt` 为准）。冻结点积评分最大复现误差 2.38×10⁻⁷，λ=0 精确等于原图像分数。独立核验来源划分、图像像素守恒、对照候选排名、矩阵评分和原缓存 SHA；原缓存未改写。编码一致性失败被单独保留，不能与这些已通过项目混称“全部验证通过”。', '',
  '入口：[REPRODUCE.md](REPRODUCE.md)、[输入校验清单](input_manifest.json)、[代码校验清单](code_manifest.json)、[最终核验](validation.json)。本轮归档是补充实验包，完整原始冻结图库仍来自之前两个归档；不包含 7B 权重。','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))


def main():
 torch.set_num_threads(2);d=data();audit,ledger,full=manual_audit(d)
 verify=read(OUT/'stage2_geometry/verification.json');verify.pop('no_gt_in_alignment_or_proposals',None)
 verify.update(no_identity_gt_in_alignment_or_proposals=True,confirmed_query_subject_boxes_used=True,bbox_gt_free_system_not_claimed=True)
 write(OUT/'stage2_geometry/verification.json',verify)
 supplement_margins_and_controls(d);majority_controls(d);identity_pair_ranking(d)
 write(OUT/'stage2_geometry/score_axes.json',dict(query_ids=d['qb']['ids'],reference_ids=d['rb']['ids'],
  reference_sha256=d['rb']['sha256'],deduplication='stable descending row score; keep first occurrence of each SHA'))
 analysis=OUT/'stage4_interventions/analysis_validated_runtime';assert (analysis/'verification.json').exists()
 plots(analysis);report(audit,ledger,analysis);inputs=provenance()
 stage4=OUT/'stage4_interventions';manifest=read(stage4/'encoding_manifest.json');portable={};repeats=[]
 for r in manifest:
  row=dict(r);source=Path(r['path'])
  if r['arm']=='base':
   target=stage4/'original_images'/(r['source_id']+source.suffix);target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
   row['path']=str(target.relative_to(stage4))
  else:row['path']=str(source.relative_to(stage4))
  portable[row['id']]=row
 write(stage4/'portable_encoding_manifest.json',list(portable.values()))
 for f in sorted((stage4/'validated_embeddings').glob('*.pt')):
  old=stage4/'embeddings'/f.name
  if old.exists():
   a=torch.load(old,weights_only=True,map_location='cpu')['embedding'];b=torch.load(f,weights_only=True,map_location='cpu')['embedding']
   repeats.append(dict(id=f.stem,max_error=float((a.float()-b.float()).abs().max()),exact=torch.equal(a,b)))
 assert repeats and all(r['exact'] for r in repeats)
 write(stage4/'within_runtime_repeatability.json',dict(unique_encoded_outputs=len(repeats),max_error=0.,records=repeats,
  primary_manifest_records=len(manifest),primary_unique_ids=len(portable),duplicated_base_record='reference_000229; identical image, also focal candidate',
  source_runtime='validated_embeddings/complete.json'))
 # Retain the byte-for-byte executed worker, before adding relative-path CLI support.
 old_worker=(ROOT/'scripts/colnomic_mechanism/encode_worker.py').read_text().replace(
  "  path=Path(record['path'])\n  if not path.is_absolute():path=args.manifest.parent/path\n  target=args.output/(record['id']+'.pt')",
  "  path=Path(record['path']);target=args.output/(record['id']+'.pt')")
 expected=read(stage4/'validated_embeddings/complete.json')['worker_sha256']
 assert hashlib.sha256(old_worker.encode()).hexdigest()==expected
 target=OUT/'source_code/executed/encode_worker.py';target.parent.mkdir(exist_ok=True);target.write_text(old_worker)
 write(OUT/'source_code/executed/manifest.json',dict(worker_sha256=expected,
  numeric_encoder_sha256=read(stage4/'validated_embeddings/complete.json')['experimental_encoder_sha256'],
  later_cli_change='relative manifest paths supported; absolute-path numerical execution unchanged and repeatability verified'))
 control=read(analysis/'verification.json')
 write(OUT/'validation.json',dict(status='experiments_executed_with_reported_limitations',
  geometry=read(OUT/'stage2_geometry/verification.json'),probes=read(OUT/'stage3_probes/verification.json'),intervention=control,
  original_cache_unchanged=True,input_files_verified=len(inputs),face_audit_all_cases_completed=False,
  same_face_success_missing_for_focal_package=True,primary_parameters_validation_only=True,
  posthoc_gate_control_declared=True,original_cache_causal_mechanism_proven=False,
  report='REPORT.md',tests_log='test_results.txt'))
 print('Finalized four-stage experimental evidence; old-cache causal replay:',control['original_cache_reproduction_passed'],flush=True)

if __name__=='__main__':main()
