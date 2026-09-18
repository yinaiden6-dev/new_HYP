#!/usr/bin/env python3
"""Independently reconstruct action/rank/counts from sealed logits after all five folds."""
import sys,json,math,hashlib,os
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
OUT=ROOT/'results/rc_h593_group_risk_strong_base_v1'

def main():
 M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 seal=M.read(OUT/'result_review_source_seal.json');M.need(seal['program']==M.bind(__file__),'REVIEW_PROGRAM_PIN')
 for b in seal['sources'].values():M.checked(b)
 result=M.read(OUT/'result.json');recorded={r['query_id']:r for r in result['rows']};M.need(len(recorded)==593,'RESULT_QUERY_COUNT')
 models=['ALL_BASE','ALL_CONST','ALL_COND','SMALL_BASE','SMALL_CONST','SMALL_COND','GROUP_BASE','GROUP_CONST','GROUP_COND'];payloads=[]
 for f in range(5):
  folder=OUT/'fits'/f'fold{f}';v=M.read(folder/'validation.json');r=M.read(folder/'receipt.json');M.need(v['status']=='GROUP_RISK_FOLD_RETRAIN_PREDICTION_REPLAY_PASS' and v['payload']==r['payload'] and v['receipt']==M.bind(folder/'receipt.json'),'FOLD_VALIDATION')
  payloads.append(torch.load(M.checked(r['payload']),map_location='cpu',weights_only=True))
 roles={r['query_id']:r for r in M.read(M.OUT/'metadata/curator_roles.json')['records']}
 from rc_aslo_xf.conditional_rep_sources import build_gallery_source
 labels=build_gallery_source(verify_cache_file_sha256=True).corrected_identities
 counts=defaultdict(int);ranks=defaultdict(list);groupchanges=defaultdict(lambda:defaultdict(list));n=0;recall=0
 for f,pay in enumerate(payloads):
  train=M.read(M.checked(pay['train_role']))['records'];trainids={r['query_id'] for r in train};testids={p['query_id'] for p in pay['predictions']};M.need(not trainids&testids and len(trainids)+len(testids)==593,'FOLD_DISJOINTNESS')
  for p in pay['predictions']:
   q=p['query_id'];role=roles[q];rr=recorded[q];M.need(role['outer_fold']==f and rr['fold']==f,'FOLD_LABEL_MEMBERSHIP');identity=role['identity'];axis=p['candidate_physical_rows'];order=p['raw_ranked_physical_rows'];target=[x for x in order if labels[x]==identity];M.need(len(target)==1,'SINGLE_TARGET_GALLERY');target=target[0]
   expected={'RAW':(axis[p['winner']]==target,order.index(target)+1)};recall+=int(target in axis)
   for name in models:
    for mode in ('REAL','INCREMENT_BIND','CBIND'):
     pred=p['models'][name][mode];zs=pred['logits'].tolist();M.need(len(zs)==127 and all(math.isfinite(v) for v in zs),'FULL_FINITE_LOGITS')
     # Python max keeps earliest physical candidate on equal scores.
     best=max(range(127),key=lambda i:zs[i]);pos=p['challenger_positions'][best] if zs[best]>0 else p['winner'];M.need(pos==pred['selected_position'],'INDEPENDENT_ACTION')
     chosen=axis[pos];reordered=[chosen]+[x for x in order if x!=chosen];key=name if mode=='REAL' else name+'_'+mode;expected[key]=(chosen==target,reordered.index(target)+1)
   for name,(ok,rank) in expected.items():
    M.need(rr['correct'][name]==ok and rr['ranks'][name]==rank,'PER_QUERY_CORRECTNESS_RANK');counts[name]+=ok;ranks[name].append(rank)
   for base,new in [('ALL_COND','GROUP_COND'),('SMALL_COND','GROUP_COND'),('ALL_BASE','GROUP_BASE'),('ALL_CONST','GROUP_CONST'),('SMALL_BASE','SMALL_CONST'),('SMALL_BASE','SMALL_COND'),('SMALL_CONST','SMALL_COND'),('GROUP_CONST','GROUP_COND')]:groupchanges[base+'__to__'+new][role['component']].append(int(expected[new][0])-int(expected[base][0]))
   n+=1
 M.need(n==593 and recall==result['recall_C128'],'COMPLETE_DENOMINATOR_AND_RECALL')
 for name,count in counts.items():
  M.need(count==result['scores'][name]['correct'],'TOTAL_CORRECT');mrr=sum(Fraction(1,r) for r in ranks[name])/593;M.need(abs(float(mrr)-result['scores'][name]['MRR'])<1e-14,'EXACT_RATIONAL_MRR')
 for name,groups in groupchanges.items():
  c=result['comparisons'][name];gain=sum(x==1 for v in groups.values() for x in v);loss=sum(x==-1 for v in groups.values() for x in v);mean=sum(Fraction(sum(v),len(v)) for v in groups.values())/len(groups)
  M.need(gain==c['rescue'] and loss==c['loss'] and abs(float(mean)-c['equal_component_difference'])<1e-14,'PAIRED_GROUP_STATISTICS')
 M.write(OUT/'result_validation.json',dict(status='GROUP_RISK_INDEPENDENT_ACTION_RANK_AND_GROUP_COUNTS_PASS',result=M.bind(OUT/'result.json'),program=M.bind(__file__),queries=593,all_five_fold_replay_gates=True,MRR_checked_using_exact_rationals=True,bootstrap_interval_not_independently_recomputed=True))
 lines=['# 统一假说检验：组风险与强基线补偿','', '593张、68身份、64组；RAW自然C128，完整127 challenger HOLD/SWITCH。所有基头和补偿均折内训练。这是开发复用交叉验证，不与旧99/128直接比较。','', '| 模型 | 正确/593 | MRR |','| --- | ---: | ---: |']
 for name in ['RAW']+models:lines.append(f"| {name} | {counts[name]} | {result['scores'][name]['MRR']:.6f} |")
 lines+=['',f"自然C128召回：{recall}/593。",'', '所有预测经过新进程重训重放；最终动作、完整gallery排名、总正确数及分组净增再次独立复算。组bootstrap区间沿用producer，验证器未声称独立重算该区间。','']
 for name,c in result['comparisons'].items():lines.append(f"- {name}: {c['rescue']}救/{c['loss']}损；等权组差{c['equal_component_difference']:.6f}；组bootstrap95% {c['component_bootstrap95']}。")
 lines+=['','[机器结果](../results/rc_h593_group_risk_strong_base_v1/result.json)，[独立复核](../results/rc_h593_group_risk_strong_base_v1/result_validation.json)。','未自动替换旧部署，未将正净增自动宣布为普遍new HYP或外部确认。']
 report=ROOT/'reports/REPORT_H593_GROUP_RISK_STRONG_BASE_RESULT_V1_20260911.md'
 with report.open('x') as f:f.write('\n'.join(lines)+'\n')
 print('GROUP_RISK_INDEPENDENT_ACTION_RANK_AND_GROUP_COUNTS_PASS')
if __name__=='__main__':main()
