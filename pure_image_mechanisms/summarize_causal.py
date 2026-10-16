"""Summarize completed pixel experiments without forcing exclusive root causes."""
from pathlib import Path
import csv,sys,collections,json
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent))
from experiment_utils import read,write,csvwrite
def cr(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def summarize(tag):
 n=int(str(tag).split('_')[0]);done=read(O/f'causal_scoring_done_{tag}.json');cohort=read(O/f'cohort{n}.json');rr=cr(f'causal_ranks_{tag}.csv');pairs=cr(f'causal_pair_decomposition_{tag}.csv');stages=cr(f'causal_stage_changes_{tag}.csv');parity=read(O/f'causal_parity_{tag}.json');arms=done['arms'];scopes=done['scopes'];assert len(rr)==n*len(arms)*len(scopes)
 ix={(r['query'],r['arm'],r['scope']):r for r in rr};sx={(r['query'],r['arm'],r['region'],r['stage']):r for r in stages};metrics=[];ledger=[];maxdecomp=max(abs(float(r['total_margin_change'])-float(r['fixed_match_representation_effect'])-float(r['rematching_effect'])) for r in pairs);assert maxdecomp<2e-7
 for role in sorted({r['role'] for r in cohort}):
  selected=[r for r in cohort if r['role']==role]
  for arm in arms:
   for scope in scopes:
    before=[int(ix[r['query'],'original',scope]['rank'] or -1) for r in selected];after=[int(ix[r['query'],arm,scope]['rank'] or -1) for r in selected];valid=[i for i,(x,y) in enumerate(zip(before,after)) if x>0 and y>0]
    metrics.append(dict(role=role,arm=arm,scope=scope,n=len(selected),valid_n=len(valid),baseline_correct=sum(before[i]==1 for i in valid),correct=sum(after[i]==1 for i in valid),repairs=sum(before[i]>1 and after[i]==1 for i in valid),harms=sum(before[i]==1 and after[i]>1 for i in valid)))
 for r in cohort:
  q=r['query'];row=dict(query=q,role=r['role'],package_id=r['package_id'],old_image_rank=int(r['image_rank']),old_center_rank=int(r['old_center_rank']))
  row['fresh_original_rank']=int(ix[q,'original','all']['rank']);row['fresh_subject_rank']=int(ix[q,'original','subject_template']['rank']);row['exact_old_query_parity']=next(x['bitwise_equal'] for x in parity if x['query']==q)
  for arm in arms:
   for scope in ['all','subject_template']:row[arm+'_'+scope+'_rank']=int(ix[q,arm,scope]['rank'])
  for arm in ['far_background_gray','far_background_blur']:
   u=sx[q,arm,'subject_unmodified_input','embedding'];v=sx[q,arm,'subject_unmodified_input','vision'];a=sx[q,arm,'subject_changed_input','embedding'];row[arm+'_foreground_input_changed_count']=int(a['n']);row[arm+'_unmodified_foreground_embedding_changed']=int(u['l2_changed_tokens']);row[arm+'_unmodified_foreground_vision_changed']=int(v['l2_changed_tokens']);row[arm+'_foreground_embedding_l2']=float(u['l2_change_mean']) if u['l2_change_mean'] else None
  # These flags record interventions, not semantic background labels.
  basecenter=row['fresh_subject_rank'];row['subject_repaired_by_both_far_fills']=basecenter>1 and all(row[a+'_subject_template_rank']==1 for a in ['far_background_gray','far_background_blur']);row['subject_repaired_by_either_far_fill']=basecenter>1 and any(row[a+'_subject_template_rank']==1 for a in ['far_background_gray','far_background_blur']);row['full_repaired_by_both_far_fills']=row['fresh_original_rank']>1 and all(row[a+'_all_rank']==1 for a in ['far_background_gray','far_background_blur']);row['full_repaired_by_either_far_fill']=row['fresh_original_rank']>1 and any(row[a+'_all_rank']==1 for a in ['far_background_gray','far_background_blur']);ledger.append(row)
 sham_mismatch=[r['query'] for r in cohort for scope in scopes if ix[r['query'],'original',scope]['rank']!=ix[r['query'],'lossless_sham',scope]['rank'] or ix[r['query'],'original',scope]['margin']!=ix[r['query'],'lossless_sham',scope]['margin']];assert not sham_mismatch
 groups={}
 for role in sorted({r['role'] for r in cohort}):
  group=[r for r in ledger if r['role']==role];groups[role]=dict(n=len(group),exact_old_query_parity=sum(r['exact_old_query_parity'] for r in group),original_rank_reproduced=sum(r['fresh_original_rank']==r['old_image_rank'] for r in group),subject_rank_reproduced=sum(r['fresh_subject_rank']==r['old_center_rank'] for r in group),subject_repaired_both_fills=sum(r['subject_repaired_by_both_far_fills'] for r in group),subject_repaired_either_fill=sum(r['subject_repaired_by_either_far_fill'] for r in group),full_repaired_both_fills=sum(r['full_repaired_by_both_far_fills'] for r in group),full_repaired_either_fill=sum(r['full_repaired_by_either_far_fill'] for r in group),foreground_input_violation=sum(any(r[a+'_foreground_input_changed_count']>0 for a in ['far_background_gray','far_background_blur']) for r in group),any_unmodified_subject_vision_changed=sum(any(r[a+'_unmodified_foreground_vision_changed']>0 for a in ['far_background_gray','far_background_blur']) for r in group))
 csvwrite(O/f'causal_metrics_{tag}.csv',metrics);csvwrite(O/f'causal_ledger_{tag}.csv',ledger);summary=dict(status='complete',queries=n,groups=groups,sham_rank_and_margin_exact=True,decomposition_max_residual=maxdecomp,ranking_decisions=len(rr),limitations=['Cohort includes selected failures and only31 success controls; not a1117 deployment accuracy estimate.','A changed foreground representation with unchanged foreground processor pixels proves outside-pixel contextual dependence; it does not prove background semantic identity.','Both-fill rescue is stronger than one-fill sensitivity but does not establish invariance to arbitrary backgrounds.','Single-family illustrative ROI findings do not label all247 failures.'])
 write(O/f'causal_summary_{tag}.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':summarize(sys.argv[1])
