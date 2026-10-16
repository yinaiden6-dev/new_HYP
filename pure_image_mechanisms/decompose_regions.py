"""Region score arithmetic on fixed historical candidates; not unique causal attribution."""
from pathlib import Path
import csv,sys
import numpy as np
O=Path(__file__).resolve().parent;P=O.parent;sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite
def cr(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
cohort=read(O/'cohort309.json');z=np.load(P/'masks.npz');pp={(r['query'],r['arm'],r['scope'],r['negative']):r for r in cr('causal_pair_decomposition_309.csv')};ll={r['query']:r for r in cr('causal_ledger_309.csv')};out=[];err=0
for c in cohort:
 q=c['query'];neg=c['image_negative'];l=ll[q];f=z[q+'__foreground_center'];t=z[q+'__template'];b=~f&~t;weights={'subject_visual':f.mean(),'background_visual':b.mean(),'template':t.mean()}
 for arm in ['far_background_gray','far_background_blur']:
  p=pp[q,arm,'all',neg];old=float(p['original_margin']);changes={};original={}
  for s,w in weights.items():
   r=pp.get((q,arm,s,neg));changes[s]=float(r['total_margin_change'])*w if r else 0;original[s]=float(r['original_margin'])*w if r else 0
  residual=max(abs(sum(original.values())-old),abs(sum(changes.values())-float(p['total_margin_change'])));assert residual<2e-6;err=max(err,residual)
  fg=changes['subject_visual']+changes['template'];bg=changes['background_visual'];out.append(dict(query=q,role=c['role'],package_id=c['package_id'],arm=arm,old_image_rank=c['image_rank'],old_subject_rank=c['old_center_rank'],edited_image_rank=l[arm+'_all_rank'],both_fill_full_rescue=l['full_repaired_by_both_far_fills'],original_margin=old,edited_margin=p['free_margin'],subject_visual_score_change=changes['subject_visual'],background_score_change=bg,template_score_change=changes['template'],only_subject_and_template_change_margin=old+fg,only_background_change_margin=old+bg,subject_and_template_update_sufficient=old+fg>0,background_update_sufficient=old+bg>0,locked_old_match_margin=p['locked_margin']))
csvwrite(O/'region_score_decomposition309.csv',out);strong=[r for r in out if r['role']=='original_image_failure' and r['both_fill_full_rescue']=='True'];result=dict(max_additive_error=err,dual_fill_full_rescue_queries=len({r['query'] for r in strong}),already_subject_correct_queries=len({r['query'] for r in strong if int(r['old_subject_rank'])==1}),outside_subject_repair_queries=len({r['query'] for r in strong if int(r['old_subject_rank'])>1}),arms={})
for arm in ['far_background_gray','far_background_blur']:
 rr=[r for r in strong if r['arm']==arm];result['arms'][arm]=dict(n=len(rr),background_region_update_sufficient=sum(r['background_update_sufficient'] for r in rr),subject_and_template_update_sufficient=sum(r['subject_and_template_update_sufficient'] for r in rr),neither_alone_sufficient=sum(not r['background_update_sufficient'] and not r['subject_and_template_update_sufficient'] for r in rr),positive_without_rematching=sum(float(r['locked_old_match_margin'])>0 for r in rr))
result['limits']='Fixed original correct/wrong pair. Region hybrids are score arithmetic, not additional realistic pixel interventions; sufficiency against this pair is not full-gallery sufficiency. Groups can overlap.';write(O/'REGION_SCORE_SUMMARY.json',result);print(result)
