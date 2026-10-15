"""Relate full-gallery repairs to fixed candidates and locked-argmax scores."""
import sys,csv,json,collections
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent))
from experiment_utils import read,write,csvwrite
def cr(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
def main(tag):
 cohort=read(O/f'cohort{tag.split("_")[0]}.json');ledger={r['query']:r for r in cr(f'causal_ledger_{tag}.csv')};pairs=cr(f'causal_pair_decomposition_{tag}.csv');index={(r['query'],r['arm'],r['scope'],r['negative']):r for r in pairs};out=[]
 prior=cr('../matching_controls_per_query.csv');px={(r['query'],r['method']):int(r['rank']) for r in prior};controls=['foreground_only','center_fixed_template_fraction','any_overlap_template','center_refvisual']
 for r in cohort:
  q=r['query'];l=ledger[q];negative=r['old_center_top1'] if r['role']=='harm' else r['image_negative']
  for arm in ['far_background_gray','far_background_blur']:
   p=index[q,arm,'subject_template',negative];out.append(dict(query=q,role=r['role'],package_id=r['package_id'],arm=arm,negative=negative,original_subject_rank=l['fresh_subject_rank'],edited_subject_rank=l[arm+'_subject_template_rank'],both_fill_subject_repair=l['subject_repaired_by_both_far_fills'],prior_controls_repaired=any(px.get((q,m),999999)==1 for m in controls),**{k:p[k] for k in ['original_margin','free_margin','locked_margin','fixed_match_representation_effect','rematching_effect','positive_changed_match_fraction','negative_changed_match_fraction']}))
 csvwrite(O/f'context_pair_mechanisms_{tag}.csv',out)
 groups={}
 for role in sorted({r['role'] for r in out}):
  rr=[r for r in out if r['role']==role];groups[role]={}
  for arm in ['far_background_gray','far_background_blur']:
   aa=[r for r in rr if r['arm']==arm];rescues=[r for r in aa if float(r['original_margin'])<0 and float(r['free_margin'])>0];groups[role][arm]=dict(fixed_pair_rescues=len(rescues),locked_old_matches_already_positive=sum(float(r['locked_margin'])>0 for r in rescues),requires_rematching_for_positive=sum(float(r['locked_margin'])<=0 for r in rescues),full_gallery_subject_repairs=sum(int(r['original_subject_rank'])>1 and int(r['edited_subject_rank'])==1 for r in aa))
 write(O/f'context_pair_summary_{tag}.json',groups);print(json.dumps(groups,indent=2))
 print('BOTH FILLS',json.dumps([r for r in out if r['both_fill_subject_repair']=='True'],indent=2))
if __name__=='__main__':main(sys.argv[1])
