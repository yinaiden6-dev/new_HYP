"""Paired identity-cluster uncertainty and repair overlap; no parameter selection."""
from pathlib import Path
import sys,csv,math
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent))
from experiment_utils import read,write
def cr(n):return list(csv.DictReader((O/n).open(encoding='utf-8-sig')))
c=cr('pure_image_all1117.csv');cc={(r['query'],r['method']):r for r in cr('reference_contrast_per_query.csv')};crop={r['query']:int(r['prior_crop_rank']) for r in cr('prior_crop_link1117.csv')};out={};rng=np.random.default_rng(20261014)
for scope in ['all','val','test']:
 rr=[r for r in c if scope=='all' or r['split']==scope];groups={}
 for r in rr:groups.setdefault(r['package_id'],[]).append(r)
 count=np.array([len(v) for v in groups.values()]);old=np.array([sum(int(r['image_rank'])==1 for r in v) for v in groups.values()]);new=np.array([sum(int(cc[r['query'],'hard20_std']['rank'])==1 for r in v) for v in groups.values()]);draw=rng.integers(0,len(groups),(10000,len(groups)));micro=(new[draw]-old[draw]).sum(1)/count[draw].sum(1);macro=((new-old)/count)[draw].mean(1)
 repairs=[r for r in rr if int(r['image_rank'])>1 and int(cc[r['query'],'hard20_std']['rank'])==1];harms=[r for r in rr if int(r['image_rank'])==1 and int(cc[r['query'],'hard20_std']['rank'])>1];n=len(repairs)+len(harms);p=min(1.,2*sum(math.comb(n,k) for k in range(min(len(repairs),len(harms))+1))/2**n) if n else 1
 out[scope]=dict(queries=len(rr),identities=len(groups),repairs=len(repairs),harms=len(harms),repair_identities=len({r['package_id'] for r in repairs}),harm_identities=len({r['package_id'] for r in harms}),micro_difference=float((new.sum()-old.sum())/count.sum()),micro_identity_cluster95=np.quantile(micro,[.025,.975]).tolist(),macro_difference=float(((new-old)/count).mean()),macro_identity_cluster95=np.quantile(macro,[.025,.975]).tolist(),nominal_paired_exact_p=p,repair_overlap_subject=sum(int(r['subject_rank'])==1 for r in repairs),repair_overlap_crop=sum(crop[r['query']]==1 for r in repairs),repair_outside_corrected_fusion=sum(int(r['corrected_fusion_rank'])>1 for r in repairs),repaired_original_ranks=[int(r['image_rank']) for r in repairs])
s=read(O/'reference_contrast_summary.json');out['nulls']=dict(n=10,min_correct=min(r['correct'] for r in s['nulls']),max_correct=max(r['correct'] for r in s['nulls']),mean_correct=float(np.mean([r['correct'] for r in s['nulls']])),matched_visual_template_weight_mass=True);out['limits']=['Historical identity split, repeatedly inspected; not a fresh blinded test.','All-cohort and validation uncertainty are descriptive after selection, not unbiased confirmatory inference.','Identity-cluster bootstrap has only18 historical test identities; uncertainty can be wide.','Ten shuffles are a mechanistic negative control, insufficient for fine permutation significance.']
write(O/'REFERENCE_CONTRAST_STATISTICS.json',out);print(out)
