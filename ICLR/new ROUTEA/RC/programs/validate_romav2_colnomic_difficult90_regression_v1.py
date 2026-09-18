#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import validate_romav2_colnomic_new_difficult_sealed_v1 as independent
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
REAL=ROOT/'results/romav2_colnomic_difficult90_real_prejoin_v1';CBIND=ROOT/'results/romav2_colnomic_difficult90_cbind_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_difficult90_token_fullrank_prejoin_v1';JOIN=ROOT/'results/romav2_colnomic_difficult90_regression_manifest_v1/target_join_manifest.json';AUTH=ROOT/'registry/romav2_colnomic_difficult90_reduction_authority_v1_20260901.json';RESULT=ROOT/'results/romav2_colnomic_difficult90_frozen_regression_v1/result.json';OUT=ROOT/'results/romav2_colnomic_difficult90_frozen_regression_v1/independent_validation.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def jr(root):
 d={}
 for s in range(10):
  for r in json.load(open(root/f'shard{s:02d}/result.json'))['rows']:d[int(r['query_ordinal'])]=r
 return d
def tr():
 d={}
 for s in range(10):
  p=torch.load(TOK/f'shard{s:02d}/payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for r in p['records']:d[int(r['query_ordinal'])]=r
 return d
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 independent validation exists')
 result=json.load(open(RESULT));authority=json.load(open(AUTH));real=jr(REAL);cbind=jr(CBIND);token=tr();targets={int(x['query_ordinal']):x for x in json.load(open(JOIN))['rows']};gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;actions={m:[] for m in ('REAL','CBIND','Q','R')};join_ok=True
 for o in range(90):
  t=targets[o];target=t['target_exact_label'];join_ok &= t['query_id']==real[o]['query_id']==cbind[o]['query_id']==token[o]['query_id'] and labels[int(t['target_gallery_physical_row'])]==target and sha(gallery.raw_paths[int(t['target_gallery_physical_row'])])==t['target_reference_sha256'];full={'ranked_physical_rows':token[o]['fullrank_physical_rows'],'ranked_score_values':token[o]['fullrank_score_values'].tolist()}
  for m in actions:
   a=independent.action(m,real[o],cbind[o],full,target,labels);a.update({'query_ordinal':o,'query_id':t['query_id'],'opened_split':t['opened_split']});actions[m].append(a)
 summary={m:independent.summary(v) for m,v in actions.items()};split_summary={split:{m:independent.summary([x for x in v if x['opened_split']==split]) for m,v in actions.items()} for split in ('val','test')};rs=summary['REAL'];reg=rs['final_top1']>=rs['base_top1'] and rs['final_mrr']>=rs['base_mrr'] and rs['break']<=1 and ((rs['rescue']>rs['break']) if rs['rescue'] else rs['break']==0);real_res=set(rs['rescue_ordinals']);controls={}
 for m in ('CBIND','Q','R'):
  kept=len(real_res&set(summary[m]['rescue_ordinals']));controls[m]={'real_rescue_retained_count':kept,'real_rescue_retention':kept/len(real_res) if real_res else 1.0,'real_increment_gt_control':rs['r1_increment']>summary[m]['r1_increment']}
 binding=bool(real_res) and controls['CBIND']['real_increment_gt_control'] and controls['CBIND']['real_rescue_retention']<1;checks={'authority':authority['status']=='ROMAV2_COLNOMIC_DIFFICULT90_REDUCTION_AUTHORIZED','envelope':result['logical_sha256']==logical(result) and result['authority_sha256']==sha(AUTH),'join':join_ok,'actions':actions==result['actions'],'summaries':summary==result['summaries'],'split_summaries':split_summary==result['split_summaries'],'controls':controls==result['controls'],'decisions':result['regression_pass']==reg and result['candidate_binding_support']==binding and result['strict_spatial_causal_claim_authorized'] is False,'all90':all(len(v)==90 for v in actions.values())};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_difficult90_frozen_regression_independent_validation_v1_20260901','status':'ROMAV2_COLNOMIC_DIFFICULT90_REGRESSION_INDEPENDENT_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_DIFFICULT90_REGRESSION_INDEPENDENT_VALIDATION_FAIL','checks':checks,'recomputed_status':'DIFFICULT90_FROZEN_REGRESSION_PASS' if reg else 'DIFFICULT90_FROZEN_REGRESSION_FAIL','recomputed_summaries':summary,'recomputed_controls':controls,'result_sha256':sha(RESULT),'target_join_read_count':1,'target_label_read_count':90,'model_update_count':0};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'recomputed_status':v['recomputed_status'],'summaries':summary},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
