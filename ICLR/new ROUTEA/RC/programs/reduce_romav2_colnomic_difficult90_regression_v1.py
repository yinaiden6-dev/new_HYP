#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'programs'))
import reduce_romav2_colnomic_new_difficult_sealed_v1 as frozen_reduce
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
REAL=ROOT/'results/romav2_colnomic_difficult90_real_prejoin_v1';CBIND=ROOT/'results/romav2_colnomic_difficult90_cbind_prejoin_v1';TOK=ROOT/'results/romav2_colnomic_difficult90_token_fullrank_prejoin_v1';JOIN=ROOT/'results/romav2_colnomic_difficult90_regression_manifest_v1/target_join_manifest.json';AUTH=ROOT/'registry/romav2_colnomic_difficult90_reduction_authority_v1_20260901.json';OUT=ROOT/'results/romav2_colnomic_difficult90_frozen_regression_v1/result.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def json_rows(root):
 d={}
 for s in range(10):
  for r in json.load(open(root/f'shard{s:02d}/result.json'))['rows']:d[int(r['query_ordinal'])]=r
 return d
def token_rows():
 d={}
 for s in range(10):
  p=torch.load(TOK/f'shard{s:02d}/payload.pt',map_location='cpu',weights_only=False,mmap=True)
  for r in p['records']:d[int(r['query_ordinal'])]=r
 return d
def main():
 if OUT.exists():raise RuntimeError('immutable difficult90 result exists')
 authority=json.load(open(AUTH));assert authority['status']=='ROMAV2_COLNOMIC_DIFFICULT90_REDUCTION_AUTHORIZED';real=json_rows(REAL);cbind=json_rows(CBIND);token=token_rows();join=json.load(open(JOIN));assert join['status']=='ROMAV2_COLNOMIC_DIFFICULT90_TARGET_JOIN_BOUND_REDUCER_ONLY' and join['logical_sha256']==logical(join);targets={int(x['query_ordinal']):x for x in join['rows']};gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;actions={m:[] for m in ('REAL','CBIND','Q','R')};join_checks=[]
 for o in range(90):
  t=targets[o];rr=real[o];cc=cbind[o];tt=token[o];target_row=int(t['target_gallery_physical_row']);target=str(t['target_exact_label']);ok=t['query_id']==rr['query_id']==cc['query_id']==tt['query_id'] and t['opened_split']==rr['opened_split']==cc['opened_split']==tt['opened_split'] and labels[target_row]==target and sha(gallery.raw_paths[target_row])==t['target_reference_sha256'];join_checks.append({'query_ordinal':o,'exact':ok});
  if not ok:raise RuntimeError(f'difficult90 join mismatch {o}')
  full={'ranked_physical_rows':tt['fullrank_physical_rows'],'ranked_score_values':tt['fullrank_score_values'].tolist()}
  for m in actions:
   a=frozen_reduce.evaluate_path(m,rr,cc,full,target,labels);a.update({'query_ordinal':o,'query_id':t['query_id'],'opened_split':t['opened_split']});actions[m].append(a)
 summary={m:frozen_reduce.summarize(v) for m,v in actions.items()};split_summary={split:{m:frozen_reduce.summarize([x for x in v if x['opened_split']==split]) for m,v in actions.items()} for split in ('val','test')};rs=summary['REAL'];regression_pass=rs['final_top1']>=rs['base_top1'] and rs['final_mrr']>=rs['base_mrr'] and rs['break']<=1 and ((rs['rescue']>rs['break']) if rs['rescue'] else rs['break']==0);real_res=set(rs['rescue_ordinals']);controls={}
 for m in ('CBIND','Q','R'):
  kept=len(real_res&set(summary[m]['rescue_ordinals']));controls[m]={'real_rescue_retained_count':kept,'real_rescue_retention':kept/len(real_res) if real_res else 1.0,'real_increment_gt_control':rs['r1_increment']>summary[m]['r1_increment']}
 cbind_support=bool(real_res) and controls['CBIND']['real_increment_gt_control'] and controls['CBIND']['real_rescue_retention']<1;value={'schema_version':'rc_romav2_colnomic_difficult90_frozen_regression_v1_20260901','status':'DIFFICULT90_FROZEN_REGRESSION_PASS' if regression_pass else 'DIFFICULT90_FROZEN_REGRESSION_FAIL','claim_level':'OPENED_REGRESSION_ONLY_NOT_INDEPENDENT_EVIDENCE','authority_sha256':sha(AUTH),'population':{'query_count':90,'identity_count':9,'val_count':38,'test_count':52,'gallery_physical_rows':5413,'corrected_exact_labels':5412},'summaries':summary,'split_summaries':split_summary,'controls':controls,'regression_pass':regression_pass,'positive_retrieval_increment':rs['final_top1']>rs['base_top1'] and rs['final_mrr']>=rs['base_mrr'],'candidate_binding_support':cbind_support,'strict_spatial_causal_claim_authorized':False,'join_checks':join_checks,'actions':actions,'target_join_read_count':1,'target_label_read_count':90,'model_update_count':0,'threshold_scan_count':0,'next_authorized_stage':'NEW_UNTOUCHED_EXTERNAL_DATASET' if regression_pass else None,'logical_sha256':''};value['logical_sha256']=logical(value);OUT.parent.mkdir(parents=True,exist_ok=False);OUT.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':value['status'],'summaries':summary,'controls':controls},sort_keys=True))
if __name__=='__main__':main()
