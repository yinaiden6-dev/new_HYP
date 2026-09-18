#!/usr/bin/env python3
"""Independent sealed-result recomputation; does not import reducer/gate module."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
REAL=ROOT/'results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1';CBIND=ROOT/'results/romav2_colnomic_new_difficult_sealed_cbind_prejoin_v1';FULL=ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1';JOIN=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/target_join_manifest.json';AUTH=ROOT/'registry/romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901.json';RESULT=ROOT/'results/romav2_colnomic_new_difficult_sealed_directional_v1/result.json';OUT=ROOT/'results/romav2_colnomic_new_difficult_sealed_directional_v1/independent_validation.json';W=torch.tensor([0.9812819097198987,-3.812628067996388,5.826320131219255,5.860970045711693,-0.14878670951682949,-0.2442796885686483],dtype=torch.float64);B=-1.5548565799571785
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sym(a,b):return (float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12)
def feat(raw,c,ch,w):
 mean=sum(raw)/len(raw);std=(sum((x-mean)**2 for x in raw)/len(raw))**.5;lc,lw=c[ch],c[w];sc=lc['real_score']/max(lc['visibility_mass'],1e-12);sw=lw['real_score']/max(lw['visibility_mass'],1e-12);qc=lc['real_score']-lc['query_control_score'];qw=lw['real_score']-lw['query_control_score'];rc=lc['real_score']-lc['reference_control_score'];rw=lw['real_score']-lw['reference_control_score'];return torch.tensor([(raw[ch]-raw[w])/max(std,1e-12),sym(lc['real_score'],lw['real_score']),sym(lc['visibility_mass'],lw['visibility_mass']),sym(sc,sw),sym(qc,qw),sym(rc,rw)],dtype=torch.float64)
def rows(root):
 d={}
 for s in range(8):
  for r in json.load(open(root/f'shard{s:02d}/result.json'))['rows']:d[int(r['query_ordinal'])]=r
 return d
def view(real,control,mode):
 c={int(x['candidate_position']):dict(x) for x in (control if mode=='CBIND' else real)['candidates']}
 if mode=='Q':
  for z in c.values():q=z['query_control_score'];z['real_score']=q;z['query_control_score']=q
 if mode=='R':
  for z in c.values():r=z['reference_control_score'];z['real_score']=r;z['reference_control_score']=r
 return c
def action(mode,real,control,full,target,labels):
 c=view(real,control,mode);axis=[int(c[i]['physical_row']) for i in range(128)];raw=[float(c[i]['raw_score']) for i in range(128)];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));values=[]
 for ch in range(128):
  if ch!=winner:values.append((float((W*feat(raw,c,ch,winner)).sum()+B),ch))
 z,best=max(values,key=lambda x:(x[0],-axis[x[1]]));winner_label=labels[axis[winner]];best_label=labels[axis[best]];switch=z>0 and best_label!=winner_label;base_labels=[labels[int(x)] for x in full['ranked_physical_rows']];final=([best_label]+[x for x in base_labels if x!=best_label]) if switch else list(base_labels);br=base_labels.index(target)+1;fr=final.index(target)+1;tp=[i for i,row in enumerate(axis) if labels[row]==target];target_position=tp[0] if tp else None;target_logit=None if target_position is None or target_position==winner else float((W*feat(raw,c,target_position,winner)).sum()+B);scores=list(map(float,full['ranked_score_values']));return {'mode':mode,'target_present_c128':target_position is not None,'target_position':target_position,'base_winner_position':winner,'proposed_challenger_position':best,'switch_logit':z,'target_switch_logit':target_logit,'raw_action':'SWITCH' if z>0 else 'HOLD','effective_action':'SWITCH' if switch else 'HOLD','base_rank':br,'final_rank':fr,'base_correct':br==1,'final_correct':fr==1,'raw_target_minus_winner_margin':scores[br-1]-scores[0],'wrong_to_wrong':br!=1 and fr!=1 and switch}
def summary(a):
 n=len(a);res={x['query_ordinal'] for x in a if not x['base_correct'] and x['final_correct']};brk={x['query_ordinal'] for x in a if x['base_correct'] and not x['final_correct']};base=sum(x['base_correct'] for x in a);final=sum(x['final_correct'] for x in a);return {'query_count':n,'target_present_c128_count':sum(x['target_present_c128'] for x in a),'base_top1':base,'final_top1':final,'base_r1':base/n,'final_r1':final/n,'base_mrr':sum(1/x['base_rank'] for x in a)/n,'final_mrr':sum(1/x['final_rank'] for x in a)/n,'rescue':len(res),'break':len(brk),'wrong_to_wrong':sum(x['wrong_to_wrong'] for x in a),'switch_count':sum(x['effective_action']=='SWITCH' for x in a),'rescue_ordinals':sorted(res),'break_ordinals':sorted(brk),'r1_increment':(final-base)/n}
def main():
 if OUT.exists():raise RuntimeError('immutable sealed validation exists')
 result=json.load(open(RESULT));authority=json.load(open(AUTH));real=rows(REAL);cbind=rows(CBIND);full=rows(FULL);join=json.load(open(JOIN));targets={int(x['query_ordinal']):x for x in join['rows']};gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;acts={m:[] for m in ('REAL','CBIND','Q','R')};join_ok=True
 for o in range(31):
  t=targets[o];target=str(t['target_exact_label']);join_ok &= t['query_id']==real[o]['query_id']==cbind[o]['query_id']==full[o]['query_id'] and labels[int(t['target_gallery_physical_row'])]==target and sha(gallery.raw_paths[int(t['target_gallery_physical_row'])])==t['target_reference_sha256']
  for m in acts:
   a=action(m,real[o],cbind[o],full[o],target,labels);a.update({'query_ordinal':o,'query_id':t['query_id']});acts[m].append(a)
 sums={m:summary(v) for m,v in acts.items()};rs=sums['REAL'];retrieval=rs['final_top1']>rs['base_top1'] and rs['final_mrr']>=rs['base_mrr'] and rs['rescue']>rs['break'] and rs['break']<=1;real_res=set(rs['rescue_ordinals']);controls={}
 for m in ('CBIND','Q','R'):
  kept=len(real_res&set(sums[m]['rescue_ordinals']));controls[m]={'real_rescue_retained_count':kept,'real_rescue_retention':kept/len(real_res) if real_res else 1.0,'real_increment_gt_control':rs['r1_increment']>sums[m]['r1_increment']}
 binding=retrieval and controls['CBIND']['real_increment_gt_control'] and controls['CBIND']['real_rescue_retention']<1;coord=all(controls[m]['real_increment_gt_control'] and controls[m]['real_rescue_retention']<1 for m in ('Q','R'));status='SEALED_DIRECTIONAL_CANDIDATE_BOUND_RETRIEVAL_GO' if retrieval and binding else ('SEALED_DIRECTIONAL_RETRIEVAL_ONLY_GO' if retrieval else 'SEALED_DIRECTIONAL_RETRIEVAL_NO_GO');checks={'result_envelope':result['logical_sha256']==logical(result) and result['authority_sha256']==sha(AUTH),'authority':authority['status']=='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORIZED','join':join_ok,'actions':acts==result['actions'],'summaries':sums==result['summaries'],'controls':controls==result['control_decisions'],'decisions':result['status']==status and result['directional_retrieval_go']==retrieval and result['candidate_binding_support']==binding and result['coordinate_sensitivity_diagnostic']==coord and result['strict_spatial_causal_claim_authorized'] is False,'all31':all(len(v)==31 for v in acts.values())};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_directional_independent_validation_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_DIRECTIONAL_INDEPENDENT_VALIDATION_PASS' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_DIRECTIONAL_INDEPENDENT_VALIDATION_FAIL','checks':checks,'recomputed_status':status,'recomputed_summaries':sums,'recomputed_controls':controls,'result_sha256':sha(RESULT),'target_join_manifest_read_count':1,'target_label_read_count':31,'model_update_count':0};OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'recomputed_status':status,'summaries':sums,'controls':controls},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
