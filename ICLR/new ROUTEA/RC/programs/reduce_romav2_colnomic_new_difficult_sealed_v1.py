#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as gate
REAL=ROOT/'results/romav2_colnomic_new_difficult_sealed_roma_prejoin_v1';CBIND=ROOT/'results/romav2_colnomic_new_difficult_sealed_cbind_prejoin_v1';FULL=ROOT/'results/romav2_colnomic_new_difficult_sealed_fullrank_prejoin_v1';JOIN=ROOT/'results/romav2_colnomic_new_difficult_sealed_manifest_v1/target_join_manifest.json';AUTH=ROOT/'registry/romav2_colnomic_new_difficult_sealed_reduction_authority_v1_20260901.json';OUT=ROOT/'results/romav2_colnomic_new_difficult_sealed_directional_v1/result.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load_rows(root):
 out={}
 for s in range(8):
  for r in json.load(open(root/f'shard{s:02d}/result.json'))['rows']:out[int(r['query_ordinal'])]=r
 if sorted(out)!=list(range(31)):raise RuntimeError('prejoin population drift')
 return out
def path_candidates(real,control,mode):
 base={int(x['candidate_position']):dict(x) for x in (control if mode=='CBIND' else real)['candidates']}
 if mode=='Q':
  for x in base.values():x['real_score']=x['query_control_score'];x['query_control_score']=x['real_score']
 elif mode=='R':
  for x in base.values():x['real_score']=x['reference_control_score'];x['reference_control_score']=x['real_score']
 return base
def evaluate_path(mode,real,control,full,target_label,labels):
 c=path_candidates(real,control,mode);axis=[int(c[i]['physical_row']) for i in range(128)];raw=[float(c[i]['raw_score']) for i in range(128)];winner=max(range(128),key=lambda i:(raw[i],-axis[i]));values=[]
 for challenger in range(128):
  if challenger!=winner:values.append((gate.logit(raw,c,challenger,winner),challenger))
 best_logit,best=max(values,key=lambda x:(x[0],-axis[x[1]]));raw_switch=best_logit>gate.SWITCH_THRESHOLD;winner_label=labels[axis[winner]];challenger_label=labels[axis[best]];effective_switch=raw_switch and challenger_label!=winner_label
 base_rows=list(map(int,full['ranked_physical_rows']));base_labels=[labels[x] for x in base_rows];base_scores=list(map(float,full['ranked_score_values']));
 if len(base_labels)!=5412 or len(set(base_labels))!=5412:raise RuntimeError('full exact-label order drift')
 final_labels=([challenger_label]+[x for x in base_labels if x!=challenger_label]) if effective_switch else list(base_labels)
 base_rank=base_labels.index(target_label)+1;final_rank=final_labels.index(target_label)+1;target_positions=[i for i,row in enumerate(axis) if labels[row]==target_label];target_position=target_positions[0] if target_positions else None;target_logit=None if target_position is None or target_position==winner else gate.logit(raw,c,target_position,winner);raw_margin=base_scores[base_rank-1]-base_scores[0]
 return {'mode':mode,'target_present_c128':target_position is not None,'target_position':target_position,'base_winner_position':winner,'proposed_challenger_position':best,'switch_logit':best_logit,'target_switch_logit':target_logit,'raw_action':'SWITCH' if raw_switch else 'HOLD','effective_action':'SWITCH' if effective_switch else 'HOLD','base_rank':base_rank,'final_rank':final_rank,'base_correct':base_rank==1,'final_correct':final_rank==1,'raw_target_minus_winner_margin':raw_margin,'wrong_to_wrong':base_rank!=1 and final_rank!=1 and effective_switch}
def summarize(actions):
 n=len(actions);base=sum(x['base_correct'] for x in actions);final=sum(x['final_correct'] for x in actions);rescues={x['query_ordinal'] for x in actions if not x['base_correct'] and x['final_correct']};breaks={x['query_ordinal'] for x in actions if x['base_correct'] and not x['final_correct']};return {'query_count':n,'target_present_c128_count':sum(x['target_present_c128'] for x in actions),'base_top1':base,'final_top1':final,'base_r1':base/n,'final_r1':final/n,'base_mrr':sum(1/x['base_rank'] for x in actions)/n,'final_mrr':sum(1/x['final_rank'] for x in actions)/n,'rescue':len(rescues),'break':len(breaks),'wrong_to_wrong':sum(x['wrong_to_wrong'] for x in actions),'switch_count':sum(x['effective_action']=='SWITCH' for x in actions),'rescue_ordinals':sorted(rescues),'break_ordinals':sorted(breaks),'r1_increment':(final-base)/n}
def main():
 if OUT.exists():raise RuntimeError('immutable sealed directional result exists')
 authority=json.load(open(AUTH));
 if authority.get('status')!='ROMAV2_COLNOMIC_NEW_DIFFICULT_SEALED_REDUCTION_AUTHORIZED' or authority.get('permissions',{}).get('target_join_attempt_count')!=1:raise RuntimeError('sealed reduction authority absent')
 real=load_rows(REAL);cbind=load_rows(CBIND);full=load_rows(FULL);join=json.load(open(JOIN));
 if join['status']!='NEW_DIFFICULT_SEALED_TARGET_JOIN_BOUND_REDUCER_ONLY' or join['logical_sha256']!=logical(join) or len(join['rows'])!=31:raise RuntimeError('target join drift')
 targets={int(x['query_ordinal']):x for x in join['rows']};gallery=build_gallery_source(verify_cache_file_sha256=True);labels=gallery.corrected_identities;join_checks=[];actions={m:[] for m in ('REAL','CBIND','Q','R')}
 for ordinal in range(31):
  t=targets[ordinal];rr=real[ordinal];cc=cbind[ordinal];ff=full[ordinal];target_row=int(t['target_gallery_physical_row']);target_label=str(t['target_exact_label']);query_exact=t['query_id']==rr['query_id']==cc['query_id']==ff['query_id'];label_exact=labels[target_row]==target_label;sha_exact=sha(gallery.raw_paths[target_row])==t['target_reference_sha256'];join_checks.append({'query_ordinal':ordinal,'query_id_exact':query_exact,'target_row_label_exact':label_exact,'target_reference_sha256_exact':sha_exact})
  if not(query_exact and label_exact and sha_exact):raise RuntimeError(f'target join mismatch {ordinal}')
  for mode in actions:
   a=evaluate_path(mode,rr,cc,ff,target_label,labels);a.update({'query_ordinal':ordinal,'query_id':t['query_id']});actions[mode].append(a)
 summary={m:summarize(v) for m,v in actions.items()};real_s=summary['REAL'];retrieval_go=real_s['final_top1']>real_s['base_top1'] and real_s['final_mrr']>=real_s['base_mrr'] and real_s['rescue']>real_s['break'] and real_s['break']<=1
 real_rescue=set(real_s['rescue_ordinals']);control={}
 for mode in ('CBIND','Q','R'):
  retained=len(real_rescue & set(summary[mode]['rescue_ordinals']));retention=(retained/len(real_rescue)) if real_rescue else 1.0;control[mode]={'real_rescue_retained_count':retained,'real_rescue_retention':retention,'real_increment_gt_control':real_s['r1_increment']>summary[mode]['r1_increment']}
 cbind_support=retrieval_go and control['CBIND']['real_increment_gt_control'] and control['CBIND']['real_rescue_retention']<1.0;coordinate_diagnostic=(control['Q']['real_increment_gt_control'] and control['Q']['real_rescue_retention']<1.0 and control['R']['real_increment_gt_control'] and control['R']['real_rescue_retention']<1.0)
 if retrieval_go and cbind_support:status='SEALED_DIRECTIONAL_CANDIDATE_BOUND_RETRIEVAL_GO'
 elif retrieval_go:status='SEALED_DIRECTIONAL_RETRIEVAL_ONLY_GO'
 else:status='SEALED_DIRECTIONAL_RETRIEVAL_NO_GO'
 value={'schema_version':'rc_romav2_colnomic_new_difficult_sealed_directional_v1_20260901','status':status,'claim_level':'ONE_SHOT_31_QUERY_DIRECTIONAL_NOT_PAPER_STRENGTH','authority_sha256':sha(AUTH),'frozen_gate_module_sha256':sha(ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py'),'population':{'query_count':31,'gallery_physical_row_count':5413,'corrected_exact_label_count':5412},'summaries':summary,'control_decisions':control,'directional_retrieval_go':retrieval_go,'candidate_binding_support':cbind_support,'coordinate_sensitivity_diagnostic':coordinate_diagnostic,'strict_spatial_causal_claim_authorized':False,'join_checks':join_checks,'actions':actions,'target_join_manifest_read_count':1,'target_label_read_count':31,'model_update_count':0,'threshold_scan_count':0,'next_authorized_stage':'NEW_UNTOUCHED_EXTERNAL_DATASET' if retrieval_go else None,'logical_sha256':''};value['logical_sha256']=logical(value);OUT.parent.mkdir(parents=True,exist_ok=False);OUT.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':status,'summaries':summary,'controls':control},sort_keys=True))
if __name__=='__main__':main()
