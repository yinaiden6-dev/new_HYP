#!/usr/bin/env python3
import json,math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"programs"))
import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa:E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import FrozenColNomicBaseV1  # noqa:E402
V1=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json";V2=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json";FULL=ROOT/"results/romav2_colnomic_visibility_xf_full_c128_gate_v1/result.json";PRE=ROOT/"results/romav2_colnomic_visibility_xf_full_c128_prejoin_v1";SOURCE=ROOT/"results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json";OUT=ROOT/"results/romav2_colnomic_no_regret_action_gate_v1/result.json"
def sym(a,b):return (float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12)
def candidate_feature(raw,local,c,w):
 mean=sum(raw)/len(raw);std=(sum((x-mean)**2 for x in raw)/len(raw))**.5;z=(raw[c]-raw[w])/max(std,1e-12);lc,lw=local[c],local[w];sc=lc['real_score']/max(lc['visibility_mass'],1e-12);sw=lw['real_score']/max(lw['visibility_mass'],1e-12);qc=lc['real_score']-lc['query_control_score'];qw=lw['real_score']-lw['query_control_score'];rc=lc['real_score']-lc['reference_control_score'];rw=lw['real_score']-lw['reference_control_score'];return torch.tensor([z,sym(lc['real_score'],lw['real_score']),sym(lc['visibility_mass'],lw['visibility_mass']),sym(sc,sw),sym(qc,qw),sym(rc,rw)],dtype=torch.float64)
def pair_data(path,base,source_rows):
 doc=json.loads(path.read_text());features=[];labels=[];meta=[]
 for row in doc['rows']:
  e=int(row['execution_ordinal']);axis=source_rows[e];raw=base.scores(e,axis).tolist();cand={x['candidate_position']:x for x in row['candidates']};target=next(x['candidate_position'] for x in row['candidates'] if x['role']=='TARGET');other=next(x['candidate_position'] for x in row['candidates'] if x['role']=='COMPETITOR')
  if row['base_correct']:winner,challenger,label=target,other,0.0
  else:winner,challenger,label=other,target,1.0
  features.append(candidate_feature(raw,cand,challenger,winner));labels.append(label);meta.append({'execution_ordinal':e,'label_switch':bool(label),'winner':winner,'challenger':challenger})
 return torch.stack(features),torch.tensor(labels,dtype=torch.float64),meta
def train(x,y):
 torch.manual_seed(17);head=nn.Linear(6,1,dtype=torch.float64);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=1e-3)
 weights=torch.where(y.eq(0),torch.full_like(y,4.0),torch.ones_like(y))
 for _ in range(1000):
  opt.zero_grad();logits=head(x).squeeze(1);loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).mean();loss.backward();opt.step()
 return head,float(loss.detach())
def evaluate(head,x,y):
 logits=head(x).squeeze(1).detach();pred=logits>0;return {'accuracy':int(pred.eq(y.bool()).sum()),'rescue':int((pred&y.bool()).sum()),'hold':int((~pred&~y.bool()).sum()),'switch_count':int(pred.sum()),'logits':logits.tolist()}
def main():
 if OUT.exists():raise RuntimeError('immutable action gate output exists')
 source=json.loads(SOURCE.read_text())['records'];axes={int(r['execution_ordinal']):r['candidate_physical_rows'] for r in source};base=FrozenColNomicBaseV1(ROOT);x3,y3,m3=pair_data(V1,base,axes);x4,y4,m4=pair_data(V2,base,axes);h3,l3=train(x3,y3);h4,l4=train(x4,y4);e34=evaluate(h3,x4,y4);e43=evaluate(h4,x3,y3);cross_gates={'fold3_to_4':e34['accuracy']>=24 and e34['rescue']>=10 and e34['hold']>=14,'fold4_to_3':e43['accuracy']>=24 and e43['rescue']>=10 and e43['hold']>=14};cross=all(cross_gates.values());combined,loss=train(torch.cat((x3,x4)),torch.cat((y3,y4)));pre=[]
 for s in range(4):pre+=json.loads((PRE/f'shard{s:02d}/result.json').read_text())['rows']
 pmap={int(r['execution_ordinal']):r for r in pre};full=json.loads(FULL.read_text());actions=[]
 for row in full['rows']:
  if not row['target_present']:continue
  e=int(row['execution_ordinal']);r=pmap[e];axis=[x['physical_row'] for x in r['candidates']];raw=base.scores(e,axis).tolist();winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cands={x['candidate_position']:x for x in r['candidates']};logits=[]
  for c in range(128):
   if c==winner:continue
   logits.append((float(combined(candidate_feature(raw,cands,c,winner)[None]).detach()),c))
  best_logit,best=max(logits,key=lambda x:(x[0],-axis[x[1]]));decision=best if best_logit>0 else winner;target=int(row['target_position']);actions.append({'execution_ordinal':e,'query_id':row['query_id'],'target_position':target,'base_winner':winner,'proposed_challenger':best,'switch_logit':best_logit,'decision':'SWITCH' if best_logit>0 else 'HOLD','final_position':decision,'base_correct':winner==target,'final_correct':decision==target})
 base_top=sum(x['base_correct'] for x in actions);final_top=sum(x['final_correct'] for x in actions);rescue=sum((not x['base_correct']) and x['final_correct'] for x in actions);brk=sum(x['base_correct'] and (not x['final_correct']) for x in actions);final_gate=cross and final_top>=28 and rescue>brk;value={'schema_version':'rc_romav2_colnomic_no_regret_action_gate_v1_20260831','status':'ROMAV2_COLNOMIC_NO_REGRET_ACTION_HEADROOM' if final_gate else 'ROMAV2_COLNOMIC_NO_REGRET_ACTION_NO_GO','claim_level':'INTERNAL_CROSSFIT_ACTION_GATE_ONLY','contract_sha256':core.sha(ROOT/'plan/ROMAV2_COLNOMIC_NO_REGRET_ACTION_GATE_V1_20260831.md'),'sources':{'fold3_sha256':core.sha(V1),'fold4_sha256':core.sha(V2),'full_c128_sha256':core.sha(FULL)},'parameter_count':7,'crossfit':{'fold3_to_4':e34,'fold4_to_3':e43,'gates':cross_gates,'pass':cross,'train_losses':[l3,l4]},'combined_head':{'weight':combined.weight.detach().flatten().tolist(),'bias':float(combined.bias.detach()),'train_loss':loss},'full_c128_action_summary':{'base_top1':base_top,'final_top1':final_top,'rescue':rescue,'break':brk,'switch_count':sum(x['decision']=='SWITCH' for x in actions)},'final_gate':final_gate,'actions':actions,'model_update_count':0,'romav2_update_count':0,'opened_read_count':0,'sealed_read_count':0,'P0_authorized':False,'scientific_GO_or_NO_GO':None,'next_authorized_stage':'ROMAV2_COLNOMIC_ACTION_FULL_C128_CONFIRMATION' if final_gate else None,'logical_sha256':''};value['logical_sha256']=core.logical(value);core.atomic(OUT,value);print(json.dumps({'status':value['status'],'crossfit':value['crossfit'],'action':value['full_c128_action_summary']},sort_keys=True))
if __name__=='__main__':main()

