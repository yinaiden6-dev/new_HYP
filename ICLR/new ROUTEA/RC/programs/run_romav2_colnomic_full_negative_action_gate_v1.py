#!/usr/bin/env python3
import json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"programs"))
import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa:E402
import run_romav2_colnomic_no_regret_action_gate_v1 as prior  # noqa:E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import FrozenColNomicBaseV1  # noqa:E402
PAIR3=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json";PAIR4=ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json";FULLTRAIN=ROOT/"results/romav2_colnomic_visibility_xf_full_c128_gate_v1/result.json";TRAINPRE=ROOT/"results/romav2_colnomic_visibility_xf_full_c128_prejoin_v1";EVALPRE=ROOT/"results/romav2_colnomic_visibility_xf_fullnegative_eval_prejoin_v1";SOURCE=ROOT/"results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json";ROLE=ROOT/"results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_shards";OUT=ROOT/"results/romav2_colnomic_full_negative_action_gate_v1/result.json"
def load_pre(root):
 rows=[]
 for s in range(4):
  p=root/f"shard{s:02d}/result.json";x=json.loads(p.read_text());assert x['target_role_read_count']==x['target_insertion_count']==0 and x['logical_sha256']==core.logical(x);rows+=x['rows']
 return rows
def train_queries(full,pre,base):
 pmap={int(r['execution_ordinal']):r for r in pre};queries=[]
 for joined in full['rows']:
  if not joined['target_present']:continue
  e=int(joined['execution_ordinal']);r=pmap[e];axis=[x['physical_row'] for x in r['candidates']];raw=base.scores(e,axis).tolist();winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cands={x['candidate_position']:x for x in r['candidates']};indices=tuple(c for c in range(128) if c!=winner);features=torch.stack([prior.candidate_feature(raw,cands,c,winner) for c in indices]);queries.append((winner,int(joined['target_position']),indices,features))
 return queries
def train_head(pair_x,pair_y,queries):
 torch.manual_seed(17);head=nn.Linear(6,1,dtype=torch.float64);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=1e-3);weights=torch.where(pair_y.eq(0),torch.full_like(pair_y,4.),torch.ones_like(pair_y))
 for _ in range(2000):
  opt.zero_grad();pair_loss=(F.binary_cross_entropy_with_logits(head(pair_x).squeeze(1),pair_y,reduction='none')*weights).mean();ql=[]
  for winner,target,indices,features in queries:
   values=head(features).squeeze(1)
   if target==winner:ql.append(4*F.softplus(values.max()))
   else:
    position=indices.index(target);positive=values[position];mask=torch.ones(len(indices),dtype=torch.bool);mask[position]=False;negative=values[mask].max();ql.append(F.softplus(-positive)+4*F.softplus(negative))
  loss=pair_loss+torch.stack(ql).mean();loss.backward();opt.step()
 return head,float(loss.detach()),float(pair_loss.detach()),float(torch.stack(ql).mean().detach())
def evaluate(head,pre,base):
 actions=[]
 for r in pre:
  e=int(r['execution_ordinal']);role=json.loads((ROLE/f'role_exec{e:03d}.json').read_text())
  if role.get('target_naturally_present') is not True:actions.append({'execution_ordinal':e,'query_id':r['query_id'],'target_present':False});continue
  axis=[x['physical_row'] for x in r['candidates']];raw=base.scores(e,axis).tolist();winner=max(range(128),key=lambda i:(raw[i],-axis[i]));cands={x['candidate_position']:x for x in r['candidates']};logits=[]
  for c in range(128):
   if c!=winner:logits.append((float(head(prior.candidate_feature(raw,cands,c,winner)[None]).detach()),c))
  best_logit,best=max(logits,key=lambda x:(x[0],-axis[x[1]]));decision=best if best_logit>0 else winner;target=int(role['target_candidate_position']);actions.append({'execution_ordinal':e,'query_id':r['query_id'],'target_present':True,'target_position':target,'base_winner':winner,'proposed_challenger':best,'switch_logit':best_logit,'decision':'SWITCH' if best_logit>0 else 'HOLD','final_position':decision,'base_correct':winner==target,'final_correct':decision==target})
 return actions
def main():
 if OUT.exists():raise RuntimeError('immutable full-negative gate exists')
 base=FrozenColNomicBaseV1(ROOT);records=json.loads(SOURCE.read_text())['records'];axes={int(r['execution_ordinal']):r['candidate_physical_rows'] for r in records};x3,y3,_=prior.pair_data(PAIR3,base,axes);x4,y4,_=prior.pair_data(PAIR4,base,axes);full=json.loads(FULLTRAIN.read_text());trainpre=load_pre(TRAINPRE);queries=train_queries(full,trainpre,base);head,loss,pair_loss,full_loss=train_head(torch.cat((x3,x4)),torch.cat((y3,y4)),queries);evalpre=load_pre(EVALPRE);actions=evaluate(head,evalpre,base);eligible=[x for x in actions if x['target_present']];base_top=sum(x['base_correct'] for x in eligible);final_top=sum(x['final_correct'] for x in eligible);rescue=sum((not x['base_correct']) and x['final_correct'] for x in eligible);brk=sum(x['base_correct'] and (not x['final_correct']) for x in eligible);gates={'eligible':len(eligible)>=30,'strict_top1_gain':final_top>base_top,'rescue_gt_break':rescue>brk,'break_at_most_one':brk<=1};passed=all(gates.values());value={'schema_version':'rc_romav2_colnomic_full_negative_action_gate_v1_20260831','status':'ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_HEADROOM' if passed else 'ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_NO_GO','claim_level':'ADAPTIVE_IDENTITY_DISJOINT_INTERNAL_ACTION_GATE','contract_sha256':core.sha(ROOT/'plan/ROMAV2_COLNOMIC_FULL_NEGATIVE_ACTION_GATE_V1_20260831.md'),'sources':{'pair3':core.sha(PAIR3),'pair4':core.sha(PAIR4),'full_train':core.sha(FULLTRAIN),'eval_prejoin':[core.sha(EVALPRE/f'shard{s:02d}/result.json') for s in range(4)]},'parameter_count':7,'train_query_count':len(queries),'train_pair_count':len(x3)+len(x4),'head':{'weight':head.weight.detach().flatten().tolist(),'bias':float(head.bias.detach()),'total_loss':loss,'pair_loss':pair_loss,'full_negative_loss':full_loss},'evaluation_summary':{'selected_query_count':len(actions),'eligible_target_count':len(eligible),'target_absent_count':len(actions)-len(eligible),'base_top1':base_top,'final_top1':final_top,'rescue':rescue,'break':brk,'switch_count':sum(x.get('decision')=='SWITCH' for x in eligible)},'gates':gates,'actions':actions,'target_join_after_all_eval_prejoin_shards':True,'romav2_update_count':0,'opened_read_count':0,'sealed_read_count':0,'P0_authorized':False,'scientific_GO_or_NO_GO':None,'next_authorized_stage':'ROMAV2_COLNOMIC_FULL_NEGATIVE_EXTERNAL_CONFIRMATION' if passed else None,'logical_sha256':''};value['logical_sha256']=core.logical(value);core.atomic(OUT,value);print(json.dumps({'status':value['status'],'evaluation':value['evaluation_summary'],'gates':gates,'losses':[loss,pair_loss,full_loss]},sort_keys=True))
if __name__=='__main__':main()
