#!/usr/bin/env python3
"""Same UNIT_COST96 training; replace only FULL loss with all128 log-loss."""
import argparse,ast,hashlib,json,os,sys,subprocess,uuid
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')];sys.dont_write_bytecode=True
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_unit_cost96_cause_v1 as U
TRAIN=ROOT/'results/rc_convex_train_loss_cause_v1';UNIT=TRAIN/'unit_cost96'
OUT=ROOT/'results/rc_full_candidate_identity_loss_v1';AUTH=ROOT/'registry/rc_full_candidate_identity_loss_authority_v1_20260912.json'
STEPS=2000;MODEL='LISTWISE_UNIT1'

def checked(b):
 p=Path(b['path']);p=p if p.is_absolute() else ROOT/p;M.need(M.sha(p)==b['sha256'],'SOURCE_DRIFT:'+str(p));return p

def guard(stage):
 a=M.read(AUTH)
 for b in a['sources'].values():checked(b)
 M.need(a['sources']['program']==M.bind(__file__),'WORKER_SOURCE')
 if stage!='preflight':M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
 def audit(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
  M.need(not any(x in s for x in ('rc_opened_','curator_roles','/reports/','d1-mi','d1_mi','grozi','gisc_prerecall_universe','/target_join/')),'TRAIN_ONLY_SOURCE_BARRIER')
  root=ROOT/'results/rc_new_hyp593_oof5_v1';M.need(root/'fits' not in p.parents and root/'roles' not in p.parents,'NO593_FOLD_PARAMETERS_OR_LABELS')
  M.need(p.name not in ('result.json','result_validation.json','independent_validation.json'),'NO_EVAL_OUTCOMES')
  if stage in ('fit','replay','preflight'):M.need(UNIT not in p.parents and ROOT/'results/rc_fixed_panels_train269_group_risk_v1/fits' not in p.parents,'NO_BASELINE_PARAMETERS_IN_NEW_FIT')
 sys.addaudithook(audit)
 if stage!='preflight':
  pf=M.read(OUT/'preflight.json');M.need(pf['status']=='LISTWISE_UNIT1_OBJECTIVE_PREFLIGHT_PASS' and pf['authority']==M.bind(AUTH) and pf['input_seal']==M.bind(TRAIN/'input_seal.json'),'PREFLIGHT_BINDINGS')

def training_function(steps=STEPS):
 oldfn,uc=U.training_function(steps=steps)
 text=U.LEGACY.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='train_head');body=ast.get_source_segment(text,node)
 for change in uc['replacements']:M.need(body.count(change['before'])==1,'UNIT_COST_SOURCE_TRANSFORM');body=body.replace(change['before'],change['after'])
 M.need(hashlib.sha256(body.encode()).hexdigest()==uc['changed_function_sha256'],'EXACT_EXISTING_UNIT_COST_FUNCTION')
 before='''        if target == winner: query_losses.append(1 * F.softplus(values.max()))
            else:
                position = challengers.index(target); mask = torch.ones(len(challengers), dtype=torch.bool); mask[position] = False
                query_losses.append(F.softplus(-values[position]) + 1 * F.softplus(values[mask].max()))'''
 # Keep the original first-line indentation outside the literal replacement.
 before=before[8:]
 after='''all_values = torch.cat((values.new_zeros(1), values))
            target_index = 0 if target == winner else challengers.index(target) + 1
            query_losses.append(torch.logsumexp(all_values, dim=0) - all_values[target_index])'''
 M.need(body.count(before)==1,'ONE_FULL_LOSS_BLOCK');changed=body.replace(before,after);M.need(changed.replace(after,before)==body,'ONLY_FULL_LOSS_CHANGED')
 ns=dict(oldfn.__globals__);ns['STEPS']=steps;exec(compile('from __future__ import annotations\n'+changed,str(U.LEGACY)+'#LISTWISE_UNIT1','exec'),ns)
 info=dict(unit_cost_function_sha256=uc['changed_function_sha256'],changed_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),replacement_count=1,changed_component='FULL_LOSS_ONLY',PAIR_loss='cost1_BCE_mean',FULL_loss='mean_logsumexp_RAW0_plus_all_challengers_minus_target',PAIR_FULL_mixture=[1,1],temperature=1,new_features=0,parameter_count=7)
 return ns['train_head'],info

def synthetic():
 # Use deliberately non-contiguous physical positions and both RAW/non-RAW targets.
 pair={'records':[dict(real_native_features={'C_PAIRED':torch.tensor([x],dtype=torch.float64)},switch_label=y) for x,y in [([1.,-.5,.2,0.,1.,-1.],False),([-.3,1.,-.1,.8,.2,1.],True)]]}
 xs=[[[.4,.7,-.2,1.,0.,-.5],[-.2,.3,.8,-.5,1.,.4]],[[.2,-.4,.6,.8,-.1,1.],[-1.,.2,-.8,0.,.3,.7]]]
 full=[dict(real_native_features={'C_PAIRED':torch.tensor(x,dtype=torch.float64)},base_winner_position=5,target_position=t,challenger_positions=[2,9]) for x,t in zip(xs,[5,9])]
 fn,change=training_function(1);actual,losses,finite=fn(pair,full,'NATIVE7','C_PAIRED')
 torch.manual_seed(17);head=nn.Linear(6,1,dtype=torch.float64);head.weight.data.zero_();head.bias.data.zero_();opt=torch.optim.AdamW(head.parameters(),lr=.03,weight_decay=.001)
 px=torch.cat([r['real_native_features']['C_PAIRED'] for r in pair['records']]);py=torch.tensor([0.,1.],dtype=torch.float64);pl=F.binary_cross_entropy_with_logits(head(px).squeeze(1),py)
 fl=[]
 for row in full:
  z=head(row['real_native_features']['C_PAIRED']).squeeze(1);allz=torch.cat((z.new_zeros(1),z));target=0 if row['target_position']==5 else row['challenger_positions'].index(row['target_position'])+1
  fl.append(F.cross_entropy(allz[None],torch.tensor([target])))
 f=torch.stack(fl).mean();total=pl+f;total.backward();M.need(max(abs(a-float(b.detach())) for a,b in zip(losses,[total,pl,f]))<1e-12,'INDEPENDENT_CROSS_ENTROPY_LOSS')
 M.need(all(torch.allclose(a.grad,b.grad,atol=1e-12,rtol=0) for a,b in zip(actual.parameters(),head.parameters())),'INDEPENDENT_FULL_GRADIENT');opt.step();M.need(all(torch.allclose(a,b,atol=1e-12,rtol=0) for a,b in zip(actual.parameters(),head.parameters())),'INDEPENDENT_ADAMW_STEP')
 z=torch.zeros(127,dtype=torch.float64,requires_grad=True);allz=torch.cat((z.new_zeros(1),z));rawloss=torch.logsumexp(allz,0);g=torch.autograd.grad(rawloss,z)[0];M.need(abs(float(rawloss.detach())-__import__('math').log(128))<1e-12 and torch.allclose(g,torch.full((127,),1/128,dtype=torch.float64),atol=1e-15,rtol=0),'RAW_INCLUDED_AND_ALL127_GRADIENTS')
 a=torch.tensor([0.,2.,1.,0.],dtype=torch.float64);b=a.clone();b[3]=.5;M.need(float(torch.logsumexp(b,0)-b[1])>float(torch.logsumexp(a,0)-a[1]) and float(a[[0,2,3]].max())==float(b[[0,2,3]].max()),'NONMAX_WRONG_ALSO_AFFECTS_LISTWISE_LOSS')
 M.need(training_function()[0].__globals__['STEPS']==2000 and finite['all_finite'],'PRODUCTION_BUDGET_UNCHANGED')
 return dict(function_change=change,tolerance=1e-12,synthetic_only=True,natural_training_updates=0,checks=['RAW0_in_denominator','all127_challenger_gradients','target_position_not_physical_index','nonmaximum_wrong_contributes','independent_CE_loss_gradient_and_AdamW_step'])

def preflight():
 x=synthetic();seal=M.read(TRAIN/'input_seal.json');M.need(seal['status']=='TRAIN_INPUTS_SEALED_BEFORE_EVAL_CONES','EXISTING_TRAIN_SEAL');checked(seal['sources']['original_train_pack'])
 M.write(OUT/'preflight.json',dict(status='LISTWISE_UNIT1_OBJECTIVE_PREFLIGHT_PASS',authority=M.bind(AUTH),input_seal=M.bind(TRAIN/'input_seal.json'),function_change=x['function_change'],synthetic=x,original_head_training_updates=0,baseline_head_training_updates=0,natural_training_updates=0));print('LISTWISE_UNIT1_OBJECTIVE_PREFLIGHT_PASS',flush=True)

def compute():
 seal=M.read(TRAIN/'input_seal.json');pack=torch.load(checked(seal['sources']['original_train_pack']),weights_only=True,map_location='cpu');original=M.read(checked(seal['sources']['original_train_input_seal']));pair,full=pack['pair']['records'],pack['full']
 M.need(len(pair)==64 and len(full)==32 and len({r['query_id'] for r in pair+full})==96,'ORIGINAL_MIXED96')
 M.need([r['execution_ordinal'] for r in pair]==[r[2] for r in original['original_pair_order']] and [r['execution_ordinal'] for r in full]==original['original_training_order'],'ORIGINAL_TRAIN_ORDER')
 fn,change=training_function();M.need(change==M.read(OUT/'preflight.json')['function_change'],'FROZEN_OBJECTIVE');head,losses,finite=fn(pack['pair'],full,'NATIVE7','C_PAIRED');theta=torch.cat((head.weight.detach().flatten(),head.bias.detach()))
 return dict(status='LISTWISE_UNIT1_MATCHED_TRAINING_COMPLETE',theta_binary64=[float(v).hex() for v in theta],final_preupdate_losses=dict(zip(['total','PAIR','FULL'],losses)),function_change=change,finite_training=finite,authority=M.bind(AUTH),input_seal=M.bind(TRAIN/'input_seal.json'),original_training_pack=seal['sources']['original_train_pack'],train_count={'PAIR':64,'FULL':32,'total':96},original_head_training_updates=0,baseline_head_training_updates=0,new_head_training_updates=STEPS,heldout_label_reads=0,optimizer=dict(name='AdamW',lr=.03,weight_decay=.001,steps=STEPS,seed=17,initialization='zero',dtype='float64'))

def fit(replay,nonce):
 if not replay:M.need(not (OUT/'fit.json').exists(),'IMMUTABLE_FIT')
 value=compute()
 if replay:
  M.need(nonce and nonce==os.environ.get('RC_LISTWISE_NONCE'),'FRESH_REPLAY');old=M.read(OUT/'fit.json');M.need(value==old,'NEW_PARAMETERS_AND_INPUTS_BIT_REPLAY')
  M.write(OUT/'fit_validation.json',dict(status='LISTWISE_UNIT1_FRESH_PARAMETER_REPLAY_PASS',fit=M.bind(OUT/'fit.json'),authority=M.bind(AUTH),input_seal=M.bind(TRAIN/'input_seal.json'),fresh_nonce=nonce,original_head_training_updates=0,baseline_head_training_updates=0,new_head_training_updates=STEPS,heldout_label_reads=0))
 else:
  M.write(OUT/'fit.json',value);nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,'replay','--nonce',nonce],env=dict(os.environ,RC_LISTWISE_NONCE=nonce),check=True);print('LISTWISE_UNIT1_FRESH_PARAMETER_REPLAY_PASS',flush=True)

def predict():
 import run_rc_new_hyp593_oof5_v1 as P
 val=M.read(OUT/'fit_validation.json');M.need(val['status']=='LISTWISE_UNIT1_FRESH_PARAMETER_REPLAY_PASS' and val['fit']==M.bind(OUT/'fit.json') and val['authority']==M.bind(AUTH),'VALIDATED_NEW_FIT');new=M.read(OUT/'fit.json');theta=torch.tensor([float.fromhex(x) for x in new['theta_binary64']],dtype=torch.float64)
 seal=M.read(UNIT/'prediction_seal.json');fv=M.read(UNIT/'fit_validation.json');M.need(seal['status']=='UNIT_COST96_TRAIN_ONLY_ALL160_PREDICTIONS_SEALED' and fv['status']=='UNIT_COST96_FRESH_PARAMETER_REPLAY_PASS' and seal['unit_cost_fit_validation']==M.bind(UNIT/'fit_validation.json'),'SEALED_BASELINE_PREDICTIONS')
 old=torch.load(checked(seal['payload']),weights_only=True,map_location='cpu');oldrows={r['query_id']:r for r in old['predictions']};rows,sources=P.features();test=[r for r in rows if r['query_id'] in oldrows];M.need(len(test)==len(oldrows)==160,'FULL160')
 preds=P.predict({MODEL:dict(theta=theta)},test)
 for row in preds:
  prev=oldrows[row['query_id']];M.need(all(row[k]==prev[k] for k in ('candidate_physical_rows','raw_ranked_physical_rows','winner','challenger_positions')),'FIXED_INPUT_AXES')
  for name in ('ORIGINAL7','TRAIN_UNIT_COST7'):row['models'][name]=prev['models'][name]
 common=dict(fit=M.bind(OUT/'fit.json'),fit_validation=M.bind(OUT/'fit_validation.json'),authority=M.bind(AUTH),input_seal=M.bind(TRAIN/'input_seal.json'),previous_payload=seal['payload'],heldout_label_reads=0,original_head_training_updates=0,baseline_head_training_updates=0)
 value=dict(parameters={MODEL:dict(theta=theta),**old['parameters']},predictions=preds,feature_sources=sources,**common)
 p=OUT/'predictions.pt'
 with p.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
 M.write(OUT/'prediction_seal.json',dict(status='LISTWISE_UNIT1_ALL160_PREDICTIONS_SEALED',payload=M.bind(p),**common));print('LISTWISE_UNIT1_ALL160_PREDICTIONS_SEALED',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','fit','replay','predict']);ap.add_argument('--nonce');args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);guard(args.stage)
 if args.stage=='preflight':preflight()
 elif args.stage=='predict':predict()
 else:fit(args.stage=='replay',args.nonce)
