#!/usr/bin/env python3
"""Fixed product-score and response-feature factorial; original retrieval training."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];NAME='rc_product_response_factorial_v1';OUT=ROOT/'results'/NAME
AUTH=ROOT/'registry/rc_product_response_factorial_authority_v1_20260909.json'
PLAN=ROOT/'plan/RC_PRODUCT_RESPONSE_FACTORIAL_V1_20260909.md'
LAUNCH=ROOT/'slurm/rc_product_response_factorial_v1_dev_cpuonly_59m.sbatch'
SOURCE=ROOT/'programs/run_rc_absolute_evidence_scale_calibration_v1.py'
PARENT_AUTH=ROOT/'registry/rc_absolute_evidence_scale_calibration_authority_v1_20260909.json'
COLUMNS={'JOINT4':(0,2,3),'PRODUCT5':(0,1,2,3),'RESPONSE6':(0,2,3,4,5),'ORIGINAL7':(0,1,2,3,4,5)}
MODES=('REAL','CBIND');ARM='C_PAIRED';U=None;H=None
PREVIOUS=ROOT/'results/rc_retrained_evidence_sufficiency_v1'
PREVIOUS_PINS={'parameters.json':'8d1781e0abaec6bcd2cdacb6f0637140398764ca1f533452f9bb1cf956aee20b',
 'result.json':'20102901242c734c2b5c0404e0e5571999ab1dc25ceefa5cee2f37f884a12601',
 'input_closure.json':'0ed30bda7a237d350ee5b636792bd66b00c3e7690e869322a806781051c83aad',
 'independent_validation.json':'ebac4a7d704e121d972f8999dbcbe767e588681e3f9a7352974f78d9e809e195'}
CONTINUATION=ROOT/'registry/rc_retrieval_only_post_submission_continuation_addendum_v1_20260909.json'
COMPARISONS=(('PRODUCT5','JOINT4'),('ORIGINAL7','RESPONSE6'),('RESPONSE6','JOINT4'),('ORIGINAL7','PRODUCT5'),
 ('PRODUCT5','ORIGINAL7'),('RESPONSE6','ORIGINAL7'),('ORIGINAL7','JOINT4'))

def need(x,m):
 if not bool(x):raise RuntimeError(m)

def setup():
 global U,H
 original=json.loads(PARENT_AUTH.read_text());need(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original['sources']['program']['sha256'],'SOURCE_WORKER_PIN')
 spec=importlib.util.spec_from_file_location('fixed_inputs_for_retrained_sufficiency',SOURCE)
 U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
 U.helper();H=U.H;U.deadline();need(U.sources()==original['sources'],'PARENT_INPUT_AUTHORITY')

def sources():
 for name,digest in PREVIOUS_PINS.items():need(H.sha(PREVIOUS/name)==digest,'PREVIOUS_PIN:'+name)
 return {'program':H.binding(__file__),'plan':H.binding(PLAN),'launcher':H.binding(LAUNCH),'parent_worker':H.binding(SOURCE),
         'parent_authority':H.binding(PARENT_AUTH),'continuation':H.binding(U.EXT),'split_qualification':H.binding(U.SPLIT),
         'previous_trial':{name:H.binding(PREVIOUS/name) for name in PREVIOUS_PINS},'post_submission_continuation':H.binding(CONTINUATION),
         'source_input_closure':H.binding(ROOT/'results/rc_absolute_evidence_scale_calibration_v1/input_closure.json')}

def keys(name):return ('real_native_features','cbind_native_features') if name=='ORIGINAL7' else ('subset_'+name+'_real','subset_'+name+'_cbind')
def family(name):return 'NATIVE7' if name=='ORIGINAL7' else name

def prepare():
 import torch
 frozen,pure,pair,train,evals,entries,labels,parent=U.prepare()
 need(parent==H.read(ROOT/'results/rc_absolute_evidence_scale_calibration_v1/input_closure.json'),'ORIGINAL_INPUT_CLOSURE')
 H.BARRIER.outcomes.update(str((PREVIOUS/k).resolve()) for k in ('result.json','independent_validation.json'))
 previous=H.read(PREVIOUS/'input_closure.json');old_seals={(r['kind'],r['query_id'],r['execution_ordinal']):r for r in previous['subset_feature_seals']}
 need(len(old_seals)==128,'PREVIOUS_FULL_PAIR_SEALS')
 seals=[]
 def subset(row,kind):
  out=dict(row);record={'kind':kind,'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'features':{}}
  for mode in (('REAL',) if kind=='PAIR' else MODES):
   original=row['real_native_features' if mode=='REAL' else 'cbind_native_features'][ARM];record['features'][mode]={}
   for name,cols in COLUMNS.items():
    x=original[:,list(cols)].contiguous();need(x.shape==(len(row['challenger_positions']),len(cols)) and bool(torch.isfinite(x).all()),'SUBSET_SHAPE_FINITE')
    # Independent literal element indexing preserves every selected binary64 value.
    literal=torch.tensor([[float(original[i,j]) for j in cols] for i in range(len(original))],dtype=torch.float64)
    need(H.tensor_sha(x)==H.tensor_sha(literal),'SUBSET_LITERAL_BITS')
    for j,c in enumerate(cols):need([H.hx(v) for v in x[:,j]]==[H.hx(v) for v in original[:,c]],'SUBSET_COLUMN_BITS')
    key=keys(name)[0 if mode=='REAL' else 1]
    if name=='ORIGINAL7':need(H.tensor_sha(x)==H.tensor_sha(original),'ORIGINAL7_UNCHANGED')
    else:out[key]={ARM:x}
    record['features'][mode][name]=H.tensor_sha(x)
    if name in ('JOINT4','ORIGINAL7'):need(H.tensor_sha(x)==old_seals[(kind,row['query_id'],row['execution_ordinal'])]['features'][mode][name],'OLD_JOINT4_ORIGINAL7_FEATURE_SEAL')
  seals.append(record);return out
 pair={**pair,'records':[subset(r,'PAIR') for r in pair['records']]}
 train=[subset(r,'FULL') for r in train];evals=[subset(r,'FULL') for r in evals]
 dispatch=pure['train_head'].__globals__['FAMILIES'];need(dispatch is pure['actions'].__globals__['FAMILIES'],'SHARED_FROZEN_FEATURE_DISPATCH')
 ledger={}
 for name,cols in COLUMNS.items():
  if name!='ORIGINAL7':dispatch[name]=(*keys(name),tuple(frozen.FEATURE_NAMES[c] for c in cols))
  x=torch.cat([r[keys(name)[0]][ARM] for r in pair['records']+train]);need(x.shape==(4128,len(cols)),'FIXED_TRAIN_DESIGN')
  bias=torch.ones((len(x),1),dtype=torch.float64)
  ledger[name]={'columns':list(cols),'names':[frozen.FEATURE_NAMES[c] for c in cols],'feature_sha256':H.tensor_sha(x),
    'design_rank_with_bias':int(torch.linalg.matrix_rank(torch.cat([x,bias],dim=1))),
    'exact_zero_columns':torch.nonzero(x.eq(0).all(0)).flatten().tolist(),'parameter_count':len(cols)+1}
 closure={'parent_input_closure':parent,'subset_feature_seals':seals,'training_design':ledger,'columns':{k:list(v) for k,v in COLUMNS.items()},
          'runtime_EVAL_target_reads':0,'new_encoder_or_RoMa_forwards':0,'original_scalar_feature_values_preserved':True}
 return pure,pair,train,evals,entries,labels,closure

def fit(pure,pair,train):
 params={};expected=H.read(PREVIOUS/'parameters.json');native=H.read(U.NATIVE);started=time.monotonic()
 need(expected['ORIGINAL7']['weight_binary64']==native['weight_binary64'] and expected['ORIGINAL7']['bias_binary64']==native['bias_binary64'],'PREVIOUS_ORIGINAL_NATIVE_PARAMETER_BINDING')
 source=(ROOT/H.PINS['old_runner'][0]).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='train_head')
 digest=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
 for name,cols in COLUMNS.items():
  head,loss,finite=pure['train_head'](pair,train,family(name),ARM);w=head.weight.detach().flatten();b=float(head.bias.detach())
  p={'weight_binary64':[H.hx(v) for v in w],'bias_binary64':H.hx(b),'parameter_sha256':pure['parameter_sha'](w,b),
     'parameter_count':len(cols)+1,'original_feature_columns':list(cols),'loss_total_last_recorded':loss[0],
     'PAIR_loss_last_recorded':loss[1],'FULL_loss_last_recorded':loss[2],'finite_training':finite,
     'unchanged_training_function_sha256':digest}
  if name in ('JOINT4','ORIGINAL7'):need(p['weight_binary64']==expected[name]['weight_binary64'] and p['bias_binary64']==expected[name]['bias_binary64'],'OLD_JOINT4_ORIGINAL7_PARAMETER_REGRESSION')
  params[name]=p;print(json.dumps({'event':'PRODUCT_RESPONSE_HEAD_FIT','model':name,'seconds':time.monotonic()-started,'parameter_sha256':p['parameter_sha256']}),flush=True)
 return params

def tensors(p):
 import torch
 return torch.tensor([float.fromhex(v) for v in p['weight_binary64']],dtype=torch.float64),float.fromhex(p['bias_binary64'])

def predict(row,p,name,mode):
 w,b=tensors(p);values=row[keys(name)[0 if mode=='REAL' else 1]][ARM]@w+b
 cs=row['challenger_positions'];axis=row['candidate_physical_rows'];i=max(range(len(cs)),key=lambda i:(float(values[i]),-axis[cs[i]]));switch=float(values[i])>0
 return {'final_position':int(cs[i]) if switch else int(row['base_winner_position']),'decision':'SWITCH' if switch else 'HOLD',
         'all127_logits_binary64':[H.hx(x) for x in values]}

def paired(new,base,rows):
 value=U.paired(new,base);mapping={r['query_id']:r['supergroup'] for r in rows};groups={}
 for n,b in zip(new,base):
  g=groups.setdefault(mapping[n['query_id']],{'queries':0,'new_correct':0,'base_correct':0})
  g['queries']+=1;g['new_correct']+=int(n['final_correct']);g['base_correct']+=int(b['final_correct'])
 for g in groups.values():g['net']=g['new_correct']-g['base_correct']
 value['supergroups']=groups;value['group_balanced_accuracy_difference']=sum(g['net']/g['queries'] for g in groups.values())/len(groups)
 value['groups_improved']=sum(g['net']>0 for g in groups.values());value['groups_harmed']=sum(g['net']<0 for g in groups.values())
 return value

def summarize(params,predictions,pure,pair,train,evals,entries,labels,closure):
 seal=H.read(OUT/'eval_prejoin_seal.json')
 need(seal['parameters_sha256']==H.sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==H.sha(OUT/'eval_prejoin.json') and H.BARRIER.blocked==0,'PREJOIN_SEAL_REQUIRED')
 H.BARRIER.release();joined=H.role_join(evals,entries,labels)
 need(all(not ({r[k] for r in train}&{r[k] for r in joined}) for k in ('query_id','target_identity','supergroup')),'TRAIN_EVAL_DISJOINT')
 acts={};metrics={};comparisons={};pred_by_id={r['query_id']:r for r in predictions}
 for role,rows in [('TRAIN',train),('EVAL',joined)]:
  need(len(rows)==32,'WHOLE32');acts[role]={};metrics[role]={}
  for name in COLUMNS:
   w,b=tensors(params[name]);acts[role][name]={mode:pure['actions'](w,b,rows,family(name),ARM,control=mode=='CBIND') for mode in MODES}
   metrics[role][name]={mode:pure['summary'](a) for mode,a in acts[role][name].items()}
   for row in rows:
    for mode in MODES:need(pred_by_id[row['query_id']]['predictions'][name][mode]==predict(row,params[name],name,mode),'PREJOIN_LOGITS_REPLAY')
  comparisons[role]={new+'_vs_'+base:paired(acts[role][new]['REAL'],acts[role][base]['REAL'],rows) for new,base in COMPARISONS}
 old=H.read(ROOT/H.PINS['old_result'][0]);previous=H.read(PREVIOUS/'result.json');pv=H.read(PREVIOUS/'independent_validation.json')
 need(pv['result_sha256']==H.sha(PREVIOUS/'result.json') and all(pv['checks'].values()),'PREVIOUS_INDEPENDENT_VALIDATION')
 for mode,key in [('REAL','actions'),('CBIND','cbind_actions')]:
  need(H.encode(acts['EVAL']['ORIGINAL7'][mode])==H.encode(old['evaluations']['NATIVE7'][ARM][key]),'ORIGINAL_NATIVE7_ACTION_REGRESSION')
  need(H.encode(acts['EVAL']['JOINT4'][mode])==H.encode(previous['actions']['EVAL']['JOINT4'][mode]),'OLD_JOINT4_ACTION_REGRESSION')
 pair_metrics={}
 for name in COLUMNS:
  w,b=tensors(params[name]);correct=sum(int((float((r[keys(name)[0]][ARM]@w+b)[0])>0)==bool(r['switch_label'])) for r in pair['records'])
  pair_metrics[name]={'query_count':64,'correct_pair_decisions':correct,'scope':'labelled optimization pool, not held-out performance'}
 c=comparisons['EVAL'];candidates={};improvements={}
 for name in ('PRODUCT5','RESPONSE6'):
  base=c[name+'_vs_JOINT4'];full=c[name+'_vs_ORIGINAL7']
  candidates[name]=full['break']==0 and full['net']>=0 and base['net']>0 and base['group_balanced_accuracy_difference']>0
  improvements[name]=full['break']==0 and full['net']>0
 status='PRODUCT_RESPONSE_FACTORIAL_INTERNAL_SIMPLIFICATION_SUPPORTED' if any(candidates.values()) else 'PRODUCT_RESPONSE_FACTORIAL_INTERNAL_SIMPLIFICATION_NOT_ESTABLISHED'
 return {'status':status,'theory_name':'new HYP','authority_sha256':H.sha(AUTH),'input_closure_sha256':H.sha(OUT/'input_closure.json'),
         'parameters_sha256':H.sha(OUT/'parameters.json'),'eval_prejoin_seal_sha256':H.sha(OUT/'eval_prejoin_seal.json'),
         'metrics':metrics,'actions':acts,'comparisons':comparisons,'PAIR_training_pool_description':pair_metrics,'training_design':closure['training_design'],
         'C_BIND_rescue_retention':{name:pure['retention'](acts['EVAL'][name]['REAL'],acts['EVAL'][name]['CBIND']) for name in COLUMNS},
         'primary_checks':candidates,'internal_mechanism_simplification_candidates':candidates,'internal_model_improvement_candidates':improvements,
         'old_JOINT4_and_ORIGINAL7_parameters_actions_exact':True,
         'factorial_comparisons':['PRODUCT5_vs_JOINT4','ORIGINAL7_vs_RESPONSE6','RESPONSE6_vs_JOINT4','ORIGINAL7_vs_PRODUCT5'],
         'candidate_source':'original RAW full-gallery C128; all127-challenger zero-threshold HOLD/SWITCH',
         'TRAIN_query_count':32,'EVAL_query_count':32,'PAIR_query_count':64,'candidate_count':128,
         'EVAL_supergroup_count':len({r['supergroup'] for r in joined}),'new_encoder_or_RoMa_forwards':0,
         'task_supervision':'retrieval identity and pair labels only, no task spatial annotation',
         'evidence_level':'previously opened internal EVAL32, not untouched population confirmation','HYP_GO_claimed':False,'deployment_changed':False,
         'limits':['Four nested subsets retain original numerical features and training protocol; their parameter counts explicitly differ.',
                  'S is the original joint score contrast, not a newly recomputed product or independently added raw information.',
                  'The ideal positive epsilon-free dS identity supplies a nonlinear basis interpretation, not statistical interaction proof.',
                  'Both factor contributions at both backgrounds are reported; no post-hoc selection of a favorable comparison.',
                  'A simplified internal model need not exceed28 to explain old results; external reliability is separately unverified.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','freeze','run','validate'],required=True);phase=p.parse_args().phase
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);setup();src=sources();pre=OUT.parent/(NAME+'_preflight')/'result.json'
 if phase=='freeze':
  need(not AUTH.exists() and not OUT.exists(),'APPEND_ONLY_AUTHORITY');v=H.read(pre)
  need(v['sources']==src and v['status']=='PRODUCT_RESPONSE_FACTORIAL_PREFLIGHT_PASS','PREFLIGHT_REQUIRED')
  H.atomic(AUTH,{'status':'PRODUCT_RESPONSE_FACTORIAL_AUTHORIZED','sources':src,'columns':{k:list(v) for k,v in COLUMNS.items()},
                'factorial_comparisons':[list(x) for x in COMPARISONS[:4]],'secondary_comparisons':[list(x) for x in COMPARISONS],
                'steps':2000,'seed':17,'loss':'original PAIR_SIGN','cutoff_UTC':H.read(U.EXT)['cutoff_UTC'],'preflight':H.binding(pre)})
  print(json.dumps({'authority_sha256':H.sha(AUTH)}),flush=True);return
 pure,pair,train,evals,entries,labels,closure=prepare()
 if phase=='preflight':
  H.atomic(pre,{'status':'PRODUCT_RESPONSE_FACTORIAL_PREFLIGHT_PASS','sources':src,'input_closure':closure,
                'training_updates':0,'runtime_EVAL_target_reads':0,'checks':{'original_C_REAL_CBIND_features_rebuilt':True,
                'all_subset_features_independently_literal_rebuilt':True,'no_new_scalar_or_map_values':True,'four_fixed_input_dimensions':True}})
  print(json.dumps({'status':'PRODUCT_RESPONSE_FACTORIAL_PREFLIGHT_PASS','training_design':closure['training_design']}),flush=True);return
 need(H.read(AUTH)['sources']==src,'AUTHORITY_DRIFT');validate=phase=='validate';need(OUT.exists() if validate else not OUT.exists(),'APPEND_ONLY_OUTPUT_STATE')
 if validate:need(H.read(OUT/'input_closure.json')==closure,'INDEPENDENT_INPUT_REBUILD')
 else:H.atomic(OUT/'input_closure.json',closure)
 params=fit(pure,pair,train)
 if validate:need(H.read(OUT/'parameters.json')==params,'INDEPENDENT_PARAMETER_RETRAIN')
 else:H.atomic(OUT/'parameters.json',params)
 rows=sorted(train+evals,key=lambda r:r['execution_ordinal'])
 preds=[{'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],
         'predictions':{name:{mode:predict(r,params[name],name,mode) for mode in MODES} for name in COLUMNS}} for r in rows]
 if validate:need(H.read(OUT/'eval_prejoin.json')==preds,'INDEPENDENT_ALL_LOGITS_REPLAY')
 else:
  H.atomic(OUT/'eval_prejoin.json',preds);H.atomic(OUT/'eval_prejoin_seal.json',{'parameters_sha256':H.sha(OUT/'parameters.json'),
    'eval_prejoin_sha256':H.sha(OUT/'eval_prejoin.json'),'EVAL_count':32,'TRAIN_count':32,'candidate_count':128,'runtime_EVAL_target_reads':0})
 value=summarize(params,preds,pure,pair,train,evals,entries,labels,closure)
 if validate:
  need(H.read(OUT/'result.json')==value,'INDEPENDENT_RESULT_REPLAY')
  H.atomic(OUT/'independent_validation.json',{'status':'PRODUCT_RESPONSE_FACTORIAL_INDEPENDENT_REEXECUTION_PASS','result_sha256':H.sha(OUT/'result.json'),
   'checks':{'all_original_and_subset_features_rebuilt':True,'all_four_heads_retrained_exactly':True,'old_JOINT4_and_ORIGINAL7_parameters_actions_exact':True,
             'all_full_C128_predictions_and_comparisons_replayed':True},'scope':'fresh-process frozen optimizer replay plus literal subset feature verification'})
 else:H.atomic(OUT/'result.json',value)
 print(json.dumps({'status':value['status'],'validation':validate,'EVAL':value['metrics']['EVAL'],'primary_checks':value['primary_checks']}),flush=True)
if __name__=='__main__':main()
