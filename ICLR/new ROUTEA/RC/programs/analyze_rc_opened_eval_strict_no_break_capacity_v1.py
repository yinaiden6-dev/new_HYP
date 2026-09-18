#!/usr/bin/env python3
"""Four label-aware EVAL feasibility certificates; not learned/deployable models."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,math,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_opened_eval_strict_no_break_capacity_v1';PLAN=ROOT/'plan/RC_OPENED_EVAL_STRICT_NO_BREAK_CAPACITY_DIAGNOSTIC_V1_20260909.md'
VALIDATOR=ROOT/'programs/validate_rc_opened_eval_strict_no_break_capacity_v1.py'
LABEL='NON_DEPLOYABLE_LABEL_AWARE_EVAL_DIAGNOSTIC';ERRORS=('DIFFICULT-0050','OUTCOME-0213','OUTCOME-0373','OUTCOME-0676')
BASE=ROOT/'results/rc_product_response_factorial_v1';FULL=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1'
PINS={'certificate_helper':('programs/analyze_rc_train_action_constraint_feasibility_v1.py','6b8c6fae75c43a30e3b32fc2568db8641ccb54cef167426695c75ddb473d548f'),
 'old_certificate_validator':('programs/validate_rc_train_action_constraint_feasibility_v1.py','0d3422c18a0972600b36475d919b68065138a72fe83d673621957f298b4d1ea1'),
 'feature_gate':('src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py','96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7'),
 'full_validation':('results/routea_matched_three_arm_fullnegative_features_v1/validation.json','4f6c514b7e7d619d32cf312e06ee9a39494eb7ef4113aaebc45e10b38ce1c827'),
 'baseline_result':('results/rc_product_response_factorial_v1/result.json','e9dae1e5837e72d3f38ba3e4473d0ff555b6ee90fadfbe3d3dbf64376759c0bf'),
 'baseline_validation':('results/rc_product_response_factorial_v1/independent_validation.json','f91c01caac1824ceb8cde358a9c22f95b94fe0d846965b35815c66cc778ed3bb'),
 'baseline_parameters':('results/rc_product_response_factorial_v1/parameters.json','019cfeac14575b97371f66f81e5ad6ec55ec321af502a0b9110e4cc207c79b43'),
 'baseline_feature_closure':('results/rc_product_response_factorial_v1/input_closure.json','ee1a9fc70aff90c361e8b8b336400b90a07cad000267d99eb1a6fd32f316279a'),
 'continuation':('registry/rc_retrieval_only_research_extension_authority_v1_20260909.json','b9bdcd4b1c847fc22755a2e5565f30a4e77f3bda4f92e4273769601caf10ceda')}
def need(x,m):
 if not bool(x):raise RuntimeError(m)
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def binding(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def atomic(p,d):need(not p.exists(),'APPEND_ONLY:'+str(p));p.write_bytes(encode(d)+b'\n')
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def manual_feature(raw,e,c,w):
 import torch
 mean=sum(float(x) for x in raw)/len(raw);std=(sum((float(x)-mean)**2 for x in raw)/len(raw))**.5
 def sym(a,b):return (float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12)
 ec,ew=e[c],e[w];s=float(ec['real_score']);t=float(ew['real_score']);m=float(ec['visibility_mass']);n=float(ew['visibility_mass'])
 return torch.tensor([(float(raw[c])-float(raw[w]))/max(std,1e-12),sym(s,t),sym(m,n),sym(s/max(m,1e-12),t/max(n,1e-12)),
 sym(s-float(ec['query_control_score']),t-float(ew['query_control_score'])),sym(s-float(ec['reference_control_score']),t-float(ew['reference_control_score']))],dtype=torch.float64)

def inputs(independent=False):
 import torch
 sources={k:{'path':p,'sha256':h} for k,(p,h) in PINS.items()}
 for x in sources.values():need(sha(ROOT/x['path'])==x['sha256'],'SOURCE_PIN:'+x['path'])
 sources.update(program=binding(Path(__file__).resolve()),validator=binding(VALIDATOR),plan=binding(PLAN))
 ext=read(ROOT/PINS['continuation'][0]);need(datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(ext['cutoff_UTC']),'RESEARCH_CUTOFF')
 result=read(BASE/'result.json');v=read(BASE/'independent_validation.json');need(v['result_sha256']==PINS['baseline_result'][1] and all(v['checks'].values()) and v['status']=='PRODUCT_RESPONSE_FACTORIAL_INDEPENDENT_REEXECUTION_PASS','BASELINE_VALIDATION')
 params=read(BASE/'parameters.json')['ORIGINAL7'];need(params['parameter_sha256']=='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263','ORIGINAL_NATIVE7_PARAMETERS')
 w=torch.tensor([float.fromhex(x) for x in params['weight_binary64']],dtype=torch.float64);b=float.fromhex(params['bias_binary64']);need(hashlib.sha256(encode({'weight':w.tolist(),'bias':b})).hexdigest()==params['parameter_sha256'],'PARAMETER_SHA')
 old=result['actions']['EVAL']['ORIGINAL7']['REAL'];need(len(old)==32 and sum(a['final_correct'] for a in old)==28 and sum(a['base_correct'] for a in old)==25,'OPENED_NATIVE7_32_28_25');byid={a['query_id']:a for a in old}
 need(tuple(a['query_id'] for a in old if not a['final_correct'])==ERRORS,'FOUR_PREDECLARED_ERRORS')
 closure=read(BASE/'input_closure.json');seals={(x['kind'],x['query_id'],x['execution_ordinal']):x for x in closure['subset_feature_seals']}
 preds=read(BASE/'eval_prejoin.json');need(sha(BASE/'eval_prejoin.json')==read(BASE/'eval_prejoin_seal.json')['eval_prejoin_sha256'],'BASELINE_PREJOIN_SEAL');sources['baseline_predictions']=binding(BASE/'eval_prejoin.json');sources['baseline_prediction_seal']=binding(BASE/'eval_prejoin_seal.json');pb={x['query_id']:x for x in preds}
 gate=load(ROOT/PINS['feature_gate'][0],'capacity_frozen_feature_definition');fullv=read(FULL/'validation.json');need(all(fullv['checks'].values()),'FULL_SOURCE_VALIDATION');rows=[]
 for ent in fullv['shards']:
  path=FULL/f"shard{int(ent['shard']):02d}/payload.pt";need(sha(path)==ent['payload_sha256'],'FULL_SHARD_HASH');sources['full_shard_'+str(ent['shard'])]=binding(path)
  doc=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
  rows.extend(r for r in doc['records'] if r['data_split_role']=='EVAL')
 rows.sort(key=lambda r:r['execution_ordinal']);need(len(rows)==32 and [r['query_id'] for r in rows]==[a['query_id'] for a in old],'CURRENT_EVAL_AXIS')
 endpoints=[];joined=[]
 for r in rows:
  a=byid[r['query_id']];axis=[int(x) for x in r['candidate_physical_rows']];cs=[int(x) for x in r['challenger_positions']];raw=r['base_scores'].tolist();winner=int(r['base_winner_position']);t=int(a['target_position'])
  need(len(axis)==len(set(axis))==128 and cs==[i for i in range(128) if i!=winner],'FULL127_CHALLENGER_ORDER');need(winner==max(range(128),key=lambda i:(raw[i],-axis[i])) and winner==a['base_winner'] and axis[winner]==a['base_winner_physical_row'] and axis[t]==a['target_physical_row'],'TARGET_WINNER_AXIS')
  fn=manual_feature if independent else gate.candidate_feature;x=torch.stack([fn(raw,r['evidence']['C_PAIRED'],c,winner) for c in cs]);saved=r['real_native_features']['C_PAIRED'];need(x.dtype==torch.float64 and x.shape==(127,6) and bool(torch.isfinite(x).all()) and x.contiguous().numpy().tobytes()==saved.contiguous().numpy().tobytes(),'RAW_SCALAR_NATIVE6_EXACT')
  # Source tensor seals hash the exact tensor bytes and schema in the old helper.
  sx=seals[('FULL',r['query_id'],r['execution_ordinal'])]['features']['REAL']['ORIGINAL7']
  importstruct={'shape':list(x.shape),'dtype':str(x.dtype),'bytes':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
  # Full source byte equality above and old parameter/logit replay below are
  # independent of how the historical tensor seal encodes its metadata.
  z=x@w+b;expected=pb[r['query_id']]['predictions']['ORIGINAL7']['REAL'];need([float(v).hex() for v in z]==expected['all127_logits_binary64'],'ORIGINAL_ALL127_LOGIT_BITS')
  i=max(range(127),key=lambda i:(float(z[i]),-axis[cs[i]]));final=cs[i] if float(z[i])>0 else winner
  need(final==a['final_position'] and (final==t)==a['final_correct'] and float(z[i]).hex()==float(a['switch_logit']).hex() and cs[i]==a['proposed_challenger'],'ORIGINAL_ACTION_REPLAY')
  joined.append({**r,'real_native_features':{'C_PAIRED':x},'target_position':t})
  endpoints.append({'query_id':r['query_id'],'execution_ordinal':r['execution_ordinal'],'candidate_physical_rows':axis,'base_winner_position':winner,'target_position':t,'challenger_positions':cs,
  'original_final_correct':bool(a['final_correct']),'native6_binary64':[[float(v).hex() for v in f] for f in x],'source_feature_tensor_seal':sx})
 correct=[a['query_id'] for a in old if a['final_correct']]
 return sources,joined,endpoints,correct,[Fraction.from_float(float(x)) for x in w]+[Fraction.from_float(b)]

def main():
 import torch,numpy as np,scipy
 torch.set_num_threads(8);torch.set_num_interop_threads(1);need(not OUT.exists(),'APPEND_ONLY_OUTPUT')
 sources,rows,endpoints,correct,theta=inputs();helper=load(ROOT/PINS['certificate_helper'][0],'original_train_feasibility_functions');preserved=[r for r in rows if r['query_id'] in correct]
 _,terms,_=helper.build({'records':[]},preserved,'real_native_features',False,False);margins=[sum((x*y for x,y in zip(helper.row_exact(t),theta)),Fraction(0)) for t in terms];need(len(margins)==28*127 and min(margins)>0,'OLD28_EXACT_STRICT_FEASIBILITY')
 manifest={'sources':sources,'diagnostic_label':LABEL,'error_query_order':list(ERRORS),'preserved_correct_query_ids':correct,'parameter_count':7,'constraints_per_system':3683,'query_count_per_system':29,
 'input_query_count':32,'solver_use':'floating LP search followed by exact rational certificate','label_aware_EVAL_diagnostic':True,'new_trained_model':False,'baseline_28_exact_min_margin':str(min(margins)),
 'input_scope':'Only32 EVAL rows enter constraints; frozen mixedFULL source shards are read without TRAIN/PAIR role joins or constraints.'}
 OUT.mkdir();atomic(OUT/'execution_manifest.json',manifest);atomic(OUT/'source_feature_endpoints.json',{'diagnostic_label':LABEL,'rows':endpoints});systems={}
 for err in ERRORS:
  required=[r for r in rows if r['query_id'] in correct or r['query_id']==err];need(len(required)==29,'REQUIRED29');a,terms,meta=helper.build({'records':[]},required,'real_native_features',False,False);need(a.shape==(3683,7),'SYSTEM_SHAPE')
  certificate=helper.certify(a,terms,meta);certificate.update(diagnostic_label=LABEL,required_query_ids=[r['query_id'] for r in required],unconstrained_original_error_query_ids=[q for q in ERRORS if q!=err],
   constraint_order_sha256=hashlib.sha256(encode(meta)).hexdigest(),witness_or_dual_is_not_model=True)
  systems[err]=certificate;print(json.dumps({'system':err,'status':certificate['status'],'diagnostic_label':LABEL}),flush=True)
 for x in sources.values():need(sha(ROOT/x['path'])==x['sha256'],'SOURCE_CHANGED_DURING_DIAGNOSTIC')
 output={'status':'OPENED_EVAL_STRICT_NO_BREAK_CAPACITY_DIAGNOSTIC_COMPLETE','diagnostic_label':LABEL,'theory_name':'new HYP','head_class':'Unrestricted real linear readout of original NATIVE7 six frozen features plus intercept',
 'dataset':'previously opened matched EVAL32','candidate_source':'original RAW full-gallery natural C128 with all127 challenger action constraints','baseline':'frozen ORIGINAL7/NATIVE7 C_PAIRED 28/32, RAW25/32',
 'systems':systems,'sources':sources,'execution_manifest':binding(OUT/'execution_manifest.json'),'source_feature_endpoints':binding(OUT/'source_feature_endpoints.json'),
 'baseline_28_exact_min_margin':str(min(margins)),'all32_original_actions_and_all4064_logits_replayed_exactly':True,'all_four_systems_reported':True,
 'solver':{'scipy':scipy.__version__,'numpy':np.__version__,'method':'highs','floating_search_only':True},'EVAL_label_aware_constraint_queries':32,'TRAIN_or_PAIR_constraint_queries':0,
 'new_model_accuracy':None,'new_model_predictions_computed':False,'trainable_or_deployable_checkpoint_created':False,'original_head_modified':False,'HYP_GO_claimed':False,'threshold_scan_count':0,'new_encoder_or_RoMa_forward_count':0,'scheduler_calls':0,
 'interpretation':['Each system preserves all old28 correct and strictly fixes one specified olderror; other3 errors are unconstrained.',
 'Finite strict positive margins scale to unit margins because all inequalities are homogeneous in the7 readout coefficients including intercept.',
 'Exact certificates use exact rational values of original binary64 feature endpoints, not rounded difference vectors.',
 'Feasibility is a post-hoc representational possibility, not evidence that training can generalize or a new29/32 model result.',
 'Infeasibility excludes only this strict positive-margin system; exact ties/zero boundaries, changed correct sets, nonlinear readouts and other features remain outside scope.',
 'No certificate coefficient may feed current or subsequent training, thresholds, subset selection or deployment. Fixed jackknife work is independent.']}
 atomic(OUT/'result.json',output);print(json.dumps({'status':output['status'],'result_sha256':sha(OUT/'result.json'),'statuses':{q:c['status'] for q,c in systems.items()}},sort_keys=True),flush=True)
if __name__=='__main__':main()
