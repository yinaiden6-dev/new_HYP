#!/usr/bin/env python3
"""Fixed-recipe seven-parameter readouts: ORIGINAL7, FULL96 and primary FULL128.

Fit is TRAIN-only. A fresh process refits all parameters before either EVAL
feature bank is opened; all REAL/CBIND predictions are then independently
sealed before any EVAL role or outcome is opened.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,os,subprocess,sys,time,uuid
from datetime import datetime,timezone
from fractions import Fraction
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];PROGRAM=Path(__file__).resolve()
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'programs')]
import rc_original7_train128_execution_common_v1 as C
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC
AUTH=ROOT/'registry/rc_original7_train128_readout_authority_v1_20260910.json'
PLAN=ROOT/'plan/RC_ORIGINAL7_TRAIN96_TRAIN128_FIXED_RECIPE_READOUT_V1_20260910.md'
LAUNCH=ROOT/'slurm/rc_original7_train128_readout_v1_dev_cpuonly_59m.sbatch'
OUT=ROOT/'results/rc_original7_train128_readout_v1'
PREFLIGHT=ROOT/'results/rc_original7_train128_readout_v1_preflight'
TRAIN_INPUT=ROOT/'results/rc_original7_train128_inputs_v1'
TRAIN_META=ROOT/'results/rc_original7_train128_manifest_v1'
SHARED=ROOT/'results/rc_shared_query_target_prior_cache_v1'
PAIR=ROOT/'results/routea_matched_three_arm_pair64_training_features_v2'
OLD_RUNNER=ROOT/'programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py'
OLD_CPU=ROOT/'programs/run_rc_original7_eval128_full_evidence_v2_compat.py'
HEAD=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
E128=ROOT/'results/rc_original7_eval128_full_evidence_v1'
E128_META=ROOT/'results/rc_original7_expanded_eval128_manifest_v1'
OLD_ROLES=ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2'
OLD_BASELINE=ROOT/'results/rc_same_support_specificity_v1'
GALLERY_IMAGES=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images'
MODELS=('ORIGINAL7','FULL96','FULL128');MODES=('REAL','CBIND');PANELS=('EVAL32','EVAL128');ARM='C_PAIRED'
BASE_PARAMETER_SHA='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'
TRAIN_FUNCTION_SHA='45921b88203ef7065d824bdaab8e5468fc6d617dd3cb6d3fee213dfd2ccc7403'
PHASE=None;RELEASED=False;HASH_DEPTH=0;BLOCKED=[];EVAL_SHARED_PATHS=set()
CONTRACT={'models':list(MODELS),'primary_model':'FULL128','FULL96_role':'prespecified full-negative-context diagnostic; never selected after EVAL',
 'feature_count':6,'parameter_count':7,'initialization':'ALL_ZERO','seed':17,'steps':2000,'optimizer':'AdamW','lr':.03,'weight_decay':.001,
 'training_function_sha256':TRAIN_FUNCTION_SHA,'PAIR64':'original payload/order/features/labels untouched, including original mixed Q/R shifts',
 'FULL_sets':{'ORIGINAL7':32,'FULL96':96,'FULL128':128},'FULL_order':'TRAIN128 execution order; first32 oldFULL, next64 oldPAIR images, last32 added views',
 'target_absent':'retain selected record and recall miss; omit only undefined FULL positive loss, never insert target or label winner correct',
 'FULL_loss_reduction':'original mean over the effective target-present FULL rows','controls':list(MODES),'panels':list(PANELS),
 'inference':'all127 X@w+b; SWITCH iff maximum>0; physical-row ties; unchanged natural C128',
 'stage_order':'fit; explicit fresh refit; predict both panels all models/controls; fresh prediction replay; then join',
 'new_encoder_RoMa_LP_calls':0,'new_features':0,'new_thresholds':0,'deployment_changed':False}
STATIC={
 'old_training_program':(OLD_RUNNER,'546e2bc7b3df6c67bcac41e079ba1b00f15c7e65c52a8ab994cd5b2c38d81e22'),
 'old_evaluation_program':(OLD_CPU,'806a8506693e2d9e88aba4331702c2d2c8df58e4d3f693b9b30e0601de49304c'),
 'feature_core':(ROOT/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py','96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7'),
 'original7_seal':(HEAD,'42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b'),
 'PAIR64_payload':(PAIR/'payload.pt','d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3'),
 'PAIR64_validation':(PAIR/'independent_validation.json','657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b'),
 'shared_manifest':(SHARED/'manifest.json','f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c'),
 'shared_validation':(SHARED/'validation.json','0258615f000d0d82f757f57298b803420395ebc3a35e3a46bfa53c3116b7488f'),
 'train_worker':(TRAIN_META/'worker_manifest.json','50d894c9643ca1ef200c18ba79cf700b598be15c5ea322010853c201fc0f22f7'),
 'train_curator':(TRAIN_META/'curator_roles.json','240be57ffe824e8e029e83052a8b608166782edfdf1df8327bec164621de8ca1'),
 'train_metadata_validation':(TRAIN_META/'independent_metadata_validation.json','03176db194d4b5f9d68cc226a1ce48b019675e99b037205bae71552202fa907d'),
 'train_legacy_parity_index':(TRAIN_META/'legacy_full32_parity_index.json','31c337c676dc17d38fb85cdcb902b3aeddfe104aebbb6840baeaaba5837c6b03'),
 'EVAL128_prejoin_records':(E128/'prejoin_records.json','206e66c57da348afd120b75aeae99410388b4bd1211b7eb21b2ce4eaaa1acf41'),
 'EVAL128_prejoin_validation':(E128/'prejoin_validation.json','b7c6e577421215ec9f72d4c59f0c4b420f13f72a11fde85dcec193c0ba9a38c4'),
 'EVAL128_prejoin_seal':(E128/'prejoin_seal.json','cdaafb85ddc158ec9480074f4cce19bb24e204273f8693a519f7191ca6e76f09'),
 'EVAL128_input_qualification':(E128/'input_qualification.json','23f0f9af334dbd2916d6bf358324b417aea2a2894b618785580d12149e420291')}
POST={
 'EVAL128_curator':(E128_META/'curator_roles.json','47277ce66f7aba445fb80a93643039c7e8a47c1f5cc8237f60157a98e3e93381'),
 'EVAL128_metadata_validation':(E128_META/'independent_metadata_validation.json','a70936a24fc2f5c6996228aee673a6de505c3022487b903878d268ad9a34cd94'),
 'EVAL128_result':(E128/'result.json','096c9575ffa05a3f9b813adbe5e65c879fbdee77507aa757de09420bef0be866'),
 'EVAL128_validation':(E128/'validation.json','ccdad5d3de3f4b42b64b69fe11a76be3bb70e79b182c90a60bea11cfb8a1978a'),
 'EVAL128_independent_review':(E128/'independent_review.json','bda61510c69399f9adf8a7cbabb84365e96a7a13eeba83a99702996b7b62ca88'),
 'EVAL32_role_manifest':(OLD_ROLES/'role_manifest.json','2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454'),
 'EVAL32_role_validation':(OLD_ROLES/'independent_validation.json','ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac'),
 'EVAL32_original7_result':(OLD_BASELINE/'result.json','174195a1d5c5bb99602ea822d6702b2317740a0a7f265b2d19227059803207d1'),
 'EVAL32_original7_validation':(OLD_BASELINE/'independent_validation.json','6cabaebb90e5453d607d33faf7c6c11c1f02278a63b57a64f891dd51096211c2')}

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def sha(p):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  return h.hexdigest()
 finally:HASH_DEPTH-=1
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==v['sha256'],'SOURCE_HASH_DRIFT:'+str(p));return p.resolve()
def hx(v):return float(v).hex()
def tensor(v):return torch.tensor([[float.fromhex(x) for x in row] for row in v],dtype=torch.float64)
def matrix_hex(v):return [[hx(x) for x in row] for row in v]
def tsha(v):
 v=v.detach().cpu().contiguous();return hashlib.sha256(str(v.dtype).encode()+json.dumps(list(v.shape),separators=(',',':')).encode()+v.view(torch.uint8).numpy().tobytes()).hexdigest()
def runtime():return {'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
def write(name,value):C.write_json(OUT/name,value)

def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower();bad=None
 if any(x in s for x in ('/rc_opened_eval_strict_','d1_mi','d1-mi','grozi','gisc_prerecall_universe','direction_capacity','angle_capacity','rc_train_origin_linear_projection_capacity')):bad='ORACLE_OR_PROTECTED_DATA_FORBIDDEN'
 if not HASH_DEPTH:
  if not RELEASED and (p in {x[0].resolve() for x in POST.values()} or '/role_shards/' in s or '/target_join/' in s):bad='EVAL_ROLE_OR_OUTCOME_BEFORE_FRESH_PREDICTION_SEAL'
  if PHASE in ('preflight','fit','validate-fit') and (p in EVAL_SHARED_PATHS or p.is_relative_to(E128) or p.is_relative_to(E128_META)):bad='EVAL_FEATURE_OR_OUTCOME_BEFORE_FIT_SEAL'
  if PHASE=='preflight' and (p==STATIC['train_curator'][0].resolve() or p==PAIR.joinpath('payload.pt').resolve() or p==TRAIN_INPUT.joinpath('feature_records.json').resolve()):bad='PREFLIGHT_NO_NATURAL_TRAINING_DATA'
 if bad:BLOCKED.append(str(p));raise RuntimeError(bad)

def shared_entries():
 d=read(SHARED/'manifest.json');v=read(SHARED/'validation.json')
 need(d['status']=='RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_COMPLETE' and v['status']=='RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_SOURCE_REPLAY_VALIDATION_PASS' and v['manifest_sha256']==sha(SHARED/'manifest.json') and all(x is True for x in v['checks'].values()),'QUALIFIED_ORIGINAL_SHARED_CACHE')
 rows=[x for x in d['records'] if x['kind']=='FULL'];need(len(rows)==64,'ORIGINAL_FULL64_MANIFEST')
 for x in rows:
  p=Path(x['path']);p=p if p.is_absolute() else ROOT/p
  if x['role']=='EVAL':EVAL_SHARED_PATHS.update((p.resolve(),p.with_name('arrays.npz').resolve()))
 return {role:sorted((x for x in rows if x['role']==role),key=lambda x:x['execution_ordinal']) for role in ('TRAIN','EVAL')}

def source_bindings():
 sources={k:bind(p) for k,(p,h) in STATIC.items()};post={k:bind(p) for k,(p,h) in POST.items()}
 for k,(_,h) in STATIC.items():need(sources[k]['sha256']==h,'STATIC_SOURCE_PIN:'+k)
 for k,(_,h) in POST.items():need(post[k]['sha256']==h,'POSTJOIN_SOURCE_PIN:'+k)
 for k,p in {'program':PROGRAM,'plan':PLAN,'launcher':LAUNCH,'common_program':Path(C.__file__),
  'gallery_identity_core':ROOT/'src/rc_aslo_xf/gallery_identity_repair.py','gallery_identity_contract':ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json',
  'gallery_identity_registry':ROOT/'registry/gallery_identity_repair_v1.json','shared_validation':SHARED/'validation.json'}.items():sources[k]=bind(p)
 for role,rows in shared_entries().items():
  need(len(rows)==32,'ORIGINAL_SPLIT32')
  for x in rows:
   p=checked(x);prefix='shared_'+role+'_'+str(x['execution_ordinal'])
   sources[prefix+'_metadata']=bind(p);sources[prefix+'_arrays']=bind(p.with_name('arrays.npz'))
 for key,p in {'train_input_features':TRAIN_INPUT/'feature_records.json','train_input_validation':TRAIN_INPUT/'validation.json','train_input_authority':C.AUTHORITY}.items():
  if p.is_file():sources[key]=bind(p)
 ready=all(k in sources for k in ('train_input_features','train_input_validation','train_input_authority'))
 return sources,post,ready

def pure_functions():
 text=OLD_RUNNER.read_text();tree=ast.parse(text);names={'finite_tensor','finite_scalar','parameter_sha','train_head'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];need({n.name for n in nodes}==names,'PURE_OLD_FUNCTIONS')
 train=next(n for n in nodes if n.name=='train_head');need(hashlib.sha256(ast.get_source_segment(text,train).encode()).hexdigest()==TRAIN_FUNCTION_SHA,'UNCHANGED_TRAIN_FUNCTION_AST_SOURCE')
 space={'torch':torch,'nn':nn,'F':F,'math':math,'hashlib':hashlib,'json':json,'STEPS':2000,
  'FAMILIES':{'NATIVE7':('real_native_features','cbind_native_features',tuple(FC.FEATURE_NAMES))},'CrossfitContractError':RuntimeError}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(OLD_RUNNER),'exec'),space)
 etext=OLD_CPU.read_text();etree=ast.parse(etext);enames={'gallery_labels','literal_feature','predict','target_action','metrics','paired_actions','exact_group_signflip','primary_statistics'}
 enodes=[n for n in etree.body if isinstance(n,ast.FunctionDef) and n.name in enames]
 need({n.name for n in enodes}==enames,'PURE_QUALIFIED_ACTION_FUNCTIONS')
 espace={'torch':torch,'np':np,'math':math,'os':os,'Path':Path,'Fraction':Fraction,'defaultdict':defaultdict,'hashlib':hashlib,'need':need,'hx':hx,'GALLERY_IMAGES':GALLERY_IMAGES}
 exec(compile(ast.fix_missing_locations(ast.Module(body=enodes,type_ignores=[])),str(OLD_CPU),'exec'),espace)
 return space,espace

def require(phase):
 need(datetime.now(timezone.utc)<C.CUTOFF and bool(os.environ.get('SLURM_JOB_ID')),'SLURM_AND_USER_DEADLINE_REQUIRED')
 a=read(AUTH);need(a['status']=='ORIGINAL7_TRAIN128_READOUT_AUTHORIZED' and a['contract']==CONTRACT,'READOUT_AUTHORITY_CONTRACT')
 need(phase in a['allowed_phases'] and a['cutoff_UTC']=='2026-09-11T16:00:00+00:00','AUTHORIZED_PHASE_AND_CUTOFF')
 sources,post,ready=source_bindings();need(ready,'ALL_TRAIN_INPUTS_REQUIRED')
 need(a['source_bindings']==sources and a['postjoin_source_bindings']==post,'COMPLETE_SOURCE_AUTHORITY')
 pre=read(checked(a['preflight']));need(pre['status']=='TRAIN128_READOUT_PREFLIGHT_READY' and pre['sources']==sources and pre['postjoin_sources']==post and pre['runtime']==runtime(),'FINAL_READY_PREFLIGHT')
 return a

def evidence_features(row,independent=False):
 _,ev=pure_functions();feature=ev['literal_feature'] if independent else FC.candidate_feature
 raw=[float.fromhex(x) for x in row['candidate_raw_scores_binary64']]
 evidence={int(p):{k:float.fromhex(x) for k,x in v.items()} for p,v in row['C4_binary64'].items()}
 need(set(evidence)==set(range(128)) and len(raw)==128,'FULL_C128_EVIDENCE')
 out={}
 for mode in MODES:
  e=evidence if mode=='REAL' else {i:evidence[(i+64)%128] for i in range(128)}
  out[mode]=torch.stack([feature(raw,e,c,row['base_winner_position']) for c in row['challenger_positions']])
  need(matrix_hex(out[mode])==row['features_binary64'][mode],'QUALIFIED_FEATURE_BITS:'+mode)
 return out

def shared_row(entry):
 m=read(checked(entry));need(m['original_C_scalars_bit_exact'] is True,'OLD_C4_SOURCE_REPLAY');ap=checked(m['arrays'])
 with np.load(ap,allow_pickle=False) as z:raw=z['raw_scores'].tolist()
 axis=m['axis'];winner=m['winner'];cs=[i for i in range(128) if i!=winner]
 evidence={str(c['candidate_position']):c['old_scalars_binary64'] for c in m['candidates']}
 ranked=sorted(range(128),key=lambda p:(-raw[p],axis[p]))
 need(ranked[0]==winner and len(evidence)==128,'OLD_FULL128_AXIS')
 row={'query_id':m['query_id'],'execution_ordinal':m['execution_ordinal'],'track':entry.get('track','OLD_PANEL'),
  'candidate_physical_rows':axis,'candidate_raw_scores_binary64':[hx(x) for x in raw],'base_winner_position':winner,
  'challenger_positions':cs,'C4_binary64':evidence,'raw_ranked_physical_rows':[axis[p] for p in ranked],
  'raw_ranked_scores_binary64':[hx(raw[p]) for p in ranked],'query_tokens_sha256':m['q_tokens_sha256'],'source_image_sha256':m['query_source_image_sha256'],
  'candidate_map_token_bindings':[{k:c[k] for k in ('candidate_position','physical_row','query_map_sha256','reference_map_sha256','reference_tokens_sha256')} for c in m['candidates']]}
 e={int(p):{k:float.fromhex(x) for k,x in v.items()} for p,v in evidence.items()}
 row['features_binary64']={mode:matrix_hex(torch.stack([FC.candidate_feature(raw,e if mode=='REAL' else {i:e[(i+64)%128] for i in range(128)},c,winner) for c in cs])) for mode in MODES}
 return row

def training_inputs(independent=False):
 v=read(TRAIN_INPUT/'validation.json');need(v['status']=='ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS','TRAIN_INPUTS_QUALIFIED')
 need(v['query_count']==128 and v['candidate_occurrences']==16384 and v['independent_C4_scalar_checks']==65536 and v['curator_reads']==v['EVAL_result_reads']==v['training_updates']==v['forbidden_read_attempts']==0,'WHOLE_TRAIN_INPUTS_AND_BOUNDARY')
 need(checked(v['feature_records'])==(TRAIN_INPUT/'feature_records.json').resolve() and checked(v['worker_manifest'])==(TRAIN_META/'worker_manifest.json').resolve() and checked(v['authority'])==C.AUTHORITY.resolve(),'TRAIN_INPUT_BINDINGS')
 ia=read(checked(v['authority']));need(ia['status']=='ORIGINAL7_TRAIN128_INPUTS_AUTHORIZED' and v['program']==ia['source_bindings']['cpu_program'],'QUALIFIED_INPUT_CPU_PRODUCER');checked(v['program'])
 records=read(TRAIN_INPUT/'feature_records.json');roles=read(TRAIN_META/'curator_roles.json');workers=read(TRAIN_META/'worker_manifest.json')['records']
 need(roles['status']=='CURATOR_ONLY_ORIGINAL7_TRAIN128_ROLES_FROZEN' and roles['training_labels_only'] is True,'TRAIN_CURATOR_SCOPE')
 need(len(records)==len(roles['records'])==len(workers)==128,'ALL128_TRAIN_RECORDS')
 _,ops=pure_functions();labels,mapping=ops['gallery_labels']();joined=[]
 for i,(r,role,worker) in enumerate(zip(records,roles['records'],workers,strict=True)):
  need(r['execution_ordinal']==role['execution_ordinal']==worker['execution_ordinal']==i and r['query_id']==role['query_id']==worker['query_id'],'TRAIN_QUERY_ORDER')
  need(r['source_image_sha256']==role['source_image_sha256']==worker['source_image_sha256'] and role['role']=='TRAIN','TRAIN_IMAGE_ROLE_BINDING')
  need(role['selection_origin']==('ORIGINAL_FULL32' if i<32 else 'ORIGINAL_PAIR64' if i<96 else 'ADDED32'),'FIXED_TRAIN_ORIGIN_ORDER')
  f=evidence_features(r,independent);positions=[p for p,h in enumerate(r['candidate_physical_rows']) if labels[h]==role['identity']]
  need(len(positions)<=1 and sum(labels[h]==role['identity'] for h in r['raw_ranked_physical_rows'])==1,'NATURAL_C128_AND_FULL_GALLERY_IDENTITY')
  joined.append({**r,'target_position':positions[0] if positions else None,'target_identity':role['identity'],'supergroup':role['group'],
   'original_query_id':role['original_query_id'],'original_training_execution_ordinal':role['original_training_execution_ordinal'],
   'real_native_features':{ARM:f['REAL']},'cbind_native_features':{ARM:f['CBIND']}})
 old=shared_entries()['TRAIN'];need(len(old)==32,'ORIGINAL_TRAIN32')
 for new,entry in zip(joined[:32],old,strict=True):
  original=shared_row(entry);need(new['original_training_execution_ordinal']==entry['execution_ordinal'] and new['original_query_id']==entry['query_id'],'OLD_FULL32_ORDER')
  for k in ('candidate_physical_rows','candidate_raw_scores_binary64','base_winner_position','challenger_positions','query_tokens_sha256','C4_binary64','features_binary64','candidate_map_token_bindings'):
   need(new[k]==original[k],'OLD_FULL32_INPUT_FEATURE_BIT_REPLAY:'+k)
  need(new['target_position'] is not None,'OLD32_TARGET_PRESENT_REQUIRED')
 pv=read(PAIR/'independent_validation.json');need(pv['status']=='ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED' and pv['checks'] and all(x is True for x in pv['checks'].values()),'OLD_PAIR_VALIDATION')
 pair=torch.load(PAIR/'payload.pt',map_location='cpu',mmap=True,weights_only=True);need(len(pair['records'])==64,'ORIGINAL_PAIR64')
 for pairrow,fullrow in zip(pair['records'],joined[32:96],strict=True):
  need(pairrow['query_id']==fullrow['original_query_id'],'PAIR64_ORIGINAL_IMAGE_ORDER')
  x=pairrow['real_native_features'][ARM];need(x.shape==(1,6) and x.dtype==torch.float64 and bool(torch.isfinite(x).all()),'PAIR64_NATIVE_FEATURE_DOMAIN')
 sets={name:[r for r in joined[:n] if r['target_position'] is not None] for name,n in CONTRACT['FULL_sets'].items()}
 counts={name:{'selected_FULL_images':n,'effective_FULL_loss_rows':len(sets[name]),'recall_miss_count':n-len(sets[name]),
  'recall_miss_query_ids':[r['query_id'] for r in joined[:n] if r['target_position'] is None],'PAIR_auxiliary_rows':64,
  'effective_execution_order':[r['execution_ordinal'] for r in sets[name]]} for name,n in CONTRACT['FULL_sets'].items()}
 closure={'counts':counts,'old_FULL32_bit_exact':True,'PAIR64_feature_label_order_unchanged':True,'gallery_mapping_sha256':mapping,
  'TRAIN_feature_sha256':{name:[tsha(r['real_native_features'][ARM]) for r in rows] for name,rows in sets.items()},
  'PAIR_feature_sha256':[tsha(r['real_native_features'][ARM]) for r in pair['records']],
  'PAIR_switch_labels':[bool(r['switch_label']) for r in pair['records']],
  'TRAIN_target_positions':{r['query_id']:r['target_position'] for r in joined},'target_insertions':0,'EVAL_features_roles_or_outcomes_read':0}
 return pair,sets,closure

def fit_all(independent=False):
 pair,sets,closure=training_inputs(independent);old,_=pure_functions();expected=read(HEAD);params={}
 for name in MODELS:
  began=time.monotonic();head,loss,finite=old['train_head'](pair,sets[name],'NATIVE7',ARM)
  w=head.weight.detach().flatten();b=float(head.bias.detach())
  value={'weight_binary64':[hx(x) for x in w],'bias_binary64':hx(b),'parameter_sha256':old['parameter_sha'](w,b),
   'parameter_count':7,'loss_last_recorded':list(loss),'finite_training':finite,'training_function_sha256':TRAIN_FUNCTION_SHA,'effective_FULL_loss_rows':len(sets[name])}
  if name=='ORIGINAL7':need(value['parameter_sha256']==BASE_PARAMETER_SHA and value['weight_binary64']==expected['weight_binary64'] and value['bias_binary64']==expected['bias_binary64'],'ORIGINAL7_RETRAIN_REGRESSION_ABORT')
  params[name]=value;print(json.dumps({'event':'FIXED_RECIPE_HEAD_FITTED','model':name,'parameter_sha256':value['parameter_sha256'],'effective_FULL_rows':len(sets[name]),'seconds':time.monotonic()-began}),flush=True)
 return params,closure

def fit():
 require('fit');need(not (OUT/'parameter_seal.json').exists(),'APPEND_ONLY_PARAMETERS')
 p,closure=fit_all();write('training_input_closure.json',closure);write('parameters.json',p)
 write('parameter_seal.json',{'status':'TRAIN128_ALL_THREE_PARAMETERS_SEALED_BEFORE_EVAL_FEATURES','authority':bind(AUTH),'parameters':bind(OUT/'parameters.json'),
  'training_input_closure':bind(OUT/'training_input_closure.json'),'models':list(MODELS),'primary_model':'FULL128','EVAL_features_roles_or_outcomes_read':0,'forbidden_read_attempts':len(BLOCKED)})
 child('validate-fit')

def fit_seal(require_validation=True):
 seal=read(OUT/'parameter_seal.json');need(seal['status']=='TRAIN128_ALL_THREE_PARAMETERS_SEALED_BEFORE_EVAL_FEATURES' and seal['authority']==bind(AUTH),'PARAMETER_SEAL_SCOPE')
 need(checked(seal['parameters'])==(OUT/'parameters.json').resolve() and checked(seal['training_input_closure'])==(OUT/'training_input_closure.json').resolve(),'PARAMETER_SOURCE_SEAL')
 need(seal['EVAL_features_roles_or_outcomes_read']==seal['forbidden_read_attempts']==0,'PARAMETER_PREJOIN_BOUNDARY')
 if require_validation:
  v=read(OUT/'fit_validation.json');need(v['status']=='TRAIN128_FRESH_INDEPENDENT_REFIT_PASS' and v['parameter_seal']==bind(OUT/'parameter_seal.json') and v['parameters']==bind(OUT/'parameters.json') and v['explicit_fresh_subprocess'] and v['all_three_heads_bit_exact'],'INDEPENDENT_REFIT_REQUIRED')
 return read(OUT/'parameters.json')

def validate_fit():
 require('validate-fit');expected=fit_seal(False);actual,closure=fit_all(True)
 need(actual==expected and closure==read(OUT/'training_input_closure.json'),'FRESH_ALL_THREE_HEADS_AND_TRAIN_CLOSURE_REPLAY')
 write('fit_validation.json',{'status':'TRAIN128_FRESH_INDEPENDENT_REFIT_PASS','authority':bind(AUTH),'parameter_seal':bind(OUT/'parameter_seal.json'),
  'parameters':bind(OUT/'parameters.json'),'training_input_closure':bind(OUT/'training_input_closure.json'),'explicit_fresh_subprocess':True,
  'all_three_heads_bit_exact':True,'original7_parameter_sha256':BASE_PARAMETER_SHA,'training_updates':6000,'EVAL_features_roles_or_outcomes_read':0,'forbidden_read_attempts':len(BLOCKED)})

def evaluation_inputs(independent=False):
 fit_seal();need(PHASE in ('predict','validate-predictions','join','validate-result'),'EVAL_FEATURES_ONLY_AFTER_ALL_PARAMETERS_VALIDATED')
 entries=shared_entries()['EVAL'];small=[shared_row(e) for e in entries]
 v=read(E128/'prejoin_validation.json');seal=read(E128/'prejoin_seal.json')
 need(v['status']=='ORIGINAL7_EVAL128_PREJOIN_INDEPENDENT_CPU_REPLAY_PASS' and seal['status']=='ORIGINAL7_EVAL128_ALL_PREDICTIONS_SEALED_BEFORE_CURATOR','QUALIFIED_EVAL128_LABEL_FREE_SOURCE')
 need(v['checks'] and all(x is True for x in v['checks'].values()) and seal['query_count']==128 and seal['head_parameter_sha256']==BASE_PARAMETER_SHA,'EVAL128_FULL_INPUT_CLOSURE')
 need(checked(v['seal'])==(E128/'prejoin_seal.json').resolve() and checked(v['records'])==checked(seal['records'])==(E128/'prejoin_records.json').resolve(),'EVAL128_SOURCE_RECORD_SEAL')
 need(checked(v['input_qualification'])==checked(seal['input_qualification'])==(E128/'input_qualification.json').resolve(),'EVAL128_QUALIFICATION_BINDING')
 large=read(E128/'prejoin_records.json');need(len(large)==128 and [r['execution_ordinal'] for r in large]==list(range(128)),'ALL_EVAL128')
 banks={'EVAL32':small,'EVAL128':large}
 for panel,rows in banks.items():
  for row in rows:evidence_features(row,independent)
 return banks

def score_banks(params,banks,independent=False):
 _,ops=pure_functions();outputs={};count=0
 for panel,rows in banks.items():
  outputs[panel]=[]
  for row in rows:
   f=evidence_features(row,independent);predictions={}
   for name in MODELS:
    p=params[name];w=torch.tensor([float.fromhex(x) for x in p['weight_binary64']],dtype=torch.float64);b=float.fromhex(p['bias_binary64'])
    predictions[name]={mode:ops['predict'](row,f[mode]@w+b,independent) for mode in MODES};count+=127*len(MODES)
   if panel=='EVAL128':
    for mode in MODES:need(predictions['ORIGINAL7'][mode]==row['predictions'][mode],'EVAL128_ORIGINAL7_ALL127_PREDICTION_REPLAY')
   outputs[panel].append({'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'predictions':predictions,
    'feature_sha256':{mode:tsha(f[mode]) for mode in MODES}})
 need(count==160*127*3*2,'ALL160_ALLTHREE_ALLMODES_LOGITS')
 return outputs,count

def predict_all():
 require('predict');p=fit_seal();need(not (OUT/'prediction_seal.json').exists(),'APPEND_ONLY_PREDICTION_SEAL')
 banks=evaluation_inputs();predictions,count=score_banks(p,banks);write('predictions.json',predictions)
 write('prediction_seal.json',{'status':'TRAIN128_ALL160_ALLTHREE_REAL_CBIND_PREDICTIONS_SEALED','authority':bind(AUTH),
  'parameters':bind(OUT/'parameters.json'),'parameter_seal':bind(OUT/'parameter_seal.json'),'fit_validation':bind(OUT/'fit_validation.json'),
  'predictions':bind(OUT/'predictions.json'),'models':list(MODELS),'modes':list(MODES),'panels':{'EVAL32':32,'EVAL128':128},
  'all127_logit_count':count,'EVAL_role_outcome_reads':0,'forbidden_read_attempts':len(BLOCKED)})
 child('validate-predictions')

def prediction_seal(require_validation=True):
 fit_seal();s=read(OUT/'prediction_seal.json')
 need(s['status']=='TRAIN128_ALL160_ALLTHREE_REAL_CBIND_PREDICTIONS_SEALED' and s['authority']==bind(AUTH),'PREDICTION_SEAL_SCOPE')
 need(s['models']==list(MODELS) and s['modes']==list(MODES) and s['panels']=={'EVAL32':32,'EVAL128':128} and s['all127_logit_count']==121920,'COMPLETE_PREDICTION_SCOPE')
 for k,n in [('parameters','parameters.json'),('parameter_seal','parameter_seal.json'),('fit_validation','fit_validation.json'),('predictions','predictions.json')]:need(checked(s[k])==(OUT/n).resolve(),'PREDICTION_COMPONENT:'+k)
 need(s['EVAL_role_outcome_reads']==s['forbidden_read_attempts']==0,'PREJOIN_NO_TARGET_OUTCOME_ACCESS')
 if require_validation:
  v=read(OUT/'prediction_validation.json');need(v['status']=='TRAIN128_FRESH_ALL160_PREDICTION_REPLAY_PASS' and v['prediction_seal']==bind(OUT/'prediction_seal.json') and v['predictions']==bind(OUT/'predictions.json') and v['explicit_fresh_subprocess'] and v['all127_logit_count']==121920,'FRESH_ALL_PREDICTIONS_REQUIRED')
 return read(OUT/'predictions.json')

def validate_predictions():
 require('validate-predictions');expected=prediction_seal(False);banks=evaluation_inputs(True)
 actual,count=score_banks(fit_seal(),banks,True);need(actual==expected,'INDEPENDENT_LITERAL_FEATURE_AND_PREDICTION_REPLAY')
 write('prediction_validation.json',{'status':'TRAIN128_FRESH_ALL160_PREDICTION_REPLAY_PASS','authority':bind(AUTH),
  'prediction_seal':bind(OUT/'prediction_seal.json'),'predictions':bind(OUT/'predictions.json'),'explicit_fresh_subprocess':True,
  'all127_logit_count':count,'independent_literal_feature_and_physical_tie_replay':True,'EVAL_role_outcome_reads':0,'forbidden_read_attempts':len(BLOCKED)})

def postjoin_roles(banks):
 need(RELEASED and PHASE in ('join','validate-result'),'POSTJOIN_RELEASE_REQUIRED')
 rm=read(OLD_ROLES/'role_manifest.json');rv=read(OLD_ROLES/'independent_validation.json')
 need(rm['status']=='RGH_P0_A0_ROLE_MANIFEST_READY' and rv['status']=='RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS','ORIGINAL_ROLE_SOURCES_QUALIFIED')
 entries={int(e['execution_ordinal']):e for e in rm['shards']};small={}
 for row in banks['EVAL32']:
  role=read(checked(entries[row['execution_ordinal']]))
  need(role['query_id']==row['query_id'],'EVAL32_ROLE_QUERY_JOIN')
  small[row['query_id']]={'identity':role['identity'],'group':role['supergroup'],'original_query_id':role['query_id'],'track':role['track']}
 meta=read(E128_META/'independent_metadata_validation.json');curator=read(E128_META/'curator_roles.json')
 need(meta['status']=='EXPANDED_EVAL128_INDEPENDENT_METADATA_SELECTION_PASS' and curator['status']=='CURATOR_ONLY_EXPANDED_EVAL128_ROLES_FROZEN','QUALIFIED_EVAL128_CURATOR')
 need(checked(meta['curator_ledger'])==(E128_META/'curator_roles.json').resolve(),'EVAL128_CURATOR_BINDING')
 large={x['query_id']:x for x in curator['records']};need(len(large)==128 and set(large)=={x['query_id'] for x in banks['EVAL128']},'ALL128_CURATOR_RECORDS')
 return {'EVAL32':small,'EVAL128':large}

def group_comparison(new,baseline,ops):
 need([x['query_id'] for x in new]==[x['query_id'] for x in baseline],'GROUP_PAIRED_ORDER')
 value=ops['primary_statistics']([{**x,'base_correct':b['final_correct']} for x,b in zip(new,baseline,strict=True)])
 for group in value['groups']:
  group['baseline_correct']=group.pop('RAW_correct');group['model_correct']=group.pop('REAL_correct')
 value['baseline']='ORIGINAL7'
 return value

def summary_result():
 global RELEASED
 expected=prediction_seal();banks=evaluation_inputs(True)
 recomputed,count=score_banks(fit_seal(),banks,True);need(recomputed==expected,'POSTJOIN_ALL_PREDICTIONS_REPLAY_BEFORE_RELEASE')
 RELEASED=True;roles=postjoin_roles(banks);_,ops=pure_functions();labels,mapping=ops['gallery_labels']()
 actions={};metrics={};comparisons={};preservation={};controls={};groups={};group_baseline={}
 for panel,rows in banks.items():
  pred={x['query_id']:x for x in expected[panel]};actions[panel]={};metrics[panel]={};preservation[panel]={};controls[panel]={};groups[panel]={}
  for name in MODELS:
   actions[panel][name]={mode:[] for mode in MODES}
   for row in rows:
    role=roles[panel][row['query_id']];report_row={**row,'track':role.get('track',row['track'])}
    for mode in MODES:actions[panel][name][mode].append(ops['target_action'](report_row,pred[row['query_id']]['predictions'][name][mode],role,labels))
   metrics[panel][name]={mode:ops['metrics'](aa) for mode,aa in actions[panel][name].items()}
   groups[panel][name]=ops['primary_statistics'](actions[panel][name]['REAL'])
   controls[panel][name]=ops['paired_actions'](actions[panel][name]['CBIND'],actions[panel][name]['REAL'])
  need(groups[panel]['ORIGINAL7']['group_count']==(11 if panel=='EVAL32' else 21),'COMPLETE_OPENED_GROUP_POPULATION')
  base=actions[panel]['ORIGINAL7']['REAL'];base_good={r['query_id'] for r in base if r['final_correct']};raw_good={r['query_id'] for r in base if r['base_correct']}
  comparisons[panel]={name+'_vs_ORIGINAL7':ops['paired_actions'](actions[panel][name]['REAL'],base) for name in ('FULL96','FULL128')}
  comparisons[panel]['FULL128_vs_FULL96']=ops['paired_actions'](actions[panel]['FULL128']['REAL'],actions[panel]['FULL96']['REAL'])
  group_baseline[panel]={name:group_comparison(actions[panel][name]['REAL'],base,ops) for name in ('FULL96','FULL128')}
  for name in MODELS:
   good={r['query_id'] for r in actions[panel][name]['REAL'] if r['final_correct']}
   preservation[panel][name]={'original7_correct_count':len(base_good),'retained_original7_correct':len(good&base_good),
    'lost_original7_correct_query_ids':sorted(base_good-good),'added_vs_original7_query_ids':sorted(good-base_good),
    'RAW_correct_count':len(raw_good),'retained_RAW_correct':len(good&raw_good),'lost_RAW_correct_query_ids':sorted(raw_good-good)}
 old32=read(OLD_BASELINE/'result.json');old32v=read(OLD_BASELINE/'independent_validation.json')
 need(old32v['result_sha256']==sha(OLD_BASELINE/'result.json') and old32v['status']=='SAME_SUPPORT_SPECIFICITY_INDEPENDENT_REEXECUTION_PASS','QUALIFIED_OLD_EVAL32_BASELINE')
 old128=read(E128/'result.json');old128v=read(E128/'validation.json');review=read(E128/'independent_review.json')
 need(checked(old128v['result'])==(E128/'result.json').resolve() and old128v['status']=='ORIGINAL7_EVAL128_POSTJOIN_LITERAL_REPLAY_PASS','QUALIFIED_OLD_EVAL128_BASELINE')
 need('PASS' in review['status'],'INDEPENDENT_EVAL128_BASELINE_REVIEW')
 for panel,prior in [('EVAL32',old32['actions']['EVAL']['ORIGINAL7']),('EVAL128',old128['actions'])]:
  for mode in MODES:
   before={x['query_id']:x for x in prior[mode]};after=actions[panel]['ORIGINAL7'][mode]
   need(set(before)=={x['query_id'] for x in after},'BASELINE_ALL_QUERY_REPLAY')
   for row in after:
    old=before[row['query_id']]
    need(row['final_correct']==old['final_correct'] and row['base_correct']==old['base_correct'] and row['final_physical_row']==old['final_physical_row'] and row['decision']==old['decision'] and hx(row['switch_logit'])==hx(old['switch_logit']),'ORIGINAL7_ACTION_AND_CORRECT_SET_REPLAY:'+panel)
 need(metrics['EVAL32']['ORIGINAL7']['REAL']['final_top1']==28 and metrics['EVAL128']['ORIGINAL7']['REAL']['final_top1']==99,'ORIGINAL28_AND99_BASELINE_REGRESSION')
 need(all(x['candidate_recall'] for x in actions['EVAL32']['ORIGINAL7']['REAL']),'OLD32_TARGETS_IN_NATURAL_C128')
 return {'status':'ORIGINAL7_TRAIN128_FIXED_RECIPE_READOUT_COMPLETE','authority':bind(AUTH),'program':bind(PROGRAM),'plan':bind(PLAN),
  'models':list(MODELS),'primary_model':'FULL128','FULL96_role':CONTRACT['FULL96_role'],'feature_count':6,'parameter_count_per_head':7,
  'parameters':bind(OUT/'parameters.json'),'fit_validation':bind(OUT/'fit_validation.json'),'prediction_seal':bind(OUT/'prediction_seal.json'),
  'prediction_validation':bind(OUT/'prediction_validation.json'),'training_input_closure':read(OUT/'training_input_closure.json'),
  'metrics':metrics,'paired_comparisons':comparisons,'original7_and_RAW_correct_preservation':preservation,'CBIND_vs_REAL':controls,
  'group_results_vs_RAW':groups,'group_results_vs_ORIGINAL7':group_baseline,'actions':actions,'baseline28_and99_correct_sets_exact':True,'RAW_and_original7_preservation_reported_separately':True,
  'source_gallery_mapping_sha256':mapping,'all127_logit_count':count,'new_encoder_RoMa_LP_calls':0,'new_features':0,'threshold_changes':0,
  'target_insertions':0,'EVAL_target_absent_queries_dropped':0,'deployment_changed':False,'scientific_GO_or_NO_GO':None,
  'evidence_level':'historically opened internal panels; no untouched external or ownership claim',
  'limits':['FULL96 adds full-C128 negative context to the original PAIR64 images; FULL128 additionally includes32 training views.',
   'Original PAIR64 auxiliary features retain their historical mixed Q/R shifts; newly materialized FULL128 uses the frozen half-roll controls.',
   'Candidate-absent TRAIN records remain selected and counted; only their undefined FULL positive loss is omitted.',
   'No selection between FULL96 and FULL128 or parameter change is made from these evaluation outcomes.']}

def join():
 require('join');need(not (OUT/'result.json').exists(),'APPEND_ONLY_RESULT')
 value=summary_result();write('result.json',value);child('validate-result')
 print(json.dumps({'status':value['status'],'result':bind(OUT/'result.json'),'REAL_correct':{panel:{n:value['metrics'][panel][n]['REAL']['final_top1'] for n in MODELS} for panel in PANELS}}),flush=True)

def validate_result():
 require('validate-result');value=summary_result();need(value==read(OUT/'result.json'),'FRESH_POSTJOIN_ALL_RESULTS_REPLAY')
 write('validation.json',{'status':'ORIGINAL7_TRAIN128_READOUT_FRESH_RESULT_REPLAY_PASS','result':bind(OUT/'result.json'),
  'authority':bind(AUTH),'parameters':bind(OUT/'parameters.json'),'fit_validation':bind(OUT/'fit_validation.json'),
  'prediction_validation':bind(OUT/'prediction_validation.json'),'explicit_fresh_subprocess':True,
  'all_actions_metrics_correct_sets_and_controls_replayed':True,'new_training_updates':0})

def metadata_schema_probe(ready):
 pair=read(PAIR/'independent_validation.json')
 need(pair['status']=='ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED' and pair['v2_payload_sha256']==STATIC['PAIR64_payload'][1] and pair['checks'] and all(x is True for x in pair['checks'].values()),'PAIR_METADATA_SCHEMA_AND_BINDING')
 worker=read(TRAIN_META/'worker_manifest.json');meta=read(TRAIN_META/'independent_metadata_validation.json');parity=read(TRAIN_META/'legacy_full32_parity_index.json')
 need(worker['status']=='TRAIN128_WORKER_MANIFEST_FROZEN_METADATA_ONLY' and worker['query_count']==128 and len(worker['records'])==128,'TRAIN_WORKER_SCHEMA')
 need(meta['status']=='TRAIN128_INDEPENDENT_METADATA_SELECTION_PASS' and meta['worker_manifest']==bind(TRAIN_META/'worker_manifest.json') and meta['curator_ledger']==bind(TRAIN_META/'curator_roles.json'),'TRAIN_METADATA_BINDING')
 need(parity['status']=='ORIGINAL7_TRAIN128_LEGACY_FULL32_PARITY_INDEX_METADATA_ONLY' and len(parity['records'])==32 and [x['execution_ordinal'] for x in parity['records']]==list(range(32)),'TRAIN_PARITY_METADATA_SCHEMA')
 head=read(HEAD);need(head['parameter_sha256']==BASE_PARAMETER_SHA and len(head['weight_binary64'])==6,'ORIGINAL7_PARAMETER_SEAL_SCHEMA')
 if ready:
  v=read(TRAIN_INPUT/'validation.json');ia=read(checked(v['authority']))
  need(v['status']=='ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS' and v['query_count']==128 and v['candidate_occurrences']==16384 and v['independent_C4_scalar_checks']==65536,'READY_TRAIN_INPUT_VALIDATION_SCHEMA')
  need(v['program']==ia['source_bindings']['cpu_program'] and checked(v['feature_records'])==(TRAIN_INPUT/'feature_records.json').resolve(),'READY_TRAIN_INPUT_BINDINGS')
 return {'PAIR_validation_exact_status':pair['status'],'TRAIN_metadata_exact_status':meta['status'],'full32_parity_index_entries':32,'TRAIN_input_validation_ready_checked':ready,'feature_tensor_deserializations':0,'curator_semantic_reads':0}

def preflight():
 sources,post,ready=source_bindings();old,ops=pure_functions();metadata_check=metadata_schema_probe(ready)
 raw=[i/128 for i in range(128)];ev={i:{'real_score':.1+(i%9)*.01,'visibility_mass':.2+(i%7)*.01,'query_control_score':.03,'reference_control_score':.04} for i in range(128)}
 for c in range(127):need(tsha(FC.candidate_feature(raw,ev,c,127))==tsha(ops['literal_feature'](raw,ev,c,127)),'SYNTHETIC_SIX_FEATURE_BITS')
 toy={'candidate_physical_rows':list(range(128)),'base_winner_position':0,'challenger_positions':list(range(1,128)),'raw_ranked_physical_rows':list(range(5412)),
  'query_id':'SYNTHETIC','execution_ordinal':0,'track':'SYNTHETIC'}
 labels=[str(i) for i in range(5412)];z=torch.zeros(127,dtype=torch.float64)
 held=ops['predict'](toy,z);need(held['decision']=='HOLD','FIXED_ZERO_THRESHOLD_HOLD')
 role={'identity':'151','group':'SYNTHETIC','original_query_id':'SYNTHETIC'}
 absent=ops['target_action'](toy,held,role,labels);need(not absent['candidate_recall'] and not absent['final_correct'] and absent['final_target_rank_full_gallery']==152,'ABSENT_TARGET_RETAINED')
 z[4]=1.;chosen=ops['predict'](toy,z);need(chosen['decision']=='SWITCH' and chosen['final_position']==5,'ALL127_ACTION')
 rows=[{'target_position':0},{'target_position':None},{'target_position':17}];effective=[r for r in rows if r['target_position'] is not None]
 need(len(rows)==3 and len(effective)==2 and effective[0]['target_position']==0,'ABSENT_FULL_LOSS_ONLY_EXCLUSION')
 group_probe=ops['primary_statistics']([{'group':'g0','base_correct':False,'final_correct':True},{'group':'g1','base_correct':True,'final_correct':False}]);need(group_probe['group_count']==2 and group_probe['equal_group_mean_accuracy_difference']==0. and group_probe['two_sided_group_signflip']['two_sided_p']==1.,'FROZEN_GROUP_STATISTICS_SYNTHETIC')
 value={'status':'TRAIN128_READOUT_PREFLIGHT_READY' if ready else 'TRAIN128_READOUT_PREFLIGHT_STAGED_INPUTS_PENDING','sources':sources,'postjoin_sources':post,
  'runtime':runtime(),'contract':CONTRACT,'metadata_schema_probe':metadata_check,'original_training_function_sha256':TRAIN_FUNCTION_SHA,'synthetic_six_feature_checks':127,
  'synthetic_absent_target_and_fixed_action_checks':4,'synthetic_group_statistics_checks':1,'natural_optimizer_updates':0,'natural_feature_payload_reads':0,'EVAL_role_outcome_reads':0,'forbidden_read_attempts':len(BLOCKED)}
 key=hashlib.sha256(json.dumps({'sources':sources,'postjoin':post},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 path=PREFLIGHT/key/'result.json';C.write_json(path,value)
 print(json.dumps({'status':value['status'],'preflight':bind(path),'natural_optimizer_updates':0,'natural_feature_payload_reads':0}),flush=True)

def child(phase):
 nonce=uuid.uuid4().hex;env=dict(os.environ,TRAIN128_READOUT_VALIDATOR_NONCE=nonce,TRAIN128_READOUT_VALIDATOR_PARENT_PID=str(os.getpid()))
 subprocess.run([sys.executable,str(PROGRAM),'--phase',phase,'--validator-nonce',nonce],check=True,env=env)

def main():
 global PHASE
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('preflight','fit','validate-fit','predict','validate-predictions','join','validate-result'),required=True);p.add_argument('--validator-nonce');args=p.parse_args();PHASE=args.phase
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(17);sys.addaudithook(audit)
 if PHASE.startswith('validate-'):need(args.validator_nonce==os.environ.get('TRAIN128_READOUT_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('TRAIN128_READOUT_VALIDATOR_PARENT_PID'),'EXPLICIT_FRESH_VALIDATOR_SUBPROCESS')
 {'preflight':preflight,'fit':fit,'validate-fit':validate_fit,'predict':predict_all,'validate-predictions':validate_predictions,'join':join,'validate-result':validate_result}[PHASE]()
if __name__=='__main__':main()
