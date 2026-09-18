#!/usr/bin/env python3
"""Fixed ALL readouts and net-four capacity, not a new HYP evaluation."""
import ast, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_full64_all_reference_capacity_v1"
CACHE = ROOT / "results/rc_full64_full_reference_cache_v1"
ROLES = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json"
EVAL = ROOT / "results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json"
TRAIN = ROOT / "results/rc_shared_query_prior_train_eval_accounting_v1/result.json"
PINS = {
 "manifest": (CACHE/"manifest.json", "51618b74113bb8c3064e36116c6db0f6d559419d1da7a0fcb4b0b3725ce12234"),
 "validation": (CACHE/"validation.json", "65760c2a528e31c6fe9246b6bbe76869a74d81599d5b8cdeedf60ecaaca431f8"),
 "roles": (ROLES, "2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454"),
 "role_validation": (ROLES.parent/"independent_validation.json", "ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac"),
 "identity_core": (ROOT/"src/rc_aslo_xf/gallery_identity_repair.py", "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d"),
 "identity_manifest": (ROOT/"registry/gallery_identity_repair_v1.json", "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f"),
 "label_helper": (ROOT/"programs/run_rc_shared_query_target_prior_development_v1.py", "b004d6b3c518773d9f31c65f8e9a40f0d43593fe8e06b39b99bd4f33bb951307"),
 "EVAL_system_reference": (EVAL, "591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3"),
 "TRAIN_system_reference": (TRAIN, "603e4b764a5d58c23263b7b46878ac3df4ed35616e95adccbd52e59c16a91df5")}
released = False
hash_only = 0
blocked = []

def need(x, message):
 if not bool(x): raise RuntimeError(message)
def sha(path):
 global hash_only
 hash_only += 1
 try: return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 finally: hash_only -= 1
def read(path): return json.loads(Path(path).read_text())
def checked(v):
 p=Path(v['path']); need(sha(p)==v['sha256'], 'SOURCE_HASH_DRIFT'); return p
def bind(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def save(path,value):
 with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 path.chmod(0o444)
def barrier(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
 p=str(Path(os.fsdecode(args[0])).resolve())
 if not released and not hash_only and ('/role_shards/' in p or p in (str(EVAL),str(TRAIN))):
  blocked.append(p);raise RuntimeError('TARGET_READ_BEFORE_ALL64_SCORE_SEAL')
def metrics(rows):
 return {'query_count':len(rows),'strict_correct':sum(x['strict_correct'] for x in rows),
  'argmax_correct':sum(x['argmax_correct'] for x in rows),'target_coverage':sum(bool(x['target_equivalent_positions']) for x in rows),
  'MRR':sum(x['target_MRR'] for x in rows)/len(rows)}

def main():
 global released
 torch.set_num_threads(8);torch.set_num_interop_threads(1);sys.addaudithook(barrier)
 need(not (OUT/'result.json').exists(),'APPEND_ONLY_RESULT_EXISTS');sources={}
 for key,(path,digest) in PINS.items(): need(sha(path)==digest,'PIN:'+key);sources[key]=bind(path)
 m=read(CACHE/'manifest.json');v=read(CACHE/'validation.json')
 need(v['manifest_sha256']==sources['manifest']['sha256'] and all(x is True for x in v['checks'].values()),'CACHE_NOT_VALIDATED')
 if OUT.exists():
  seal=read(OUT/'prejoin_seal.json');need(seal['sources']==sources and seal['prejoin_sha256']==sha(OUT/'prejoin.json'),'SEALED_PREJOIN_RESUME_BINDING')
  pending=read(OUT/'prejoin.json')
 else:
  pending=[]
  for entry in m['records']:
   meta=read(checked(entry));scores={}
   with np.load(checked(meta['arrays']),allow_pickle=False) as arrays:
    for label,key in (('R_FULL','a_RFULL'),('R_IMAGE','a_RIMAGE_same_source')):
     a=torch.from_numpy(arrays[key].copy());need(a.dtype==torch.float64 and a.shape==(128,meta['query_token_count']),'FULL_ARRAY_AXIS')
     scores[label]=[float(x).hex() for x in a.mean(dim=1)]
   pending.append({'query_id':meta['query_id'],'execution_ordinal':meta['execution_ordinal'],'role':meta['role'],
    'candidate_physical_rows':meta['candidate_physical_rows'],'scores_binary64':scores,'target_label_reads':0})
  OUT.mkdir(parents=True);save(OUT/'prejoin.json',pending)
  save(OUT/'prejoin_seal.json',{'status':'ALL64_BOTH_R_FULL128_MEAN_SCORES_SEALED','prejoin_sha256':sha(OUT/'prejoin.json'),
   'query_count':64,'candidate_count':128,'representations':['R_FULL','R_IMAGE'],'sources':sources,'target_label_reads':0})
 need(len(pending)==64 and sum(x['role']=='TRAIN' for x in pending)==32,'FULL64_POPULATION')
 need(not blocked,'EARLY_LABEL_READ');released=True
 sys.path.insert(0,str(ROOT/'src'))
 helper=PINS['label_helper'][0];nodes=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='corrected_labels']
 need(len(nodes)==1,'FROZEN_LABEL_HELPER');ns={'ROOT':ROOT,'Path':Path,'os':os,'need':need};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(helper),'exec'),ns)
 labels=ns['corrected_labels']();rolemap={x['execution_ordinal']:x for x in read(ROLES)['shards']}
 refs={'EVAL':{x['execution_ordinal']:x for x in read(EVAL)['evaluations']['NATIVE7']['C_PAIRED']['actions']},
       'TRAIN':{x['execution_ordinal']:x for x in read(TRAIN)['FULL_TRAIN32_actions']['UNIFORM']}}
 joined=[]
 for row in pending:
  ex=row['execution_ordinal'];r=read(checked(rolemap[ex]));axis=row['candidate_physical_rows']
  need(r['query_id']==row['query_id'] and r['target_insertion_count']==r['raw_d1_field_count']==r['target_spatial_supervision_count']==0,'ROLE_BINDING')
  targets=[i for i,p in enumerate(axis) if labels[p]==r['identity']];wrong=[i for i in range(128) if i not in targets]
  need(bool(wrong),'NO_WRONG_CANDIDATE');views={}
  for rep,hexes in row['scores_binary64'].items():
   scores=[float.fromhex(x) for x in hexes];order=sorted(range(128),key=lambda i:(-scores[i],axis[i]));winner=order[0]
   rival=max(wrong,key=lambda i:(scores[i],-axis[i]));best=max(targets,key=lambda i:(scores[i],-axis[i])) if targets else None
   margin=None if best is None else scores[best]-scores[rival];rank=min((order.index(i)+1 for i in targets),default=None)
   views[rep]={'target_equivalent_positions':targets,'best_target_position':best,'strongest_wrong_position':rival,
    'strongest_wrong_physical_row':axis[rival],'strict_margin_binary64':None if margin is None else margin.hex(),
    'strict_correct':margin is not None and margin>0,'argmax_correct':winner in targets,'argmax_position':winner,
    'target_rank':rank,'target_MRR':0. if rank is None else 1./rank,'full_candidate_rank_order':order}
  ref=refs[row['role']][ex];need(ref['query_id']==row['query_id'] and ref['target_position'] in targets,'SYSTEM_REFERENCE_TARGET_BINDING')
  joined.append({**row,'target_identity':r['identity'],'supergroup':r['supergroup'],'views':views,
   'separate_system_reference':{'RAW_argmax_correct':ref['base_winner'] in targets,'NativeC_action_correct':ref['final_position'] in targets}})
 summary={};capacity={};system={}
 for role in ('TRAIN','EVAL'):
  rows=[x for x in joined if x['role']==role];summary[role]={};capacity[role]={}
  system[role]={'query_count':32,**{k:sum(x['separate_system_reference'][k] for x in rows) for k in ('RAW_argmax_correct','NativeC_action_correct')}}
  for rep in ('R_FULL','R_IMAGE'):
   values=[x['views'][rep] for x in rows];s=metrics(values);groups={g:[x['views'][rep] for x in rows if x['supergroup']==g] for g in sorted({x['supergroup'] for x in rows})}
   gm={g:metrics(xs) for g,xs in groups.items()};s['supergroup_metrics']=gm;s['supergroup_count']=len(groups)
   s['group_balanced_strict_rate']=sum(x['strict_correct']/x['query_count'] for x in gm.values())/len(gm)
   s['group_balanced_argmax_rate']=sum(x['argmax_correct']/x['query_count'] for x in gm.values())/len(gm)
   s['group_balanced_MRR']=sum(x['MRR'] for x in gm.values())/len(gm);summary[role][rep]=s
   capacity[role][rep]={'required_paired_net':4,'maximum_possible_strict_net_even_if_every_query_wins':32-s['strict_correct'],
    'maximum_possible_argmax_net_even_if_every_query_wins':32-s['argmax_correct'],
    'net4_mathematically_possible_under_strict_definition':32-s['strict_correct']>=4,
    'net4_mathematically_possible_under_argmax_definition':32-s['argmax_correct']>=4}
 result={'status':'RC_FULL64_ALL_REFERENCE_CAPACITY_COMPLETE','sources':sources,'program':bind(Path(__file__)),
  'prejoin_seal':bind(OUT/'prejoin_seal.json'),'readout':'ALL_ORIGINAL_QUERY_IMAGE_TOKENS_MEAN_OF_CACHED_CPU_FP64_NORMALIZED_MAXSIM',
  'metrics':summary,'net4_capacity':capacity,'separate_original_system_references':system,'rows':joined,
  'limits':['No new P or HYP scoring is performed; this is the fixed ALL baseline and its arithmetic net-gain ceiling.',
   'Target-equivalent candidates are grouped; strict margins exclude all target-equivalent positions and ties fail.',
   'RAW uses full query/image+template GPU FP32 SUM-MaxSim; NativeC adds its old action. They are separate references, not the ALL baseline.',
   'No primary R representation or gate is changed after observing these counts.'],
  'training_updates':0,'encoder_or_RoMa_forward_count':0,'new_P_count':0,'HYP_GO_claimed':False,'scientific_GO_or_NO_GO':None}
 save(OUT/'result.json',result)
 print(json.dumps({'status':result['status'],'result_sha256':sha(OUT/'result.json'),'metrics':summary,'net4_capacity':capacity,'separate_system_references':system},sort_keys=True),flush=True)

if __name__=='__main__':main()
