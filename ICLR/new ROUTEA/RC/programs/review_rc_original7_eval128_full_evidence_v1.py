#!/usr/bin/env python3
"""Post-completion independent128 review: frozen head, all actions, absent targets, groups; no encoder/RoMa."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_original7_eval128_full_evidence_v1'
AUTH=ROOT/'registry/rc_original7_eval128_full_evidence_execution_authority_v2_compat_20260910.json'
AUTH_SHA='bc671f19d6b975b96e1e6541ac11356889bac9f8e9f138f8977520b9a888a519'
PARENT=ROOT/'registry/rc_original7_eval128_full_evidence_execution_authority_v1_20260910.json'
PARENT_SHA='95a15a6d83a53e70ca2e2750344f42c4c2483f0e9d454260835a2961f1996b6c'
CPU=ROOT/'programs/run_rc_original7_eval128_full_evidence_v2_compat.py'
CPU_SHA='806a8506693e2d9e88aba4331702c2d2c8df58e4d3f693b9b30e0601de49304c'
COMMON=ROOT/'programs/rc_original7_eval128_execution_common_v2_compat.py'
COMMON_SHA='b624c30d3d156aa28ba6c2193447018be846159defb726ab5ca48599cdca36d3'
COHORT=ROOT/'results/rc_original7_expanded_eval128_manifest_v1'
WORKER=COHORT/'worker_manifest.json';CURATOR=COHORT/'curator_roles.json';META=COHORT/'independent_metadata_validation.json'
HEAD=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
RAW_BRIDGE=ROOT/'results/rc_original7_eval128_token_raw_v1/bridge'
RAW_BRIDGE_VALIDATION_SHA='0f9bc42fecce900bf845b06ba1e21f17f09515217143c70d416da48d4459e557'
REPORT=ROOT/'reports/REPORT_ORIGINAL7_EVAL128_INDEPENDENT_REVIEW_V1_20260910.md'
PARAMETER_SHA='ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'
PINS={WORKER:'f2c2e6f9181e103a61a181edf43e3309aa1de72743cfa6cf6097a9deb108e4e7',CURATOR:'47277ce66f7aba445fb80a93643039c7e8a47c1f5cc8237f60157a98e3e93381',META:'a70936a24fc2f5c6996228aee673a6de505c3022487b903878d268ad9a34cd94',HEAD:'42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b',ROOT/'src/rc_aslo_xf/gallery_identity_repair.py':'995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d',ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json':'867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650',ROOT/'registry/gallery_identity_repair_v1.json':'9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f'}
CHANGED={'token_raw_program','token_raw_common_program','token_raw_bridge_launcher','token_raw_shard_launcher','token_raw_preflight','roma_program','roma_bridge_launcher','roma_shards_launcher','roma_preflight','cpu_program','cpu_preflight','cpu_launcher_raw_aggregate','cpu_launcher_finalize_prejoin','cpu_launcher_join_validated','execution_common_program'}
ZERO={'DROP_RAW':[0],'DROP_S':[1],'DROP_M':[2],'DROP_L':[3],'DROP_Q':[4],'DROP_R':[5],'DROP_QR':[4,5],'DROP_S_QR':[1,4,5]}
CONDITIONS=('REAL','CBIND',*ZERO);FEATURES=('RAW','S','M','L','Q','R');C4=('real_score','visibility_mass','query_control_score','reference_control_score')
RELEASED=False;HASH_DEPTH=0;BLOCKED=[]

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
 if any(x in s for x in ('/rc_opened_eval_strict_','direction_capacity','angle_capacity','d1_mi','d1-mi','/grozi/','gisc_prerecall_universe')):
  BLOCKED.append(str(p));raise PermissionError('PROTECTED_OR_ORACLE_INPUT')
 if not RELEASED and not HASH_DEPTH and p in {CURATOR.resolve(),META.resolve(),(OUT/'result.json').resolve()}:
  BLOCKED.append(str(p));raise PermissionError('PRIVATE_OUTCOMES_BEFORE_REVIEW_PREJOIN_CLOSURE')
sys.addaudithook(audit)
def sha(p):
 global HASH_DEPTH
 HASH_DEPTH+=1
 try:
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for c in iter(lambda:f.read(8<<20),b''):h.update(c)
  return h.hexdigest()
 finally:HASH_DEPTH-=1
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
 need(sha(p)==v['sha256'],'SOURCE_BINDING:'+str(p));return p
def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def hx(v):return float(v).hex()
def eq(a,b,path):
 if isinstance(a,dict):
  need(isinstance(b,dict) and set(a)==set(b),'DICT_SCHEMA:'+path)
  for k in a:eq(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list):
  need(isinstance(b,list) and len(a)==len(b),'LIST_SCHEMA:'+path)
  for i,(x,y) in enumerate(zip(a,b)):eq(x,y,path+'/'+str(i))
 elif isinstance(a,float):need(isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(float(b)) and abs(a-float(b))<=2e-15,'METRIC_FLOAT:'+path)
 else:need(type(a)==type(b) and a==b,'EXACT_VALUE:'+path)
def append(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():need(p.read_bytes()==data,'APPEND_ONLY_REVIEW_DRIFT');return
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 p.chmod(0o444)

def source_lineage(authority,qualification):
 need(sha(PARENT)==PARENT_SHA and authority['parent_execution_authority']==bind(PARENT),'PARENT95_AUTHORITY_REUSED_WITHOUT_RELABELLING')
 parent=read(PARENT);unchanged=set(parent['source_bindings'])-CHANGED
 need(unchanged==set(authority['unchanged_parent_binding_names']),'EXACT_PARENT_SCIENTIFIC_SOURCE_COVERAGE')
 for name in unchanged:need(authority['source_bindings'].get(name)==parent['source_bindings'][name],'SCIENTIFIC_INPUT_CHANGED:'+name)
 reused={k:bind(RAW_BRIDGE/f) for k,f in [('payload','payload.pt'),('receipt','receipt.json'),('validation','validation.json')]}
 need(authority['reused_token_raw_bridge']==reused and reused['validation']['sha256']==RAW_BRIDGE_VALIDATION_SHA,'EXACT_COMPLETED_PARENT_RAW_BRIDGE')
 validation=read(RAW_BRIDGE/'validation.json');receipt=read(RAW_BRIDGE/'receipt.json')
 need(validation['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_PASS' and validation['authority']==receipt['authority']==bind(PARENT),'RAW_BRIDGE_AUTHORITY_STAYS95')
 need(validation['payload']==receipt['payload']==reused['payload'] and validation['receipt']==reused['receipt'],'RAW_BRIDGE_PAYLOAD_RECEIPT_BINDINGS')
 need(receipt['query_count']==1 and receipt['status']=='RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_READY' and all(x is True for x in receipt['bridge_checks'].values()),'RAW_BRIDGE_PARITY')
 current=bind(AUTH);shards=[]
 need(len(qualification['raw_shards'])==len(qualification['roma_shards'])==16,'ALL16_STAGE_SHARDS')
 for kind,key,status in [('RAW','raw_shards','RC_ORIGINAL7_EVAL128_TOKEN_RAW_CPU_REPLAY_PASS'),('ROMA','roma_shards','RC_ORIGINAL7_EVAL128_ROMA_CPU_REPLAY_PASS')]:
  need([s['shard'] for s in qualification[key]]==list(range(16)),'EXACT_STAGE_SHARD_AXIS')
  for s in qualification[key]:
   payload=checked(s['payload']);rp=checked(s['receipt']);vp=checked(s['validation']);r=read(rp);v=read(vp)
   need(v['status']==status and r['authority']==v['authority']==current,'NEW_SHARDS_BIND_CURRENT_V2_AUTHORITY')
   need(checked(v['payload'])==checked(r['payload'])==payload and checked(v['receipt'])==rp,'CURRENT_SHARD_PAYLOAD_VALIDATION')
   need(r['query_count']==v['query_count']==8,'EIGHT_ANONYMOUS_QUERIES_PER_SHARD')
   if kind=='RAW':need(v['fresh_CPU_process'] is True and all(x is True for x in v['checks'].values()),'RAW_FRESH_VALIDATION')
   else:need(v['candidate_occurrence_count']==1024 and v['independently_recomputed_C4_scalars']==4096 and v['all_C4_bits_exact'] is True and v['target_role_read_count']==v['model_update_count']==0,'ROMA_ALL_C4_SOURCE_QUALIFICATION')
   shards.append({'kind':kind,**s})
 return {'parent_authority':bind(PARENT),'current_authority':current,'reused_RAW_bridge':reused,'unchanged_scientific_binding_count':len(unchanged),'all32_current_shards':shards,'RAW_parent_provenance_preserved':True}

def gallery_labels():
 sys.path.insert(0,str(ROOT/'src'))
 from rc_aslo_xf.gallery_identity_repair import build_identity_map
 paths=[];images=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images'
 for directory,dirs,names in os.walk(images):
  dirs.sort()
  paths.extend(Path(directory)/n for n in sorted(names) if (Path(directory)/n).is_file() and Path(n).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})
 need(len(paths)==5413,'PHYSICAL_GALLERY5413')
 mapping=build_identity_map([p.stem.strip() for p in paths]);need(len(set(mapping.labels))==5412,'CORRECTED_IDENTITY5412')
 return mapping.labels,mapping.corrected_row_identity_mapping_sha256

def independent_feature_matrix(row,mode):
 import torch
 raw=[float.fromhex(v) for v in row['candidate_raw_scores_binary64']];n=len(raw);need(n==128,'C128_RAW_AXIS')
 mean=sum(raw)/n;std=(sum((v-mean)**2 for v in raw)/n)**.5
 evidence={int(k):{f:float.fromhex(v) for f,v in e.items()} for k,e in row['C4_binary64'].items()}
 need(set(evidence)==set(range(128)) and all(set(e)==set(C4) and all(math.isfinite(v) for v in e.values()) for e in evidence.values()),'QUALIFIED_C4_RECORD_DOMAIN')
 donor=[(i+64)%128 for i in range(128)] if mode=='CBIND' else list(range(128));winner=row['base_winner_position'];out=[]
 def v(pos):
  e=evidence[donor[pos]];return (e['real_score'],e['visibility_mass'],e['real_score']/max(e['visibility_mass'],1e-12),e['real_score']-e['query_control_score'],e['real_score']-e['reference_control_score'])
 base=v(winner)
 for c in row['challenger_positions']:
  own=v(c);contrasts=[(x-y)/(abs(x)+abs(y)+1e-12) for x,y in zip(own,base)]
  out.append([(raw[c]-raw[winner])/max(std,1e-12),*contrasts])
 return torch.tensor(out,dtype=torch.float64)

def independent_prediction(row,logits):
 values=list(map(float,logits));axis=row['candidate_physical_rows'];cs=row['challenger_positions']
 need(len(values)==127 and all(math.isfinite(x) for x in values),'ALL127_FINITE_LOGITS')
 # Sort whole challenger axis instead of sharing producer argmax implementation.
 best_index=sorted(range(127),key=lambda i:(-values[i],axis[cs[i]]))[0];best=cs[best_index]
 switch=values[best_index]>0.;final=best if switch else row['base_winner_position']
 return {'decision':'SWITCH' if switch else 'HOLD','proposed_challenger':best,'proposed_physical_row':axis[best],
  'max_logit_binary64':hx(values[best_index]),'final_position':final,'final_physical_row':axis[final],
  'all127_logits_binary64':[hx(v) for v in values]}

def rebuild_raw_and_predictions(records,qualification,labels,w,b):
 import torch
 byid={r['query_id']:r for r in records};need(len(records)==len(byid)==128,'ALL128_UNIQUE')
 need([r['execution_ordinal'] for r in records]==list(range(128)),'OPAQUE_EXECUTION_AXIS')
 # Read only already-completed scalar RAW payload fields; mmap leaves token banks unused.
 raw_ids=set()
 for s in qualification['raw_shards']:
  payload=torch.load(checked(s['payload']),map_location='cpu',mmap=True,weights_only=True)
  for r in payload['records']:
   q=r['query_id'];need(q in byid and q not in raw_ids,'RAW_QUERY_COVERAGE');raw_ids.add(q);saved=byid[q]
   physical=r['raw_physical_scores'];need(physical.dtype==torch.float64 and physical.shape==(5413,) and bool(torch.isfinite(physical).all()),'RAW_PHYSICAL_SCORE_DOMAIN')
   rank=[];seen=set()
   for p in sorted(range(5413),key=lambda p:(-float(physical[p]),p)):
    if labels[p] not in seen:seen.add(labels[p]);rank.append(p)
   need(len(rank)==5412 and rank==r['raw_ranked_physical_rows']==saved['raw_ranked_physical_rows'],'INDEPENDENT_FULL5412_RAW_RANK')
   need([hx(physical[p]) for p in rank]==saved['raw_ranked_scores_binary64'],'FULL_RANKED_SCORES')
   axis=sorted(rank[:128]);need(axis==saved['candidate_physical_rows']==r['candidate_physical_rows'],'NATURAL_C128_NO_INSERTION')
   need([hx(physical[p]) for p in axis]==saved['candidate_raw_scores_binary64'],'CANONICAL_RAW_VECTOR_BITS')
   need(saved['base_winner_position']==axis.index(rank[0]) and saved['challenger_positions']==[i for i in range(128) if i!=axis.index(rank[0])],'RAW_WINNER_AND_ALL127_CHALLENGERS')
   need(saved['query_tokens_sha256']==r['query_tokens_sha256'] and saved['source_image_sha256']==r['query_source_sha256'],'RAW_QUERY_SOURCE_METADATA')
 need(raw_ids==set(byid),'ALL128_RAW_REPLAYED')
 count=0
 for row in records:
  need(set(row['predictions'])==set(CONDITIONS) and row['C_BIND_source_positions']==[(i+64)%128 for i in range(128)],'TEN_CONDITIONS_AND_WHOLE_EVIDENCE_BINDING')
  matrices={mode:independent_feature_matrix(row,mode) for mode in ('REAL','CBIND')}
  for mode,x in matrices.items():need([[hx(v) for v in r] for r in x]==row['features_binary64'][mode],'ORIGINAL6_FEATURE_BITS:'+mode)
  need([[hx(v) for v in r] for r in matrices['REAL']*w]==row['REAL_all127_signed_terms_binary64'],'REAL_SIGNED_SIX_TERM_BITS')
  for condition in CONDITIONS:
   x=matrices['CBIND'] if condition=='CBIND' else matrices['REAL']
   if condition in ZERO:x=x.clone();x[:,ZERO[condition]]=0.
   values=x@w+b;expected=independent_prediction(row,values)
   need(expected==row['predictions'][condition],'ALL162560_LOGITS_AND1280_TARGET_FREE_ACTIONS_BIT_EXACT')
   count+=len(values)
 need(count==162560,'FULL_LOGIT_COUNT')
 return count

def action(row,pred,role,labels):
 axis=row['candidate_physical_rows'];cs=row['challenger_positions'];raw_rank=row['raw_ranked_physical_rows'];target=role['identity'];winner=row['base_winner_position']
 rank_labels=[labels[p] for p in raw_rank];need(len(rank_labels)==len(set(rank_labels))==5412 and rank_labels.count(target)==1,'ONE_TARGET_IN_FULL_GALLERY')
 raw_position=rank_labels.index(target);inside=[i for i,p in enumerate(axis) if labels[p]==target];need(len(inside)<=1,'IDENTITY_UNIQUE_C128')
 target_pos=inside[0] if inside else None;z=[float.fromhex(v) for v in pred['all127_logits_binary64']]
 i=sorted(range(127),key=lambda k:(-z[k],axis[cs[k]]))[0];best=cs[i];switch=z[i]>0.;final=best if switch else winner
 ordered=list(raw_rank)
 if switch:ordered.remove(axis[best]);ordered.insert(0,axis[best])
 final_rank=next(j for j,p in enumerate(ordered) if labels[p]==target)+1
 need(pred['final_position']==final and pred['decision']==('SWITCH' if switch else 'HOLD'),'SEALED_ACTION_MATCH')
 bc=labels[axis[winner]]==target;fc=labels[axis[final]]==target
 if target_pos is None:target_logit=None;threshold=None;comp=None;kind='CANDIDATE_RECALL_MISS'
 elif target_pos==winner:target_logit=None;threshold=-max(z);comp=None;kind='RAW_CORRECT_HELD' if fc else 'RAW_CORRECT_BROKEN'
 else:
  ti=cs.index(target_pos);target_logit=z[ti];threshold=z[ti];comp=z[ti]-max(v for j,v in enumerate(z) if cs[j]!=target_pos)
  kind='RAW_WRONG_RESCUED' if fc else 'TARGET_CANNOT_TRIGGER_SWITCH' if threshold<=0 else 'TARGET_LOSES_CHALLENGER_COMPETITION'
 wrong=[j for j,c in enumerate(cs) if c!=target_pos];wi=sorted(wrong,key=lambda k:(-z[k],axis[cs[k]]))[0]
 return {'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'original_query_id':role['original_query_id'],
  'identity':target,'group':role['group'],'track':row['track'],'candidate_recall':bool(inside),'target_position':target_pos,
  'target_physical_row_in_full_rank':raw_rank[raw_position],'RAW_winner_physical_row':axis[winner],
  'proposed_challenger':best,'proposed_physical_row':axis[best],'switch_logit':z[i],'decision':'SWITCH' if switch else 'HOLD',
  'final_position':final,'final_physical_row':axis[final],'base_correct':bc,'final_correct':fc,
  'wrong_to_wrong':not bc and not fc and final!=winner,'raw_target_rank_full_gallery':raw_position+1,'final_target_rank_full_gallery':final_rank,
  'target_challenger_logit':target_logit,'RAW_winner_policy_score':0.,'switch_or_HOLD_protection_margin':threshold,
  'target_vs_strongest_wrong_margin':comp,'strongest_wrong_challenger':cs[wi],'strongest_wrong_physical_row':axis[cs[wi]],
  'strongest_wrong_logit':z[wi],'outcome_kind':kind}

def metrics(aa):
 n=len(aa);base=sum(a['base_correct'] for a in aa);final=sum(a['final_correct'] for a in aa);recall=sum(a['candidate_recall'] for a in aa)
 return {'query_count':n,'candidate_recall_count':recall,'candidate_recall_at128':recall/n if n else None,
  'RAW_top1':base,'final_top1':final,'RAW_accuracy':base/n if n else None,'final_accuracy':final/n if n else None,
  'rescue':sum(not a['base_correct'] and a['final_correct'] for a in aa),'break':sum(a['base_correct'] and not a['final_correct'] for a in aa),'net':final-base,
  'RAW_full_gallery_MRR':float(sum((Fraction(1,a['raw_target_rank_full_gallery']) for a in aa),Fraction())/n) if n else None,
  'final_full_gallery_MRR':float(sum((Fraction(1,a['final_target_rank_full_gallery']) for a in aa),Fraction())/n) if n else None,
  'switch_count':sum(a['decision']=='SWITCH' for a in aa),'wrong_to_wrong_count':sum(a['wrong_to_wrong'] for a in aa)}

def compare(aa,baseline):
 need([a['query_id'] for a in aa]==[a['query_id'] for a in baseline],'PAIRED_QUERY_ORDER')
 gain=[a['query_id'] for a,b in zip(aa,baseline) if a['final_correct'] and not b['final_correct']]
 loss=[a['query_id'] for a,b in zip(aa,baseline) if b['final_correct'] and not a['final_correct']]
 rr=sum((Fraction(1,a['final_target_rank_full_gallery'])-Fraction(1,b['final_target_rank_full_gallery']) for a,b in zip(aa,baseline)),Fraction())/len(aa)
 return {'rescue':len(gain),'break':len(loss),'net':len(gain)-len(loss),'rescue_query_ids':gain,'break_query_ids':loss,'full_gallery_MRR_difference':float(rr)}

def signflip_independent(groups):
 values=[Fraction(g['net'],g['query_count']) for g in groups];scale=math.lcm(*(v.denominator for v in values)) if values else 1
 signed=[int(v*scale) for v in values if v];weights=[abs(v) for v in signed];observed=abs(sum(signed));total=sum(weights)
 # Independent subset-sum polynomial: each subset maps to2*subset_sum-total, instead of producer +/- recursion.
 counts={0:1}
 for weight in weights:
  next_counts=defaultdict(int)
  for s,n in counts.items():next_counts[s]+=n;next_counts[s+weight]+=n
  counts=dict(next_counts)
 denominator=2**len(weights);numerator=sum(n for s,n in counts.items() if abs(2*s-total)>=observed)
 need(sum(counts.values())==denominator,'EXACT_ALL_SIGN_CONFIGURATIONS')
 return {'two_sided_p':numerator/denominator,'p_numerator':numerator,'p_denominator':denominator,'nonzero_groups':len(weights),
  'total_groups':len(groups),'integer_scale_LCM':scale,'observed_absolute_scaled_sum':observed}

def group_statistics(aa,expected):
 import numpy as np
 grouped={}
 for a in aa:
  g=grouped.setdefault(a['group'],{'group':a['group'],'query_count':0,'RAW_correct':0,'REAL_correct':0})
  g['query_count']+=1;g['RAW_correct']+=int(a['base_correct']);g['REAL_correct']+=int(a['final_correct'])
 groups=[grouped[k] for k in sorted(grouped)]
 for g in groups:g['net']=g['REAL_correct']-g['RAW_correct'];g['accuracy_difference']=g['net']/g['query_count']
 need(len(groups)==21 and sum(g['query_count'] for g in groups)==128,'GROUP21_IMAGE128')
 eq(groups,expected['groups'],'SOURCE_GROUP_ROWS')
 mean=float(sum((Fraction(g['net'],g['query_count']) for g in groups),Fraction())/len(groups))
 need(mean==expected['equal_group_mean_accuracy_difference'],'EXACT_GROUP_PAIRED_MEAN')
 test=signflip_independent(groups)
 for k,v in test.items():need(v==expected['two_sided_group_signflip'][k],'INDEPENDENT_SIGNFLIP:'+k)
 values=np.array([g['accuracy_difference'] for g in groups],dtype=np.float64);rng=np.random.default_rng(20260910)
 chosen=rng.integers(0,len(values),size=(10000,len(values)));samples=values[chosen].mean(axis=1);ci=np.quantile(samples,[.025,.975],method='linear')
 boot={'seed':20260910,'draws':10000,'sampling':'source groups with replacement, all group means equally weighted','interval_level':.95,
  'percentile_method':'numpy.quantile linear','lower':float(ci[0]),'upper':float(ci[1]),'numpy_version':np.__version__,
  'bit_generator':type(rng.bit_generator).__name__,'samples_binary64_sha256':hashlib.sha256(samples.tobytes()).hexdigest()}
 need(boot==expected['group_bootstrap'],'SAME_NUMPY_SEEDED_BOOTSTRAP_BIT_REPLAY')
 return {'groups':groups,'group_count':21,'equal_group_mean_accuracy_difference':mean,'two_sided_group_signflip':test,'group_bootstrap':boot,
  'signflip_reviewer_algorithm':'integer subset-sum DP; no Monte Carlo approximation','bootstrap_scope':'same NumPy seeded implementation replay, not an independent population replication'}

def six_ledger(row,a,w,b):
 x=[[float.fromhex(v) for v in r] for r in row['features_binary64']['REAL']];z=[float.fromhex(v) for v in row['predictions']['REAL']['all127_logits_binary64']]
 cs=row['challenger_positions'];axis=row['candidate_physical_rows'];wi=cs.index(a['strongest_wrong_challenger']);target=a['target_position'];winner=row['base_winner_position']
 def point(i):
  terms=[Fraction.from_float(x[i][k])*Fraction.from_float(float(w[k])) for k in range(6)]
  exact=sum(terms,Fraction.from_float(b));residual=Fraction.from_float(z[i])-exact
  return {'candidate_position':cs[i],'physical_row':axis[cs[i]],'six_terms':{n:float(terms[k]) for k,n in enumerate(FEATURES)},'bias':b,'logit':z[i],'FP64_minus_exact_affine_roundoff':float(residual)}
 wrong=point(wi)
 value={'query_id':row['query_id'],'execution_ordinal':row['execution_ordinal'],'candidate_recall':a['candidate_recall'],'outcome_kind':a['outcome_kind'],
  'selected_challenger_terms':point(cs.index(a['proposed_challenger'])),'strongest_wrong_terms':wrong,'target_challenger_terms':None,
  'target_minus_wrong_six_terms':None,'bias_cancels_in_challenger_comparison':True,'RAW_winner_policy_score':0.,
  'threshold_or_HOLD_protection_margin':a['switch_or_HOLD_protection_margin'],'target_vs_wrong_margin':a['target_vs_strongest_wrong_margin']}
 if target is not None and target!=winner:
  ti=cs.index(target);value['target_challenger_terms']=point(ti)
  value['target_minus_wrong_six_terms']={n:float((Fraction.from_float(x[ti][k])-Fraction.from_float(x[wi][k]))*Fraction.from_float(float(w[k]))) for k,n in enumerate(FEATURES)}
 elif target==winner:value['HOLD_protection_six_terms']={n:-wrong['six_terms'][n] for n in FEATURES};value['HOLD_protection_bias']=-b
 else:value['absence_note']='Target absent from original naturalC128; no target logit or invented target feature is computed.'
 return value


def main():
 global RELEASED
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review-validated-result',action='store_true',required=True);p.parse_args()
 vp=OUT/'validation.json';need(vp.is_file(),'EXPANDED128_CPU_POSTJOIN_VALIDATION_NOT_READY')
 validation=read(vp)
 need(validation['status']=='ORIGINAL7_EVAL128_POSTJOIN_LITERAL_REPLAY_PASS' and all(v is True for v in validation['checks'].values()),'COMPLETED_CPU_VALIDATION_REQUIRED')
 need(checked(validation['result'])==(OUT/'result.json').resolve() and validation['prejoin_logit_checks']==162560 and validation['fitted_models']==0,'VALIDATION_RESULT_AND_COUNTS')
 need(sha(AUTH)==AUTH_SHA and sha(CPU)==CPU_SHA and sha(COMMON)==COMMON_SHA,'FROZEN_V2_CONTROL_PINS')
 for path,digest in PINS.items():need(sha(path)==digest,'FROZEN_COHORT_HEAD_IDENTITY_PIN:'+path.name)
 authority=read(AUTH);need(authority['status']=='ORIGINAL7_EVAL128_FULL_EVIDENCE_AUTHORIZED' and authority['query_count']==128 and authority['primary_model']=='ORIGINAL7_NATIVE7','CURRENT_AUTHORITY_SCOPE')
 need(checked(authority['source_bindings']['cpu_program'])==CPU.resolve() and checked(authority['source_bindings']['execution_common_program'])==COMMON.resolve(),'CURRENT_PROGRAM_SOURCE_BINDINGS')
 seal=read(OUT/'prejoin_seal.json');preval=read(OUT/'prejoin_validation.json');qualification=read(OUT/'input_qualification.json')
 need(seal['status']=='ORIGINAL7_EVAL128_ALL_PREDICTIONS_SEALED_BEFORE_CURATOR' and preval['status']=='ORIGINAL7_EVAL128_PREJOIN_INDEPENDENT_CPU_REPLAY_PASS','PREJOIN_QUALIFICATION_STATUS')
 for key,name in [('records','prejoin_records.json'),('parameters','prejoin_parameters.json'),('input_qualification','input_qualification.json')]:need(checked(seal[key])==checked(preval[key])==(OUT/name).resolve(),'ALL_PREJOIN_COMPONENT_BINDINGS')
 need(checked(preval['seal'])==(OUT/'prejoin_seal.json').resolve() and checked(seal['authority'])==AUTH.resolve(),'PREJOIN_CURRENT_AUTHORITY')
 need(seal['query_count']==128 and seal['condition_names']==list(CONDITIONS) and seal['all127_logit_count']==162560 and seal['head_parameter_sha256']==PARAMETER_SHA and seal['curator_ledger_sha256']==PINS[CURATOR],'FROZEN_ALL128_TEN_CONDITIONS')
 need(seal['curator_payload_reads']==seal['forbidden_read_attempts']==0 and all(v is True for v in preval['checks'].values()),'PREJOIN_ACCESS_AND_QUALIFICATION')
 need(qualification['status']=='ORIGINAL7_EVAL128_ALL_INPUTS_CPU_QUALIFIED' and qualification['query_count']==128 and qualification['candidate_occurrences']==16384 and qualification['independent_C4_scalar_checks']==65536 and qualification['head_predictions_computed_at_this_gate']==qualification['curator_payload_reads']==0,'ALL65536_C4_SOURCE_VALIDATED_BEFORE_HEAD')
 lineage=source_lineage(authority,qualification)
 import torch,numpy as np
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 pref=read(checked(authority['source_bindings']['cpu_preflight']))
 need(pref['runtime']['numpy']==np.__version__ and pref['runtime']['torch']==str(torch.__version__),'REVIEW_CPU_NUMERICAL_RUNTIME')
 head=read(HEAD);parameters=read(OUT/'prejoin_parameters.json');w=torch.tensor([float.fromhex(v) for v in head['weight_binary64']],dtype=torch.float64);b=float.fromhex(head['bias_binary64'])
 need(len(w)==6 and head['parameter_sha256']==PARAMETER_SHA==hashlib.sha256(encode({'weight':list(map(float,w)),'bias':b})).hexdigest(),'CURRENT_NATIVE7_PARAMETER_DIGEST')
 need(parameters=={'source':bind(HEAD),'parameter_sha256':PARAMETER_SHA,'weight_binary64':head['weight_binary64'],'bias_binary64':head['bias_binary64'],'parameter_count':7},'ONLY_FROZEN_CURRENT_SEVEN_PARAMETERS')
 records=read(OUT/'prejoin_records.json');labels,mapping=gallery_labels();logit_checks=rebuild_raw_and_predictions(records,qualification,labels,w,b)
 private=[v for v in authority['postjoin_source_bindings'].values() if Path(v['path']).resolve()==CURATOR.resolve()]
 need(len(private)==1 and private[0]==bind(CURATOR),'EXACT_PRIVATE_CURATOR_BINDING')
 need(not BLOCKED and not RELEASED,'NO_PRIVATE_JOIN_BEFORE_INDEPENDENT_PREDICTION_CLOSURE');RELEASED=True
 result=read(OUT/'result.json');curator=read(CURATOR);meta=read(META);worker=read(WORKER)
 need(result['authority']==bind(AUTH) and result['program']==bind(CPU) and result['prejoin_seal']==bind(OUT/'prejoin_seal.json') and result['prejoin_validation']==bind(OUT/'prejoin_validation.json'),'FINAL_RESULT_PROVENANCE')
 need(result['curator_ledger']==bind(CURATOR) and result['curator_metadata_validation']==bind(META) and result['worker_manifest']==bind(WORKER),'FINAL_COHORT_PROVENANCE')
 need(curator['status']=='CURATOR_ONLY_EXPANDED_EVAL128_ROLES_FROZEN' and meta['status']=='EXPANDED_EVAL128_INDEPENDENT_METADATA_SELECTION_PASS' and checked(meta['curator_ledger'])==CURATOR.resolve() and checked(meta['worker_manifest'])==WORKER.resolve(),'INDEPENDENT_COHORT_SELECTION')
 roles={r['query_id']:r for r in curator['records']};workers={r['query_id']:r for r in worker['records']}
 need(len(roles)==len(workers)==len(curator['records'])==len(worker['records'])==128 and set(roles)==set(workers)=={r['query_id'] for r in records},'ALL128_JOIN_WITHOUT_FILTERING')
 need(len({r['identity'] for r in roles.values()})==24 and len({r['group'] for r in roles.values()})==21 and len({r['source_image_sha256'] for r in roles.values()})==128,'128_IMAGES_24_IDENTITIES_21_GROUPS')
 need(result['query_count']==128 and result['identity_count']==24 and result['source_group_count']==21 and result['corrected_gallery_mapping_sha256']==mapping and result['parameter_sha256']==PARAMETER_SHA,'RESULT_POPULATION_AND_PARAMETER_LINEAGE')
 acts={c:[] for c in CONDITIONS}
 for row in records:
  role=roles[row['query_id']];wr=workers[row['query_id']]
  need(role['execution_ordinal']==wr['execution_ordinal']==row['execution_ordinal'] and role['track']==wr['track']==row['track'] and role['source_image_sha256']==wr['source_image_sha256']==row['source_image_sha256'],'CURATOR_QUERY_AND_IMAGE_MATCH')
  for condition in CONDITIONS:acts[condition].append(action(row,row['predictions'][condition],role,labels))
 need(set(result['actions'])==set(CONDITIONS),'ALL_FIXED_CONDITIONS_REPORTED')
 for c in CONDITIONS:need(encode(acts[c])==encode(result['actions'][c]),'ALL1280_ACTIONS_EXACT:'+c)
 all_metrics={c:metrics(a) for c,a in acts.items()};conditional={c:metrics([a for a in aa if a['candidate_recall']]) for c,aa in acts.items()}
 eq(all_metrics,result['metrics_all128'],'ALL128_METRICS');eq(conditional,result['metrics_conditional_target_in_C128'],'CONDITIONAL_METRICS_NEVER_REPLACE_ALL128')
 contrasts={c:compare(acts[c],acts['REAL']) for c in CONDITIONS if c!='REAL'};eq(contrasts,result['paired_controls_vs_REAL'],'FIXED_CONTROL_PAIRED_CONTRASTS')
 rescues=[a['query_id'] for a in acts['REAL'] if not a['base_correct'] and a['final_correct']];breaks=[a['query_id'] for a in acts['REAL'] if a['base_correct'] and not a['final_correct']]
 absent=[a['query_id'] for a in acts['REAL'] if not a['candidate_recall']]
 need(rescues==result['primary_rescue_query_ids'] and breaks==result['primary_break_query_ids'] and absent==result['candidate_recall_miss_query_ids'],'PRIMARY_RESCUE_BREAK_RECALL_MISS_SETS')
 for c in CONDITIONS:
  for a in acts[c]:
   if not a['candidate_recall']:need(a['target_challenger_logit'] is None and a['target_vs_strongest_wrong_margin'] is None and not a['base_correct'] and not a['final_correct'] and a['final_target_rank_full_gallery']==a['raw_target_rank_full_gallery'],'ABSENT_TARGET_RETAINED_AS_MISS_FULL_GALLERY_RANK_UNCHANGED')
 retention={}
 for c in CONDITIONS:
  correct={a['query_id'] for a in acts[c] if a['final_correct']}
  retention[c]={'REAL_RAW_rescue_count':len(rescues),'retained_query_ids':[q for q in rescues if q in correct],'lost_query_ids':[q for q in rescues if q not in correct]}
 need(retention==result['REAL_rescue_retention'],'ALL_CONDITION_RESCUE_RETENTION')
 stats=group_statistics(acts['REAL'],result['primary_group_statistics'])
 ledger=[six_ledger(r,a,w,b) for r,a in zip(records,acts['REAL'],strict=True)];eq(ledger,result['six_feature_decision_ledger'],'INDEPENDENT_SIX_TERM_AND_MARGIN_LEDGER')
 real=all_metrics['REAL'];net=real['net']>0 and stats['equal_group_mean_accuracy_difference']>0;preserved=net and real['break']==0
 need(result['internal_gain_and_preservation']==preserved and result['internal_net_gain_with_group_positive']==net,'NO_GOAL_REDEFINITION')
 status='ORIGINAL7_EVAL128_INTERNAL_GAIN_AND_PRESERVATION' if preserved else 'ORIGINAL7_EVAL128_INTERNAL_NET_GAIN_WITH_BREAKS' if net else 'ORIGINAL7_EVAL128_NO_INTERNAL_POSITIVE_PAIRED_GAIN'
 need(result['status']==status and not result['technical_or_metadata_errors'] and result['target_insertion_count']==result['target_absent_queries_dropped']==result['new_training_updates']==result['new_J_T_or_LP_calls']==0,'FINAL_RESULT_STATUS_AND_ERROR_BOUNDARY')
 need(result['deployment_changed'] is False and result['universal_HYP_claimed'] is False and result['ownership_claimed'] is False,'NO_AUTOMATIC_DEPLOYMENT_OR_CAUSAL_GO')
 value={'status':'ORIGINAL7_EVAL128_POST_RESULT_INDEPENDENT_REVIEW_PASS','reviewer':bind(Path(__file__).resolve()),
  'source_cpu_program':bind(CPU),'source_authority':bind(AUTH),'source_validation':bind(vp),'result':bind(OUT/'result.json'),
  'prejoin_seal':bind(OUT/'prejoin_seal.json'),'prejoin_validation':bind(OUT/'prejoin_validation.json'),'curator_ledger':bind(CURATOR),
  'worker_manifest':bind(WORKER),'curator_selection_validation':bind(META),'source_lineage':lineage,
  'query_count':128,'identity_count':24,'source_group_count':21,'source_group_image_counts':dict(sorted(Counter(a['group'] for a in acts['REAL']).items())),
  'independent_full5412_RAW_rank_replays':128,'independent_feature_conditions':1280,'independent_logit_bit_checks':logit_checks,'independent_action_checks':1280,
  'source_C4_scalar_checks_reused':65536,'new_RoMa_or_encoder_forwards':0,'new_training_updates':0,'new_LP_calls':0,
  'metrics_all128':all_metrics,'metrics_conditional_target_in_C128':conditional,'primary_group_statistics':stats,
  'primary_rescue_query_ids':rescues,'primary_break_query_ids':breaks,'candidate_recall_miss_query_ids':absent,
  'paired_controls_vs_REAL':contrasts,'REAL_rescue_retention':retention,'internal_gain_and_preservation':preserved,
  'full_gallery_MRR_replayed_by_literal_move_to_front':True,'all_target_absent_queries_retained':True,
  'six_feature_threshold_competition_and_absent_target_ledger_rebuilt':True,'curator_opened_only_after_this_process_prediction_replay':True,
  'aggregate_float_comparison_absolute_tolerance':2e-15,'limitations':['Post-completion artifact replay cannot retroactively prove wall-clock chronology beyond source guards, bindings and counters.','Bootstrap replays the prespecified NumPy implementation; sign-flip is independently enumerated by subset-sum DP.','Identity/group-held-out source selection is not untouched external confirmation; fixed feature-zero controls are not physical causality.'],
  'report_path':str(REPORT)}
 append(OUT/'independent_review.json',encode(value)+b'\n')
 lines=['# ORIGINAL7新128图：独立复核补充','',
  f"独立复核通过：128图、24身份、21来源组；完整5412排序、1,280份六列条件矩阵、162,560个logits及1,280个动作均已重放。",'',
  f"Candidate recall@128：{real['candidate_recall_count']}/128。RAW {real['RAW_top1']}/128 → 冻结ORIGINAL7 {real['final_top1']}/128；救回{real['rescue']}、损失{real['break']}。",
  f"完整gallery MRR：{real['RAW_full_gallery_MRR']:.9f} → {real['final_full_gallery_MRR']:.9f}。缺席C128的{len(absent)}图保留为miss，没有补入target或伪造target logit。",'',
  f"21组等权差：{stats['equal_group_mean_accuracy_difference']:.9f}；预定10,000次组bootstrap95%区间[{stats['group_bootstrap']['lower']:.9f}, {stats['group_bootstrap']['upper']:.9f}]。",
  f"独立精确sign-flip：p={stats['two_sided_group_signflip']['two_sided_p']:.9g}（{stats['two_sided_group_signflip']['p_numerator']}/{stats['two_sided_group_signflip']['p_denominator']}）。保持原正确且新增：{preserved}。",'',
  '| 固定条件 | 正确/128 | 对RAW救/损 | 对REAL净变化 |','| --- | ---: | --- | ---: |']
 for c in CONDITIONS:
  m=all_metrics[c];change=0 if c=='REAL' else contrasts[c]['net'];lines.append(f"| {c} | {m['final_top1']} | {m['rescue']}/{m['break']} | {change:+d} |")
 lines+=['','旧RAW桥接仍保留V1/95授权来源；复用有明确V2授权，所有新分片绑定当前V2。没有改写旧产物或把旧桥接重新标成新执行。',
  '本复核不重复65,536项已资格化C4计算，不运行RoMa、encoder或训练；只独立复核冻结读出、完整排名和评价。',
  '来源目录历史已打开；组内相关、身份/组数量及条件子集均单列，不把128图当128独立身份，不自动宣称外部确认、空间因果或部署。',
  f"复核SHA：{sha(OUT/'independent_review.json')}。",'']
 append(REPORT,'\n'.join(lines).encode())
 print(json.dumps({'status':value['status'],'review':bind(OUT/'independent_review.json'),'report':bind(REPORT),'REAL':real,
  'group_mean':stats['equal_group_mean_accuracy_difference'],'group_signflip':stats['two_sided_group_signflip'],'gain_and_preservation':preserved},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
