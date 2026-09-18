#!/usr/bin/env python3
"""One new cached D arm; frozen-order old training and delayed EVAL postjoin."""
from __future__ import annotations
import argparse,ast,hashlib,importlib,importlib.util,json,math,os,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PREFIX='rc_full_mass_free_identity_development_v1'
AUTH=ROOT/'registry/rc_full_mass_free_identity_development_authority_v1_20260909.json'
PLAN=ROOT/'plan/RC_FULL_MASS_FREE_IDENTITY_DEVELOPMENT_V1_20260909.md'
LAUNCH=ROOT/'slurm/rc_full_mass_free_identity_development_v1_dev_cpuonly_59m.sbatch'
OUT=ROOT/('results/'+PREFIX)
FULL=ROOT/'results/routea_matched_three_arm_fullnegative_features_v1'
PAIR=ROOT/'results/routea_matched_three_arm_pair64_training_features_v2'
ROLE_ROOT=ROOT/'results/cw0_rgh_xf_v2_p0_a0_manifest_v2'
PARENT=ROOT/'results/routea_matched_three_arm_common3_native7_crossfit_v1'
D='D_FULL_MASS_FREE_IDENTITY';ARMS=('B_QUERY','C_PAIRED',D);FAMILIES=('COMMON3','NATIVE7');MODES=('REAL','CBIND','Q','R')
PINS={
 'old_runner':('programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py','546e2bc7b3df6c67bcac41e079ba1b00f15c7e65c52a8ab994cd5b2c38d81e22'),
 'frozen_head':('src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py','96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7'),
 'core':('src/rc_aslo_xf/reference_visibility_full_mass_free_identity_v1.py','04a5d664a358c8896072865dc623261697ee556a9f393d3f9cea79777e45bc78'),
 'lossless_core':('src/rc_aslo_xf/reference_visibility_pv_lossless_v1.py','f5fcfa1ce6fa6582a3628035e2ba54414185160fa36515fa27fd55d8aa81adcc'),
 'full_validation':(str((FULL/'validation.json').relative_to(ROOT)),'4f6c514b7e7d619d32cf312e06ee9a39494eb7ef4113aaebc45e10b38ce1c827'),
 'pair_validation':(str((PAIR/'independent_validation.json').relative_to(ROOT)),'657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b'),
 'pair_payload':(str((PAIR/'payload.pt').relative_to(ROOT)),'d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3'),
 'role_manifest':(str((ROLE_ROOT/'role_manifest.json').relative_to(ROOT)),'2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454'),
 'role_validation':(str((ROLE_ROOT/'independent_validation.json').relative_to(ROOT)),'ae735624176e5e400ff5c16874b7f73b0505ce71f6814d0c4ed515d94455f8ac'),
 'old_result':(str((PARENT/'result.json').relative_to(ROOT)),'591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3'),
 'old_validation':(str((PARENT/'independent_validation.json').relative_to(ROOT)),'1a7b08edfad7823db62f86c5f0b213106daead6f2ddba2d299102b69e546d8df'),
 'identity_manifest':('registry/gallery_identity_repair_v1.json','9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f')}
CONTRACT={'new_arm':D,'formula':'rho=FP64(C.M/B.M);D.M=C.M;D.real=B.real*rho;D.Q=B.Q*rho;D.R=B.R*rho',
 'new_method_variant_count':1,'families':list(FAMILIES),'baseline_retrains':['B_QUERY','C_PAIRED'],
 'updates_per_head':2000,'seed':17,'initialization':'FP64_LINEAR_ALL_ZERO','optimizer':'AdamW',
 'lr':.03,'weight_decay':.001,'betas':[.9,.999],'eps':1e-8,'amsgrad':False,'foreach':None,'fused':None,
 'train_order':'32_TRAIN_EXECUTION_ORDINAL_ASCENDING;PAIR64_ORIGINAL_PAYLOAD_ORDER;FULL_BATCH_EVERY_UPDATE',
 'early_stopping':False,'checkpoint_selection':False,'cpu_threads':8,
 'pair_loss':'MEAN_BCE_WITH_LOGITS_WITH_LABEL0_WEIGHT4_LABEL1_WEIGHT1',
 'query_loss':'BASE_CORRECT:4softplus(max_logits);BASE_WRONG:softplus(-target_logit)+4softplus(max_other_logits)',
 'total_loss':'pair_loss+mean_over_all32_query_losses','eval_query_count':32,'candidate_count':128,
 'postjoin':'TRAIN_ROLES_ONLY_BEFORE_FIT;ALL_HEADS_AND_ALL_EVAL_TARGET_FREE_LOGITS_SEALED_BEFORE_EVAL_ROLES',
 'old_result_read':'HASH_ONLY_UNTIL_EVAL_PREJOIN_SEAL','controls':list(MODES),
 'Q_R_controls':'OLD_FROZEN_PATH_CANDIDATES_RELABEL_REAL_SCORE_ONLY_MATCHING_ORIGINAL_CONTROL',
 'scientific_GO_or_NO_GO':None,'formal_panel_authorized':False,'deployment_replacement_authorized':False,'automatic_stage_advance':False}
BARRIER=None

def need(x,message):
    if not bool(x):raise RuntimeError(message)

def safe(path):
    path=Path(path).resolve();need(path.is_relative_to(ROOT),'SOURCE_OUTSIDE_ROOT')
    need(not any(x in str(path).lower() for x in ('d1_mi','d1-mi','d1_minimal_intervention','grozi','gisc_prerecall_universe')),'PROTECTED_SOURCE')
    return path

def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def logical(x):return hashlib.sha256(encode(x)).hexdigest()
def sha(path):
    global BARRIER
    path=safe(path);h=hashlib.sha256()
    if BARRIER is not None:BARRIER.hash_only+=1
    try:
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    finally:
        if BARRIER is not None:BARRIER.hash_only-=1
    return h.hexdigest()

def read(path):return json.loads(safe(path).read_text())
def binding(path):return {'path':str(Path(path).relative_to(ROOT)),'sha256':sha(path)}
def checked(v):
    path=safe(ROOT/v['path']);need(sha(path)==v['sha256'],'SOURCE_HASH_DRIFT:'+v['path']);return path

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=encode(value)+b'\n'
    if path.exists():need(path.read_bytes()==data,'IMMUTABLE_REPLAY_DRIFT:'+str(path));return
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.chmod(temp,0o444);os.link(temp,path)
    finally:os.unlink(temp)

def hx(x):return float(x).hex()
def tensor_sha(t):
    x=t.detach().cpu().contiguous();return hashlib.sha256(str(x.dtype).encode()+encode(list(x.shape))+x.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()

class ReadBarrier:
    def __init__(self,train_paths,eval_paths,outcome_paths):
        self.train_paths={str(Path(p).resolve()) for p in train_paths};self.eval_paths={str(Path(p).resolve()) for p in eval_paths}
        self.outcomes={str(Path(p).resolve()) for p in outcome_paths};self.released=False;self.hash_only=0;self.blocked=0;self.role_reads=[]
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=str(Path(os.fsdecode(args[0])).resolve())
        if path in self.outcomes and not self.released and not self.hash_only:self.blocked+=1;raise RuntimeError('OLD_OUTCOME_READ_BEFORE_EVAL_SEAL')
        if str(ROLE_ROOT/'role_shards') in path and path not in self.train_paths and (not self.released or path not in self.eval_paths):
            self.blocked+=1;raise RuntimeError('EVAL_OR_UNRELATED_ROLE_READ_BEFORE_SEAL')
    def release(self):self.released=True

def load_modules():
    import torch
    from torch import nn
    from torch.nn import functional as F
    for key in ('old_runner','frozen_head','core','lossless_core'):checked({'path':PINS[key][0],'sha256':PINS[key][1]})
    sys.path.insert(0,str(ROOT/'src'))
    frozen=importlib.import_module('rc_aslo_xf.romav2_colnomic_frozen_gate_v1')
    core=importlib.import_module('rc_aslo_xf.reference_visibility_full_mass_free_identity_v1')
    text=(ROOT/PINS['old_runner'][0]).read_text();tree=ast.parse(text)
    names={'finite_tensor','finite_scalar','parameter_sha','train_head','actions','summary','retention','feature_ledger'}
    body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    need({node.name for node in body}==names,'ORIGINAL_PURE_FUNCTIONS_MISSING')
    family_def={'COMMON3':('real_common_features','cbind_common_features',('standardized_raw_gap','symmetric_local_score')),
      'NATIVE7':('real_native_features','cbind_native_features',tuple(frozen.FEATURE_NAMES))}
    namespace={'torch':torch,'nn':nn,'F':F,'math':math,'hashlib':hashlib,'json':json,'STEPS':2000,
       'FAMILIES':family_def,'ARMS':ARMS,'CrossfitContractError':RuntimeError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[])),str(ROOT/PINS['old_runner'][0]),'exec'),namespace)
    return frozen,core,namespace

def freeze():
    need(not AUTH.exists(),'AUTHORITY_EXISTS')
    sources={k:{'path':p,'sha256':h} for k,(p,h) in PINS.items()}
    for v in sources.values():checked(v)
    sources.update(program=binding(Path(__file__)),plan=binding(PLAN),launcher=binding(LAUNCH),
      identity_core=binding(ROOT/'src/rc_aslo_xf/gallery_identity_repair.py'),
      identity_contract=binding(ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'))
    e0path=ROOT/('results/'+PREFIX+'_e0/result.json');atomic(e0path,synthetic_e0());sources['e0']=binding(e0path)
    a={'status':'RC_FULL_MASS_FREE_IDENTITY_DEVELOPMENT_AUTHORIZED','sources':sources,'contract':CONTRACT,
       'output_rel':str(OUT.relative_to(ROOT)),'scientific_GO_or_NO_GO':None,'automatic_stage_advance':False}
    atomic(AUTH,a);print(json.dumps({'status':a['status'],'authority_sha256':sha(AUTH)}),flush=True)

def authority():
    a=read(AUTH);need(a['status']=='RC_FULL_MASS_FREE_IDENTITY_DEVELOPMENT_AUTHORIZED' and a['contract']==CONTRACT
      and a['output_rel']==str(OUT.relative_to(ROOT)) and a['scientific_GO_or_NO_GO'] is None,'AUTHORITY_SCOPE')
    for v in a['sources'].values():checked(v)
    for k,(p,h) in PINS.items():need(a['sources'][k]=={'path':p,'sha256':h},'SOURCE_PIN_DRIFT')
    need(a['sources']['program']==binding(Path(__file__)) and a['sources']['plan']==binding(PLAN)
      and a['sources']['launcher']==binding(LAUNCH),'IMPLEMENTATION_BINDING')
    e0=read(checked(a['sources']['e0']));need(e0['status']=='RC_FULL_MASS_FREE_IDENTITY_RUNNER_E0_PASS'
       and e0['checks'] and all(x is True for x in e0['checks'].values()),'E0_NOT_CLOSED');return a

def transform_evidence(evidence,mode,source_positions=None):
    out={i:dict(evidence[source_positions[i] if mode=='CBIND' else i]) for i in evidence}
    if mode=='Q':
        for x in out.values():x['real_score']=x['query_control_score'];x['query_control_score']=x['real_score']
    if mode=='R':
        for x in out.values():x['real_score']=x['reference_control_score'];x['reference_control_score']=x['real_score']
    return out

def add_features(record,frozen,core,pair=False):
    import torch
    out=dict(record);out['evidence']={arm:dict(record['evidence'][arm]) for arm in ('B_QUERY','C_PAIRED')}
    out['evidence'][D]={i:core.combine_full_mass_with_free_identity(record['evidence']['B_QUERY'][i],record['evidence']['C_PAIRED'][i]['visibility_mass'])
       for i in record['evidence']['B_QUERY']}
    raw=record['base_scores'].tolist();winner=int(record['base_winner_position']);challengers=list(record['challenger_positions'])
    out['mode_features']={mode:{} for mode in (('REAL',) if pair else MODES)}
    for mode in out['mode_features']:
        out['mode_features'][mode]={'COMMON3':{},'NATIVE7':{}}
        for arm in ARMS:
            evidence=transform_evidence(out['evidence'][arm],mode,record.get('cbind_source_positions'))
            native=torch.stack([frozen.candidate_feature(raw,evidence,c,winner) for c in challengers])
            out['mode_features'][mode]['NATIVE7'][arm]=native;out['mode_features'][mode]['COMMON3'][arm]=native[:,:2].contiguous()
            if arm!=D and mode in ('REAL','CBIND'):
                for family,key in (('COMMON3','common'),('NATIVE7','native')):
                    saved=record[('real_' if mode=='REAL' else 'cbind_')+key+'_features'][arm]
                    need(tensor_sha(saved)==tensor_sha(out['mode_features'][mode][family][arm]),'ORIGINAL_B_C_FEATURE_BITS_DRIFT')
    out['real_common_features']=out['mode_features']['REAL']['COMMON3'];out['real_native_features']=out['mode_features']['REAL']['NATIVE7']
    if not pair:
        out['cbind_common_features']=out['mode_features']['CBIND']['COMMON3'];out['cbind_native_features']=out['mode_features']['CBIND']['NATIVE7']
    return out

def corrected_labels():
    from rc_aslo_xf.gallery_identity_repair import build_identity_map,PHYSICAL_ROW_COUNT
    gallery=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images';paths=[]
    for directory,subdirs,names in os.walk(gallery):
        subdirs.sort()
        paths.extend(Path(directory)/name for name in sorted(names) if (Path(directory)/name).is_file() and Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})
    need(len(paths)==PHYSICAL_ROW_COUNT,'GALLERY_CATALOGUE_COUNT')
    return build_identity_map([p.stem.strip() for p in paths]).labels

def role_join(rows,role_entries,labels):
    joined=[]
    for row in rows:
        ex=int(row['execution_ordinal']);entry=role_entries[ex];path=safe(entry['path']);need(sha(path)==entry['sha256'],'ROLE_HASH')
        role=read(path);need(role['query_id']==row['query_id'] and role['track']==row['track'] and role['target_insertion_count']==0
           and role['raw_d1_field_count']==0 and role['target_spatial_supervision_count']==0,'ROLE_BINDING')
        targets=[i for i,p in enumerate(row['candidate_physical_rows']) if labels[p]==role['identity']]
        need(len(targets)==1,'TARGET_NOT_UNIQUE_IN_ORIGINAL_RAW_C128')
        joined.append({**row,'target_position':targets[0],'target_identity':role['identity'],'supergroup':role['supergroup']})
    return joined

def prepare(a):
    global BARRIER
    import torch
    frozen,core,pure=load_modules();fullv=read(checked(a['sources']['full_validation']));pairv=read(checked(a['sources']['pair_validation']))
    need(fullv['status']=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURES_VALIDATED' and fullv['checks']
      and all(v is True for v in fullv['checks'].values()) and pairv['status']=='ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED'
      and pairv['checks'] and all(v is True for v in pairv['checks'].values()) and pairv['v2_payload_sha256']==PINS['pair_payload'][1],'INPUT_VALIDATION')
    full=[];source_shards=[]
    for entry in fullv['shards']:
        shard=int(entry['shard']);path=FULL/f'shard{shard:02d}/payload.pt'
        need(sha(path)==entry['payload_sha256'],'FULL_PAYLOAD_SHA');payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        need(payload['status']=='ROUTEA_MATCHED_THREE_ARM_FULLNEGATIVE_FEATURE_SHARD_READY','FULL_SCHEMA')
        full.extend(payload['records']);source_shards.append({'shard':shard,**binding(path)})
    need(len(full)==64 and len({r['execution_ordinal'] for r in full})==64,'FULL64_AXIS')
    raw_train=sorted((r for r in full if r['data_split_role']=='TRAIN'),key=lambda r:int(r['execution_ordinal']))
    raw_eval=sorted((r for r in full if r['data_split_role']=='EVAL'),key=lambda r:int(r['execution_ordinal']))
    need(len(raw_train)==len(raw_eval)==32,'TRAIN_EVAL_AXIS')
    rolem=read(checked(a['sources']['role_manifest']));entries={int(e['execution_ordinal']):e for e in rolem['shards']}
    BARRIER=ReadBarrier([entries[r['execution_ordinal']]['path'] for r in raw_train],
      [entries[r['execution_ordinal']]['path'] for r in raw_eval],[ROOT/PINS[k][0] for k in ('old_result','old_validation')]);sys.addaudithook(BARRIER.hook)
    pair=torch.load(checked(a['sources']['pair_payload']),map_location='cpu',mmap=True,weights_only=True)
    need(len(pair['records'])==64,'PAIR64_AXIS')
    pair={**pair,'records':[add_features(r,frozen,core,pair=True) for r in pair['records']]}
    train_unjoined=[add_features(r,frozen,core) for r in raw_train];evals=[add_features(r,frozen,core) for r in raw_eval]
    labels=corrected_labels();train=role_join(train_unjoined,entries,labels)
    closure={'status':'RC_FULL_MASS_FREE_IDENTITY_INPUT_CLOSURE','source_shards':source_shards,
      'pair_payload_sha256':PINS['pair_payload'][1],'train_execution_order':[r['execution_ordinal'] for r in train],
      'pair_order':[[r['pair_cohort'],r['pair_row_ordinal'],r['execution_ordinal']] for r in pair['records']],
      'eval_execution_order':[r['execution_ordinal'] for r in evals],
      'full_candidate_axes':{str(r['execution_ordinal']):list(r['candidate_physical_rows']) for r in full},
      'feature_seals':{str(r['execution_ordinal']):{mode:{family:{arm:tensor_sha(r['mode_features'][mode][family][arm]) for arm in ARMS} for family in FAMILIES} for mode in MODES} for r in train_unjoined+evals},
      'pair_D_feature_seals':[tensor_sha(r['real_native_features'][D]) for r in pair['records']],
      'feature_degeneracy_ledger':pure['feature_ledger'](pair,train),
      'TRAIN_role_reads':32,'EVAL_role_reads':0,'old_outcome_semantic_reads':0,'source_tensor_new_model_forwards':0}
    return frozen,core,pure,pair,train,evals,entries,labels,closure

def train_models(pair,train,pure):
    heads={}
    for family in FAMILIES:
        heads[family]={}
        for arm in ARMS:
            head,losses,finite=pure['train_head'](pair,train,family,arm);w=head.weight.detach().flatten();bias=float(head.bias.detach())
            heads[family][arm]={'weight':[float(x) for x in w],'bias':bias,'weight_binary64':[hx(x) for x in w],
              'bias_binary64':hx(bias),'parameter_sha256':pure['parameter_sha'](w,bias),'parameter_count':len(w)+1,
              'loss_total':losses[0],'loss_pair':losses[1],'loss_fullnegative':losses[2],'finite_training':finite}
            print(json.dumps({'event':'HEAD_2000_UPDATES_COMPLETE','family':family,'arm':arm,'parameter_sha256':heads[family][arm]['parameter_sha256']}),flush=True)
    return heads

def predict(row,weight,bias,family,arm,mode):
    matrix=row['mode_features'][mode][family][arm];logits=matrix@weight+float(bias)
    axis=list(row['candidate_physical_rows']);challengers=list(row['challenger_positions']);winner=int(row['base_winner_position'])
    i=max(range(127),key=lambda i:(float(logits[i]),-axis[challengers[i]]));best=challengers[i];switch=float(logits[i])>0
    return {'candidate_physical_rows':axis,'challenger_positions':challengers,'base_scores_binary64':[hx(x) for x in row['base_scores']],
      'all127_logits_binary64':[hx(x) for x in logits],'base_winner_position':winner,'proposed_challenger_position':best,
      'final_position':best if switch else winner,'final_physical_row':axis[best if switch else winner],
      'decision':'SWITCH' if switch else 'HOLD','switch_logit_binary64':hx(logits[i])}

def evaluation_prejoin(evals,heads,frozen):
    import torch
    records=[]
    for row in evals:
        predictions={family:{arm:{mode:predict(row,torch.tensor(heads[family][arm]['weight'],dtype=torch.float64),heads[family][arm]['bias'],family,arm,mode)
          for mode in MODES} for arm in ARMS} for family in FAMILIES}
        predictions['FROZEN_C']={'C_PAIRED':{mode:predict(row,frozen.WEIGHT,frozen.BIAS,'NATIVE7','C_PAIRED',mode) for mode in MODES}}
        records.append({'execution_ordinal':row['execution_ordinal'],'query_id':row['query_id'],
          'predictions':predictions,'D_full_C128_evidence_binary64':{str(i):{k:hx(v) for k,v in e.items()} for i,e in row['evidence'][D].items()},'target_reads':0})
    return records

def model_actions(rows,heads,pure,frozen,family,arm,mode):
    import torch
    if family=='FROZEN_C':weight,bias,feature_family=frozen.WEIGHT,frozen.BIAS,'NATIVE7'
    else:weight,bias,feature_family=torch.tensor(heads[family][arm]['weight'],dtype=torch.float64),heads[family][arm]['bias'],family
    key='real_common_features' if feature_family=='COMMON3' else 'real_native_features'
    prepared=[{**r,key:{**r[key],arm:r['mode_features'][mode][feature_family][arm]}} for r in rows]
    return pure['actions'](weight,bias,prepared,feature_family,arm)

def postjoin(a,heads,prejoin,train,evals,entries,labels,pure,frozen):
    need((OUT/'parameters.json').is_file() and (OUT/'eval_prejoin_seal.json').is_file(),'POSTJOIN_WITHOUT_SEALS')
    seal=read(OUT/'eval_prejoin_seal.json')
    need(seal['parameters_sha256']==sha(OUT/'parameters.json') and seal['eval_prejoin_sha256']==sha(OUT/'eval_prejoin.json')
      and seal['eval_query_count']==32 and BARRIER.blocked==0,'POSTJOIN_SEAL_DRIFT')
    BARRIER.release();joined=role_join(evals,entries,labels)
    disjoint={field:not bool({r[field] for r in train}&{r[field] for r in joined}) for field in ('execution_ordinal','target_identity','supergroup')}
    need(all(disjoint.values()),'TRAIN_EVAL_IDENTITY_OR_GROUP_OVERLAP')
    old=read(checked(a['sources']['old_result']));oldv=read(checked(a['sources']['old_validation']))
    need(oldv['status']=='ROUTEA_MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_INDEPENDENT_VALIDATION_PASS'
      and oldv['checks'] and all(v is True for v in oldv['checks'].values()) and oldv['producer_result_sha256']==PINS['old_result'][1],'OLD_RESULT_VALIDATION')
    by_ex={r['execution_ordinal']:r for r in prejoin};all_actions={};baseline_checks={}
    for family in FAMILIES:
        all_actions[family]={}
        for arm in ('B_QUERY','C_PAIRED'):
            expected=old['heads'][family][arm];h=heads[family][arm]
            baseline_checks[family+'/'+arm+'/parameters']=h['weight_binary64']==[hx(x) for x in expected['weight']] and h['bias_binary64']==hx(expected['bias'])
            all_actions[family][arm]={mode:model_actions(joined,heads,pure,frozen,family,arm,mode) for mode in MODES}
            for mode,field in (('REAL','actions'),('CBIND','cbind_actions')):
                baseline_checks[family+'/'+arm+'/'+mode+'_actions']=encode(all_actions[family][arm][mode])==encode(old['evaluations'][family][arm][field])
    frozen_actions={mode:model_actions(joined,heads,pure,frozen,'FROZEN_C','C_PAIRED',mode) for mode in MODES}
    baseline_checks['FROZEN_C_REAL']=encode(frozen_actions['REAL'])==encode(old['frozen_c_regression']['real_actions'])
    baseline_checks['FROZEN_C_CBIND']=encode(frozen_actions['CBIND'])==encode(old['frozen_c_regression']['cbind_actions'])
    passed=all(baseline_checks.values())
    if passed:
        for family in FAMILIES:all_actions[family][D]={mode:model_actions(joined,heads,pure,frozen,family,D,mode) for mode in MODES}
    for family,arms in all_actions.items():
        for arm,modes in arms.items():
            for mode,actions in modes.items():
                for action in actions:
                    pred=by_ex[action['execution_ordinal']]['predictions'][family][arm][mode]
                    need(pred['final_position']==action['final_position'] and pred['decision']==action['decision']
                      and pred['switch_logit_binary64']==hx(action['switch_logit']),'PREJOIN_PREDICTION_POSTJOIN_DRIFT')
    metrics={family:{arm:{mode:pure['summary'](acts) for mode,acts in modes.items()} for arm,modes in arms.items()} for family,arms in all_actions.items()}
    cbind_retention={family:{arm:pure['retention'](modes['REAL'],modes['CBIND']) for arm,modes in arms.items()} for family,arms in all_actions.items()}
    paired={}
    if passed:
        for family in FAMILIES:
            new=all_actions[family][D]['REAL'];paired[family]={}
            for tag,base in (('RETRAINED_C',all_actions[family]['C_PAIRED']['REAL']),('RETRAINED_B',all_actions[family]['B_QUERY']['REAL']),('FROZEN_C',frozen_actions['REAL'])):
                rescues=sum(n['final_correct'] and not b['final_correct'] for n,b in zip(new,base));breaks=sum(b['final_correct'] and not n['final_correct'] for n,b in zip(new,base))
                paired[family][tag]={'rescue':rescues,'break':breaks,'paired_net':rescues-breaks}
    return {'status':'RC_FULL_MASS_FREE_IDENTITY_DEVELOPMENT_COMPLETE' if passed else 'RC_FULL_MASS_FREE_IDENTITY_BASELINE_REGRESSION_ABORT',
      'claim_level':'OPENED_TRAIN32_PLUS_HISTORICAL_PAIR64_FIXED_D_ARM_DEVELOPMENT_EVAL32','authority_sha256':sha(AUTH),
      'baseline_regression_pass':passed,'baseline_checks':baseline_checks,'D_interpretation_authorized':passed,
      'metrics':metrics,'paired_D_REAL_comparisons':paired,'C_BIND_rescue_retention':cbind_retention,
      'input_closure_sha256':sha(OUT/'input_closure.json'),
      'feature_degeneracy_ledger':read(OUT/'input_closure.json')['feature_degeneracy_ledger'],
      'actions':all_actions,'frozen_C_actions':frozen_actions,
      'parameters_sha256':sha(OUT/'parameters.json'),'eval_prejoin_seal_sha256':sha(OUT/'eval_prejoin_seal.json'),
      'no_family_selection_performed':True,'families_reported':list(FAMILIES),'candidate_count':128,'train_queries':32,'pair_queries':64,'eval_queries':32,
      'train_eval_disjoint':disjoint,'EVAL_role_opened_only_after_all_parameters_and_predictions_sealed':True,'early_blocked_read_count':BARRIER.blocked,
      'control_semantics':'CBIND complete evidence shift64; Q/R old frozen path relabel on corresponding cached scalar scores',
      'limits':['One new cached-scalar D arm; no bit-identity claim to a separately recomputed raw D equation.',
        'Both existing head families reported without selecting a winner; native D reference-response degeneracy disclosed.',
        'EVAL32 is previously opened internal development; no untouched generalization or formal HYP GO.',
        'Original B/C and FROZEN_C sources remain immutable; failed regression blocks D interpretation.'],
      'new_backbone_count':0,'new_P_geometry_count':0,'formal_panel_consumed':False,'deployment_replacement_authorized':False,
      'scientific_GO_or_NO_GO':None,'automatic_stage_advance':False}

def execute(validate=False):
    import torch
    a=authority();frozen,core,pure,pair,train,evals,entries,labels,closure=prepare(a)
    if validate:need(OUT.exists(),'OUTPUT_MISSING')
    else:need(not OUT.exists(),'OUTPUT_EXISTS');OUT.mkdir(parents=True)
    if validate:need(read(OUT/'input_closure.json')==closure,'INPUT_CLOSURE_REPLAY')
    else:atomic(OUT/'input_closure.json',closure)
    heads=train_models(pair,train,pure)
    if validate:need(read(OUT/'parameters.json')==heads,'INDEPENDENT_2000_UPDATE_PARAMETER_REPLAY')
    else:atomic(OUT/'parameters.json',heads)
    prejoin=evaluation_prejoin(evals,heads,frozen)
    if validate:need(read(OUT/'eval_prejoin.json')==prejoin,'FULL_EVAL_LOGIT_PREJOIN_REPLAY')
    else:
        atomic(OUT/'eval_prejoin.json',prejoin)
        atomic(OUT/'eval_prejoin_seal.json',{'status':'RC_FULL_MASS_FREE_IDENTITY_ALL_EVAL_PREDICTIONS_PREJOIN_SEALED',
          'authority_sha256':sha(AUTH),'parameters_sha256':sha(OUT/'parameters.json'),'eval_prejoin_sha256':sha(OUT/'eval_prejoin.json'),
          'eval_query_count':32,'candidate_count':128,'head_training_updates_each':2000,'EVAL_role_reads':0,'old_outcome_semantic_reads':0})
    result=postjoin(a,heads,prejoin,train,evals,entries,labels,pure,frozen)
    if validate:
        need(read(OUT/'result.json')==result,'INDEPENDENT_RESULT_REPLAY')
        receipt={'status':'RC_FULL_MASS_FREE_IDENTITY_INDEPENDENT_VALIDATION_PASS','result_sha256':sha(OUT/'result.json'),
          'authority_sha256':sha(AUTH),'baseline_regression_pass':result['baseline_regression_pass'],
          'checks':{'all_raw_only_inputs_and_D_features_rebuilt':True,'all_six_heads_2000_updates_independently_retrained':True,
            'all_parameters_and_all_eval_control_logits_replayed':True,'TRAIN_only_labels_before_parameter_freeze':True,
            'EVAL_roles_and_old_outcomes_only_after_full_prejoin':True,'all_old_B_C_FROZEN_regression_checks_replayed':True,
            'all32_actions_metrics_and_paired_comparisons_exact':True},'scientific_GO_or_NO_GO':None,'automatic_stage_advance':False}
        atomic(OUT/'independent_validation.json',receipt);print(json.dumps(receipt),flush=True)
    else:atomic(OUT/'result.json',result);print(json.dumps({'status':result['status'],'baseline_regression_pass':result['baseline_regression_pass'],'metrics':result['metrics'],'scientific_GO_or_NO_GO':None}),flush=True)
    return 0 if result['baseline_regression_pass'] else 4

def synthetic_e0():
    import torch
    frozen,core,pure=load_modules()
    evidence={'real_score':.125,'visibility_mass':.25,'query_control_score':.0625,'reference_control_score':.125}
    d=core.combine_full_mass_with_free_identity(evidence,.5);need(d=={'real_score':.25,'visibility_mass':.5,'query_control_score':.125,'reference_control_score':.25},'D_EXACT_BRIDGE')
    raw=[float(128-i) for i in range(128)];ev={i:{**evidence,'real_score':.125+i/1024,'reference_control_score':.125+i/1024} for i in range(128)}
    rows=[]
    for ex in (1,2):
        native=torch.stack([frozen.candidate_feature(raw,ev,i,0) for i in range(1,128)])
        row={'execution_ordinal':ex,'query_id':'synthetic'+str(ex),'track':'synthetic','heldout_fold':1,
          'candidate_physical_rows':list(range(128)),'base_scores':torch.tensor(raw,dtype=torch.float64),'base_winner_position':0,
          'challenger_positions':list(range(1,128)),'target_position':0 if ex==1 else 1,
          'real_native_features':{D:native},'real_common_features':{D:native[:,:2].contiguous()}}
        row['mode_features']={mode:{'NATIVE7':{D:native},'COMMON3':{D:native[:,:2].contiguous()}} for mode in MODES};rows.append(row)
    pair={'records':[{'switch_label':bool(i),'real_native_features':{D:r['real_native_features'][D][:1]},
      'real_common_features':{D:r['real_common_features'][D][:1]}} for i,r in enumerate(rows)]}
    original_steps=pure['STEPS'];pure['STEPS']=4
    for family in FAMILIES:
        h,_,_=pure['train_head'](pair,rows,family,D);acts=pure['actions'](h.weight.detach().flatten(),float(h.bias),rows,family,D)
        for row,act in zip(rows,acts):
            pred=predict(row,h.weight.detach().flatten(),float(h.bias),family,D,'REAL')
            need(pred['final_position']==act['final_position'] and pred['switch_logit_binary64']==hx(act['switch_logit']),'TARGET_FREE_PREDICTION_PARITY')
    pure['STEPS']=original_steps
    need(original_steps==2000 and not any('load_inputs'==name or 'main'==name for name in pure),'PURE_OLD_FUNCTION_SCOPE')
    train_path=ROLE_ROOT/'role_shards/role_exec998.json';eval_path=ROLE_ROOT/'role_shards/role_exec999.json';old_path=ROOT/'results/synthetic_old_outcome.json'
    barrier=ReadBarrier([train_path],[eval_path],[old_path]);barrier.hook('open',(str(train_path),'r'))
    for blocked_path in (eval_path,old_path):
        try:barrier.hook('open',(str(blocked_path),'r'))
        except RuntimeError:pass
        else:raise AssertionError('EARLY_EVAL_OR_OUTCOME_READ_ACCEPTED')
    barrier.hash_only=1;barrier.hook('open',(str(old_path),'r'));barrier.hash_only=0
    barrier.release();barrier.hook('open',(str(eval_path),'r'));barrier.hook('open',(str(old_path),'r'))
    return {'status':'RC_FULL_MASS_FREE_IDENTITY_RUNNER_E0_PASS','checks':{'D_same_rho_four_field_interface':True,
      'early_EVAL_role_and_old_outcome_reads_rejected':True,'old_outcome_hash_only_access_separate':True,'postseal_role_access_released':True,
      'original_training_and_action_functions_AST_reused':True,'synthetic_four_update_optimizer_paths_both_families':True,
      'target_free_prediction_equals_original_action_after_synthetic_join':True,'full127_logit_and_physical_tie_axes':True,
      'production_update_constant_2000_unmodified':True},'synthetic_only':True,'natural_model_training_updates':0,'scientific_GO_or_NO_GO':None}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('e0','freeze','run','validate'),required=True);args=parser.parse_args()
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    if args.phase=='e0':print(json.dumps(synthetic_e0(),sort_keys=True));return 0
    if args.phase=='freeze':freeze();return 0
    return execute(validate=args.phase=='validate')
if __name__=='__main__':raise SystemExit(main())
