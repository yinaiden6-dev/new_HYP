#!/usr/bin/env python3
"""Incremental FIT/SELECT group split; outer OOF stays unchanged; TRAIN only."""
from __future__ import annotations
import argparse,ast,hashlib,json,math,os,subprocess,sys,uuid
from collections import defaultdict
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy
from scipy.optimize import linprog
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];PROGRAM=Path(__file__).resolve()
sys.path.insert(0,str(ROOT/'src'))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC
OUT=ROOT/'results/rc_train128_group_held_selection_v1'
PREFLIGHT=ROOT/'results/rc_train128_group_held_selection_v1_preflight'
INPUT=ROOT/'results/rc_original7_train128_inputs_v1'
META=ROOT/'results/rc_original7_train128_manifest_v1'
PARENT=ROOT/'results/rc_train128_disagreement_oof4_v1'
OLD=ROOT/'programs/run_rc_train128_disagreement_oof4_v1.py'
OLD_SHA='d7302eca66c7c6469a7f5ef99a77afeae8af4b25c3a253a1bdcbb2da7af3a715'
PURE=ROOT/'programs/run_rc_train128_protected_projection_oof4_v1.py'
PURE_SHA='adda0ee973c6e3074a172458ead8c77a70b11ee77194624d26e14590995ee7ec'
PLAN=ROOT/'plan/RC_TRAIN128_GROUP_HELD_SELECTION_V1_20260911.md'
AUTH=ROOT/'registry/rc_train128_group_held_selection_authority_v1_20260911.json'
LAUNCH=ROOT/'slurm/rc_train128_group_held_selection_v1_dev_cpuonly_59m.sbatch'
GALLERY_IMAGES=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images'
SPLIT_SEED='RC_TRAIN128_GROUP_HELD_SELECTION_V1_20260911'
DEADLINE=datetime(2026,9,11,16,tzinfo=timezone.utc)
MODELS=('BASE7','FIT_SELECT','SELECT_RANK','SELECT_PROTECT');ARM='C_PAIRED'
PHASE,FOLD=None,None;HASH_DEPTH=0;BLOCKED=[];PAYLOAD_READS=[];OUTER_LABELS_RELEASED=False;LP_COUNT=0
LP_OPTIONS={'presolve':True,'time_limit':30.0,'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9}
CONTRACT={
 'scope':'post-hoc TRAIN128 grouped outer OOF; no EVAL or deployment',
 'outer_fold_count':4,'outer_folds_and_BASE7':'unchanged previously validated group OOF BASE7',
 'incremental_split':'24 outer-training groups sorted by SHA256(seed|group); first8 SELECT, remaining16 FIT',
 'split_seed':SPLIT_SEED,'fit_group_count':16,'select_group_count':8,
 'baseline_selection_overlap':'BASE7 may have seen SELECT original FULL/PAIR labels; only incremental FIT/SELECT is isolated; outer holdout excludes both',
 'FIT':'one original PROTECTED7 LP per FIT BASE error; P,E,delta and all training scores use FIT only; BASE included',
 'pool_source':'fresh FIT-only candidates; previous24group pool forbidden',
 'pool_before_SELECT':'seal complete theta and FIT scores; distinct fresh FIT replay before SELECT label reads',
 'models':list(MODELS),'primary':'SELECT_PROTECT',
 'FIT_SELECT':'original FIT group/count/L1/ordinal rank',
 'SELECT_RANK':'same sealed pool; SELECT group/count/L1/ordinal rank; no preservation filter; strict SELECT group-mean gain required else exact BASE',
 'SELECT_PROTECT':'same pool, reject any SELECT BASE-correct loss; same SELECT rank; strict SELECT group-mean gain required else exact BASE',
 'selection_training_updates':0,'final_theta':'bit-exact selected sealed pool member',
 'outer_labels':'only after all4 parameter and complete all127 prediction seals plus fresh SELECT replay',
 'gate':['SELECT_PROTECT correct>BASE7','all BASE7-correct outer images retained','equal-group SELECT_PROTECT-BASE7>0'],
 'controls_are_additional_gates':False,'group_gate_is_consistency':True,
 'features':'unchanged original6, no F or new features','dtype':'float64',
 'action':'complete natural C128/all127; SWITCH iff max logit>0; physical-row tie',
 'LP_solver':'highs-ds','LP_options':LP_OPTIONS,'base_refits':0,
 'EVAL_access':False,'cutoff_UTC':DEADLINE.isoformat(),
 'limits':'finite candidate pool and reused TRAIN; no whole-class capacity claim or universal no-regret guarantee',
}
STATISTICS_CONTRACT={'paired_comparisons':[['SELECT_PROTECT','BASE7'],['SELECT_RANK','BASE7'],['FIT_SELECT','BASE7'],['SELECT_PROTECT','SELECT_RANK'],['SELECT_PROTECT','FIT_SELECT'],['SELECT_RANK','FIT_SELECT']],
 'unit':'source group; equal-weight group accuracy difference','bootstrap_draws':10000,'bootstrap_seed':20260910,
 'bootstrap_interval':'95% percentile numpy.quantile linear','signflip':'exact two-sided inclusive rational group signflip','p_value_is_gate':False}

def need(value, message):
    if not bool(value):
        raise RuntimeError(message)

def sha(path):
    global HASH_DEPTH
    HASH_DEPTH += 1
    try:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    finally:
        HASH_DEPTH -= 1

def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}

def checked(binding):
    p = Path(binding['path'])
    p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    need(sha(p) == binding['sha256'], 'ARTIFACT_HASH_DRIFT:' + str(p))
    return p

def write(path, value):
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'APPEND_ONLY_OUTPUT:' + str(path))
        return
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def hx(value):
    return float(value).hex()

def matrix_hex(value):
    return [[hx(v) for v in row] for row in value]

def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'CUDA_used': False,
            'solver': 'highs-ds', 'solver_options': LP_OPTIONS, 'flush_denormal': False}

def audit(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
    hard=any(t in s for t in ('/rc_opened_','/rc_original7_eval','/grozi/','d1-mi','d1_mi','/target_join/','/role_shards/','direction_capacity','angle_capacity','origin_linear_projection','/rc_train128_protected_projection_oof4_v1/'))
    if hard:
        BLOCKED.append(str(p));raise PermissionError('HARD_EVAL_OR_OLD_POOL_READ_FORBIDDEN')
    # Role boundaries precede hash and own-output allowances.
    role=False;allowed=True
    if p.name in ('fit_roles.json','select_roles.json') and OUT in p.parents:
        role=True
        allowed=PHASE in ('metadata-prepare','validate-metadata') or (p.parent==OUT/f'fold{FOLD:02d}' if FOLD is not None else False) and ((p.name=='fit_roles.json' and PHASE in ('fit-fold','validate-fit')) or (p.name=='select_roles.json' and PHASE in ('select-fold','validate-select')))
    elif PARENT in p.parents and p.name in ('train_roles.json','heldout_roles.json'):
        role=True
        allowed=(p.name=='train_roles.json' and PHASE in ('metadata-prepare','validate-metadata')) or (p.name=='heldout_roles.json' and PHASE in ('join','validate-join') and OUTER_LABELS_RELEASED)
    if role and not allowed:
        BLOCKED.append(str(p));raise PermissionError('ROLE_PHASE_OR_FOLD_FORBIDDEN')
    if ROOT/'results' in p.parents:
        allowed=OUT in p.parents or PREFLIGHT in p.parents or p in (INPUT/'validation.json',INPUT/'feature_records.json',META/'worker_manifest.json',PARENT/'fold_manifest.json',PARENT/'validation.json')
        if PARENT in p.parents and p.parent.name.startswith('fold'):
            allowed=p.name in ('parameters.json','predictions.json','seal.json','validation.json') or role
        mode=args[1] if len(args)>1 else None
        is_write=isinstance(mode,str) and any(c in mode for c in 'wax+')
        if OUT in p.parents and not is_write and PHASE in ('fit-fold','validate-fit','select-fold','validate-select','run'):
            public=p in (OUT/'split_manifest.json',OUT/'metadata_validation.json')
            if PHASE in ('fit-fold','validate-fit'):
                allowed=public or p.parent==OUT/f'fold{FOLD:02d}' and p.name in ('fit_roles.json','fit_pool.json','fit_seal.json','fit_validation.json')
            elif PHASE in ('select-fold','validate-select'):
                allowed=public or p.parent==OUT/f'fold{FOLD:02d}' and p.name in ('select_roles.json','fit_pool.json','fit_seal.json','fit_validation.json','parameters.json','selection_ledger.json','predictions.json','seal.json','validation.json')
            else:allowed=public
        if PHASE in ('metadata-prepare','validate-metadata'):
            allowed=p in (PARENT/'fold_manifest.json',PARENT/'validation.json',META/'worker_manifest.json') or role or OUT in p.parents
        if PHASE=='preflight':allowed=PREFLIGHT in p.parents
        if not allowed:
            BLOCKED.append(str(p));raise PermissionError('RESULT_PHASE_ALLOWLIST_FORBIDDEN:'+str(p))
        mode=args[1] if len(args)>1 else None
        if not HASH_DEPTH and isinstance(mode,str) and 'r' in mode:PAYLOAD_READS.append(str(p))

def read(path):
    p=Path(path)
    return json.loads(p.read_text())

def guarded_linprog(*args,**kwargs):
    global LP_COUNT
    need(PHASE in ('preflight','fit-fold','validate-fit'),'LP_FORBIDDEN_OUTSIDE_FIT_PHASE')
    LP_COUNT+=1
    return linprog(*args,**kwargs)

def pure_ops():
    need(sha(PURE)==PURE_SHA,'FROZEN_PURE_LP_SOURCE')
    names={'top_index','serialize_prediction','exact_constraints','dot','parameter_values','parameter_tensor','train_score','rank_candidate','build_context','solve_training','manual_feature'}
    nodes=[n for n in ast.parse(PURE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==names,'ONLY_PURE_LP_AND_ACTION_OPERATORS')
    ns=dict(globals());ns['MODELS']=('BASE7','PROTECTED7','REPAIR_ONLY7');ns['linprog']=guarded_linprog
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PURE),'exec'),ns)
    return {k:ns[k] for k in names}

def reused_operators(independent=False):
    need(sha(OLD)==OLD_SHA,'FROZEN_PARENT_SOURCE')
    names={'qualified_rows','gallery_labels','exact_group_signflip','paired_group_statistics'}
    nodes=[n for n in ast.parse(OLD.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==names,'ONLY_UNQUARANTINED_PARENT_OPERATORS')
    ns=dict(globals())
    if independent:ns['FC']=SimpleNamespace(candidate_feature=pure_ops()['manual_feature'])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OLD),'exec'),ns)
    return {k:ns[k] for k in names}

def metadata_build():
    parent=read(PARENT/'fold_manifest.json');need(len(parent['records'])==128,'UNCHANGED_OUTER128')
    folds={};public=[]
    for fold in range(4):
        source=checked(parent['role_projections'][f'fold{fold}_train_roles.json'])
        need(source==(PARENT/f'fold{fold:02d}'/'train_roles.json').resolve(),'EXACT_PARENT_TRAIN_PROJECTION')
        roles=read(source)['records'];groups=sorted({r['group'] for r in roles},key=lambda g:(hashlib.sha256((SPLIT_SEED+'|'+g).encode()).hexdigest(),g))
        need(len(groups)==24 and len({r['identity'] for r in roles})==24 and all(len({r['identity'] for r in roles if r['group']==g})==1 for g in groups),'24_ONE_TO_ONE_OUTER_TRAIN_GROUPS_IDENTITIES')
        assigned={g:('SELECT' if i<8 else 'FIT') for i,g in enumerate(groups)};projections={};foldpub=[]
        need({r['query_id'] for r in roles}=={r['query_id'] for r in parent['records'] if r['fold']!=fold},'EXACT_OUTER_TRAIN_MEMBERSHIP')
        for name in ('FIT','SELECT'):
            keep=[{k:r[k] for k in ('query_id','original_query_id','execution_ordinal','source_image_sha256','group','identity')} for r in roles if assigned[r['group']]==name]
            need(len({r['group'] for r in keep})==(16 if name=='FIT' else 8),'FIXED_GROUP_SPLIT_SIZE')
            projections[name]={'status':'INCREMENTAL_GROUP_ROLE_PROJECTION','fold':fold,'role':name,'source_train_roles':bind(source),'records':keep}
        need({r['identity'] for r in projections['FIT']['records']}.isdisjoint(r['identity'] for r in projections['SELECT']['records']),'FIT_SELECT_IDENTITY_DISJOINT')
        lookup={r['query_id']:r for r in roles}
        for r in parent['records']:
            role='OUTER' if r['fold']==fold else assigned[lookup[r['query_id']]['group']]
            foldpub.append({k:r[k] for k in ('query_id','execution_ordinal','source_image_sha256','group_key')}|{'role':role})
        folds[str(fold)]={'groups_SHA_order':[hashlib.sha256(g.encode()).hexdigest() for g in groups],'public_records':foldpub,'projections':projections}
    return {'folds':folds,'parent_fold_manifest':bind(PARENT/'fold_manifest.json'),'split_seed':SPLIT_SEED}

def prepare(replay=False,nonce=None):
    if replay:check_fresh(nonce)
    value=metadata_build();public={};bindings={}
    for fold,part in value['folds'].items():
        directory=OUT/f'fold{int(fold):02d}'
        for name,payload in part['projections'].items():
            path=directory/(name.lower()+'_roles.json')
            if replay:need(read(path)==payload,'FRESH_METADATA_ROLE_REPLAY')
            else:write(path,payload)
            bindings[f'{fold}:{name}']=bind(path)
        public[fold]={'group_hash_order':part['groups_SHA_order'],'records':part['public_records']}
    manifest={'status':'TRAIN128_GROUP_HELD_SELECTION_METADATA_FROZEN','contract':CONTRACT,'parent_fold_manifest':value['parent_fold_manifest'],'split_seed':SPLIT_SEED,'folds':public,'role_projections':bindings,'model_or_feature_reads':0,'training_updates':0,'EVAL_reads':0}
    if replay:
        need(read(OUT/'split_manifest.json')==manifest,'FRESH_METADATA_MANIFEST_REPLAY')
        write(OUT/'metadata_validation.json',{'status':'TRAIN128_GROUP_HELD_SELECTION_METADATA_REPLAY_PASS','manifest':bind(OUT/'split_manifest.json'),'program':bind(PROGRAM),'plan':bind(PLAN),'fold_count':4,'FIT_groups_per_fold':16,'SELECT_groups_per_fold':8,'OUTER_groups_per_fold':8,'model_or_feature_reads':0,'training_updates':0,'EVAL_reads':0,'fresh_process_nonce':nonce,'process_PID':os.getpid(),'forbidden_read_attempts':len(BLOCKED)})
    else:
        write(OUT/'split_manifest.json',manifest);fresh('validate-metadata')
    print(json.dumps({'phase':PHASE,'status':'PASS','metadata':bind(OUT/'split_manifest.json')}),flush=True)

def static_sources():
    paths={'program':PROGRAM,'launcher':LAUNCH,'plan':PLAN,'pure_LP_program':PURE,'parent_program':OLD,'prior_root_validation':PARENT/'validation.json','parent_fold_manifest':PARENT/'fold_manifest.json','metadata_manifest':OUT/'split_manifest.json','metadata_validation':OUT/'metadata_validation.json','input_validation':INPUT/'validation.json','input_features':INPUT/'feature_records.json','worker_manifest':META/'worker_manifest.json','feature_core':Path(FC.__file__),'gallery_identity_program':ROOT/'src/rc_aslo_xf/gallery_identity_repair.py','gallery_identity_registry':ROOT/'registry/gallery_identity_repair_v1.json','gallery_identity_contract':ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'}
    for fold in range(4):
        for name in ('validation','seal','parameters','predictions'):paths[f'parent_fold{fold}_{name}']=PARENT/f'fold{fold:02d}'/(name+'.json')
    return {k:bind(p) for k,p in paths.items()}

def require_authority():
    need(datetime.now(timezone.utc)<DEADLINE,'RESEARCH_DEADLINE_REACHED')
    need(os.environ.get('SLURM_JOB_ID','').isdigit(),'NATURAL_WORK_REQUIRES_SLURM')
    a=read(AUTH)
    need(a['status']=='TRAIN128_GROUP_HELD_SELECTION_EXECUTION_AUTHORIZED' and PHASE in a['allowed_stages'],'STAGE_AUTHORIZED')
    need(a['contract']==CONTRACT and a['statistics_contract']==STATISTICS_CONTRACT and a['cutoff_UTC']==DEADLINE.isoformat(),'EXACT_AUTHORITY_CONTRACT')
    need(a['source_bindings']==static_sources(),'EXACT_NATURAL_SOURCE_BINDINGS')
    pre=read(checked(a['preflight']));need(pre['status']=='TRAIN128_GROUP_HELD_SELECTION_SYNTHETIC_PREFLIGHT_PASS' and pre['program']==bind(PROGRAM) and pre['runtime']==runtime() and pre['contract']==CONTRACT,'EXACT_SYNTHETIC_QUALIFICATION')
    md=read(OUT/'metadata_validation.json');need(md['status']=='TRAIN128_GROUP_HELD_SELECTION_METADATA_REPLAY_PASS' and md['manifest']==bind(OUT/'split_manifest.json') and md['program']==bind(PROGRAM) and md['plan']==bind(PLAN),'METADATA_REPLAY_QUALIFIED')
    need(sha(OLD)==OLD_SHA and sha(PURE)==PURE_SHA,'FROZEN_ANCESTRY')
    return bind(AUTH)

def baseline_context(fold,independent=False):
    op=reused_operators(independent);rows=op['qualified_rows']();parent=read(PARENT/'fold_manifest.json');split=read(OUT/'split_manifest.json')
    directory=PARENT/f'fold{fold:02d}';v=read(directory/'validation.json');seal=read(checked(v['seal']));base=read(checked(seal['parameters']))['BASE7'];previous=read(checked(seal['predictions']))
    rootval=read(PARENT/'validation.json');need(v['status']=='TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS' and v['fold']==fold and v['program']==rootval['program'] and v['authority']==rootval['authority'],'PRIOR_FOLD_EXACT_LINEAGE')
    need(v['heldout_label_reads']==v['EVAL_reads']==v['forbidden_read_attempts']==0,'PRIOR_FOLD_READ_QUALIFICATION')
    public=split['folds'][str(fold)]['records'];need(len(rows)==len(public)==len(parent['records'])==128,'FULL_OUTER_AXIS')
    byrole={r:[] for r in ('FIT','SELECT','OUTER')}
    for row,p,old in zip(rows,public,parent['records'],strict=True):
        need(row['query_id']==p['query_id']==old['query_id'] and row['execution_ordinal']==p['execution_ordinal']==old['execution_ordinal'] and row['source_image_sha256']==p['source_image_sha256']==old['source_image_sha256'],'SOURCE_AXIS_PARITY')
        need((p['role']=='OUTER')==(old['fold']==fold),'UNCHANGED_OUTER_MEMBERSHIP');byrole[p['role']].append(row)
    ops=pure_ops();w,b=ops['parameter_tensor'](base)
    for row,old in zip(byrole['OUTER'],previous,strict=True):
        need(row['query_id']==old['query_id'] and row['candidate_physical_rows']==old['candidate_physical_rows'] and ops['serialize_prediction'](row,row['X']@w+b)==old['predictions']['BASE7'],'ALL127_PRIOR_BASE_OUTER_BIT_PARITY')
    closure={'prior_fold_validation':bind(directory/'validation.json'),'prior_fold_seal':bind(directory/'seal.json'),'prior_parameters':seal['parameters'],'prior_predictions':seal['predictions'],'split_manifest':bind(OUT/'split_manifest.json'),'role_ordinals':{k:[r['execution_ordinal'] for r in v] for k,v in byrole.items()}}
    return byrole,base,closure

def attach_roles(rows,fold,role):
    manifest=read(OUT/'split_manifest.json');path=checked(manifest['role_projections'][f'{fold}:{role}'])
    need(path==(OUT/f'fold{fold:02d}'/(role.lower()+'_roles.json')).resolve(),'EXACT_ROLE_PROJECTION_PATH')
    values=read(path);need(values['fold']==fold and values['role']==role,'EXACT_ROLE_PROJECTION_ENVELOPE')
    lookup={r['query_id']:r for r in values['records']};need(set(lookup)=={r['query_id'] for r in rows},'EXACT_ROLE_MEMBERSHIP')
    labels,mapping=reused_operators()['gallery_labels']()
    for row in rows:
        r=lookup[row['query_id']];need(row['execution_ordinal']==r['execution_ordinal'] and row['source_image_sha256']==r['source_image_sha256'],'ROLE_IMAGE_BINDING')
        targets=[i for i,p in enumerate(row['candidate_physical_rows']) if labels[p]==r['identity']];need(len(targets)<=1,'NATURAL_C128_IDENTITY_AXIS');row['target_position']=targets[0] if targets else None;row['group']=r['group']
    return {'roles':bind(path),'gallery_mapping_sha256':mapping}

def counters():
    unique=sorted(set(PAYLOAD_READS))
    return {'payload_read_paths':unique,'FIT_role_payload_reads':sum(p.endswith('/fit_roles.json') for p in PAYLOAD_READS),'SELECT_role_payload_reads':sum(p.endswith('/select_roles.json') for p in PAYLOAD_READS),'outer_role_payload_reads':sum(p.endswith('/heldout_roles.json') for p in PAYLOAD_READS),'EVAL_reads':0,'base_refits':0,'gradient_optimizer_steps':0,'actual_LP_calls':LP_COUNT,'forbidden_read_attempts':len(BLOCKED)}

def fit_fold(fold,replay=False,nonce=None):
    auth=require_authority()
    if replay:check_fresh(nonce)
    byrole,base,closure=baseline_context(fold,replay);closure['FIT']=attach_roles(byrole['FIT'],fold,'FIT');ops=pure_ops();w,b=ops['parameter_tensor'](base)
    pool=ops['solve_training'](byrole['FIT'],w,b,'PROTECTED7',replay)
    payload={'status':'FIT_ONLY_COMPLETE_PROTECTED_LP_POOL','fold':fold,'BASE7':base,'fit_result':pool,'fit_query_count':len(byrole['FIT']),'fit_group_count':16,'SELECT_labels_used':0}
    directory=OUT/f'fold{fold:02d}'
    if replay:
        seal=read(directory/'fit_seal.json');need(seal['authority']==auth and seal['program']==bind(PROGRAM) and seal['closure']==closure,'FRESH_FIT_SOURCE_BINDING');need(payload==read(checked(seal['pool'])),'FRESH_ALL_FIT_LP_POOL_BITS');need(os.getpid()!=seal['process_PID'],'FRESH_FIT_DISTINCT_PROCESS')
        write(directory/'fit_validation.json',{'status':'GROUP_HELD_SELECTION_FRESH_FIT_POOL_REPLAY_PASS','fold':fold,'seal':bind(directory/'fit_seal.json'),'authority':auth,'program':bind(PROGRAM),'LP_calls_replayed':pool['LP_call_count'],'all_pool_parameter_FIT_score_bits_replayed':True,'fresh_process_nonce':nonce,'process_PID':os.getpid(),'runtime':runtime(),**counters()})
    else:
        write(directory/'fit_pool.json',payload);write(directory/'fit_seal.json',{'status':'FIT_POOL_SEALED_BEFORE_SELECT_LABELS','fold':fold,'authority':auth,'program':bind(PROGRAM),'closure':closure,'pool':bind(directory/'fit_pool.json'),'runtime':runtime(),'process_PID':os.getpid(),**counters()});fresh('validate-fit',fold)
    print(json.dumps({'phase':PHASE,'fold':fold,'LP_calls':pool['LP_call_count'],'pool_size':len(pool['candidate_pool']),'status':'PASS'}),flush=True)

def qualify_pool(fold,auth,base):
    directory=OUT/f'fold{fold:02d}';val=read(directory/'fit_validation.json');seal=read(checked(val['seal']))
    need(val['status']=='GROUP_HELD_SELECTION_FRESH_FIT_POOL_REPLAY_PASS' and val['fold']==fold and val['authority']==auth and val['program']==bind(PROGRAM) and val['SELECT_role_payload_reads']==val['outer_role_payload_reads']==val['EVAL_reads']==val['forbidden_read_attempts']==0,'FRESH_FIT_BEFORE_SELECT_LABELS')
    need(seal['fold']==fold and seal['authority']==auth and seal['program']==bind(PROGRAM),'FIT_SEAL_LINEAGE');pool=read(checked(seal['pool']));need(pool['BASE7']==base and pool['fold']==fold,'POOL_UNCHANGED_BASE7')
    candidates=pool['fit_result']['candidate_pool'];need(candidates and candidates[0]['candidate_id']=='BASE','POOL_HAS_EXACT_BASE')
    need(len({c['candidate_id'] for c in candidates})==len(candidates),'UNIQUE_POOL_CANDIDATES')
    return candidates,pool,{'fit_validation':bind(directory/'fit_validation.json'),'fit_seal':bind(directory/'fit_seal.json'),'fit_pool':seal['pool']}

def select_candidates(candidates,rows,base):
    op=pure_ops();bw,bb=op['parameter_tensor'](base);theta0=np.asarray([float(v) for v in bw]+[bb],dtype=np.float64)
    scored=[]
    for c in candidates:
        w,b=op['parameter_tensor'](c['parameters']);theta=np.asarray([float(v) for v in w]+[b],dtype=np.float64);s=op['train_score'](rows,theta,theta0);need(s is not None,'FINITE_ALL_SELECT_ACTIONS')
        scored.append({**c,'FIT_score':c['training_score'],'training_score':s})
    b=next(c for c in scored if c['candidate_id']=='BASE');base_good=set(b['training_score']['correct_query_ids'])
    ranked=min(scored,key=op['rank_candidate']);fit=min(candidates,key=op['rank_candidate'])
    rank_strict=Fraction(ranked['training_score']['equal_group_accuracy_fraction'])>Fraction(b['training_score']['equal_group_accuracy_fraction'])
    if not rank_strict:ranked=b
    eligible=[c for c in scored if base_good<=set(c['training_score']['correct_query_ids'])]
    primary=min(eligible,key=op['rank_candidate']);strict=Fraction(primary['training_score']['equal_group_accuracy_fraction'])>Fraction(b['training_score']['equal_group_accuracy_fraction'])
    if not strict:primary=b
    chosen={'BASE7':b,'FIT_SELECT':fit,'SELECT_RANK':ranked,'SELECT_PROTECT':primary}
    ledger={'all_candidate_SELECT_scores':[{k:c[k] for k in ('candidate_id','source_error_execution_ordinal','parameters','FIT_score','training_score')} for c in scored],'SELECT_BASE_correct_query_ids':sorted(base_good),'SELECT_protected_eligible_candidate_ids':[c['candidate_id'] for c in eligible],'SELECT_RANK_strict_group_gain':rank_strict,'SELECT_RANK_fallback_reason':None if rank_strict else 'NO_STRICT_SELECT_GROUP_GAIN','strict_SELECT_group_gain':strict,'primary_fallback_reason':None if strict else 'NO_STRICT_SELECT_GROUP_GAIN','selected_candidate_ids':{m:c['candidate_id'] for m,c in chosen.items()},'selected_parameter_sha256':{m:c['parameters']['parameter_sha256'] for m,c in chosen.items()},'SELECT_LP_calls':0,'selection_training_updates':0}
    parameters={m:c['parameters'] for m,c in chosen.items()};parameters['BASE7']=base
    for m,c in chosen.items():
        original=next(x for x in candidates if x['candidate_id']==c['candidate_id']);need(c['parameters']==original['parameters'],'EVERY_SELECTED_THETA_IS_BIT_EXACT_POOL_MEMBER')
    need(base_good<=set(primary['training_score']['correct_query_ids']),'PRIMARY_SELECT_BASE_PROTECTION')
    return parameters,ledger

def select_fold(fold,replay=False,nonce=None):
    auth=require_authority()
    if replay:check_fresh(nonce)
    byrole,base,closure=baseline_context(fold,replay);candidates,pool,poolclosure=qualify_pool(fold,auth,base);closure.update(poolclosure)
    # SELECT labels are opened only after complete sealed pool and fresh FIT replay.
    closure['SELECT']=attach_roles(byrole['SELECT'],fold,'SELECT');parameters,ledger=select_candidates(candidates,byrole['SELECT'],base)
    op=pure_ops();predictions=[]
    for row in byrole['OUTER']:
        preds={}
        for m in MODELS:
            w,b=op['parameter_tensor'](parameters[m]);z=row['X']@w+b;need(bool(torch.isfinite(z).all()),'FINITE_OUTER_ALL127');preds[m]=op['serialize_prediction'](row,z)
        predictions.append({k:row[k] for k in ('query_id','execution_ordinal','source_image_sha256','candidate_physical_rows','base_winner_position')}|{'predictions':preds})
    directory=OUT/f'fold{fold:02d}'
    if replay:
        seal=read(directory/'seal.json');need(seal['authority']==auth and seal['program']==bind(PROGRAM) and seal['closure']==closure,'FRESH_SELECT_SOURCE_BINDING');need(parameters==read(checked(seal['parameters'])) and ledger==read(checked(seal['selection_ledger'])) and predictions==read(checked(seal['predictions'])),'FRESH_POOL_SELECTION_ALL_OUTER_LOGIT_BITS');need(os.getpid()!=seal['process_PID'],'FRESH_SELECT_DISTINCT_PROCESS')
        write(directory/'validation.json',{'status':'GROUP_HELD_SELECTION_FRESH_SELECT_REPLAY_PASS','fold':fold,'seal':bind(directory/'seal.json'),'authority':auth,'program':bind(PROGRAM),'heldout_label_reads':0,'heldout_logit_checks':len(predictions)*127*4,'prior_BASE_logit_parity_checks':len(predictions)*127,'all_pool_SELECT_scores_selection_and_outer_logits_replayed':True,'SELECT_LP_calls':0,'selection_training_updates':0,'fresh_process_nonce':nonce,'process_PID':os.getpid(),'runtime':runtime(),**counters()})
    else:
        write(directory/'parameters.json',parameters);write(directory/'selection_ledger.json',ledger);write(directory/'predictions.json',predictions)
        write(directory/'seal.json',{'status':'GROUP_HELD_SELECTION_ALL_PARAMETERS_OUTER_PREDICTIONS_SEALED','fold':fold,'authority':auth,'program':bind(PROGRAM),'closure':closure,'parameters':bind(directory/'parameters.json'),'selection_ledger':bind(directory/'selection_ledger.json'),'predictions':bind(directory/'predictions.json'),'runtime':runtime(),'heldout_label_reads':0,'process_PID':os.getpid(),**counters()});fresh('validate-select',fold)
    print(json.dumps({'phase':PHASE,'fold':fold,'status':'PASS','selected':ledger['selected_candidate_ids']}),flush=True)

def check_fresh(nonce):
    need(nonce and os.environ.get('RC_GROUP_HELD_NONCE')==nonce and str(os.getppid())==os.environ.get('RC_GROUP_HELD_PARENT_PID'),'EXPLICIT_DISTINCT_FRESH_PROCESS')

def fresh(phase,fold=None):
    nonce=uuid.uuid4().hex;env=os.environ.copy();env['RC_GROUP_HELD_NONCE']=nonce;env['RC_GROUP_HELD_PARENT_PID']=str(os.getpid())
    command=[sys.executable,str(PROGRAM),'--phase',phase,'--nonce',nonce]
    if fold is not None:command+=['--fold',str(fold)]
    subprocess.run(command,check=True,env=env)


def joined_result():
    global OUTER_LABELS_RELEASED
    authority = require_authority()
    manifest = read(PARENT / 'fold_manifest.json')
    predictions, fold_sources = [], {}
    for fold in range(4):
        directory = OUT / ('fold%02d' % fold)
        val = read(directory / 'validation.json')
        need(val['status'] == 'GROUP_HELD_SELECTION_FRESH_SELECT_REPLAY_PASS' and val['fold'] == fold and
             val['authority'] == authority and val['program'] == bind(PROGRAM) and
             val['heldout_label_reads'] == val['EVAL_reads'] == val['gradient_optimizer_steps'] == val['base_refits'] == val['forbidden_read_attempts'] == 0,
             'ALL_FRESH_FOLD_GATES_BEFORE_HELDOUT_ROLE_READ')
        need(val['actual_LP_calls']==val['FIT_role_payload_reads']==val['outer_role_payload_reads']==0,'SELECT_REAL_COUNTER_BOUNDARIES')
        seal = read(checked(val['seal']))
        need(seal['authority'] == authority and seal['program'] == bind(PROGRAM) and seal['fold'] == fold,
             'SEALED_FOLD_LINEAGE')
        checked(seal['parameters'])
        rows = read(checked(seal['predictions']))
        expected = [r['execution_ordinal'] for r in manifest['records'] if r['fold'] == fold]
        need([r['execution_ordinal'] for r in rows] == expected and val['heldout_logit_checks'] == len(rows)*127*4,
             'EXACT_HELDOUT_GROUP_COVERAGE')
        predictions.extend(dict(r, fold=fold) for r in rows)
        fold_sources[str(fold)] = bind(directory / 'validation.json')
    need(sorted(r['execution_ordinal'] for r in predictions) == list(range(128)), 'ALL128_SEALED_BEFORE_LABELS')
    OUTER_LABELS_RELEASED = True
    roles = {}
    for fold in range(4):
        path = checked(manifest['role_projections']['fold%d_heldout_roles.json' % fold])
        need(path == (PARENT / ('fold%02d' % fold) / 'heldout_roles.json').resolve(), 'EXACT_OLD_HELDOUT_ROLES')
        for role in read(path)['records']:
            need(role['query_id'] not in roles, 'DISJOINT_HELDOUT_ROLE_GROUPS')
            roles[role['query_id']] = role
    operators = reused_operators()
    labels, mapping = operators['gallery_labels']()
    rows = []
    for pred in sorted(predictions, key=lambda r: r['execution_ordinal']):
        role = roles[pred['query_id']]
        need(pred['execution_ordinal'] == role['execution_ordinal'] and pred['source_image_sha256'] == role['source_image_sha256'],
             'POST_SEAL_HELDOUT_ROLE_JOIN')
        axis, identity = pred['candidate_physical_rows'], role['identity']
        correct = {m: labels[pred['predictions'][m]['final_physical_row']] == identity for m in MODELS}
        rows.append({'query_id': pred['query_id'], 'original_query_id': role['original_query_id'],
                     'execution_ordinal': pred['execution_ordinal'], 'group': role['group'], 'identity': identity,
                     'fold': pred['fold'], 'RAW_correct': labels[axis[pred['base_winner_position']]] == identity,
                     'target_in_natural_C128': identity in {labels[p] for p in axis}, 'correct': correct})
    groups = sorted({r['group'] for r in rows})
    need(len(groups) == len({r['identity'] for r in rows}) == 32, 'ALL32_GROUPS_IDENTITIES')
    scores, means = {}, {}
    for model in MODELS:
        per_group = {g: Fraction(sum(r['correct'][model] for r in rows if r['group'] == g), sum(r['group'] == g for r in rows)) for g in groups}
        average = sum(per_group.values(), Fraction(0))/32
        means[model] = average
        scores[model] = {'correct': sum(r['correct'][model] for r in rows),
                         'rescue_vs_RAW': sum(r['correct'][model] and not r['RAW_correct'] for r in rows),
                         'break_vs_RAW': sum(not r['correct'][model] and r['RAW_correct'] for r in rows),
                         'equal_group_accuracy': float(average), 'equal_group_accuracy_fraction': str(average),
                         'group_metrics': {g: {'correct': sum(r['correct'][model] for r in rows if r['group'] == g),
                             'count': sum(r['group'] == g for r in rows), 'accuracy_fraction': str(value)} for g, value in per_group.items()}}
    retained = {m: [r['original_query_id'] for r in rows if r['correct']['BASE7'] and not r['correct'][m]] for m in MODELS[1:]}
    gates = {'count_strictly_above_BASE7': scores['SELECT_PROTECT']['correct'] > scores['BASE7']['correct'],
             'all_BASE_correct_retained': not retained['SELECT_PROTECT'],
             'equal_group_strictly_above_BASE7': means['SELECT_PROTECT'] > means['BASE7']}
    return {'status': 'TRAIN128_GROUP_HELD_SELECTION_JOINED', 'authority': authority, 'program': bind(PROGRAM),
            'contract': CONTRACT, 'posthoc_TRAIN_reuse': True, 'prior_OOF_validation': bind(PARENT / 'validation.json'),
            'fold_validations': fold_sources, 'gallery_mapping_sha256': mapping,
            'query_count': 128, 'group_count': 32, 'identity_count': 32,
            'RAW_correct': sum(r['RAW_correct'] for r in rows), 'target_absent_count': sum(not r['target_in_natural_C128'] for r in rows),
            'scores': scores, 'BASE_correct_losses': retained,
            'fold_metrics': {str(f): {'count': sum(r['fold'] == f for r in rows),
                 'RAW_correct': sum(r['RAW_correct'] for r in rows if r['fold'] == f),
                 'correct': {m: sum(r['correct'][m] for r in rows if r['fold'] == f) for m in MODELS}} for f in range(4)},
            'paired_group_statistics': operators['paired_group_statistics'](rows, groups),
            'gates': gates, 'OOF_gate_pass': all(gates.values()),
            'next_stage': 'SEPARATE_FINAL_FIT_AUTHORITY_ELIGIBLE' if all(gates.values()) else 'STOP_THIS_FAMILY_WITHOUT_EVAL_ACCESS',
            'rows': rows, 'EVAL_reads': 0, 'gradient_optimizer_steps': 0, 'base_refits': 0,
            'final_ec7_fit_performed': False, 'forbidden_read_attempts': len(BLOCKED)}

def join(replay=False, nonce=None):
    if replay:
        need(nonce and os.environ.get('RC_GROUP_HELD_NONCE') == nonce, 'FRESH_JOIN_NONCE')
    result = joined_result()
    if replay:
        need(read(OUT / 'result.json') == result, 'FRESH_JOIN_COUNTS_GROUP_STATS_GATES_EXACT')
        write(OUT / 'validation.json', {'status': 'TRAIN128_GROUP_HELD_SELECTION_ALL_FOLDS_JOIN_REPLAY_PASS',
              'authority': bind(AUTH), 'program': bind(PROGRAM), 'result': bind(OUT / 'result.json'),
              'process_PID': os.getpid(), 'fresh_process_nonce': nonce, 'query_count': 128, 'group_count': 32,
              'all_heldout_logit_checks': 128*127*4, 'OOF_gate_pass': result['OOF_gate_pass'],
              'EVAL_reads': 0, 'gradient_optimizer_steps': 0, 'base_refits': 0, 'forbidden_read_attempts': len(BLOCKED)})
    else:
        write(OUT / 'result.json', result)
        fresh('validate-join')
    print(json.dumps({'phase': PHASE, 'gate': result['OOF_gate_pass'],
                      'correct': {m: result['scores'][m]['correct'] for m in MODELS}}), flush=True)

def synthetic_preflight():
    op=pure_ops();weight=torch.tensor([1.,0.,0.,0.,0.,0.],dtype=torch.float64)
    def toy(key,ordinal,target,first=-1.,second=0.):
        x=torch.zeros((127,6),dtype=torch.float64);x[:,0]=-3.;x[0,0]=first;x[0,1]=second
        return {'query_id':key,'execution_ordinal':ordinal,'group':'g'+str(ordinal),'candidate_physical_rows':list(range(128)),'challenger_positions':list(range(1,128)),'base_winner_position':0,'target_position':target,'X':x}
    fit=[toy('fit_hold',0,0,-2.),toy('fit_error',1,1,-1.,1.)];a=op['solve_training'](fit,weight,0.,'PROTECTED7');b=op['solve_training'](fit,weight,0.,'PROTECTED7',True);need(a==b and a['selected_candidate_id']=='fit_error','SYNTHETIC_FIT_POOL_REPLAY')
    base=op['parameter_values']([1.,0.,0.,0.,0.,0.,0.]);pool=a['candidate_pool']
    # FIT gain can break an unseen SELECT hold; primary must preserve and fall back.
    selection=[toy('select_hold',2,0,-1.,1.)];parameters,ledger=select_candidates(pool,selection,base)
    need(ledger['selected_candidate_ids']['SELECT_PROTECT']=='BASE' and parameters['SELECT_PROTECT']==base,'SELECT_PROTECT_FALLBACK_ON_HOLD_BREAK')
    selection=[toy('select_hold',2,0,-2.),toy('select_error',3,1,-1.,1.)];parameters,ledger=select_candidates(pool,selection,base)
    need(ledger['selected_candidate_ids']['SELECT_PROTECT']=='fit_error','SELECT_PROTECT_ACCEPTS_INDEPENDENT_GAIN')
    for c in pool:need(any(parameters['SELECT_PROTECT']==z['parameters'] for z in pool),'POOL_MEMBERSHIP_ONLY')
    probes=[('fit-fold',0,OUT/'fold00/select_roles.json'),('fit-fold',0,OUT/'fold01/fit_roles.json'),('fit-fold',0,OUT/'fold00/selection_ledger.json'),('select-fold',0,OUT/'fold00/fit_roles.json'),('select-fold',0,PARENT/'fold00/heldout_roles.json'),('fit-fold',0,ROOT/'results/rc_opened_synthetic/no_file'),('fit-fold',0,ROOT/'results/rc_train128_protected_projection_oof4_v1/fold00/parameters.json')]
    global PHASE,FOLD,HASH_DEPTH
    save=(PHASE,FOLD,HASH_DEPTH);count=len(BLOCKED)
    for phase,fold,path in probes:
        PHASE,FOLD,HASH_DEPTH=phase,fold,1
        try:audit('open',(str(path),'r',0))
        except PermissionError:pass
        else:raise RuntimeError('SYNTHETIC_ROLE_OR_HASH_GUARD_FAILURE')
    PHASE,FOLD,HASH_DEPTH=save;del BLOCKED[count:]
    value={'status':'TRAIN128_GROUP_HELD_SELECTION_SYNTHETIC_PREFLIGHT_PASS','program':bind(PROGRAM),'launcher':bind(LAUNCH),'plan':bind(PLAN),'pure_LP_program':bind(PURE),'contract':CONTRACT,'statistics_contract':STATISTICS_CONTRACT,'runtime':runtime(),'synthetic_LP_calls':2,'synthetic_guard_probes':len(probes),'SELECT_selection_updates':0,'natural_feature_reads':0,'natural_label_reads':0,'natural_solver_calls':0,'EVAL_reads':0,'forbidden_read_attempts':len(BLOCKED)}
    path=PREFLIGHT/sha(PROGRAM)/'result.json';write(path,value);print(json.dumps({'status':value['status'],'preflight':bind(path)}),flush=True)

def main():
    global PHASE,FOLD
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('preflight','metadata-prepare','validate-metadata','fit-fold','validate-fit','select-fold','validate-select','join','validate-join','run'));parser.add_argument('--fold',type=int,choices=range(4));parser.add_argument('--nonce');args=parser.parse_args();PHASE,FOLD=args.phase,args.fold
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_flush_denormal(False);sys.addaudithook(audit)
    if PHASE=='preflight':synthetic_preflight()
    elif PHASE in ('metadata-prepare','validate-metadata'):prepare(PHASE=='validate-metadata',args.nonce)
    elif PHASE in ('fit-fold','validate-fit'):
        need(FOLD is not None,'FOLD_REQUIRED');fit_fold(FOLD,PHASE=='validate-fit',args.nonce)
    elif PHASE in ('select-fold','validate-select'):
        need(FOLD is not None,'FOLD_REQUIRED');select_fold(FOLD,PHASE=='validate-select',args.nonce)
    elif PHASE in ('join','validate-join'):join(PHASE=='validate-join',args.nonce)
    else:
        require_authority()
        for fold in range(4):
            subprocess.run([sys.executable,str(PROGRAM),'--phase','fit-fold','--fold',str(fold)],check=True)
            subprocess.run([sys.executable,str(PROGRAM),'--phase','select-fold','--fold',str(fold)],check=True)
        subprocess.run([sys.executable,str(PROGRAM),'--phase','join'],check=True)

if __name__=='__main__':main()
