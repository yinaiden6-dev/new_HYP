#!/usr/bin/env python3
"""TRAIN-only one-pass protected L1 projection screen with frozen fold BASE7.

Numerical LPs propose parameters. Rational endpoint margins and actual FP64
all-challenger actions determine acceptance. No EVAL or oracle artifact access.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, subprocess, sys, uuid
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy
from scipy.optimize import linprog
import torch
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / 'src'))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC
OUT = ROOT / 'results/rc_train128_protected_projection_oof4_v1'
PREFLIGHT = ROOT / 'results/rc_train128_protected_projection_oof4_v1_preflight'
PARENT = ROOT / 'results/rc_train128_disagreement_oof4_v1'
OLD = ROOT / 'programs/run_rc_train128_disagreement_oof4_v1.py'
OLD_SHA = 'd7302eca66c7c6469a7f5ef99a77afeae8af4b25c3a253a1bdcbb2da7af3a715'
OLD_VALIDATION_SHA = '43ccd7e30a50e52f4c4783d4d17304f9b4ec205bac1ca09674e2a82fca20d8c0'
OLD_RESULT_SHA = '9fdf72f68ab3b1444c1961a9d842b4a8d9267bdfebaee30a081f7b7947a79d0d'
INPUT = ROOT / 'results/rc_original7_train128_inputs_v1'
META = ROOT / 'results/rc_original7_train128_manifest_v1'
PLAN = ROOT / 'plan/RC_TRAIN128_PROTECTED_PROJECTION_OOF4_V1_20260910.md'
AUTH = ROOT / 'registry/rc_train128_protected_projection_oof4_authority_v1_20260910.json'
LAUNCH = ROOT / 'slurm/rc_train128_protected_projection_oof4_v1_dev_cpuonly_59m.sbatch'
GALLERY_IMAGES = ROOT.parents[2] / 'dailymed/data/box_flat_20000_images/data/raw_images'
MODELS = ('BASE7', 'PROTECTED7', 'REPAIR_ONLY7')
ARM = 'C_PAIRED'
DEADLINE = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
PHASE, FOLD = None, None
HASH_DEPTH = 0
BLOCKED = []
LP_OPTIONS = {'presolve': True, 'time_limit': 30.0, 'dual_feasibility_tolerance': 1e-9,
              'primal_feasibility_tolerance': 1e-9}
CONTRACT = {
 'scope': 'post-hoc TRAIN128 grouped OOF; no EVAL or deployment',
 'models': list(MODELS), 'primary': 'PROTECTED7', 'fold_count': 4,
 'prior_OOF_program_sha256': OLD_SHA, 'base_refits': 0, 'new_features': 0,
 'training_labels': 'current fold train_roles only; no heldout role reads before four seals',
 'protected': 'all TRAIN images correctly classified by fixed fold BASE7',
 'errors': 'all original TRAIN BASE errors whose target is naturally in C128',
 'delta': 'exact rational minimum of original strict action margins over protected; empty or nonpositive -> exact BASE fallback',
 'candidate_search': 'one LP per original error per arm; one pass only; BASE included in each pool',
 'LP': 'min sum(Delta_plus+Delta_minus), theta=theta0+Delta_plus-Delta_minus; nonnegative increments',
 'PROTECTED7_rows': 'all127 action constraints for protected TRAIN queries and one error',
 'REPAIR_ONLY7_rows': 'all127 action constraints for that one error only',
 'numeric_rhs': 'float(delta); search goal only, never certified exact delta floor',
 'solver': 'scipy.optimize.linprog highs-ds', 'solver_options': LP_OPTIONS,
 'acceptance': 'numerical success then all exact endpoint margins>0 and actual FP64 error repair; PROTECTED additionally preserves every protected TRAIN image',
 'failure_labels': 'numerical/action rejection, not mathematical infeasibility',
 'candidate_selection': 'descending exact equal-group accuracy, descending correct images, ascending exact FP64 L1 from BASE, ascending original error execution ordinal; BASE ordinal=-1',
 'dtype': 'float64', 'threshold': 'unchanged >0 SWITCH, physical-row tie, complete natural C128',
 'gate': ['PROTECTED correct images>BASE', 'every BASE-correct heldout image retained', 'equal-group PROTECTED-BASE>0'],
 'group_gate_interpretation': 'logical consistency under complete preservation, not independent evidence',
 'REPAIR_ONLY_gate': False, 'fresh_validation': 'rerun all LPs; bit exact parameters, candidate choices, complete heldout logits',
 'claim_limits': 'one-pass finite candidate pool; no whole-class maximal rescue or unique global closest-head claim',
 'EVAL_access': False,
}
STATISTICS_CONTRACT = {
 'paired_comparisons': [['PROTECTED7','BASE7'],['REPAIR_ONLY7','BASE7'],['PROTECTED7','REPAIR_ONLY7']],
 'unit': 'source group; equal-weight group accuracy difference',
 'bootstrap_draws': 10000, 'bootstrap_seed': 20260910,
 'bootstrap_interval': '95% percentile numpy.quantile linear',
 'signflip': 'exact two-sided inclusive rational group signflip', 'p_value_is_gate': False,
}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)

def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    s = str(p).lower()
    hard = any(token in s for token in ('/grozi/', 'd1-mi', 'd1_mi', '/target_join/', '/role_shards/',
                  '/rc_opened_', '/rc_original7_eval', 'direction_capacity', 'angle_capacity', 'origin_linear_projection'))
    if hard:
        BLOCKED.append(str(p))
        raise PermissionError('EVAL_OR_PROTECTED_INPUT_FORBIDDEN')
    if HASH_DEPTH:
        return  # Byte binding only; no role or evaluation payload deserialization.
    forbidden = False
    if ROOT / 'results' in p.parents:
        allowed = (OUT in p.parents or PREFLIGHT in p.parents or
                   p in (INPUT / 'validation.json', INPUT / 'feature_records.json',
                         META / 'worker_manifest.json', PARENT / 'fold_manifest.json', PARENT / 'validation.json'))
        if PARENT in p.parents and p.parent.name.startswith('fold'):
            allowed = p.name in ('validation.json', 'seal.json', 'parameters.json', 'predictions.json')
            if p.name == 'train_roles.json':
                allowed = PHASE in ('solve-fold', 'validate-fold') and p.parent == PARENT / ('fold%02d' % FOLD)
            if p.name == 'heldout_roles.json':
                allowed = PHASE in ('join', 'validate-join')
        forbidden = not allowed
        if PHASE == 'preflight':
            forbidden |= p == INPUT / 'feature_records.json' or PARENT in p.parents
    if forbidden:
        BLOCKED.append(str(p))
        raise PermissionError('PROTECTED_PROJECTION_TRAIN_FOLD_READ_BOUNDARY:' + str(p))

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

def read(path):
    return json.loads(Path(path).read_text())

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

def top_index(logits, row):
    candidates, axis = row['challenger_positions'], row['candidate_physical_rows']
    return max(range(len(candidates)), key=lambda j: (float(logits[j]), -axis[candidates[j]]))

def serialize_prediction(row, logits):
    index = top_index(logits, row)
    cs, axis = row['challenger_positions'], row['candidate_physical_rows']
    final = cs[index] if float(logits[index]) > 0 else row['base_winner_position']
    return {'all127_logits_binary64': [hx(v) for v in logits], 'best_challenger_position': cs[index],
            'best_logit_binary64': hx(logits[index]), 'final_position': final, 'final_physical_row': axis[final],
            'action': 'SWITCH' if float(logits[index]) > 0 else 'HOLD'}

def load_fold(fold, independent=False):
    operators = reused_operators(independent)
    rows = operators['qualified_rows']()
    manifest = read(PARENT / 'fold_manifest.json')
    directory = PARENT / ('fold%02d' % fold)
    val = read(directory / 'validation.json')
    seal = read(checked(val['seal']))
    parent_root = read(PARENT / 'validation.json')
    need(val['status'] == 'TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS' and val['fold'] == fold and
         val['program'] == parent_root['program'] and val['authority'] == parent_root['authority'] and
         val['heldout_label_reads'] == val['EVAL_reads'] == val['forbidden_read_attempts'] == 0, 'PRIOR_BASE_FOLD_QUALIFICATION')
    need(seal['fold'] == fold and seal['authority'] == val['authority'] and seal['program'] == val['program'], 'PRIOR_BASE_SEAL_LINEAGE')
    base = read(checked(seal['parameters']))['BASE7']
    old_predictions = read(checked(seal['predictions']))
    weight = torch.tensor(list(map(float.fromhex, base['weight_binary64'])), dtype=torch.float64)
    bias = float.fromhex(base['bias_binary64'])
    need(weight.shape == (6,) and torch.isfinite(weight).all() and math.isfinite(bias), 'EXACT_SIX_PLUS_BIAS_PARAMETERS')
    roles_path = checked(manifest['role_projections']['fold%d_train_roles.json' % fold])
    need(roles_path == (directory / 'train_roles.json').resolve(), 'SAME_FOLD_TRAIN_ROLES')
    roles = read(roles_path)['records']
    rolemap = {r['query_id']: r for r in roles}
    need(set(rolemap) == {r['query_id'] for r in manifest['records'] if r['fold'] != fold} and
         len({r['group'] for r in roles}) == 24, 'ALL24_TRAINING_GROUPS_ONLY')
    labels, mapping = operators['gallery_labels']()
    train, held = [], []
    for row, assignment in zip(rows, manifest['records'], strict=True):
        need(row['query_id'] == assignment['query_id'], 'OLD_FOLD_AXIS')
        if assignment['fold'] == fold:
            need(row['query_id'] not in rolemap, 'HELDOUT_ROLES_NOT_IN_TRAIN')
            held.append(row)
        else:
            identity = rolemap[row['query_id']]['identity']
            targets = [p for p, physical in enumerate(row['candidate_physical_rows']) if labels[physical] == identity]
            need(len(targets) <= 1, 'NATURAL_C128_IDENTITY_DEDUP')
            row['target_position'] = targets[0] if targets else None
            row['group'] = rolemap[row['query_id']]['group']
            train.append(row)
    need(len(train)+len(held) == 128 and len(old_predictions) == len(held), 'ALL128_RETAINED')
    for row, previous in zip(held, old_predictions, strict=True):
        need(row['query_id'] == previous['query_id'] and row['execution_ordinal'] == previous['execution_ordinal'] and
             row['candidate_physical_rows'] == previous['candidate_physical_rows'], 'PRIOR_HELDOUT_AXIS')
        need(serialize_prediction(row, row['X'] @ weight + bias) == previous['predictions']['BASE7'],
             'PRIOR_BASE_ALL127_LOGITS_AND_ACTION_EXACT')
    closure = {'prior_fold_validation': bind(directory / 'validation.json'),
               'prior_fold_seal': bind(directory / 'seal.json'), 'prior_parameters': seal['parameters'],
               'prior_predictions': seal['predictions'], 'train_roles': bind(roles_path),
               'fold_manifest': bind(PARENT / 'fold_manifest.json'), 'gallery_mapping_sha256': mapping,
               'training_ordinals': [r['execution_ordinal'] for r in train],
               'heldout_ordinals': [r['execution_ordinal'] for r in held]}
    return train, held, weight, bias, base, closure

def joined_result():
    authority = require_authority()
    manifest = read(PARENT / 'fold_manifest.json')
    predictions, fold_sources = [], {}
    for fold in range(4):
        directory = OUT / ('fold%02d' % fold)
        val = read(directory / 'validation.json')
        need(val['status'] == 'PROTECTED_PROJECTION_OOF4_FRESH_LP_REPLAY_PASS' and val['fold'] == fold and
             val['authority'] == authority and val['program'] == bind(PROGRAM) and
             val['heldout_label_reads'] == val['EVAL_reads'] == val['gradient_optimizer_steps'] == val['base_refits'] == val['forbidden_read_attempts'] == 0,
             'ALL_FRESH_FOLD_GATES_BEFORE_HELDOUT_ROLE_READ')
        seal = read(checked(val['seal']))
        need(seal['authority'] == authority and seal['program'] == bind(PROGRAM) and seal['fold'] == fold,
             'SEALED_FOLD_LINEAGE')
        checked(seal['parameters'])
        rows = read(checked(seal['predictions']))
        expected = [r['execution_ordinal'] for r in manifest['records'] if r['fold'] == fold]
        need([r['execution_ordinal'] for r in rows] == expected and val['heldout_logit_checks'] == len(rows)*127*3,
             'EXACT_HELDOUT_GROUP_COVERAGE')
        predictions.extend(dict(r, fold=fold) for r in rows)
        fold_sources[str(fold)] = bind(directory / 'validation.json')
    need(sorted(r['execution_ordinal'] for r in predictions) == list(range(128)), 'ALL128_SEALED_BEFORE_LABELS')
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
    gates = {'count_strictly_above_BASE7': scores['PROTECTED7']['correct'] > scores['BASE7']['correct'],
             'all_BASE_correct_retained': not retained['PROTECTED7'],
             'equal_group_strictly_above_BASE7': means['PROTECTED7'] > means['BASE7']}
    return {'status': 'TRAIN128_PROTECTED_PROJECTION_OOF4_JOINED', 'authority': authority, 'program': bind(PROGRAM),
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
        need(nonce and os.environ.get('RC_PROTECTED_PROJECTION_FRESH_NONCE') == nonce, 'FRESH_JOIN_NONCE')
    result = joined_result()
    if replay:
        need(read(OUT / 'result.json') == result, 'FRESH_JOIN_COUNTS_GROUP_STATS_GATES_EXACT')
        write(OUT / 'validation.json', {'status': 'TRAIN128_PROTECTED_PROJECTION_OOF4_ALL_FOLDS_JOIN_REPLAY_PASS',
              'authority': bind(AUTH), 'program': bind(PROGRAM), 'result': bind(OUT / 'result.json'),
              'process_PID': os.getpid(), 'fresh_process_nonce': nonce, 'query_count': 128, 'group_count': 32,
              'all_heldout_logit_checks': 128*127*3, 'OOF_gate_pass': result['OOF_gate_pass'],
              'EVAL_reads': 0, 'gradient_optimizer_steps': 0, 'base_refits': 0, 'forbidden_read_attempts': len(BLOCKED)})
    else:
        write(OUT / 'result.json', result)
        fresh('validate-join')
    print(json.dumps({'phase': PHASE, 'gate': result['OOF_gate_pass'],
                      'correct': {m: result['scores'][m]['correct'] for m in MODELS}}), flush=True)


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'CUDA_used': False,
            'solver': 'highs-ds', 'solver_options': LP_OPTIONS, 'flush_denormal': False}


def manual_feature(raw, evidence, challenger, winner):
    mean = sum(float(v) for v in raw) / len(raw)
    sd = (sum((float(v)-mean)**2 for v in raw) / len(raw))**.5
    def sym(a, b):
        return (float(a)-float(b))/(abs(float(a))+abs(float(b))+1e-12)
    ec, ew = evidence[challenger], evidence[winner]
    s,t,m,n = float(ec['real_score']),float(ew['real_score']),float(ec['visibility_mass']),float(ew['visibility_mass'])
    return torch.tensor([(float(raw[challenger])-float(raw[winner]))/max(sd,1e-12),sym(s,t),sym(m,n),
      sym(s/max(m,1e-12),t/max(n,1e-12)),sym(s-float(ec['query_control_score']),t-float(ew['query_control_score'])),
      sym(s-float(ec['reference_control_score']),t-float(ew['reference_control_score']))],dtype=torch.float64)


def reused_operators(independent=False):
    need(sha(OLD) == OLD_SHA, 'FROZEN_PARENT_PROGRAM')
    names = {'qualified_rows','gallery_labels','exact_group_signflip','paired_group_statistics'}
    nodes = [n for n in ast.parse(OLD.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == names, 'REUSED_UNQUARANTINED_METHODS_ONLY')
    namespace = dict(globals())
    if independent:
        namespace['FC'] = SimpleNamespace(candidate_feature=manual_feature)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OLD),'exec'),namespace)
    return {name: namespace[name] for name in names}


def static_sources():
    paths = {'program':PROGRAM,'launcher':LAUNCH,'plan':PLAN,'prior_program':OLD,
       'prior_root_validation':PARENT/'validation.json','prior_result_bytes':PARENT/'result.json',
       'prior_authority':ROOT/'registry/rc_train128_disagreement_oof4_execution_authority_v1_20260910.json',
       'fold_manifest':PARENT/'fold_manifest.json','input_validation':INPUT/'validation.json',
       'input_features':INPUT/'feature_records.json','worker_manifest':META/'worker_manifest.json',
       'feature_core':Path(FC.__file__),
       'gallery_identity_program':ROOT/'src/rc_aslo_xf/gallery_identity_repair.py',
       'gallery_identity_registry':ROOT/'registry/gallery_identity_repair_v1.json',
       'gallery_identity_contract':ROOT/'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'}
    for fold in range(4):
        for name in ('train_roles','heldout_roles','validation','seal','parameters','predictions'):
            paths[f'prior_fold{fold}_{name}'] = PARENT/f'fold{fold:02d}'/(name+'.json')
    # Hard-coded allowlist contains no EVAL or opened-capacity path. audit applies
    # hard oracle exclusions even during this byte-hash-only qualification.
    need(not any('/rc_opened_' in str(p).lower() for p in paths.values()), 'NO_ORACLE_STATIC_SOURCE')
    return {key:bind(path) for key,path in paths.items()}


def require_authority():
    need(os.environ.get('SLURM_JOB_ID') and datetime.now(timezone.utc)<DEADLINE, 'SLURM_AND_DEADLINE_REQUIRED')
    a=read(AUTH)
    need(a['status']=='TRAIN128_PROTECTED_PROJECTION_OOF4_AUTHORIZED' and a['contract']==CONTRACT,'AUTHORITY_CONTRACT')
    need(PHASE in a['allowed_stages'] and a['cutoff_UTC']==DEADLINE.isoformat(),'AUTHORIZED_PHASE_AND_DEADLINE')
    need(a['source_bindings']==static_sources(),'COMPLETE_FROZEN_SOURCES')
    pre=read(checked(a['preflight']))
    need(pre['status']=='TRAIN128_PROTECTED_PROJECTION_OOF4_SYNTHETIC_PREFLIGHT_PASS' and
         pre['program']==bind(PROGRAM) and pre['launcher']==bind(LAUNCH) and pre['plan']==bind(PLAN)
         and pre['prior_program']==bind(OLD) and pre['contract']==CONTRACT and pre['runtime']==runtime(),'SYNTHETIC_PREFLIGHT_BINDING')
    need(sha(PARENT/'validation.json')==OLD_VALIDATION_SHA and sha(PARENT/'result.json')==OLD_RESULT_SHA,'PRIOR_OOF_FIXED_PINS')
    v=read(PARENT/'validation.json')
    need(v['status']=='TRAIN128_DISAGREEMENT_OOF4_ALL_FOLDS_AND_JOIN_REPLAY_PASS' and
         v['result']==bind(PARENT/'result.json') and v['program']==bind(OLD) and
         v['all_heldout_logit_checks']==128*127*3 and v['EVAL_reads']==v['forbidden_read_attempts']==0,'PRIOR_FOLD_BASE_QUALIFIED')
    return bind(AUTH)


def exact_constraints(row):
    target=row['target_position']; need(target is not None,'NO_CONSTRAINT_FOR_ABSENT_TARGET')
    cs=row['challenger_positions']; winner=row['base_winner_position']
    vectors={c:[Fraction.from_float(float(v)) for v in row['X'][i]]+[Fraction(1)] for i,c in enumerate(cs)}
    if target==winner:
        result=[[-v for v in vectors[c]] for c in cs]
    else:
        need(target in vectors,'TARGET_ON_CHALLENGER_AXIS')
        result=[vectors[target]]+[[a-b for a,b in zip(vectors[target],vectors[c])] for c in cs if c!=target]
    need(len(result)==127 and all(len(r)==7 for r in result),'ALL127_EXACT_ACTION_ROWS')
    return result


def dot(row,theta):
    return sum((a*b for a,b in zip(row,theta)),Fraction(0))


def parameter_values(theta):
    values=list(map(float,theta));need(len(values)==7 and all(math.isfinite(v) for v in values),'FINITE7_PARAMETERS')
    payload={'weight':values[:6],'bias':values[6]}
    return {'weight_binary64':[hx(v) for v in values[:6]],'bias_binary64':hx(values[6]),
            'parameter_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()}


def parameter_tensor(parameters):
    return torch.tensor(list(map(float.fromhex,parameters['weight_binary64'])),dtype=torch.float64),float.fromhex(parameters['bias_binary64'])


def train_score(rows,theta,theta0):
    w=torch.tensor(list(map(float,theta[:6])),dtype=torch.float64); b=float(theta[6])
    correct=[]
    for row in rows:
        logits=row['X']@w+b
        if not bool(torch.isfinite(logits).all()):
            return None
        final=serialize_prediction(row,logits)['final_position']
        correct.append(row['target_position'] is not None and final==row['target_position'])
    groups=sorted({r['group'] for r in rows});need(groups,'NONEMPTY_TRAIN_GROUPS')
    means=[Fraction(sum(ok for r,ok in zip(rows,correct) if r['group']==g),sum(r['group']==g for r in rows)) for g in groups]
    l1=sum((abs(Fraction.from_float(float(a))-Fraction.from_float(float(b))) for a,b in zip(theta,theta0)),Fraction(0))
    return {'equal_group_accuracy_fraction':str(sum(means,Fraction(0))/len(groups)),
      'correct':sum(correct),'L1_from_BASE_fraction':str(l1),
      'correct_query_ids':[r['query_id'] for r,ok in zip(rows,correct) if ok],
      'training_image_count':len(rows),'training_group_count':len(groups)}


def rank_candidate(candidate):
    score=candidate['training_score']
    return (-Fraction(score['equal_group_accuracy_fraction']),-score['correct'],
            Fraction(score['L1_from_BASE_fraction']),candidate['source_error_execution_ordinal'])


def build_context(rows,weight,bias):
    theta0=np.asarray([float(v) for v in weight]+[float(bias)],dtype=np.float64)
    base_score=train_score(rows,theta0,theta0);need(base_score is not None,'FINITE_BASE_TRAIN_SCORES')
    good=set(base_score['correct_query_ids'])
    protected=[r for r in rows if r['query_id'] in good]
    errors=[r for r in rows if r['query_id'] not in good and r['target_position'] is not None]
    need([r['execution_ordinal'] for r in errors]==sorted(r['execution_ordinal'] for r in errors),'ORIGINAL_ERROR_EXECUTION_ORDER')
    exact={r['query_id']:exact_constraints(r) for r in rows if r['target_position'] is not None}
    p_constraints=[a for r in protected for a in exact[r['query_id']]]
    fractions=[Fraction.from_float(float(v)) for v in theta0]
    delta=min((dot(a,fractions) for a in p_constraints),default=None)
    return {'theta0':theta0,'base_score':base_score,'protected':protected,'errors':errors,
            'exact':exact,'protected_constraints':p_constraints,'delta':delta}


def solve_training(rows,weight,bias,model,independent=False,context=None):
    need(model in MODELS[1:],'PREDECLARED_MODEL_ARM')
    context=build_context(rows,weight,bias) if context is None else context
    theta0=context['theta0'];delta=context['delta'];errors=context['errors'];protected=context['protected']
    original={'candidate_id':'BASE','source_error_query_id':None,'source_error_execution_ordinal':-1,
              'parameters':parameter_values(theta0),'training_score':context['base_score']}
    pool=[original];ledger=[]
    fallback='EMPTY_PROTECTED_SET' if delta is None else 'NONPOSITIVE_EXACT_BASE_PROTECTION_MARGIN' if delta<=0 else None
    delta_float=float(delta) if delta is not None else None
    if not fallback and (not math.isfinite(delta_float) or delta_float<=0):
        fallback='DELTA_NOT_POSITIVE_FINITE_BINARY64'
    if not fallback:
        for error in errors:
            required_rows=(protected+[error]) if model=='PROTECTED7' else [error]
            exact=[a for row in required_rows for a in context['exact'][row['query_id']]]
            A=np.asarray([[float(v) for v in a] for a in exact],dtype=np.float64)
            entry={'error_query_id':error['query_id'],'execution_ordinal':error['execution_ordinal'],
                   'required_query_count':len(required_rows),'constraint_count':len(exact),
                   'delta_search_fraction':str(delta),'delta_search_binary64':hx(delta_float)}
            # Fixed increment order plus[0:7], minus[7:14]; two eager operations
            # define the final deployable binary64 candidate, not LP real values.
            try:
                solution=linprog(np.ones(14),A_ub=np.concatenate((-A,A),axis=1),
                    b_ub=A@theta0-delta_float,bounds=[(0,None)]*14,method='highs-ds',options=LP_OPTIONS)
            except Exception as exc:
                entry.update(status='NUMERICAL_SOLVER_EXCEPTION_REJECTED',exception_type=type(exc).__name__)
                ledger.append(entry);continue
            entry['solver_status']=int(solution.status)
            entry['solver_success']=bool(solution.success)
            if not solution.success or solution.x is None:
                entry['status']='NUMERICAL_SEARCH_REJECTED_NO_MATHEMATICAL_INFEASIBILITY_CLAIM'
                ledger.append(entry);continue
            increments=np.asarray(solution.x,dtype=np.float64)
            if increments.shape!=(14,) or not np.isfinite(increments).all() or (increments<0).any():
                entry['status']='INVALID_NUMERIC_INCREMENTS_REJECTED';ledger.append(entry);continue
            theta=np.subtract(np.add(theta0,increments[:7]),increments[7:])
            if not np.isfinite(theta).all():
                entry['status']='NONFINITE_CANDIDATE_REJECTED';ledger.append(entry);continue
            exact_theta=[Fraction.from_float(float(v)) for v in theta]
            minimum=min(dot(a,exact_theta) for a in exact)
            entry.update(actual_min_required_margin_fraction=str(minimum),
                         actual_min_over_delta_fraction=str(minimum/delta),
                         exact_delta_floor_met=minimum>=delta,
                         proposed_parameters=parameter_values(theta))
            if minimum<=0:
                entry['status']='EXACT_NONPOSITIVE_MARGIN_REJECTED';ledger.append(entry);continue
            score=train_score(rows,theta,theta0)
            if score is None:
                entry['status']='NONFINITE_FULL_TRAIN_ACTION_REJECTED';ledger.append(entry);continue
            correct=set(score['correct_query_ids']);pkeys={r['query_id'] for r in protected}
            repaired=error['query_id'] in correct;retained=pkeys<=correct
            entry.update(error_repaired_in_FP64=repaired,protected_retained_in_FP64=retained,
                         training_score=score)
            if not repaired or (model=='PROTECTED7' and not retained):
                entry['status']='FP64_ACTION_REJECTED';ledger.append(entry);continue
            entry['status']='ACCEPTED_EXACT_STRICT_AND_FP64_ACTION'
            pool.append({'candidate_id':error['query_id'],'source_error_query_id':error['query_id'],
              'source_error_execution_ordinal':error['execution_ordinal'],
              'parameters':parameter_values(theta),'training_score':score})
            ledger.append(entry)
    selected=min(pool,key=rank_candidate)
    original_good=set(context['base_score']['correct_query_ids'])
    selected_good=set(selected['training_score']['correct_query_ids'])
    if model=='PROTECTED7':
        need(original_good<=selected_good,
             'PRIMARY_TRAIN_PROTECTION_REQUIRED')
    return {**selected['parameters'],'selected_candidate_id':selected['candidate_id'],
        'selected_source_error_execution_ordinal':selected['source_error_execution_ordinal'],
        'selected_training_score':selected['training_score'],'protected_training_query_ids':[r['query_id'] for r in protected],
        'original_eligible_error_query_ids':[r['query_id'] for r in errors],
        'original_error_target_absent_query_ids':[r['query_id'] for r in rows if r['target_position'] is None],
        'selected_training_rescue_query_ids':[r['query_id'] for r in rows if r['query_id'] in selected_good-original_good],
        'selected_training_BASE_loss_query_ids':[r['query_id'] for r in rows if r['query_id'] in original_good-selected_good],
        'delta_exact_fraction':str(delta) if delta is not None else None,'fallback_reason':fallback,
        'LP_call_count':len(ledger),'accepted_candidate_count_including_BASE':len(pool),
        'candidate_pool':pool,'candidate_ledger':ledger,'base_refits':0,
        'selection_scope':'finite one-pass pool, never whole-class maximum rescue',
        'delta_floor_claimed':False}


def predict(rows,parameters):
    records=[]
    for row in rows:
        models={}
        for model in MODELS:
            w,b=parameter_tensor(parameters[model]);logits=row['X']@w+b
            need(bool(torch.isfinite(logits).all()),'FINITE_ALL127_HELDOUT_LOGITS')
            models[model]=serialize_prediction(row,logits)
        records.append({'execution_ordinal':row['execution_ordinal'],'query_id':row['query_id'],
          'source_image_sha256':row['source_image_sha256'],'candidate_physical_rows':row['candidate_physical_rows'],
          'base_winner_position':row['base_winner_position'],'predictions':models})
    return records


def fresh(phase,fold=None):
    nonce=uuid.uuid4().hex
    env=dict(os.environ,RC_PROTECTED_PROJECTION_FRESH_NONCE=nonce,
             RC_PROTECTED_PROJECTION_PARENT_PID=str(os.getpid()))
    command=[sys.executable,str(PROGRAM),'--phase',phase,'--nonce',nonce]
    if fold is not None:command.extend(['--fold',str(fold)])
    subprocess.run(command,env=env,check=True)


def solve_fold(fold,replay=False,nonce=None):
    authority=require_authority()
    if replay:
        need(nonce and os.environ.get('RC_PROTECTED_PROJECTION_FRESH_NONCE')==nonce and
             str(os.getppid())==os.environ.get('RC_PROTECTED_PROJECTION_PARENT_PID'),'EXPLICIT_FRESH_LP_PROCESS')
    train,held,weight,bias,base,closure=load_fold(fold,replay)
    context=build_context(train,weight,bias)
    parameters={'BASE7':base}
    for model in MODELS[1:]:
        parameters[model]=solve_training(train,weight,bias,model,replay,context)
    predictions=predict(held,parameters);directory=OUT/f'fold{fold:02d}'
    if replay:
        seal=read(directory/'seal.json')
        need(seal['authority']==authority and seal['program']==bind(PROGRAM) and seal['closure']==closure,'FRESH_EXACT_SOURCE_CLOSURE')
        need(parameters==read(checked(seal['parameters'])),'FRESH_ALL_LP_CANDIDATES_SELECTION_PARAMETER_BITS')
        need(predictions==read(checked(seal['predictions'])),'FRESH_ALL_HELDOUT_LOGIT_AND_ACTION_BITS')
        need(os.getpid()!=seal['process_PID'],'FRESH_PROCESS_DISTINCT')
        write(directory/'validation.json',{'status':'PROTECTED_PROJECTION_OOF4_FRESH_LP_REPLAY_PASS',
          'authority':authority,'program':bind(PROGRAM),'fold':fold,'seal':bind(directory/'seal.json'),
          'fresh_process_nonce':nonce,'process_PID':os.getpid(),'runtime':runtime(),
          'LP_calls_replayed':sum(parameters[m]['LP_call_count'] for m in MODELS[1:]),
          'heldout_logit_checks':len(held)*127*3,'prior_BASE_logit_parity_checks':len(held)*127,
          'all_candidate_selection_and_parameter_bits_replayed':True,'heldout_label_reads':0,
          'EVAL_reads':0,'gradient_optimizer_steps':0,'base_refits':0,'forbidden_read_attempts':len(BLOCKED)})
    else:
        need(not (directory/'seal.json').exists(),'NO_OVERWRITE_FROZEN_FOLD')
        write(directory/'parameters.json',parameters);write(directory/'predictions.json',predictions)
        write(directory/'seal.json',{'status':'PROTECTED_PROJECTION_OOF4_PARAMETERS_PREDICTIONS_SEALED_PREJOIN',
          'authority':authority,'program':bind(PROGRAM),'fold':fold,'closure':closure,
          'parameters':bind(directory/'parameters.json'),'predictions':bind(directory/'predictions.json'),
          'process_PID':os.getpid(),'runtime':runtime(),'heldout_label_reads':0,'EVAL_reads':0,
          'gradient_optimizer_steps':0,'base_refits':0,'forbidden_read_attempts':len(BLOCKED)})
        fresh('validate-fold',fold)
    print(json.dumps({'phase':PHASE,'fold':fold,'status':'PASS',
      'LP_calls':sum(parameters[m]['LP_call_count'] for m in MODELS[1:]),
      'selected':{m:parameters[m]['selected_candidate_id'] for m in MODELS[1:]}}),flush=True)


def synthetic_preflight():
    operators=reused_operators()
    raw=[i/128 for i in range(128)]
    evidence={i:{'real_score':.1+i/100,'visibility_mass':.2+i/100,'query_control_score':.03,'reference_control_score':.02} for i in range(128)}
    for c in range(127):
        need(manual_feature(raw,evidence,c,127).numpy().tobytes()==FC.candidate_feature(raw,evidence,c,127).numpy().tobytes(),'SYNTHETIC_LITERAL_FEATURE_PARITY')
    weight=torch.tensor([1.,0.,0.,0.,0.,0.],dtype=torch.float64)
    def toy(key,ordinal,target,first=-1.,second=0.):
        x=torch.zeros((127,6),dtype=torch.float64);x[:,0]=-3.;x[0,0]=first;x[0,1]=second
        return {'query_id':key,'execution_ordinal':ordinal,'group':'g'+str(ordinal),
                'candidate_physical_rows':list(range(128)),'challenger_positions':list(range(1,128)),
                'base_winner_position':0,'target_position':target,'X':x}
    protect=toy('protect',0,0,-2.);error=toy('error',1,1,-1.,1.)
    for model in MODELS[1:]:
        first=solve_training([protect,error],weight,0.,model)
        second=solve_training([protect,error],weight,0.,model,True)
        need(first==second and first['LP_call_count']==1 and first['selected_candidate_id']=='error','SYNTHETIC_ONEPASS_REPLAY_REPAIR')
    conflict=[toy('protect',0,0),toy('error',1,1)]
    p=solve_training(conflict,weight,0.,'PROTECTED7')
    need(p['selected_candidate_id']=='BASE' and p['LP_call_count']==1 and p['accepted_candidate_count_including_BASE']==1,'CONFLICT_NUMERIC_REJECTION_AND_BASE_FALLBACK')
    c=solve_training(conflict,weight,0.,'REPAIR_ONLY7')
    need(c['LP_call_count']==1,'CONTROL_SAME_ORIGINAL_ERROR_ENUMERATION')
    empty=solve_training([toy('empty',0,1)],weight,0.,'PROTECTED7')
    need(empty['fallback_reason']=='EMPTY_PROTECTED_SET' and empty['LP_call_count']==0,'EMPTY_P_EXACT_BASE_FALLBACK')
    boundary=toy('boundary',0,0,0.)
    b=solve_training([boundary,error],weight,0.,'PROTECTED7')
    need(b['fallback_reason']=='NONPOSITIVE_EXACT_BASE_PROTECTION_MARGIN' and b['LP_call_count']==0,'ZERO_MARGIN_BASE_FALLBACK')
    tied=[{'candidate_id':'BASE','source_error_execution_ordinal':-1,'training_score':{'equal_group_accuracy_fraction':'1/2','correct':2,'L1_from_BASE_fraction':'0'}},
          {'candidate_id':'other','source_error_execution_ordinal':0,'training_score':{'equal_group_accuracy_fraction':'1/2','correct':2,'L1_from_BASE_fraction':'0'}}]
    need(min(tied,key=rank_candidate)['candidate_id']=='BASE','EXACT_BASE_TIE_PRIORITY')
    stats_rows=[{'group':'g'+str(i//2),'original_query_id':'s'+str(i),'correct':{m:i%2==0 for m in MODELS}} for i in range(6)]
    stats=operators['paired_group_statistics'](stats_rows,['g0','g1','g2'])
    need(all(v['net_correct_images']==0 and v['exact_two_sided_group_signflip']['two_sided_p']==1 for v in stats['contrasts'].values()),'FROZEN_PAIRED_STATISTICS')
    # Dry guard check: hard oracle denial is evaluated before HASH_DEPTH bypass,
    # no such file is opened and no real artifact is even hashed.
    global HASH_DEPTH
    old_depth=HASH_DEPTH;old_blocked=len(BLOCKED);HASH_DEPTH=1
    try:
        try:audit('open',(str(ROOT/'results/rc_opened_forbidden_synthetic_guard_probe/no_file'),))
        except PermissionError:pass
        else:raise RuntimeError('ORACLE_HASH_BYPASS_MUST_BE_DENIED')
    finally:
        HASH_DEPTH=old_depth;del BLOCKED[old_blocked:]
    value={'status':'TRAIN128_PROTECTED_PROJECTION_OOF4_SYNTHETIC_PREFLIGHT_PASS',
      'program':bind(PROGRAM),'launcher':bind(LAUNCH),'plan':bind(PLAN),'prior_program':bind(OLD),
      'contract':CONTRACT,'statistics_contract':STATISTICS_CONTRACT,'runtime':runtime(),
      'synthetic_feature_checks':127,'synthetic_LP_calls':6,'oracle_hash_guard_probe_pass':True,
      'natural_feature_reads':0,'natural_solver_calls':0,'natural_label_reads':0,
      'EVAL_reads':0,'base_refits':0,'forbidden_read_attempts':len(BLOCKED)}
    path=PREFLIGHT/sha(PROGRAM)/'result.json';write(path,value)
    print(json.dumps({'status':value['status'],'preflight':bind(path)}),flush=True)


def main():
    global PHASE,FOLD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('preflight','solve-fold','validate-fold','join','validate-join','run'),required=True)
    p.add_argument('--fold',type=int,choices=range(4));p.add_argument('--nonce');args=p.parse_args()
    PHASE,FOLD=args.phase,args.fold
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_flush_denormal(False)
    sys.addaudithook(audit)
    if PHASE=='preflight':synthetic_preflight()
    elif PHASE in ('solve-fold','validate-fold'):
        need(FOLD is not None,'FOLD_REQUIRED');solve_fold(FOLD,PHASE=='validate-fold',args.nonce)
    elif PHASE in ('join','validate-join'):join(PHASE=='validate-join',args.nonce)
    else:
        require_authority()
        for fold in range(4):
            subprocess.run([sys.executable,str(PROGRAM),'--phase','solve-fold','--fold',str(fold)],check=True)
        subprocess.run([sys.executable,str(PROGRAM),'--phase','join'],check=True)


if __name__=='__main__':main()
