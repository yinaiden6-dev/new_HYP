#!/usr/bin/env python3
"""TRAIN-only grouped OOF screen for a frozen-base content correction.

prepare reads metadata only. preflight uses synthetic tensors only. Natural
project/fit/refit/join stages require a separately frozen execution authority.
Each fold sees only its own training labels; heldout labels open at final join.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / 'src'))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC

OUT = ROOT / 'results/rc_train128_disagreement_oof4_v1'
PREFLIGHT = ROOT / 'results/rc_train128_disagreement_oof4_v1_preflight'
CACHE = ROOT / 'results/rc_train128_disagreement_features_v1'
INPUT = ROOT / 'results/rc_original7_train128_inputs_v1'
META = ROOT / 'results/rc_original7_train128_manifest_v1'
PAIR = ROOT / 'results/routea_matched_three_arm_pair64_training_features_v2'
PLAN = ROOT / 'plan/RC_TRAIN128_DISAGREEMENT_OOF4_V1_20260910.md'
AUTH = ROOT / 'registry/rc_train128_disagreement_oof4_execution_authority_v1_20260910.json'
LAUNCH = ROOT / 'slurm/rc_train128_disagreement_oof4_v1_dev_cpuonly_59m.sbatch'
OLD = ROOT / 'programs/run_routea_matched_three_arm_common3_native7_crossfit_v1.py'
GALLERY_IMAGES = ROOT.parents[2] / 'dailymed/data/box_flat_20000_images/data/raw_images'
SEED = 'RC_TRAIN128_DISAGREEMENT_OOF4_V1_20260910'
DEADLINE = datetime(2026, 9, 11, 16, tzinfo=timezone.utc)
ARM = 'C_PAIRED'
MODELS = ('BASE7', 'CONSTANT1', 'CONDITIONAL4')
STEPS = 2000
TRAIN_FUNCTION_SHA = '45921b88203ef7065d824bdaab8e5468fc6d617dd3cb6d3fee213dfd2ccc7403'
PINS = {
    OLD: '546e2bc7b3df6c67bcac41e079ba1b00f15c7e65c52a8ab994cd5b2c38d81e22',
    Path(FC.__file__): '96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7',
    META / 'worker_manifest.json': '50d894c9643ca1ef200c18ba79cf700b598be15c5ea322010853c201fc0f22f7',
    META / 'curator_roles.json': '240be57ffe824e8e029e83052a8b608166782edfdf1df8327bec164621de8ca1',
    META / 'independent_metadata_validation.json': '03176db194d4b5f9d68cc226a1ce48b019675e99b037205bae71552202fa907d',
    INPUT / 'validation.json': '770b9f9ce46f431c7bbe8479e320934c9d1dacdf23f3c3eea650f2812c1a77bf',
    PAIR / 'payload.pt': 'd7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3',
    PAIR / 'independent_validation.json': '657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b',
}
CONTRACT = {
    'scope': 'TRAIN128 grouped OOF only; no final fit or EVAL access',
    'models': list(MODELS), 'primary': 'CONDITIONAL4', 'fold_count': 4,
    'group_count': 32, 'group_seed': SEED, 'fold_rule': 'sha256(seed+"|"+group) sorted rank modulo 4',
    'fold_base': 'original FULL32 and original PAIR64 excluding all heldout groups',
    'base_train_function_sha256': TRAIN_FUNCTION_SHA,
    'base_parameters': 7, 'new_parameter_counts': {'CONSTANT1': 1, 'CONDITIONAL4': 4},
    'correction_formula': 'z=X@w+b+softplus(beta0+beta1*(-xRAW)+beta2*(D_w-D_c)+beta3*sym(mean_wr_c,mean_wr_w))*sym(F_c,F_w)',
    'constant_formula': 'z=X@w+b+softplus(alpha)*sym(F_c,F_w)',
    'initialization': 'base all zero; alpha=0; all four beta=0',
    'seed': 17, 'optimizer': 'AdamW', 'lr': 0.03, 'weight_decay': 0.001,
    'steps': STEPS, 'checkpoint': 'last only', 'dtype': 'float64',
    'correction_loss': 'original FULL query loss only, protection weight 4, mean over target-present training rows',
    'PAIR_correction_loss': False, 'normalization': 'none',
    'target_absent': 'retain all 128 heldout images; omit undefined positive FULL training loss only',
    'action': 'all 127 challenger logits, SWITCH iff maximum>0, physical row ties, natural C128 unchanged',
    'gate': ['CONDITIONAL correct count strictly greater than BASE and CONSTANT',
             'CONDITIONAL equal-group mean strictly greater than BASE and CONSTANT',
             'CONDITIONAL RAW break count no greater than BASE RAW break count'],
    'fold_label_rule': 'fold-local training curator and pair projection only; heldout roles opened after all independent seals',
    'gate_pass_action': 'eligible for separately authorized final fit; this program never opens EVAL',
}
STATISTICS_CONTRACT = {
    'paired_comparisons': [['CONDITIONAL4', 'BASE7'], ['CONDITIONAL4', 'CONSTANT1'], ['CONSTANT1', 'BASE7']],
    'unit': 'source group; equally weighted group mean accuracy difference',
    'bootstrap_draws': 10000, 'bootstrap_seed': 20260910,
    'bootstrap_generator': 'numpy.default_rng PCG64; one shared group-index draw array for all contrasts',
    'bootstrap_interval': 'percentile 95%, numpy.quantile method linear',
    'signflip': 'exact two-sided inclusive tail, abs(sum signed group differences)>=abs(observed sum)',
    'arithmetic': 'rational group differences, common-denominator integer DP; zero groups retained exactly',
    'p_value_is_gate': False,
}
PHASE = None
FOLD = None
HASH_DEPTH = 0
BLOCKED = []


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def sha(path):
    global HASH_DEPTH
    HASH_DEPTH += 1
    try:
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for data in iter(lambda: stream.read(8 << 20), b''):
                h.update(data)
        return h.hexdigest()
    finally:
        HASH_DEPTH -= 1


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(binding):
    path = Path(binding['path'])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    need(sha(path) == binding['sha256'], 'SOURCE_HASH_DRIFT:' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'IMMUTABLE_OUTPUT_EXISTS:' + str(path))
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


def write(path, value):
    write_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def hx(value):
    return float(value).hex()


def matrix_hex(value):
    return [[hx(x) for x in row] for row in value]


def runtime():
    return {'python': sys.version, 'executable': sys.executable, 'torch': str(torch.__version__),
            'numpy': np.__version__, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'CUDA_used': False}


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = Path(os.fsdecode(args[0])).resolve()
    s = str(p).lower()
    if HASH_DEPTH:
        return
    forbidden = any(x in s for x in ('/grozi/', 'd1-mi', 'd1_mi', '/target_join/',
                       '/role_shards/', 'direction_capacity', 'angle_capacity', 'origin_linear_projection', '/rc_opened_'))
    if ROOT / 'results' in p.parents:
        allowed = (p in (INPUT / 'validation.json', INPUT / 'feature_records.json',
                         META / 'worker_manifest.json', META / 'independent_metadata_validation.json',
                         PAIR / 'independent_validation.json') or OUT in p.parents or PREFLIGHT in p.parents or CACHE in p.parents)
        allowed |= p == META / 'curator_roles.json' and PHASE == 'prepare'
        allowed |= p == PAIR / 'payload.pt' and PHASE == 'project'
        forbidden |= not allowed
        if PHASE in ('prepare', 'preflight'):
            forbidden |= p == INPUT / 'feature_records.json' or CACHE in p.parents or p == PAIR / 'payload.pt'
        if OUT in p.parents and PHASE in ('fit-fold', 'validate-fold'):
            forbidden |= p.name == 'heldout_roles.json'
            if p.name in ('train_roles.json', 'pair_train.pt', 'pair_projection.json'):
                forbidden |= p.parent != OUT / ('fold%02d' % FOLD)
        if p.name == 'heldout_roles.json' and PHASE not in ('prepare', 'join', 'validate-join'):
            forbidden = True
    if forbidden:
        BLOCKED.append(str(p))
        raise PermissionError('TRAIN_OOF_READ_BOUNDARY:' + str(p))


def training_functions():
    need(sha(OLD) == PINS[OLD], 'OLD_TRAINING_PROGRAM_PIN')
    text = OLD.read_text()
    names = {'finite_tensor', 'finite_scalar', 'parameter_sha', 'train_head'}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == names, 'EXACT_OLD_TRAINING_AST')
    source = ast.get_source_segment(text, next(n for n in nodes if n.name == 'train_head'))
    need(hashlib.sha256(source.encode()).hexdigest() == TRAIN_FUNCTION_SHA, 'ORIGINAL_TRAIN_HEAD_AST_PIN')
    space = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'hashlib': hashlib, 'json': json,
             'STEPS': STEPS, 'FAMILIES': {'NATIVE7': ('real_native_features', 'cbind_native_features', FC.FEATURE_NAMES)},
             'CrossfitContractError': RuntimeError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(OLD), 'exec'), space)
    return space


def metadata_prepare():
    for p in (META / 'worker_manifest.json', META / 'curator_roles.json', META / 'independent_metadata_validation.json'):
        need(sha(p) == PINS[p], 'FROZEN_TRAIN_METADATA')
    workers = read(META / 'worker_manifest.json')['records']
    roles = read(META / 'curator_roles.json')['records']
    valid = read(META / 'independent_metadata_validation.json')
    need(valid['status'] == 'TRAIN128_INDEPENDENT_METADATA_SELECTION_PASS', 'METADATA_VALIDATED')
    need(len(workers) == len(roles) == 128, 'TRAIN128_ONLY')
    groups = {r['group'] for r in roles}
    need(len(groups) == len({r['identity'] for r in roles}) == 32, 'TRAIN32_IDENTITIES_GROUPS')
    need(all(len({r['identity'] for r in roles if r['group'] == g}) == 1 for g in groups), 'IDENTITY_GROUP_BIJECTION')
    ordered = sorted(groups, key=lambda g: (hashlib.sha256((SEED + '|' + g).encode()).hexdigest(), g))
    assignments = {g: i % 4 for i, g in enumerate(ordered)}
    public = []
    for i, (w, r) in enumerate(zip(workers, roles, strict=True)):
        need(w['execution_ordinal'] == r['execution_ordinal'] == i and w['query_id'] == r['query_id']
             and w['source_image_sha256'] == r['source_image_sha256'] and r['role'] == 'TRAIN', 'TRAIN_ROLE_JOIN')
        public.append({'execution_ordinal': i, 'query_id': w['query_id'], 'source_image_sha256': w['source_image_sha256'],
                       'fold': assignments[r['group']], 'group_key': hashlib.sha256(('group|' + r['group']).encode()).hexdigest(),
                       'selection_origin': r['selection_origin']})
    files = {}
    for fold in range(4):
        directory = OUT / ('fold%02d' % fold)
        train = [r for r in roles if assignments[r['group']] != fold]
        held = [r for r in roles if assignments[r['group']] == fold]
        need(len({r['group'] for r in held}) == 8, 'EXACT_EIGHT_HELDOUT_GROUPS')
        need(any(r['selection_origin'] == 'ORIGINAL_FULL32' for r in train) and
             any(r['selection_origin'] == 'ORIGINAL_PAIR64' for r in train), 'NONEMPTY_ORIGINAL_LOSS_DOMAINS')
        for name, rows in (('train_roles.json', train), ('heldout_roles.json', held)):
            path = directory / name
            write(path, {'status': 'TRAIN_ONLY_GROUPED_ROLE_PROJECTION', 'fold': fold, 'records': rows,
                         'source_curator': bind(META / 'curator_roles.json')})
            files['fold%d_%s' % (fold, name)] = bind(path)
    value = {'status': 'TRAIN128_OOF4_METADATA_ONLY_FOLDS_FROZEN', 'seed': SEED, 'contract': CONTRACT,
             'records': public, 'group_order': [{'group_key': hashlib.sha256(('group|' + g).encode()).hexdigest(),
                'assignment_hash': hashlib.sha256((SEED + '|' + g).encode()).hexdigest(), 'fold': assignments[g]} for g in ordered],
             'role_projections': files, 'worker_manifest': bind(META / 'worker_manifest.json'),
             'source_curator': bind(META / 'curator_roles.json'), 'source_metadata_validation': bind(META / 'independent_metadata_validation.json'),
             'image_or_feature_reads': 0, 'training_updates': 0, 'EVAL_reads': 0}
    write(OUT / 'fold_manifest.json', value)
    print(json.dumps({'status': value['status'], 'manifest': bind(OUT / 'fold_manifest.json'),
         'fold_image_counts': [sum(x['fold'] == f for x in public) for f in range(4)]}), flush=True)


def static_sources():
    paths = {'oof_program': PROGRAM, 'oof_launcher': LAUNCH, 'plan': PLAN,
             'cache_program': ROOT / 'programs/cache_rc_train128_disagreement_features_v1.py',
             'cache_authority': ROOT / 'registry/rc_train128_disagreement_cache_authority_v1_20260910.json',
             'old_training_program': OLD, 'feature_core': Path(FC.__file__),
             'fold_manifest': OUT / 'fold_manifest.json', 'train_worker': META / 'worker_manifest.json',
             'train_metadata_validation': META / 'independent_metadata_validation.json',
             'train_input_validation': INPUT / 'validation.json', 'train_features': INPUT / 'feature_records.json',
             'PAIR64_payload': PAIR / 'payload.pt', 'PAIR64_validation': PAIR / 'independent_validation.json',
             'gallery_identity_program': ROOT / 'src/rc_aslo_xf/gallery_identity_repair.py',
             'gallery_identity_registry': ROOT / 'registry/gallery_identity_repair_v1.json',
             'gallery_identity_contract': ROOT / 'protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json'}
    for fold in range(4):
        for kind in ('train_roles', 'heldout_roles'):
            paths['fold%d_%s' % (fold, kind)] = OUT / ('fold%02d' % fold) / (kind + '.json')
    return {key: bind(path) for key, path in paths.items()}


def require_authority():
    need(os.environ.get('SLURM_JOB_ID') and datetime.now(timezone.utc) < DEADLINE, 'SLURM_AND_DEADLINE_REQUIRED')
    value = read(AUTH)
    need(value['status'] == 'TRAIN128_DISAGREEMENT_OOF4_AUTHORIZED' and value['contract'] == CONTRACT, 'AUTHORITY_CONTRACT')
    need(PHASE in value['allowed_stages'] and value['cutoff_UTC'] == DEADLINE.isoformat(), 'AUTHORIZED_STAGE_DEADLINE')
    need(value['source_bindings'] == static_sources(), 'COMPLETE_OOF_SOURCE_BINDINGS')
    for p, expected in PINS.items():
        need(sha(p) == expected, 'STATIC_PIN:' + str(p))
    pre = read(checked(value['preflight']))
    need(pre['status'] == 'TRAIN128_DISAGREEMENT_OOF4_SYNTHETIC_PREFLIGHT_PASS' and pre['program'] == bind(PROGRAM)
         and pre['contract'] == CONTRACT and pre['runtime'] == runtime(), 'QUALIFIED_SYNTHETIC_PREFLIGHT')
    manifest = read(OUT / 'fold_manifest.json')
    need(manifest['contract'] == CONTRACT and manifest['status'] == 'TRAIN128_OOF4_METADATA_ONLY_FOLDS_FROZEN', 'FROZEN_FOLDS')
    return bind(AUTH)


def qualified_rows():
    validation = read(INPUT / 'validation.json')
    need(validation['status'] == 'ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS'
         and validation['query_count'] == 128 and validation['all32_original_input_bits_exact'], 'QUALIFIED_TRAIN128')
    need(checked(validation['feature_records']) == (INPUT / 'feature_records.json').resolve(), 'QUALIFIED_FEATURE_BINDING')
    rows = read(INPUT / 'feature_records.json')
    workers = read(META / 'worker_manifest.json')['records']
    need(len(rows) == 128, 'COMPLETE_TRAIN128')
    for i, (row, worker) in enumerate(zip(rows, workers, strict=True)):
        need(row['execution_ordinal'] == worker['execution_ordinal'] == i and row['query_id'] == worker['query_id']
             and row['source_image_sha256'] == worker['source_image_sha256'], 'QUALIFIED_TRAIN128_ORDER')
        need(len(row['candidate_physical_rows']) == 128 and len(set(row['candidate_physical_rows'])) == 128
             and len(row['challenger_positions']) == 127, 'NATURAL_FULL_C128')
        raw = list(map(float.fromhex, row['candidate_raw_scores_binary64']))
        axis = row['candidate_physical_rows']
        ranked = sorted(range(128), key=lambda p: (-raw[p], axis[p]))
        need(ranked[0] == row['base_winner_position'] and row['challenger_positions'] ==
             [p for p in range(128) if p != ranked[0]], 'PHYSICAL_CHALLENGER_ORDER')
        evidence = {int(p): {k: float.fromhex(v) for k, v in d.items()} for p, d in row['C4_binary64'].items()}
        x = torch.stack([FC.candidate_feature(raw, evidence, c, ranked[0]) for c in row['challenger_positions']])
        need(matrix_hex(x) == row['features_binary64']['REAL'], 'ORIGINAL_SIX_FEATURE_BIT_REPLAY')
        row['X'] = x
        row['real_native_features'] = {ARM: x}
    return rows


def gallery_labels():
    from rc_aslo_xf.gallery_identity_repair import build_identity_map
    paths = []
    for directory, subdirs, names in os.walk(GALLERY_IMAGES):
        subdirs.sort()
        paths.extend(Path(directory) / n for n in sorted(names) if (Path(directory) / n).is_file()
                     and Path(n).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'})
    need(len(paths) == 5413, 'FROZEN_PHYSICAL_GALLERY')
    mapping = build_identity_map([p.stem.strip() for p in paths])
    need(len(set(mapping.labels)) == 5412, 'CORRECTED_IDENTITY_GALLERY')
    return tuple(mapping.labels), mapping.corrected_row_identity_mapping_sha256


def project_pairs():
    authority = require_authority()
    validation = read(PAIR / 'independent_validation.json')
    need(validation['status'] == 'ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED'
         and all(x is True for x in validation['checks'].values()), 'ORIGINAL_PAIR_QUALIFICATION')
    payload = torch.load(PAIR / 'payload.pt', map_location='cpu', mmap=True, weights_only=True)
    pairrows = payload['records']
    need(len(pairrows) == 64, 'ORIGINAL_PAIR64')
    folds = read(OUT / 'fold_manifest.json')['records']
    outputs = {}
    for fold in range(4):
        directory = OUT / ('fold%02d' % fold)
        roles = read(directory / 'train_roles.json')['records']
        lookup = {r['original_query_id']: r for r in roles if r['selection_origin'] == 'ORIGINAL_PAIR64'}
        kept = []
        kept_indices = []
        for i, row in enumerate(pairrows):
            want = folds[32 + i]['fold'] != fold
            need(want == (row['query_id'] in lookup), 'PAIR_GROUP_FILTER_MATCHES_ORIGINAL_ORDER')
            if want:
                role = lookup[row['query_id']]
                need(role['execution_ordinal'] == 32 + i, 'PAIR64_ORDER_BINDING')
                kept.append({'query_id': row['query_id'], 'switch_label': row['switch_label'],
                             'real_native_features': {ARM: row['real_native_features'][ARM]}})
                kept_indices.append(i)
        need(kept, 'NONEMPTY_FOLD_PAIR_LOSS')
        stream = io.BytesIO()
        torch.save({'records': kept}, stream)
        write_bytes(directory / 'pair_train.pt', stream.getvalue())
        receipt = {'status': 'TRAIN_ONLY_GROUP_EXCLUDED_PAIR_PROJECTION', 'fold': fold, 'authority': authority,
                   'source_payload': bind(PAIR / 'payload.pt'), 'train_roles': bind(directory / 'train_roles.json'),
                   'payload': bind(directory / 'pair_train.pt'), 'original_pair_indices': kept_indices,
                   'PAIR_labels_and_features_unchanged': True, 'training_updates': 0, 'EVAL_reads': 0}
        write(directory / 'pair_projection.json', receipt)
        outputs[str(fold)] = bind(directory / 'pair_projection.json')
    write(OUT / 'pair_projections.json', {'status': 'ALL_FOUR_PAIR_PROJECTIONS_SEALED', 'authority': authority,
                                        'folds': outputs, 'training_updates': 0})


def cache_rows(rows):
    result = read(CACHE / 'result.json')
    validation = read(CACHE / 'validation.json')
    manifest = read(CACHE / 'manifest.json')
    need(validation['status'] == 'TRAIN128_DISAGREEMENT_CACHE_INDEPENDENT_REPLAY_PASS', 'DISAGREEMENT_CACHE_REPLAY_PASS')
    need(validation['manifest'] == bind(CACHE / 'manifest.json') and validation['cache'] == bind(CACHE / 'features.npz')
         and validation['source_validation'] == bind(INPUT / 'validation.json'), 'EXACT_CACHE_QUALIFICATION')
    need(validation['feature_scalar_bit_checks'] == 81920 and validation['source_C4_scalar_bit_checks'] == 65536
         and validation['curator_reads'] == validation['EVAL_reads'] == validation['training_updates'] == 0, 'FULL_CACHE_AND_NO_LABELS')
    need(manifest['source_validation'] == bind(INPUT / 'validation.json'), 'CACHE_TRAIN_SOURCE')
    # Bind the producer/authority, and completion receipt, before using feature values.
    cache_authority = bind(ROOT / 'registry/rc_train128_disagreement_cache_authority_v1_20260910.json')
    cache_program = bind(ROOT / 'programs/cache_rc_train128_disagreement_features_v1.py')
    need(result['status'] == 'TRAIN128_DISAGREEMENT_CACHE_READY' and result['authority'] == cache_authority
         and result['manifest'] == bind(CACHE / 'manifest.json') and result['validation'] == bind(CACHE / 'validation.json')
         and result['cache'] == bind(CACHE / 'features.npz'), 'CACHE_COMPLETION_BINDINGS')
    need(manifest['authority'] == validation['authority'] == cache_authority
         and manifest['program'] == validation['program'] == cache_program, 'BOUND_CACHE_PRODUCER_AND_AUTHORITY')
    need(manifest['feature_names'] == ['F', 'A', 'D', 'mean_wr', 'mean_wq']
         and manifest['C4_names'] == ['real_score', 'visibility_mass', 'query_control_score', 'reference_control_score'], 'CACHE_COLUMN_ORDER')
    with np.load(checked(manifest['cache']), allow_pickle=False) as saved:
        features = saved['features'].copy()
        c4 = saved['c4'].copy()
        axes = saved['candidate_physical_rows'].copy()
        qids = saved['query_ids'].copy()
    need(features.shape == (128, 128, 5) and features.dtype == np.float64 and np.isfinite(features).all(), 'CACHE_DOMAIN')
    for i, row in enumerate(rows):
        m = manifest['records'][i]
        need(m['execution_ordinal'] == i and m['query_id'] == row['query_id'] == str(qids[i])
             and m['source_image_sha256'] == row['source_image_sha256']
             and m['query_tokens_sha256'] == row['query_tokens_sha256']
             and m['candidate_physical_rows'] == row['candidate_physical_rows'] == axes[i].tolist(), 'CACHE_AXIS_BINDING')
        for p in range(128):
            need([hx(v) for v in c4[i, p]] == [row['C4_binary64'][str(p)][k] for k in
                 ('real_score', 'visibility_mass', 'query_control_score', 'reference_control_score')], 'CACHE_C4_QUALIFIED_BITS')
        winner = row['base_winner_position']
        covariates = []
        df = []
        for j, challenger in enumerate(row['challenger_positions']):
            fc, _, dc, rc, _ = features[i, challenger]
            fw, _, dw, rw, _ = features[i, winner]
            covariates.append([1.0, -float(row['X'][j, 0]), float(dw) - float(dc), FC.symmetric(rc, rw)])
            df.append(FC.symmetric(fc, fw))
        row['gate_X'] = torch.tensor(covariates, dtype=torch.float64)
        row['dF'] = torch.tensor(df, dtype=torch.float64)
    return {'manifest': bind(CACHE / 'manifest.json'), 'cache': bind(CACHE / 'features.npz'),
            'validation': bind(CACHE / 'validation.json'), 'result': bind(CACHE / 'result.json')}


def load_fold(fold):
    rows = qualified_rows()
    cache = cache_rows(rows)
    directory = OUT / ('fold%02d' % fold)
    roles = read(directory / 'train_roles.json')['records']
    rolemap = {r['query_id']: r for r in roles}
    split = read(OUT / 'fold_manifest.json')['records']
    wanted = {r['query_id'] for r in split if r['fold'] != fold}
    need(set(rolemap) == wanted and len({r['group'] for r in roles}) == 24, 'EXACT_24_TRAINING_GROUPS')
    labels, mapping_sha = gallery_labels()
    train, held, absent = [], [], []
    for row, assignment in zip(rows, split, strict=True):
        need(row['query_id'] == assignment['query_id'], 'FOLD_ORDER')
        if assignment['fold'] == fold:
            need(row['query_id'] not in rolemap, 'NO_HELDOUT_LABEL_IN_TRAINING')
            held.append(row)
            continue
        role = rolemap[row['query_id']]
        positions = [p for p, physical in enumerate(row['candidate_physical_rows']) if labels[physical] == role['identity']]
        need(len(positions) <= 1, 'NATURAL_C128_IDENTITY_DEDUP')
        row['target_position'] = positions[0] if positions else None
        row['group'] = role['group']
        if positions:
            train.append(row)
        else:
            absent.append(row['query_id'])
    base_train = [r for r in train if r['execution_ordinal'] < 32]
    receipt = read(directory / 'pair_projection.json')
    need(receipt['status'] == 'TRAIN_ONLY_GROUP_EXCLUDED_PAIR_PROJECTION' and receipt['fold'] == fold
         and receipt['authority'] == bind(AUTH) and receipt['train_roles'] == bind(directory / 'train_roles.json')
         and receipt['source_payload'] == bind(PAIR / 'payload.pt'), 'FOLD_PAIR_PROJECTION')
    pair = torch.load(checked(receipt['payload']), map_location='cpu', weights_only=True)
    expected_pair = {r['original_query_id'] for r in roles if r['selection_origin'] == 'ORIGINAL_PAIR64'}
    need({r['query_id'] for r in pair['records']} == expected_pair and base_train and train, 'NONEMPTY_FOLD_TRAINING')
    return pair, base_train, train, held, {'cache': cache, 'gallery_mapping_sha256': mapping_sha,
        'train_role_projection': bind(directory / 'train_roles.json'), 'pair_projection': bind(directory / 'pair_projection.json'),
        'heldout_ordinals': [r['execution_ordinal'] for r in held],
        'base_FULL_ordinals': [r['execution_ordinal'] for r in base_train],
        'correction_FULL_ordinals': [r['execution_ordinal'] for r in train],
        'target_absent_training_queries': absent, 'original_PAIR_rows': len(pair['records'])}


def full_query_loss(values, row):
    target = int(row['target_position'])
    if target == int(row['base_winner_position']):
        return 4 * F.softplus(values.max())
    position = row['challenger_positions'].index(target)
    mask = torch.ones(len(values), dtype=torch.bool)
    mask[position] = False
    return F.softplus(-values[position]) + 4 * F.softplus(values[mask].max())


def fit_correction(base_w, base_b, rows, count, steps=STEPS):
    torch.manual_seed(17)
    beta = nn.Parameter(torch.zeros(count, dtype=torch.float64))
    optimizer = torch.optim.AdamW([beta], lr=0.03, weight_decay=0.001)
    base = [(row['X'] @ base_w + base_b).detach() for row in rows]
    loss = None
    nonzero_gradient_steps = 0
    for _ in range(steps):
        optimizer.zero_grad()
        losses = []
        for row, logits in zip(rows, base, strict=True):
            context = row['gate_X'][:, :count]
            values = logits + F.softplus(context @ beta) * row['dF']
            losses.append(full_query_loss(values, row))
        loss = torch.stack(losses).mean()
        need(torch.isfinite(loss), 'FINITE_CORRECTION_LOSS')
        loss.backward()
        need(beta.grad is not None and torch.isfinite(beta.grad).all(), 'FINITE_CORRECTION_GRADIENT')
        nonzero_gradient_steps += int(bool(torch.count_nonzero(beta.grad)))
        optimizer.step()
        need(torch.isfinite(beta).all(), 'FINITE_CORRECTION_PARAMETERS')
    return {'parameters_binary64': [hx(v) for v in beta.detach()], 'parameter_count': count,
            'last_preupdate_loss_binary64': hx(loss.detach()), 'nonzero_gradient_steps': nonzero_gradient_steps,
            'finite_updates': steps}


def fit_parameters(pair, base_rows, rows):
    old = training_functions()
    head, loss, finite = old['train_head'](pair, base_rows, 'NATIVE7', ARM)
    w = head.weight.detach().flatten().clone()
    b = float(head.bias.detach())
    value = {'BASE7': {'weight_binary64': [hx(x) for x in w], 'bias_binary64': hx(b),
                      'parameter_sha256': old['parameter_sha'](w, b), 'parameter_count': 7,
                      'last_preupdate_losses_binary64': [hx(x) for x in loss], 'finite_training': finite}}
    for name, count in (('CONSTANT1', 1), ('CONDITIONAL4', 4)):
        value[name] = fit_correction(w, b, rows, count)
    need(old['parameter_sha'](w, b) == value['BASE7']['parameter_sha256'], 'BASE_FROZEN_DURING_CORRECTIONS')
    return value


def predict(parameters, rows):
    base = parameters['BASE7']
    w = torch.tensor(list(map(float.fromhex, base['weight_binary64'])), dtype=torch.float64)
    b = float.fromhex(base['bias_binary64'])
    records = []
    for row in rows:
        base_logits = row['X'] @ w + b
        predictions = {}
        for model in MODELS:
            logits = base_logits
            if model != 'BASE7':
                beta = torch.tensor(list(map(float.fromhex, parameters[model]['parameters_binary64'])), dtype=torch.float64)
                logits = base_logits + F.softplus(row['gate_X'][:, :len(beta)] @ beta) * row['dF']
            need(torch.isfinite(logits).all(), 'FINITE_HELDOUT_LOGITS')
            cs, axis = row['challenger_positions'], row['candidate_physical_rows']
            i = max(range(127), key=lambda j: (float(logits[j]), -axis[cs[j]]))
            final = cs[i] if float(logits[i]) > 0 else row['base_winner_position']
            predictions[model] = {'all127_logits_binary64': [hx(x) for x in logits], 'best_challenger_position': cs[i],
                'best_logit_binary64': hx(logits[i]), 'final_position': final, 'final_physical_row': axis[final],
                'action': 'SWITCH' if float(logits[i]) > 0 else 'HOLD'}
        records.append({'execution_ordinal': row['execution_ordinal'], 'query_id': row['query_id'],
                        'source_image_sha256': row['source_image_sha256'], 'candidate_physical_rows': row['candidate_physical_rows'],
                        'base_winner_position': row['base_winner_position'], 'predictions': predictions})
    return records


def fold_fit(fold, replay=False, nonce=None):
    authority = require_authority()
    directory = OUT / ('fold%02d' % fold)
    if replay:
        need(nonce and os.environ.get('RC_DISAGREEMENT_OOF_REPLAY_NONCE') == nonce, 'EXPLICIT_FRESH_REFIT_NONCE')
    else:
        need(not (directory / 'seal.json').exists(), 'FOLD_ALREADY_SEALED_USE_VALIDATE')
    pair, base_rows, train, held, closure = load_fold(fold)
    parameters = fit_parameters(pair, base_rows, train)
    predictions = predict(parameters, held)
    if replay:
        seal = read(directory / 'seal.json')
        need(seal['authority'] == authority and seal['program'] == bind(PROGRAM) and seal['closure'] == closure,
             'EXACT_FOLD_EXECUTION_BINDING')
        saved_parameters = read(checked(seal['parameters']))
        saved_predictions = read(checked(seal['predictions']))
        need(saved_parameters == parameters, 'FRESH_REFIT_PARAMETER_BITS')
        need(saved_predictions == predictions, 'FRESH_ALL_CHALLENGER_PREDICTION_BITS')
        need(os.getpid() != seal['process_PID'], 'DISTINCT_REFIT_PROCESS')
        value = {'status': 'TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS', 'fold': fold,
                 'authority': authority, 'program': bind(PROGRAM), 'seal': bind(directory / 'seal.json'),
                 'runtime': runtime(), 'fresh_process_nonce': nonce, 'process_PID': os.getpid(),
                 'refit_model_count': 3, 'parameter_values_replayed': 12,
                 'heldout_query_count': len(held), 'heldout_logit_checks': len(held) * 127 * 3,
                 'heldout_label_reads': 0, 'EVAL_reads': 0, 'forbidden_read_attempts': len(BLOCKED)}
        write(directory / 'validation.json', value)
    else:
        write(directory / 'parameters.json', parameters)
        write(directory / 'predictions.json', predictions)
        seal = {'status': 'TRAIN_OOF4_FOLD_PARAMETERS_PREDICTIONS_SEALED_PREJOIN', 'fold': fold,
                'authority': authority, 'program': bind(PROGRAM), 'runtime': runtime(), 'closure': closure,
                'parameters': bind(directory / 'parameters.json'), 'predictions': bind(directory / 'predictions.json'),
                'process_PID': os.getpid(), 'heldout_label_reads': 0, 'EVAL_reads': 0,
                'forbidden_read_attempts': len(BLOCKED)}
        write(directory / 'seal.json', seal)
        fresh('validate-fold', fold)
    print(json.dumps({'stage': PHASE, 'fold': fold, 'status': 'PASS', 'heldout_queries': len(held)}), flush=True)


def fresh(phase, fold=None):
    nonce = uuid.uuid4().hex
    env = os.environ.copy()
    env['RC_DISAGREEMENT_OOF_REPLAY_NONCE'] = nonce
    cmd = [sys.executable, str(PROGRAM), '--phase', phase, '--nonce', nonce]
    if fold is not None:
        cmd.extend(['--fold', str(fold)])
    subprocess.run(cmd, check=True, env=env)


def exact_group_signflip(differences):
    need(differences, 'NONEMPTY_SIGNFLIP_GROUPS')
    differences = [Fraction(v) for v in differences]
    denominator = math.lcm(*(v.denominator for v in differences))
    integers = [v.numerator * (denominator // v.denominator) for v in differences]
    observed = abs(sum(integers))
    nonzero = [v for v in integers if v]
    zero_count = len(integers) - len(nonzero)
    distribution = {0: 1}
    for value in nonzero:
        updated = defaultdict(int)
        for total, count in distribution.items():
            updated[total + value] += count
            updated[total - value] += count
        distribution = dict(updated)
    total_nonzero = 1 << len(nonzero)
    need(sum(distribution.values()) == total_nonzero, 'EXACT_SIGNFLIP_MASS')
    tail_nonzero = sum(count for total, count in distribution.items() if abs(total) >= observed)
    p = Fraction(tail_nonzero, total_nonzero)
    # Each skipped zero contributes two identical signs. Restoring this factor
    # proves equivalence to enumerating all 2**G assignments, including zeros.
    zero_factor = 1 << zero_count
    return {'method': 'exact rational-to-integer group sign-flip dynamic program',
            'group_count': len(differences), 'nonzero_group_count': len(nonzero), 'zero_group_count': zero_count,
            'observed_group_mean_difference_fraction': str(sum(differences, Fraction(0)) / len(differences)),
            'integer_common_denominator': denominator, 'absolute_observed_integer_sum': observed,
            'nonzero_sign_assignments': total_nonzero, 'nonzero_tail_assignments': tail_nonzero,
            'all_group_sign_assignments': total_nonzero * zero_factor,
            'all_group_tail_assignments': tail_nonzero * zero_factor,
            'two_sided_p_fraction': str(p), 'two_sided_p': float(p), 'inclusive_tail': True,
            'p_value_is_gate': False}


def paired_group_statistics(rows, groups):
    need(groups and len(set(groups)) == len(groups), 'ORDERED_UNIQUE_STATISTICAL_GROUPS')
    indices = np.random.default_rng(20260910).integers(0, len(groups), size=(10000, len(groups)), dtype=np.int64)
    contrasts = {}
    for model, baseline in STATISTICS_CONTRACT['paired_comparisons']:
        differences = []
        ledger = []
        for group in groups:
            members = [r for r in rows if r['group'] == group]
            need(members, 'NONEMPTY_SOURCE_GROUP')
            model_correct = sum(r['correct'][model] for r in members)
            base_correct = sum(r['correct'][baseline] for r in members)
            difference = Fraction(model_correct - base_correct, len(members))
            differences.append(difference)
            ledger.append({'group': group, 'image_count': len(members), 'model_correct': model_correct,
                           'baseline_correct': base_correct, 'accuracy_difference_fraction': str(difference)})
        array = np.asarray([float(v) for v in differences], dtype=np.float64)
        samples = array[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975], method='linear')
        effect = sum(differences, Fraction(0)) / len(differences)
        rescued = [r['original_query_id'] for r in rows if r['correct'][model] and not r['correct'][baseline]]
        broken = [r['original_query_id'] for r in rows if r['correct'][baseline] and not r['correct'][model]]
        contrasts[model + '_minus_' + baseline] = {
            'model': model, 'baseline': baseline, 'rescue': len(rescued), 'break': len(broken),
            'net_correct_images': len(rescued) - len(broken), 'rescue_query_ids': rescued, 'break_query_ids': broken,
            'equal_group_mean_difference': float(effect), 'equal_group_mean_difference_fraction': str(effect),
            'positive_group_count': sum(v > 0 for v in differences), 'negative_group_count': sum(v < 0 for v in differences),
            'zero_group_count': sum(v == 0 for v in differences), 'group_differences': ledger,
            'bootstrap_95_percentile_interval': [float(lower), float(upper)],
            'bootstrap_distribution_sha256': hashlib.sha256(samples.tobytes()).hexdigest(),
            'exact_two_sided_group_signflip': exact_group_signflip(differences)}
    return {'contract': STATISTICS_CONTRACT, 'ordered_groups': groups,
            'group_resample_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
            'contrasts': contrasts}


def joined_result():
    authority = require_authority()
    manifests = read(OUT / 'fold_manifest.json')
    all_predictions, validations = [], {}
    # Validate all four prediction seals before opening the first heldout role file.
    for fold in range(4):
        directory = OUT / ('fold%02d' % fold)
        val = read(directory / 'validation.json')
        seal = read(checked(val['seal']))
        need(val['status'] == 'TRAIN_OOF4_FOLD_FRESH_REFIT_PREDICTION_REPLAY_PASS' and val['fold'] == fold
             and val['authority'] == authority and val['program'] == bind(PROGRAM)
             and val['heldout_label_reads'] == val['EVAL_reads'] == val['forbidden_read_attempts'] == 0, 'ALL_FOUR_INDEPENDENT_FOLD_GATES')
        need(seal['fold'] == fold and seal['authority'] == authority and seal['program'] == bind(PROGRAM), 'FOLD_SEAL_LINEAGE')
        checked(seal['parameters'])
        predictions = read(checked(seal['predictions']))
        wanted = [r['execution_ordinal'] for r in manifests['records'] if r['fold'] == fold]
        need([r['execution_ordinal'] for r in predictions] == wanted and val['heldout_logit_checks'] == len(wanted) * 127 * 3,
             'EXACT_HELDOUT_COVERAGE')
        all_predictions.extend(dict(p, fold=fold) for p in predictions)
        validations[str(fold)] = bind(directory / 'validation.json')
    need(sorted(p['execution_ordinal'] for p in all_predictions) == list(range(128)), 'ALL128_OOF_PREDICTIONS_SEALED')
    roles = {}
    for fold in range(4):
        projection = read(OUT / ('fold%02d' % fold) / 'heldout_roles.json')
        for r in projection['records']:
            need(r['query_id'] not in roles, 'DISJOINT_HELDOUT_LABEL_PROJECTIONS')
            roles[r['query_id']] = r
    labels, mapping_sha = gallery_labels()
    rows = []
    for pred in sorted(all_predictions, key=lambda p: p['execution_ordinal']):
        role = roles[pred['query_id']]
        need(role['execution_ordinal'] == pred['execution_ordinal'] and role['source_image_sha256'] == pred['source_image_sha256'], 'POST_SEAL_ROLE_JOIN')
        axis, target = pred['candidate_physical_rows'], role['identity']
        raw_ok = labels[axis[pred['base_winner_position']]] == target
        correct = {model: labels[pred['predictions'][model]['final_physical_row']] == target for model in MODELS}
        rows.append({'query_id': pred['query_id'], 'original_query_id': role['original_query_id'], 'execution_ordinal': pred['execution_ordinal'],
                     'group': role['group'], 'identity': target, 'fold': pred['fold'], 'RAW_correct': raw_ok,
                     'target_in_natural_C128': target in {labels[p] for p in axis}, 'correct': correct})
    groups = sorted({r['group'] for r in rows})
    need(len(groups) == 32 and len({r['identity'] for r in rows}) == 32, 'EXACT_TRAIN_GROUP_POPULATION')
    scores, fractions = {}, {}
    for model in MODELS:
        group_values = {g: Fraction(sum(r['correct'][model] for r in rows if r['group'] == g), sum(r['group'] == g for r in rows)) for g in groups}
        average = sum(group_values.values(), Fraction(0)) / len(groups)
        fractions[model] = average
        scores[model] = {'correct': sum(r['correct'][model] for r in rows),
                         'rescue_vs_RAW': sum(not r['RAW_correct'] and r['correct'][model] for r in rows),
                         'break_vs_RAW': sum(r['RAW_correct'] and not r['correct'][model] for r in rows),
                         'equal_group_accuracy': float(average), 'equal_group_accuracy_fraction': str(average),
                         'group_metrics': {g: {'correct': sum(r['correct'][model] for r in rows if r['group'] == g),
                                               'count': sum(r['group'] == g for r in rows), 'accuracy_fraction': str(v)}
                                           for g, v in group_values.items()}}
    comparisons = {}
    for baseline in ('BASE7', 'CONSTANT1'):
        rescued = [r['original_query_id'] for r in rows if not r['correct'][baseline] and r['correct']['CONDITIONAL4']]
        broken = [r['original_query_id'] for r in rows if r['correct'][baseline] and not r['correct']['CONDITIONAL4']]
        comparisons[baseline] = {'rescue_query_ids': rescued, 'break_query_ids': broken,
                                 'rescue': len(rescued), 'break': len(broken), 'net': len(rescued) - len(broken)}
    gates = {f'count_strictly_above_{m}': scores['CONDITIONAL4']['correct'] > scores[m]['correct'] for m in ('BASE7', 'CONSTANT1')}
    gates.update({f'equal_group_strictly_above_{m}': fractions['CONDITIONAL4'] > fractions[m] for m in ('BASE7', 'CONSTANT1')})
    gates['RAW_break_not_above_BASE7'] = scores['CONDITIONAL4']['break_vs_RAW'] <= scores['BASE7']['break_vs_RAW']
    return {'status': 'TRAIN128_DISAGREEMENT_OOF4_JOINED', 'authority': authority, 'program': bind(PROGRAM),
            'contract': CONTRACT, 'fold_validations': validations, 'gallery_mapping_sha256': mapping_sha,
            'query_count': 128, 'identity_count': 32, 'group_count': 32,
            'RAW_correct': sum(r['RAW_correct'] for r in rows), 'target_absent_count': sum(not r['target_in_natural_C128'] for r in rows),
            'fold_metrics': {str(f): {'count': sum(r['fold'] == f for r in rows),
                'RAW_correct': sum(r['RAW_correct'] for r in rows if r['fold'] == f),
                'correct': {m: sum(r['correct'][m] for r in rows if r['fold'] == f) for m in MODELS}}
                for f in range(4)},
            'scores': scores, 'conditional_comparisons': comparisons,
            'paired_group_statistics': paired_group_statistics(rows, groups),
            'gates': gates, 'OOF_gate_pass': all(gates.values()),
            'next_stage': 'SEPARATE_FINAL_FIT_AUTHORITY_ELIGIBLE' if all(gates.values()) else 'STOP_THIS_FAMILY_WITHOUT_EVAL_ACCESS',
            'rows': rows, 'EVAL_reads': 0, 'final_ec7_fit_performed': False, 'forbidden_read_attempts': len(BLOCKED)}


def join(replay=False, nonce=None):
    if replay:
        need(nonce and os.environ.get('RC_DISAGREEMENT_OOF_REPLAY_NONCE') == nonce, 'FRESH_JOIN_NONCE')
    result = joined_result()
    if replay:
        need(result == read(OUT / 'result.json'), 'INDEPENDENT_JOIN_STATS_GATE_REPLAY')
        write(OUT / 'validation.json', {'status': 'TRAIN128_DISAGREEMENT_OOF4_ALL_FOLDS_AND_JOIN_REPLAY_PASS',
              'result': bind(OUT / 'result.json'), 'authority': bind(AUTH), 'program': bind(PROGRAM),
              'fresh_process_nonce': nonce, 'process_PID': os.getpid(), 'OOF_gate_pass': result['OOF_gate_pass'],
              'query_count': 128, 'group_count': 32, 'all_heldout_logit_checks': 128 * 127 * 3,
              'EVAL_reads': 0, 'forbidden_read_attempts': len(BLOCKED)})
    else:
        write(OUT / 'result.json', result)
        fresh('validate-join')
    print(json.dumps({'stage': PHASE, 'OOF_gate_pass': result['OOF_gate_pass'], 'scores': result['scores']}), flush=True)


def synthetic_preflight():
    training_functions()
    dtype = torch.float64
    x = torch.tensor([[1., -2., .25, .5], [1., .7, -.1, -.8], [1., 0., 0., 0.]], dtype=dtype)
    df = torch.tensor([.2, -.4, 0.], dtype=dtype)
    baseline = torch.tensor([-.2, .7, -.1], dtype=dtype)
    alpha = torch.zeros(1, dtype=dtype, requires_grad=True)
    beta = torch.zeros(4, dtype=dtype, requires_grad=True)
    a = baseline + F.softplus(x[:, :1] @ alpha) * df
    b = baseline + F.softplus(x @ beta) * df
    need(torch.equal(a, b) and b[2] == baseline[2], 'EQUAL_INITIAL_GATES_AND_ZERO_DF')
    b.square().sum().backward()
    need(torch.isfinite(beta.grad).all() and torch.count_nonzero(beta.grad[1:]) == 3, 'CONDITIONAL_SLOPES_RECEIVE_GRADIENT')
    row = {'target_position': 0, 'base_winner_position': 0, 'challenger_positions': [1, 2, 3]}
    need(full_query_loss(b.detach(), row) == 4 * F.softplus(b.detach().max()), 'HOLD_PROTECTION_LOSS')
    row['target_position'] = 2
    need(full_query_loss(b.detach(), row) == F.softplus(-b.detach()[1]) + 4 * F.softplus(torch.stack([b.detach()[0], b.detach()[2]]).max()), 'TARGET_AND_STRONGEST_WRONG_LOSS')
    # Negative logits HOLD; positive tie picks the smaller physical row, never tensor order.
    for logits, axis, wanted in (([-1., -1., -1.], [8, 7, 2, 9], 0), ([1., 1., .5], [8, 7, 2, 9], 2)):
        i = max(range(3), key=lambda j: (logits[j], -axis[j + 1]))
        actual = i + 1 if logits[i] > 0 else 0
        need(actual == wanted, 'UNCHANGED_HOLD_SWITCH_AND_PHYSICAL_TIES')
    # Exercise actual train_head, correction optimizer and 127-challenger serializer
    # on synthetic inputs; production always calls the frozen 2000-step defaults.
    synthetic = []
    for i, target in enumerate((0, 3)):
        features = torch.arange(127 * 6, dtype=dtype).reshape(127, 6) / 1000 - .3
        context = torch.stack([torch.ones(127, dtype=dtype), -features[:, 0],
                               features[:, 1] / 2, features[:, 2] / 3], dim=1)
        synthetic.append({'query_id': 'SYNTHETIC-%d' % i, 'execution_ordinal': i,
            'source_image_sha256': 'synthetic-no-image', 'candidate_physical_rows': list(range(128)),
            'base_winner_position': 0, 'challenger_positions': list(range(1, 128)), 'target_position': target,
            'X': features, 'real_native_features': {ARM: features}, 'gate_X': context,
            'dF': torch.linspace(-.3, .4, 127, dtype=dtype)})
    pair = {'records': [{'switch_label': bool(i), 'real_native_features': {ARM: synthetic[i]['X'][:1]}} for i in range(2)]}
    old = training_functions()
    old['STEPS'] = 2
    head, _, finite = old['train_head'](pair, synthetic, 'NATIVE7', ARM)
    need(finite['all_finite'], 'SYNTHETIC_OLD_TRAIN_FUNCTION_RUNS')
    w, bias = head.weight.detach().flatten().clone(), float(head.bias.detach())
    parameters = {'BASE7': {'weight_binary64': [hx(v) for v in w], 'bias_binary64': hx(bias)}}
    for name, count in (('CONSTANT1', 1), ('CONDITIONAL4', 4)):
        parameters[name] = fit_correction(w, bias, synthetic, count, steps=2)
        need(parameters[name] == fit_correction(w, bias, synthetic, count, steps=2), 'DETERMINISTIC_SYNTHETIC_CORRECTION_RETRAIN')
    predictions = predict(parameters, synthetic)
    need(len(predictions) == 2 and all(len(p['predictions'][m]['all127_logits_binary64']) == 127
         for p in predictions for m in MODELS), 'SYNTHETIC_FULL_CHALLENGER_SERIALIZATION')
    # Independently enumerate small sign-flip spaces, including all-zero and
    # mixed-zero groups, to validate the DP's inclusive two-sided probability.
    import itertools
    for values in ([Fraction(0), Fraction(0)], [Fraction(0), Fraction(1, 2), Fraction(1, 3), Fraction(0)],
                   [Fraction(1, 2), Fraction(1, 3), Fraction(-1, 6), Fraction(0)]):
        observed = abs(sum(values, Fraction(0)))
        totals = [sum((sign * value for sign, value in zip(signs, values, strict=True)), Fraction(0))
                  for signs in itertools.product((-1, 1), repeat=len(values))]
        expected = Fraction(sum(abs(total) >= observed for total in totals), len(totals))
        got = exact_group_signflip(values)
        need(Fraction(got['two_sided_p_fraction']) == expected and
             got['all_group_sign_assignments'] == len(totals), 'SIGNFLIP_DP_VS_COMPLETE_ENUMERATION_WITH_ZEROS')
    statistic_rows = [{'group': 'synthetic-group-%d' % (i // 2), 'original_query_id': 'synthetic-%d' % i,
                       'correct': {model: i % 2 == 0 for model in MODELS}} for i in range(6)]
    first = paired_group_statistics(statistic_rows, ['synthetic-group-%d' % i for i in range(3)])
    need(first == paired_group_statistics(statistic_rows, ['synthetic-group-%d' % i for i in range(3)]),
         'FIXED_SEED_BOOTSTRAP_REPLAY')
    need(all(v['bootstrap_95_percentile_interval'] == [0.0, 0.0] and
         v['exact_two_sided_group_signflip']['two_sided_p_fraction'] == '1'
         for v in first['contrasts'].values()), 'ALL_ZERO_PAIRED_STATISTICS')
    result = {'status': 'TRAIN128_DISAGREEMENT_OOF4_SYNTHETIC_PREFLIGHT_PASS', 'program': bind(PROGRAM),
              'contract': CONTRACT, 'statistics_contract': STATISTICS_CONTRACT, 'runtime': runtime(), 'synthetic_checks': 15,
              'natural_feature_reads': 0, 'natural_training_updates': 0, 'EVAL_reads': 0,
              'forbidden_read_attempts': len(BLOCKED)}
    path = PREFLIGHT / sha(PROGRAM) / 'result.json'
    write(path, result)
    print(json.dumps({'status': result['status'], 'preflight': bind(path)}), flush=True)


def main():
    global PHASE, FOLD
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'preflight', 'project', 'fit-fold', 'validate-fold', 'join', 'validate-join', 'run'), required=True)
    parser.add_argument('--fold', type=int, choices=range(4))
    parser.add_argument('--nonce')
    args = parser.parse_args()
    PHASE, FOLD = args.phase, args.fold
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    torch.set_num_interop_threads(1)
    sys.addaudithook(audit)
    if PHASE == 'prepare':
        metadata_prepare()
    elif PHASE == 'preflight':
        synthetic_preflight()
    elif PHASE == 'project':
        project_pairs()
    elif PHASE in ('fit-fold', 'validate-fold'):
        need(FOLD is not None, 'FOLD_REQUIRED')
        fold_fit(FOLD, PHASE == 'validate-fold', args.nonce)
    elif PHASE in ('join', 'validate-join'):
        join(PHASE == 'validate-join', args.nonce)
    else:
        require_authority()
        subprocess.run([sys.executable, str(PROGRAM), '--phase', 'project'], check=True)
        for fold in range(4):
            subprocess.run([sys.executable, str(PROGRAM), '--phase', 'fit-fold', '--fold', str(fold)], check=True)
        subprocess.run([sys.executable, str(PROGRAM), '--phase', 'join'], check=True)


if __name__ == '__main__':
    main()
