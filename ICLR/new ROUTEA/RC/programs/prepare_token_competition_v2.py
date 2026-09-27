#!/usr/bin/env python3
"""Seal the authorized native-token, all-C128 competition experiment."""
import copy
import hashlib
import json
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
OUT = RC / 'results/rc_token_competition_f128_v2'
OLD = RC / 'results/rc_rebut_qr_qrr_f128_v1'
FEATURES = RC / 'results/rc_h593_feature_fusion_cache_v1'
ARMS = ['TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI']
NEW_SOURCES = [
    'programs/prepare_token_competition_v2.py',
    'programs/build_token_competition_evidence_v2.py',
    'programs/verify_token_competition_evidence_v2.py',
    'programs/token_competition_data_v2.py',
    'programs/run_token_competition_v2.py',
    'programs/join_token_competition_v2.py',
    'programs/check_token_competition_modules_v2.py',
    'programs/run_token_competition_stage_v2.py',
    'programs/submit_token_competition_v2.py',
    'programs/check_token_competition_launcher_v2.py',
    'src/rc_aslo_xf/token_competition_v2.py',
    'slurm/token_competition_v2.sbatch',
    'plan/RC_MULTICANDIDATE_QUERY_VERIFICATION_V2_20260927.md',
]
OLD_SOURCES = [
    'programs/rebut_qr_qrr_f128_common_v1.py',
    'programs/run_rebut_qr_qrr_v1.py',
    'programs/run_rebut_qr_qrr_frozenbase_v2.py',
    'programs/join_rebut_qr_qrr_v1.py',
    'programs/join_rebut_qr_qrr_frozenbase_v2.py',
    'programs/join_rebut_qr_qrr_f128_v1.py',
    'src/rc_aslo_xf/rebut_qr_qrr_v1.py',
]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return {'path': str(path), 'sha256': h.hexdigest()}


def checked(b):
    if bind(b['path'])['sha256'] != b['sha256']:
        raise ValueError('Changed source: ' + b['path'])
    return Path(b['path'])


def immutable(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            raise ValueError('Refuse changing frozen experiment: ' + str(path))
    else:
        with path.open('x') as f:
            f.write(data)


def prepare():
    old = read(OLD / 'residual_protocol.json')
    previous = read(OLD / 'validation.json')
    if previous['status'] != 'REBUT_QR_QRR_F128_ALL60_INDEPENDENT_JOIN_PASS':
        raise ValueError('Original F128 full verification missing')
    baseline_validation = read(checked(old['source_validation']))
    if baseline_validation['status'] != 'F128_BASELINE15_INDEPENDENT_PASS':
        raise ValueError('Original fitted baseline seal missing')
    checked(old['source_protocol']); checked(old['common_features']); checked(old['gallery'])
    expected = read(OLD / 'evidence_inputs.json')['expected_records']
    if sorted(r['execution_ordinal'] for r in expected) != list(range(128)):
        raise ValueError('Natural F128 axis missing')
    ready = read(FEATURES / 'ready.json')
    if ready['status'] != 'FUSION_ALL593_FEATURE_CACHE_PASS':
        raise ValueError('Native image feature cache not complete')
    checked(ready['catalog'])
    catalog = read(ready['catalog']['path'])
    entries = {r['execution_ordinal']: r for r in catalog['queries']}
    for r in expected:
        source = entries[r['execution_ordinal']]
        if source['query_id'] != r['query_id']:
            raise ValueError('Feature catalog panel order differs')
        checked(source['payload'])
    code_sources = {p: bind(RC / p) for p in NEW_SOURCES + OLD_SOURCES}
    baseline_bindings = {}
    for fold in range(5):
        sf = old['folds'][str(fold)]
        checked(sf['train_roles'])
        b = old['baseline_bindings'][str(fold)]['0']
        for v in b.values():
            checked(v)
        result = read(b['source_result']['path'])
        for key in ('train_query_ids', 'heldout_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids'):
            if sf[key] != result[key]:
                raise ValueError('Baseline split mismatch: ' + key)
        baseline_bindings[str(fold)] = {'0': b}
    pilot = old['folds']['0']['inner_fit_query_ids'][0]
    pilot_ordinal = next(r['execution_ordinal'] for r in expected if r['query_id'] == pilot)
    p = copy.deepcopy(old)
    p.update(version='TOKEN_COMPETITION_F128_V2', status='PROTOCOL_FROZEN_BEFORE_NEW_FIT',
        user_authorization='User: 启动，提交完整任务链。All-C128 native-token V2, 2026-09-27',
        output_root=str(OUT), arms=ARMS, seeds=[0], baseline_bindings=baseline_bindings,
        evidence_manifest=str(OUT / 'evidence_manifest.json'),
        feature_ready=bind(FEATURES / 'ready.json'), feature_catalog=ready['catalog'],
        pilot_query_id=pilot, pilot_execution_ordinal=pilot_ordinal,
        expected_records=expected, source_baseline_fits_reused=5, new_baseline_fits=0,
        new_reader_fits=15, baseline_fits=0, residual_fits=15,
        sources=code_sources, code_sources=code_sources,
        dependency_bindings={'previous_final_validation': bind(OLD / 'validation.json'),
                             'previous_residual_protocol': bind(OLD / 'residual_protocol.json')},
        input_schema='Native original-query cells; normalized q128 + full-reference content argmax r128 + eight scalars; no spatial pooling',
        evidence_schema_version='NATIVE_COLNOMIC_TOKEN_COMPETITION_F128_V2',
        evidence_manifest_status_required='TOKEN_COMPETITION_EVIDENCE128_PASS',
        scientific_feature_changes='Replace pooled projected coarse E with original local ColNomic content and support interactions',
        baseline_training_population='Reuse SAME F128 original seed0 baselines, including inner-fit and outer-TRAIN states',
        primary_structural_comparison='TOKEN_QRR_MULTI vs TOKEN_QRR_ANCHOR: identical parameters, changed rival graph only',
        secondary='TOKEN_QR versus joint readers; same local input, approximately matched active capacity',
        head='Frozen SAME F128 phase-correct 18-dimensional B_CAL plus D_g-D_RAW; RAW anchor exactly0',
        rival_rule='For each g: RAW w if g!=w, plus highest frozen B_CAL logit h outside {g,w}; ties ascending physical row; deduplicate and mean; all C128 remain eligible',
        gallery_identity_rule='Frozen gallery IDs; assert all128 identities unique in this panel; no query labels for rivals',
        local_scalars=['free_cos', 'free_top1_minus_top2', 'u', 'v_at_free',
                       'u_times_free_cos', 'v_at_free_times_free_cos', 'u_times_v_at_free_times_free_cos', 'mean_v'],
        normalizing_features='FP64 token L2 normalization before free matching; FP32 evidence storage then FP64 reader; original common features/RMS unchanged',
        model_config={'input_dim': 264, 'width': 24, 'pair_chunk_size': 16},
        engineering_checks=['synthetic model invariants', 'label-free evidence replay', 'real inner-TRAIN pilot for all3arms', 'baseline zero-residual replay', 'RNG optimizer serialization'],
        scheduling={'bulk_partition': 'cpuonly', 'small_partitions': 'cpuonly,dev_cpuonly',
                    'gpus': 0, 'cpus': 8, 'memory_gb': 32, 'time_minutes': 10,
                    'evidence_shards': 16, 'evidence_parallelism': 16,
                    'training_tasks': 15, 'training_parallelism': 15,
                    'wall_seconds': 480, 'max_restarts': 96},
        prior_F128_protocol=bind(OLD / 'protocol.json'),
        prior_F128_residual_protocol=bind(OLD / 'residual_protocol.json'),
        evidence_level='F128 opened development panel, original grouped fivefold, seed0; not held external confirmation',
        F593_status='No593 fit in this chain; unchanged existing593 acquisition continues independently',
        checkpoint='Per-fit lock, optimizer/model/RNG/cursor, per-candidate evidence shards; same Job requeue on75/timeout',
        backbone_updates=0, new_backbone_forwards=0)
    p.pop('evidence_inputs', None)
    p['optimizer'] = copy.deepcopy(old['optimizer'])
    immutable(OUT / 'protocol.json', p)
    immutable(OUT / 'preparation_validation.json', {
        'status': 'TOKEN_COMPETITION_PREPARATION_PASS', 'protocol': bind(OUT / 'protocol.json'),
        'code_sources': code_sources, 'queries':128, 'baseline_fits_reused':5,
        'new_baseline_fits':0, 'new_reader_fits':15, 'pilot_query_id':pilot,
        'pilot_execution_ordinal':pilot_ordinal, 'labels_read_for_preparation':False,
        'training_submitted':False})
    print(json.dumps({'status':'TOKEN_COMPETITION_PREPARATION_PASS', 'protocol':str(OUT/'protocol.json'),
                      'new_fits':15, 'pilot':pilot, 'pilot_ordinal':pilot_ordinal}), flush=True)


if __name__ == '__main__':
    prepare()
