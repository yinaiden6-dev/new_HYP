#!/usr/bin/env python3
"""Pure-stdlib, fail-closed bindings for the F128 seed confirmation."""
import hashlib
import json
from pathlib import Path

RC = Path(__file__).resolve().parents[1]
ORIGIN = RC / 'results/rc_token_competition_f128_v2'
OUT = RC / 'results/rc_token_competition_f128_confirm_v1'
ARMS = ('TOKEN_QR', 'TOKEN_QRR_ANCHOR', 'TOKEN_QRR_MULTI')
GATE_PASS = 'TOKEN_COMPETITION_F128_SEED_CONFIRMATION_GATE_PASS'
ORIGIN_PASS = 'TOKEN_COMPETITION_F128_ALL15_INDEPENDENT_JOIN_PASS'
FINAL_PASS = 'TOKEN_COMPETITION_F128_CONFIRM_ALL30_INDEPENDENT_JOIN_PASS'
SCIENTIFIC_FIELDS = (
    'folds', 'panel_query_ids', 'panel_indices', 'common_features', 'gallery',
    'evidence_manifest', 'expected_records', 'optimizer', 'model_config', 'loss',
    'selection', 'residual_selection', 'residual_threshold', 'threshold',
    'normalization', 'normalizing_features', 'local_scalars', 'rival_rule',
    'input_schema', 'candidate_source', 'head', 'panel_rule',
    'source_protocol', 'source_validation', 'backbone_updates', 'new_backbone_forwards',
)
NEW_SOURCES = [
    'programs/token_competition_confirm_common_v1.py',
    'programs/prepare_token_competition_confirm_v1.py',
    'programs/token_competition_confirm_data_v1.py',
    'programs/run_token_competition_confirm_v1.py',
    'programs/join_token_competition_confirm_v1.py',
    'programs/run_token_competition_confirm_stage_v1.py',
    'programs/submit_token_competition_confirm_v1.py',
    'programs/check_token_competition_confirm_launcher_v1.py',
    'programs/analyze_token_competition_confirm_v1.py',
    'slurm/token_competition_confirm_v1.sbatch',
]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def checked(binding):
    if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
        raise ValueError('An explicit path/SHA256 binding is required')
    if bind(binding['path']) != binding:
        raise ValueError('Bound artifact changed: ' + binding['path'])
    return Path(binding['path'])


def immutable(path, value):
    path = Path(path)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            raise ValueError('Refuse changing frozen confirmation: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(data)


def verify_gate(path):
    gate = read(path)
    if gate.get('status') != GATE_PASS:
        raise ValueError('Seed confirmation requires an explicit current analysis gate PASS')
    for name in ('origin_protocol', 'origin_validation', 'analysis_gate', 'gate_specification'):
        checked(gate[name])
    if gate['origin_protocol'] != bind(ORIGIN / 'protocol.json'):
        raise ValueError('Promotion does not bind the current V2 seed0 protocol')
    if gate['origin_validation'] != bind(ORIGIN / 'validation.json'):
        raise ValueError('Promotion does not bind the current V2 all15 validation')
    validation = read(gate['origin_validation']['path'])
    if validation.get('status') != ORIGIN_PASS or validation.get('protocol') != gate['origin_protocol']:
        raise ValueError('V2 independent all15 join PASS is required')
    analysis = read(gate['analysis_gate']['path'])
    if (analysis.get('status') != 'TOKEN_COMPETITION_ANALYSIS_PASS' or
            analysis.get('protocol') != gate['origin_protocol'] or
            analysis.get('validation') != gate['origin_validation']):
        raise ValueError('Promotion analysis must bind the current independently validated V2 result')
    for value in analysis.get('artifacts', {}).values():
        checked(value)
    arms = gate.get('qualifying_arms', [])
    if not arms or len(arms) != len(set(arms)) or not set(arms) <= set(ARMS):
        raise ValueError('Promotion requires identified qualifying arms; all three arms still run')
    for value in validation.get('artifacts', {}).values():
        checked(value)
    return gate


def verify_confirmation(protocol):
    if protocol.get('version') != 'TOKEN_COMPETITION_F128_CONFIRM_V1':
        raise ValueError('Wrong confirmation protocol version')
    if protocol.get('seeds') != [1, 2] or tuple(protocol.get('arms', [])) != ARMS:
        raise ValueError('Confirmation requires all three arms and exactly seeds1,2')
    if Path(protocol['output_root']).resolve() != OUT:
        raise ValueError('Confirmation output must be isolated from the original V2 root')
    gate = verify_gate(checked(protocol['promotion_gate']))
    if protocol['origin_protocol'] != gate['origin_protocol'] or protocol['origin_validation'] != gate['origin_validation']:
        raise ValueError('Confirmation origin differs from promotion gate')
    origin = read(checked(protocol['origin_protocol']))
    if origin.get('version') != 'TOKEN_COMPETITION_F128_V2' or origin.get('seeds') != [0]:
        raise ValueError('Confirmation origin must be the original V2 seed0 protocol')
    if not set(SCIENTIFIC_FIELDS) <= set(protocol['immutable_origin_fields']):
        raise ValueError('Confirmation omitted a required immutable scientific field')
    for field in protocol['immutable_origin_fields']:
        if protocol[field] != origin[field]:
            raise ValueError('Unchanged scientific field differs from origin: ' + field)
    for name in ('sources', 'code_sources', 'dependency_bindings'):
        for value in protocol[name].values():
            checked(value)
    old = read(checked(protocol['prior_F128_residual_protocol']))
    for fold in range(5):
        for seed in (1, 2):
            if protocol['baseline_bindings'][str(fold)][str(seed)] != old['baseline_bindings'][str(fold)][str(seed)]:
                raise ValueError('Frozen B_CAL baseline seed binding changed')
    return origin
