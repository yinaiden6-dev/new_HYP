#!/usr/bin/env python3
"""Freeze a separate natural-prefix F128 experiment; never alter F71 artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

RC = Path(__file__).resolve().parents[1]
OLD = RC / 'results/rc_rebut_qr_qrr_v1'
OUT = RC / 'results/rc_rebut_qr_qrr_f128_v1'
FEATURES = RC / 'results/rc_h593_feature_fusion_cache_v1'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return {'path': str(path), 'sha256': h.hexdigest()}


def checked(binding):
    path = Path(binding['path'])
    assert bind(path)['sha256'] == binding['sha256'], ('SOURCE_SHA', str(path))
    return path


def immutable(path, value):
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == data, ('REFUSE_FROZEN_DRIFT', str(path))
    else:
        with path.open('x') as stream:
            stream.write(data)


def source_paths():
    return {
        'f128_preparation': Path(__file__),
        'f128_evidence_builder': RC / 'programs/build_rebut_qr_qrr_evidence_f128_v1.py',
        'f128_evidence_verifier': RC / 'programs/verify_rebut_qr_qrr_evidence_f128_v1.py',
        'f128_common': RC / 'programs/rebut_qr_qrr_f128_common_v1.py',
        'f128_runner': RC / 'programs/run_rebut_qr_qrr_f128_v1.py',
        'f128_join': RC / 'programs/join_rebut_qr_qrr_f128_v1.py',
        'v1_runner': RC / 'programs/run_rebut_qr_qrr_v1.py',
        'v2_runner': RC / 'programs/run_rebut_qr_qrr_frozenbase_v2.py',
        'v1_join': RC / 'programs/join_rebut_qr_qrr_v1.py',
        'v2_join': RC / 'programs/join_rebut_qr_qrr_frozenbase_v2.py',
        'module': RC / 'src/rc_aslo_xf/rebut_qr_qrr_v1.py',
        'v1_evidence_builder': RC / 'programs/build_rebut_qr_qrr_evidence_v1.py',
        'v1_evidence_verifier': RC / 'programs/verify_rebut_qr_qrr_evidence_v1.py',
        'f128_fold_runner': RC / 'programs/run_rebut_qr_qrr_f128_fold_v1.py',
        'f128_submitter': RC / 'programs/submit_rebut_qr_qrr_f128_v1.py',
        'f128_launcher': RC / 'slurm/rebut_qr_qrr_f128_v1.sbatch',
        'f128_plan': RC / 'plan/RC_REBUT_QR_QRR_F128_V1_20260927.md',
    }


def make_splits(old, panel_ids):
    split = read(checked(old['sources']['grouped_split']))
    folds = {}
    held_union = set()
    for fold in range(5):
        original = split['folds'][fold]
        binding = old['folds'][str(fold)]['train_roles']
        # Only this fold's pre-existing TRAIN roles are used to form its inner split.
        roles = {r['query_id']: r for r in read(checked(binding))['records']}
        train = sorted(set(original['train_query_ids']) & panel_ids)
        held = sorted(set(original['heldout_query_ids']) & panel_ids)
        assert set(train) <= set(roles) and not set(held) & set(roles)
        assert not set(train) & set(held) and set(train) | set(held) == panel_ids
        assert not held_union & set(held)
        held_union.update(held)
        components = sorted({roles[q]['component'] for q in train},
            key=lambda c: hashlib.sha256(('REBUT_QR_QRR_V1|' + c).encode()).hexdigest())
        val_components = set(components[:max(1, math.ceil(.2 * len(components)))])
        val = [q for q in train if roles[q]['component'] in val_components]
        fit = [q for q in train if roles[q]['component'] not in val_components]
        assert fit and val and held
        assert not {roles[q]['identity'] for q in fit} & {roles[q]['identity'] for q in val}
        folds[str(fold)] = dict(train_roles=binding, train_query_ids=train,
            heldout_query_ids=held, inner_fit_query_ids=fit, inner_val_query_ids=val)
    assert held_union == panel_ids
    return folds


def prepare():
    torch.set_num_threads(2)
    old = read(OLD / 'protocol.json')
    old_validation = read(OLD / 'validation.json')
    assert old_validation['status'] == 'REBUT_QR_QRR_ALL60_F71_INDEPENDENT_JOIN_PASS'
    assert old_validation['protocol'] == bind(OLD / 'protocol.json')
    common_payload = torch.load(checked(old['common_features']), map_location='cpu', weights_only=True)
    assert common_payload['labels_included'] is False
    common = {r['execution_ordinal']: r for r in common_payload['records']}
    assert set(common) == set(range(593))
    expected = [dict(query_id=common[i]['query_id'], execution_ordinal=i,
                     axis=common[i]['axis'], winner=common[i]['winner']) for i in range(128)]
    panel_ids = {r['query_id'] for r in expected}
    assert len(panel_ids) == 128
    folds = make_splits(old, panel_ids)
    manifest_path = OLD / 'evidence_manifest.json'
    manifest = read(manifest_path)
    assert manifest['status'] == 'POOLED_F71_COMPLETE'
    assert sorted(r['execution_ordinal'] for r in manifest['records']) == list(range(71))
    reused = []
    for row in sorted(manifest['records'], key=lambda r: r['execution_ordinal']):
        payload = torch.load(checked(row['payload']), map_location='cpu', weights_only=True, mmap=True)
        c = common[row['execution_ordinal']]
        assert payload['query_id'] == row['query_id'] == c['query_id']
        assert payload['axis'] == c['axis'] and payload['winner'] == c['winner']
        assert torch.equal(payload['M0'], c['M0'])
        assert payload['query'].shape == payload['reference'].shape == (128, 64, 72)
        reused.append(dict(row, reused_from_F71=True))
    ready = read(FEATURES / 'ready.json')
    assert ready['status'] == 'FUSION_ALL593_FEATURE_CACHE_PASS'
    checked(ready['catalog'])
    catalog = read(FEATURES / 'catalog.json')
    by_ordinal = {r['execution_ordinal']: r for r in catalog['queries']}
    assert all(by_ordinal[r['execution_ordinal']]['query_id'] == r['query_id'] for r in expected)
    sources = dict(old['sources'])
    code_sources = {key: bind(path) for key, path in source_paths().items()}
    sources.update(code_sources)
    evidence_inputs = dict(status='F128_EVIDENCE_INPUTS_FROZEN', panel_indices=list(range(128)),
        expected_records=expected, common_features=old['common_features'],
        feature_ready=bind(FEATURES / 'ready.json'), feature_catalog=ready['catalog'],
        old_evidence_manifest=bind(manifest_path), reused_records=reused,
        new_indices=list(range(71, 128)), schema_unchanged=True,
        sources={key: sources[key] for key in ('f128_evidence_builder', 'f128_evidence_verifier',
            'v1_evidence_builder', 'v1_evidence_verifier')},
        query_shape=[128, 64, 72], reference_shape=[128, 64, 72], labels_read=0,
        build_gate='CPU build after the original first128 coordinate export validation; no new GPU acquisition here')
    immutable(OUT / 'evidence_inputs.json', evidence_inputs)
    protocol = copy.deepcopy(old)
    protocol.update(version='REBUT_QR_QRR_F128_V1', status='F128_PROTOCOL_FROZEN_BEFORE_NEW_FIT',
        user_authorization='2026-09-27 continue QR/QRR on natural first128; keep all previous F71 protocols/results sealed',
        output_root=str(OUT), evidence_manifest=str(OUT / 'evidence_manifest.json'),
        evidence_inputs=bind(OUT / 'evidence_inputs.json'), folds=folds,
        panel_query_ids=sorted(panel_ids), panel_indices=list(range(128)),
        panel_rule='All natural execution ordinals0..127, original593 outer-fold intersection; no outcome selection',
        historical_comparator_boundary='Historical481/492/496 heads used broader TRAIN; only same-F128 trained B_CAL is the primary matched comparator; not oldEVAL128',
        sources=sources, arms=['B_CAL', 'QR', 'QR_VEC', 'QRR'], seeds=[0, 1, 2],
        code_sources=code_sources,
        input_schema='Same pooled/projection/warp F interface as F71; old71 bytes reused,57 new CPU exports',
        evidence_schema_version='POOLED_F71_COARSE17_NATIVE_CENTER_WARP_V1',
        evidence_manifest_status_required='POOLED_F128_COMPLETE',
        baseline_fits=15, residual_fits=45, baseline_training_population='New matched F128 nested TRAIN splits, not reused F71 trained heads',
        head='First train matched common18 B_CAL; residuals freeze phase-correct inner and outer B_CAL weights/RMS',
        residual_selection=read(RC / 'results/rc_rebut_qr_qrr_frozenbase_v2/protocol.json')['selection'],
        residual_threshold='fixed0 primary; inherited B_CAL inner threshold secondary; no new threshold optimization',
        prior_F71_protocol=bind(OLD / 'protocol.json'), prior_F71_validation=bind(OLD / 'validation.json'),
        source_baseline_fits_reused=0, new_baseline_fits=15, new_reader_fits=45,
        final_main_population=128, F593_status='Not a593 training experiment; only the separately frozen natural-prefix F128',
        evidence_level='Expanded grouped development F128; includes previously explored F71, not untouched external confirmation',
        scientific_feature_changes=0, new_backbone_forwards=0, backbone_updates=0)
    immutable(OUT / 'protocol.json', protocol)
    baseline = copy.deepcopy(protocol)
    baseline.update(version='REBUT_QR_QRR_F128_BASELINE_V1',
        status='F128_BASELINE_PROTOCOL_FROZEN_BEFORE_NEW_FIT',
        output_root=str(OUT / 'baseline'), arms=['B_CAL'], root_protocol=bind(OUT / 'protocol.json'),
        head=old['head'], checkpoint=old['checkpoint'], threshold=old['threshold'])
    immutable(OUT / 'baseline_protocol.json', baseline)
    receipt = dict(status='F128_DATA_PREPARATION_PASS_NOT_EVIDENCE_COMPLETION',
        protocol=bind(OUT / 'protocol.json'), baseline_protocol=bind(OUT / 'baseline_protocol.json'),
        evidence_inputs=bind(OUT / 'evidence_inputs.json'), common_features=old['common_features'],
        code_sources=code_sources,
        queries=128, existing_F71_payloads_SHA_axis_M0_checked=71, new_CPU_evidence_queries=57,
        source_common_queries=593, training_updates=0, held_label_files_read=0,
        evidence_complete_claimed=False, original_F71_changed=False,
        fold_counts={f: {key: len(value[key]) for key in ('train_query_ids', 'heldout_query_ids',
            'inner_fit_query_ids', 'inner_val_query_ids')} for f, value in folds.items()})
    immutable(OUT / 'preparation_validation.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
