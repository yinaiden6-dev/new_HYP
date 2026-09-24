#!/usr/bin/env python3
"""Seal the user-authorized source-only fit and external replay protocol."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / 'registry/rc_simple_external_transfer_authority_v1_20260924.json'
OUT = ROOT / 'results/rc_simple_external_transfer_v1'


def bind(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    assert not AUTH.exists(), 'Authority already frozen; do not overwrite it'
    old = ROOT / 'registry/rc_new_hyp_external_head_freeze_authority_v1_20260913.json'
    source = json.loads(old.read_text())
    codes = {f'legacy_{k}': v for k, v in source['code_sources'].items()}
    for k, relative in {
        'source': 'programs/run_rc_simple_external_source_freeze_v1.py',
        'replay': 'programs/run_rc_simple_external_replay_v1.py',
        'prepare': 'programs/prepare_rc_simple_external_transfer_v1.py',
        'submit': 'programs/submit_rc_simple_external_transfer_v1.py',
        'feature_formula': 'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
        'launcher': 'slurm/rc_simple_external_transfer_v1.sbatch',
        'plan': 'plan/RC_SIMPLE_EXTERNAL_TRANSFER_V1_20260924.md',
    }.items():
        codes[k] = bind(ROOT / relative)
    for b in codes.values():
        assert bind(b['path']) == b, 'Frozen dependency drift: ' + b['path']
    source_inputs = {k: bind(ROOT / r) for k, r in {
        'source_input_manifest': 'results/rc_simple_external_transfer_v1/source/input_manifest.json',
        'legacy_authority': str(old.relative_to(ROOT)),
        'legacy_head': 'results/rc_new_hyp_external_head_freeze_v1/COST1/head.json',
        'source_cache': 'results/rc_h593_simple_explanations_v1/cache.json',
        'source_cache_validation': 'results/rc_h593_simple_explanations_v1/cache_validation.json',
        'source_cache_authority': 'registry/rc_h593_simple_explanations_authority_v1_20260924.json',
    }.items()}
    external = {k: bind(ROOT / r) for k, r in {
        'grozi': 'registry/rc_new_hyp_grozi120_inference_authority_v1_20260913.json',
        'isic': 'registry/rc_new_hyp_isic_inference_authority_v1_20260914.json',
    }.items()}
    a = dict(status='SIMPLE_EXTERNAL_TRANSFER_AUTHORIZED',
             authorization='User 2026-09-24: complete three remaining claims in order 2, 3, 1',
             code_sources=codes, source_inputs=source_inputs,
             program=codes['replay'], feature_formula=codes['feature_formula'],
             external_authorities=external, external_inputs=external,
             source_bundle_path=str(OUT / 'source/bundle.json'),
             source_dataset='H593 existing source data', source_total=593,
             source_eligible=570, source_target_absent=23,
             primary='MASS5', controls=['ADDITIVE4', 'NATIVE7', 'RAW'],
             optimizer=dict(name='AdamW', lr=0.03, weight_decay=0.001, updates=2000,
                            dtype='float64', threads=8, initialization='zero', checkpoint='final'),
             loss='COST1', external_training=False, external_threshold_selection=False,
             switch_threshold=0.0, candidates='original natural C128 per external dataset',
             cpu_only=True, new_encoder_forwards=0, new_roma_forwards=0,
             datasets=dict(grozi=dict(queries=480, shards=60, task_offset=0, group='source video'),
                           isic=dict(queries=537, shards=68, task_offset=60, group='patient component')),
             seal_policy='All source heads before external predictions; all predictions before external label join',
             evidence='Fixed source-head transfer on previously examined external panels; ISIC exploratory',
             task_order=[2, 3, 1])
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / 'frozen_protocol'
    archive.mkdir(exist_ok=True)
    for key, b in codes.items():
        shutil.copyfile(b['path'], archive / (key + '__' + Path(b['path']).name))
    AUTH.write_text(json.dumps(a, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    shutil.copyfile(AUTH, archive / AUTH.name)
    print(json.dumps(dict(status='FROZEN', authority=bind(AUTH)), ensure_ascii=False))


if __name__ == '__main__':
    main()
