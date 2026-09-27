#!/usr/bin/env python3
"""Prepare exactly 30 seed1/2 fits only after a bound current seed0 gate PASS."""
import argparse
import ast
import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from token_competition_confirm_common_v1 import (
    ARMS, NEW_SOURCES, ORIGIN, OUT, RC, bind, checked, immutable, read,
    verify_confirmation, verify_gate)


def prepare(gate_path):
    gate = verify_gate(gate_path)  # No output directory/protocol before this gate.
    origin = read(checked(gate['origin_protocol']))
    old = read(checked(origin['prior_F128_residual_protocol']))
    previous = read(checked(origin['dependency_bindings']['previous_final_validation']))
    if previous.get('status') != 'REBUT_QR_QRR_F128_ALL60_INDEPENDENT_JOIN_PASS':
        raise ValueError('Prior independently verified F128 three-seed baseline lineage missing')
    baseline_bindings = {}
    for fold in range(5):
        baseline_bindings[str(fold)] = {}
        for seed in (1, 2):
            source = old['baseline_bindings'][str(fold)][str(seed)]
            for value in source.values():
                checked(value)
            result = read(source['source_result']['path'])
            if result['binding']['arm'] != 'B_CAL' or result['binding']['seed'] != seed or result['binding']['fold'] != fold:
                raise ValueError('Wrong frozen baseline identity')
            for key in ('train_query_ids', 'heldout_query_ids', 'inner_fit_query_ids', 'inner_val_query_ids'):
                if origin['folds'][str(fold)][key] != result[key]:
                    raise ValueError('Baseline fold differs from original V2: ' + key)
            checked(origin['folds'][str(fold)]['train_roles'])
            baseline_bindings[str(fold)][str(seed)] = copy.deepcopy(source)
    from token_competition_data_v2 import verify_evidence_gate
    evidence_validation = verify_evidence_gate(origin)
    sources = {**origin['code_sources'], **{name: bind(RC / name) for name in NEW_SOURCES}}
    updates = {
        'version': 'TOKEN_COMPETITION_F128_CONFIRM_V1',
        'user_authorization': 'Continue the next experiment automatically only after the current analysis gate PASS; all three arms, seeds1 and2.',
        'output_root': str(OUT), 'seeds': [1, 2], 'arms': list(ARMS),
        'baseline_bindings': baseline_bindings,
        'origin_protocol': gate['origin_protocol'], 'origin_validation': gate['origin_validation'],
        'promotion_gate': bind(gate_path), 'origin_evidence_validation': evidence_validation,
        'sources': sources, 'code_sources': sources,
        'source_baseline_fits_reused': 10, 'new_baseline_fits': 0,
        'new_reader_fits': 30, 'baseline_fits': 0, 'residual_fits': 30,
        'dependency_bindings': {**origin['dependency_bindings'],
            'origin_protocol': gate['origin_protocol'], 'origin_validation': gate['origin_validation'],
            'promotion_gate': bind(gate_path), 'origin_evidence_validation': evidence_validation},
        'baseline_training_population': 'Reuse SAME F128 seed1/2 B_CAL inner-fit and outer-TRAIN states; no baseline fits.',
        'evidence_level': 'Additional seed stability on the SAME opened F128 development panel; not new data or external confirmation.',
        'evidence_reuse': 'Read only original V2 full native cache; verify its receipt against the ORIGINAL V2 protocol and exact common/gallery/panel axes.',
        'execution_binding_note': 'Checkpoint/result binding.runner_sha256 and data_adapter_sha256 identify the immutable V2 core. The confirmation protocol pins the active wrapper and adapter; each completed fit adds execution_adapter.json. No core source files are edited.',
        'seed_reporting': 'Show original seed0 plus seeds1 and2 separately and arithmetic summaries across all three; never select a best seed or arm using held results.',
        'scientific_change': 'Registered seeds only: [0] -> [1,2]; all scientific settings and splits remain identical.',
        'scheduling': {'bulk_partition': 'cpuonly', 'small_partitions': 'cpuonly,dev_cpuonly',
            'gpus': 0, 'training_tasks': 30, 'training_parallelism': 30,
            'wall_seconds': 480, 'max_restarts': 96},
    }
    protocol = copy.deepcopy(origin)
    protocol.update(updates)
    protocol['immutable_origin_fields'] = sorted(set(origin) - set(updates))
    verify_confirmation(protocol)
    # Import every new module before sealing the protocol. This exercises the
    # wrapper/adapter names without opening labels or executing training.
    for name in NEW_SOURCES:
        if name.endswith('.py'):
            ast.parse((RC / name).read_text(), filename=name)
            importlib.import_module(Path(name).stem)
    from run_token_competition_confirm_stage_v1 import task
    if {task(i) for i in range(30)} != {(f, s, a) for f in range(5) for s in (1, 2) for a in ARMS}:
        raise ValueError('Confirmation array does not cover exactly30 fits')
    subprocess.run([sys.executable, str(RC / 'programs/check_token_competition_confirm_launcher_v1.py'),
                    '--output', str(OUT / 'launcher_engineering.json')], check=True)
    immutable(OUT / 'protocol.json', protocol)
    immutable(OUT / 'preparation_validation.json', {
        'status': 'TOKEN_COMPETITION_F128_CONFIRM_PREPARATION_PASS',
        'protocol': bind(OUT / 'protocol.json'), 'promotion_gate': bind(gate_path),
        'new_reader_fits': 30, 'frozen_baselines_reused': 10,
        'new_baseline_fits': 0, 'new_backbone_forwards': 0,
        'seeds': [1, 2], 'original_seed0_retained': True,
        'held_labels_read_for_preparation': False, 'training_submitted': False,
    })
    immutable(OUT / 'source_engineering.json', {
        'status': 'TOKEN_COMPETITION_CONFIRM_SOURCES_PASS', 'protocol': bind(OUT / 'protocol.json'),
        'sources': sources, 'all_new_modules_imported': True,
        'task_index_bijection': True, 'new_scientific_fits_run': 0,
    })
    print(json.dumps({'status': 'TOKEN_COMPETITION_F128_CONFIRM_PREPARATION_PASS',
                      'protocol': str(OUT / 'protocol.json'), 'new_reader_fits': 30}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    prepare(parser.parse_args().gate.resolve())
