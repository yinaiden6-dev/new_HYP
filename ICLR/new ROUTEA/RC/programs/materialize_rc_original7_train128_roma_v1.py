#!/usr/bin/env python3
"""TRAIN128 input materialization with unchanged qualified RoMa/C4 operators.

Only authorized natural shards run GPU inference. Completed RAW/RoMa bridges
are explicitly reused under their original authorities; no bridge is rerun.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
import uuid

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'programs'))
import rc_original7_train128_execution_common_v1 as C

PROFILE = ROOT / 'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
TOKEN = ROOT / 'results/rc_original7_train128_token_raw_v1'
OUT = ROOT / 'results/rc_original7_train128_roma_v1'
RAW_AGGREGATE = ROOT / 'results/rc_original7_train128_inputs_v1/raw_aggregate.json'
WORKER = ROOT / 'results/rc_original7_train128_manifest_v1/worker_manifest.json'
WORKER_SHA = '50d894c9643ca1ef200c18ba79cf700b598be15c5ea322010853c201fc0f22f7'
METADATA = WORKER.parent / 'independent_metadata_validation.json'
METADATA_SHA = '03176db194d4b5f9d68cc226a1ce48b019675e99b037205bae71552202fa907d'
PARITY_INDEX = WORKER.parent / 'legacy_full32_parity_index.json'
PARITY_SHA = '31c337c676dc17d38fb85cdcb902b3aeddfe104aebbb6840baeaaba5837c6b03'
PLAN = ROOT / 'plan/RC_ORIGINAL7_TRAIN128_INPUTS_V1_20260910.md'
PARENT_PROGRAM = ROOT / 'programs/materialize_rc_original7_eval128_roma_v2_runtime_compat.py'
PARENT_PROGRAM_SHA = 'f3514841ca1d1fb47a78dbe571955465fe12b7e09c88cb2175862d2967d420df'
LAUNCHER = ROOT / 'slurm/rc_original7_train128_roma_shards_v1_accelerated_30m.sbatch'
CORE_SHA = 'fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c'
CHECKPOINT_SHA = '1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7'
TRAIN_EXPORT_SHA = 'c70f75ba1023ab0366c3509e4597c5289dfacc274dfa58b535e9ac3b9fc7b8fc'
SCORE_KEYS = ('real_score', 'visibility_mass', 'query_control_score', 'reference_control_score')
BRIDGE_PASS = 'RC_ORIGINAL7_EVAL128_ROMA_BRIDGE_PASS'
SHARD_PASS = 'RC_ORIGINAL7_TRAIN128_ROMA_CPU_REPLAY_PASS'


def hten(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_sha(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode('ascii'))
    digest.update(json.dumps(list(tensor.shape), separators=(',', ':')).encode('ascii'))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bit_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def deny_protected_reads(allow_expanded_images=False):
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        name = os.path.abspath(os.fsdecode(args[0])).lower()
        forbidden = ('curator_roles', 'target_join', 'd1_mi', 'd1-mi', 'grozi',
                     'opened_eval_strict', 'direction_capacity', 'projection_capacity',
                     'gisc_prerecall_universe', '/role_shards/', '/worker_assets/tokens/')
        C.need(not any(part in name for part in forbidden), 'PROTECTED_READ_DENIED')
        C.need('/reports/' not in name, 'REPORT_READ_DENIED')
        C.need('/worker_assets/images/' not in name or allow_expanded_images, 'NEW_IMAGES_ONLY_IN_AUTHORIZED_SHARD')
    sys.addaudithook(audit)


def source_profile():
    profile = C.read(PROFILE)
    C.need(profile['status'] == 'RC_ORIGINAL7_EVAL128_ROMA_SOURCE_PROFILE', 'PROFILE_STATUS')
    C.need(profile['sources']['core']['sha256'] == CORE_SHA, 'ORIGINAL_CORE_PIN')
    C.need(profile['sources']['checkpoint']['sha256'] == CHECKPOINT_SHA, 'CHECKPOINT_PIN')
    C.need(profile['sources']['old_train56_export']['sha256'] == TRAIN_EXPORT_SHA, 'TRAIN_EXPORT_PIN')
    C.verify_public_bindings(profile['sources'])
    for tree in profile['source_trees'].values():
        current = {str(p) for p in Path(tree['root']).rglob('*.py')}
        C.need(current == {x['path'] for x in tree['python_files']}, 'MODEL_SOURCE_TREE_COVERAGE')
        for entry in tree['python_files']:
            C.need(C.sha(entry['path']) == entry['sha256'], 'MODEL_SOURCE_DRIFT')
    C.need(C.sha(WORKER) == WORKER_SHA and C.sha(METADATA) == METADATA_SHA and C.sha(PARITY_INDEX) == PARITY_SHA, 'FROZEN_TRAIN_WORKER_METADATA')
    C.need(C.sha(PARENT_PROGRAM) == PARENT_PROGRAM_SHA, 'QUALIFIED_PARENT_NUMERICAL_SOURCE')
    wm = C.read(WORKER)
    C.need(wm['status'] == 'TRAIN128_WORKER_MANIFEST_FROZEN_METADATA_ONLY' and wm['query_count'] == 128, 'WORKER_TRAIN_INPUT_SCOPE')
    C.need([r['execution_ordinal'] for r in wm['records']] == list(range(128)) and all(r['query_id'].startswith('T128-') for r in wm['records']), 'OPAQUE_TRAIN128_AXIS')
    C.need(all(not ({'identity', 'group', 'target', 'target_position', 'target_identity', 'original_query_id'} & set(r)) for r in wm['records']), 'NO_WORKER_TARGET_FIELDS')
    return profile


def legacy_core(profile):
    """Compile only the three pinned pure functions; never import old main."""
    path = Path(profile['sources']['core']['path'])
    parsed = ast.parse(path.read_text())
    names = ('oriented', 'cell_means', 'score')
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    C.need({node.name for node in nodes} == set(names) and len(nodes) == 3, 'LEGACY_FUNCTIONS')
    namespace = {'torch': torch, 'F': F, 'Image': Image, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def replay_c4(qtokens, rtokens, wq, wr):
    """Independent literal FP64 implementation, including roll-mass reductions."""
    q = qtokens.to(torch.float64)
    r = rtokens.to(torch.float64)
    q = q / q.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    r = r / r.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    similarities = q @ r.T
    def one(query_weights, reference_weights):
        local = torch.max(similarities * reference_weights.unsqueeze(0), dim=1).values
        mass = torch.sqrt(query_weights.mean() * reference_weights.mean())
        return float(mass * (query_weights * local).sum() / query_weights.sum().clamp_min(1e-12)), float(mass)
    real, mass = one(wq, wr)
    query, _ = one(torch.roll(wq, shifts=max(1, len(wq) // 2), dims=0), wr)
    reference, _ = one(wq, torch.roll(wr, shifts=max(1, len(wr) // 2), dims=0))
    return dict(zip(SCORE_KEYS, (real, mass, query, reference), strict=True))


def stage_name(kind, shard):
    C.need(kind == 'shard' and shard in range(16), 'ONLY_NATURAL_TRAIN_SHARDS_0_TO15_NO_BRIDGE_EXECUTION')
    return f'shard{shard:02d}'


def check_binding(value, path, label):
    C.need(C.bound_path(value) == Path(path).resolve() and value['sha256'] == C.sha(path), label)


def source_token(kind, shard, *, gates=True):
    folder = TOKEN / stage_name(kind, shard)
    receipt, validation = C.read(folder / 'receipt.json'), C.read(folder / 'validation.json')
    C.need(validation['status'] == 'RC_ORIGINAL7_TRAIN128_TOKEN_RAW_CPU_REPLAY_PASS', 'TRAIN_TOKEN_CPU_QUALIFIED')
    for key, name in (('payload', 'payload.pt'), ('receipt', 'receipt.json')):
        check_binding(validation[key], folder / name, 'TOKEN_VALIDATION_' + key)
    check_binding(validation['authority'], C.AUTHORITY, 'TOKEN_CURRENT_TRAIN_AUTHORITY')
    check_binding(validation['worker_manifest'], WORKER, 'TOKEN_FROZEN_WORKER')
    check_binding(receipt['authority'], C.AUTHORITY, 'TOKEN_RECEIPT_AUTHORITY')
    check_binding(receipt['payload'], folder / 'payload.pt', 'TOKEN_RECEIPT_PAYLOAD')
    C.need(receipt['status'] == 'RC_ORIGINAL7_TRAIN128_TOKEN_RAW_SHARD_READY', 'TOKEN_RECEIPT_READY')
    payload = torch.load(folder / 'payload.pt', map_location='cpu', weights_only=True, mmap=True)
    C.need(payload['status'] == receipt['status'] and payload['shard'] == shard and payload['shard_count'] == 16, 'TOKEN_SHARD_SCHEMA')
    check_binding(payload['bindings']['authority'], C.AUTHORITY, 'TOKEN_PAYLOAD_AUTHORITY')
    check_binding(payload['bindings']['worker_manifest'], WORKER, 'TOKEN_PAYLOAD_WORKER')
    check_binding(payload['bindings']['legacy_full32_parity_index'], PARITY_INDEX, 'TOKEN_LEGACY_PARITY_INDEX')
    selected = C.read(WORKER)['records'][8 * shard:8 * (shard + 1)]
    records = payload['records']
    C.need(len(records) == len(selected) == 8, 'EIGHT_TRAIN_INPUTS_PER_SHARD')
    for row, worker in zip(records, selected, strict=True):
        C.need(row['query_id'] == worker['query_id'] and row['execution_ordinal'] == worker['execution_ordinal'] and row['track'] == worker['track'], 'OPAQUE_TRAIN_QUERY_AXIS')
        C.need(row['query_source_sha256'] == worker['source_image_sha256'] and row['query_source_path'] == worker['query_image_path'], 'TRAIN_IMAGE_SOURCE')
        C.need(not ({'identity', 'group', 'target', 'target_position', 'target_identity'} & set(row)), 'NO_TRAIN_TARGET_INPUT')
    count = 8 if shard < 4 else 0
    C.need(validation['legacy_full32_parity_count'] == count and validation['all_original_TRAIN_bits_exact'] is True, 'OLD_FULL32_TOKEN_RAW_PARITY')
    C.need(len(payload['legacy_full32_parity']) == count and payload['legacy_full32_parity'] == receipt['legacy_full32_parity'], 'OLD_FULL32_PARITY_ROWS')
    C.need(all(v['all_bits_exact'] for v in payload['legacy_full32_parity']), 'OLD_FULL32_ALL_TOKEN_RAW_BITS')
    if gates:
        reused = C.require_reused_engineering_bridges()
        C.need(set(reused) == {'token_raw', 'roma'}, 'TWO_QUALIFIED_BRIDGES_REUSED_WITH_ORIGINAL_PROVENANCE')
        aggregate = C.read(RAW_AGGREGATE)
        C.need(aggregate['status'] == 'ORIGINAL7_TRAIN128_RAW_AGGREGATE_PASS' and aggregate['query_count'] == 128, 'ALL128_TRAIN_RAW_AGGREGATE')
        check_binding(aggregate['authority'], C.AUTHORITY, 'TRAIN_RAW_AGGREGATE_AUTHORITY')
        check_binding(aggregate['worker_manifest'], WORKER, 'TRAIN_RAW_AGGREGATE_WORKER')
        C.need(len(aggregate['shards']) == 16 and [v['shard'] for v in aggregate['shards']] == list(range(16)), 'TRAIN_RAW_ALL16_SHARDS')
        for key, name in (('payload', 'payload.pt'), ('receipt', 'receipt.json'), ('validation', 'validation.json')):
            check_binding(aggregate['shards'][shard][key], folder / name, 'TRAIN_RAW_AGGREGATE_SOURCE:' + key)
    for row in records:
        axis = list(row['candidate_physical_rows'])
        C.need(axis == sorted(set(axis)) and len(axis) == 128, 'NATURAL_C128_COMPLETE_NO_INSERTION')
        C.need(row['query_tokens'].dtype == torch.float16 and row['query_tokens'].shape == (math.prod(row['query_grid_shape']), 128) and token_sha(row['query_tokens']) == row['query_tokens_sha256'], 'QUERY_TOKEN_GRID_HASH')
        for physical in axis:
            ref = payload['references'][int(physical)]
            C.need(ref['physical_row'] == physical and ref['tokens'].dtype == torch.float16 and ref['tokens'].shape == (math.prod(ref['grid_shape']), 128) and token_sha(ref['tokens']) == ref['tokens_sha256'], 'REFERENCE_TOKEN_GRID_HASH')
    return payload, {'payload': C.bind(folder / 'payload.pt'), 'receipt': C.bind(folder / 'receipt.json'), 'validation': C.bind(folder / 'validation.json')}


def geometry(path, source_sha, source_key, grid, frame, processor_sha):
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2
    with Image.open(path) as im:
        raw_hw = (int(im.height), int(im.width))
        orientation = int(im.getexif().get(274, 1))
    result = build_colnomic_canonical_geometry_v2(
        source_image_sha256=source_sha, source_key=source_key,
        processor_config_sha256=processor_sha, raw_size_hw=raw_hw,
        exif_orientation=orientation, merged_grid_shape=tuple(grid), processor_input_frame=frame)
    return result, {'sha256': result.sha256, 'raw_size_hw': list(raw_hw),
                    'oriented_size_hw': list(result.oriented_size_hw), 'exif_orientation': orientation,
                    'processor_input_frame': frame, 'grid_shape': list(grid),
                    'processor_config_sha256': processor_sha}


def check_runtime(profile):
    """Compare module versions to module-version pins; distributions are diagnostics.

    CPU-callable: no CUDA discovery, RoMa import, or model construction.
    """
    C.need(Path(sys.prefix).resolve() == Path(profile['runtime_prefix']).resolve(), 'ROMA_RUNTIME_PREFIX')
    C.need(sys.version.split()[0] == profile['runtime_versions']['python'], 'PYTHON_RUNTIME_DRIFT')
    modules, distributions = {}, {}
    for dist, key in [('torch', 'torch'), ('torchvision', 'torchvision'), ('Pillow', 'PIL'), ('numpy', 'numpy')]:
        module = __import__(key)
        modules[key] = str(module.__version__)
        distributions[key] = importlib.metadata.version(dist)
        C.need(modules[key] == profile['runtime_versions'][key], 'RUNTIME_MODULE_DRIFT:' + key)
    C.need(Path(os.environ.get('TORCH_HOME', '')).resolve() == Path(profile['torch_home']), 'TORCH_HOME_SOURCE')
    return {'runtime_prefix': str(Path(sys.prefix).resolve()), 'python': sys.version.split()[0],
            'module_versions': modules, 'distribution_versions': distributions,
            'TORCH_HOME': str(Path(os.environ['TORCH_HOME']).resolve()),
            'version_comparison': 'actual module __version__ equals original module-version profile exactly'}


def gpu_model(profile):
    check_runtime(profile)
    C.need(torch.cuda.is_available(), 'GPU_REQUIRED_FOR_NATURAL_ROMA')
    def deny_network(event, args):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'):
            raise RuntimeError('ROMA_NETWORK_ACCESS_DISABLED')
    sys.addaudithook(deny_network)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(17)
    import romav2
    C.need(Path(romav2.__file__).resolve().parent == Path(profile['source_trees']['roma']['root']), 'ROMA_IMPORT_SOURCE')
    model = romav2.RoMaV2()
    C.need(not model.training, 'ROMA_EVAL_MODE')
    return model


def save_tensor_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    C.need(not path.exists(), 'IMMUTABLE_PAYLOAD_EXISTS')
    temporary = path.with_name('.payload.partial.' + str(os.getpid()))
    try:
        torch.save(payload, temporary)
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def legacy_full32_parity(token, records):
    """Numerical anchor check; changing anonymous source_key must not fail geometry metadata SHA."""
    entries = {r['query_id']: r for r in token['legacy_full32_parity']}
    output = []; cache = {}
    for current, source in zip(records, token['records'], strict=True):
        if current['query_id'] not in entries:
            continue
        entry = entries[current['query_id']]
        C.need(entry['all_bits_exact'], 'ORIGINAL_TOKEN_INTERFACE_REQUIRED')
        binding = entry['source']['maps']; path = C.bound_path(binding)
        check_binding(binding, path, 'OLD_FULL32_MAP_SOURCE')
        if str(path) not in cache:
            cache[str(path)] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        candidates = [r for r in cache[str(path)]['records'] if r['execution_ordinal'] == entry['original_execution_ordinal']]
        C.need(len(candidates) == 1 and candidates[0]['role'] == 'TRAIN', 'ONLY_ORIGINAL_FULL_TRAIN_RECORD')
        old = candidates[0]
        C.need(current['candidate_physical_rows'] == old['candidate_physical_rows'] and current['query_tokens_sha256'] == old['query_tokens_sha256'], 'ORIGINAL_TRAIN_SOURCE_AXIS')
        C.need(len(current['candidates']) == len(old['candidates']) == 128, 'ALL128_ORIGINAL_C4_PAIRS')
        rows = []
        for new, previous in zip(current['candidates'], old['candidates'], strict=True):
            C.need(new['physical_row'] == previous['physical_row'] and new['reference_tokens_sha256'] == previous['reference_tokens_sha256'], 'ORIGINAL_REFERENCE_INTERFACE')
            rows.append({'physical_row': new['physical_row'],
                         'query_map_bits_exact': bit_equal(new['query_visibility'], previous['query_visibility']),
                         'reference_map_bits_exact': bit_equal(new['reference_visibility'], previous['reference_visibility']),
                         'C4_bits_exact': all(float(new['old_scores'][k]).hex() == float(previous['old_scores'][k]).hex() for k in SCORE_KEYS),
                         'RAW_bits_exact': float(new['old_scores']['raw_score']).hex() == float(previous['old_scores']['raw_score']).hex(),
                         'query_map_max_abs_difference': float((new['query_visibility'] - previous['query_visibility']).abs().max()),
                         'reference_map_max_abs_difference': float((new['reference_visibility'] - previous['reference_visibility']).abs().max())})
        all_maps = all(r['query_map_bits_exact'] and r['reference_map_bits_exact'] for r in rows)
        all_c4 = all(r['C4_bits_exact'] for r in rows); all_raw = all(r['RAW_bits_exact'] for r in rows)
        output.append({'query_id': current['query_id'], 'execution_ordinal': current['execution_ordinal'],
                       'original_execution_ordinal': entry['original_execution_ordinal'], 'original_map_source': C.bind(path),
                       'candidate_count': 128, 'all_maps_bit_exact': all_maps, 'all_C4_bit_exact': all_c4,
                       'all_RAW_bit_exact': all_raw, 'all_bits_exact': all_maps and all_c4 and all_raw,
                       'anonymous_geometry_source_key_SHA_not_compared': True, 'rows': rows})
    C.need(len(output) == len(entries), 'COMPLETE_ORIGINAL_FULL32_PARITY_CHECK')
    return output


def produce(kind, shard):
    C.need(kind == 'shard', 'NO_NEW_ENGINEERING_BRIDGE_RUN')
    C.require_authority('roma_shard', __file__)
    profile = source_profile()
    token, token_bindings = source_token(kind, shard)
    folder = OUT / stage_name(kind, shard)
    C.need(not (folder / 'payload.pt').exists() and not (folder / 'receipt.json').exists(), 'IMMUTABLE_ROMA_STAGE_EXISTS')
    core = legacy_core(profile)
    model = gpu_model(profile)
    records = []
    reference_geometry = {}
    processor_sha = profile['sources']['processor']['sha256']
    for source in token['records']:
        qpath = Path(source['query_source_path'])  # Keep anonymous symlink spelling.
        C.need(C.sha(qpath) == source['query_source_sha256'], 'QUERY_IMAGE_BYTES')
        frame = source['processor_input_frame']
        expected_frame = 'EXIF_ORIENTED_BEFORE_RESIZE' if source['track'] == 'new_difficult_train' else 'DECODED_RAW_BEFORE_EXIF'
        C.need(frame == expected_frame, 'EXPLICIT_QUERY_PROCESSOR_FRAME')
        qgeom, qmeta = geometry(qpath, source['query_source_sha256'],
                                f"bridge-query:{source['execution_ordinal']}", source['query_grid_shape'], frame, processor_sha)
        qimage = core.oriented(qpath)
        candidates = []
        for position, physical in enumerate(source['candidate_physical_rows']):
            reference = token['references'][int(physical)]
            rpath = Path(reference['source_path'])
            if physical not in reference_geometry:
                image_sha = C.sha(rpath)
                C.need(image_sha == reference['source_image_sha256'], 'REFERENCE_IMAGE_BYTES')
                rg, rm = geometry(rpath, image_sha, f'gallery-row:{physical}', reference['grid_shape'],
                                  'DECODED_RAW_BEFORE_EXIF', processor_sha)
                reference_geometry[physical] = (rg, rm, image_sha)
            rgeom, rmeta, image_sha = reference_geometry[physical]
            prediction = model.match(qimage, core.oriented(rpath))
            wq = core.cell_means(prediction['overlap_AB'][0, ..., 0].detach().cpu(), qgeom)
            wr = core.cell_means(prediction['overlap_BA'][0, ..., 0].detach().cpu(), rgeom)
            C.need(wq.dtype == wr.dtype == torch.float64 and bool(torch.isfinite(wq).all()) and
                   bool(torch.isfinite(wr).all()) and bool(((wq >= 0) & (wq <= 1)).all()) and
                   bool(((wr >= 0) & (wr <= 1)).all()), 'VISIBILITY_DOMAIN')
            real, mass, _ = core.score(source['query_tokens'], reference['tokens'], wq, wr)
            qc = core.score(source['query_tokens'], reference['tokens'], wq.roll(max(1, len(wq) // 2)), wr)[0]
            rc = core.score(source['query_tokens'], reference['tokens'], wq, wr.roll(max(1, len(wr) // 2)))[0]
            scores = dict(zip(SCORE_KEYS, (float(real), float(mass), float(qc), float(rc)), strict=True))
            scores.update({'candidate_position': position, 'physical_row': int(physical),
                           'raw_score': float(source['candidate_raw_scores'][position]),
                           'query_map_sha256': hten(wq), 'reference_map_sha256': hten(wr)})
            candidates.append({'candidate_position': position, 'physical_row': int(physical),
                               'reference_tokens_sha256': reference['tokens_sha256'],
                               'reference_image_sha256': image_sha, 'reference_grid_shape': list(reference['grid_shape']),
                               'reference_geometry': rmeta, 'query_visibility': wq, 'reference_visibility': wr,
                               'query_map_sha256': hten(wq), 'reference_map_sha256': hten(wr),
                               'control_shifts': {'query': max(1, len(wq) // 2), 'reference': max(1, len(wr) // 2)},
                               'old_scores': scores})
            del prediction
        records.append({'query_id': source['query_id'], 'execution_ordinal': source['execution_ordinal'],
                        'track': source['track'], 'query_source_sha256': source['query_source_sha256'],
                        'query_grid_shape': list(source['query_grid_shape']), 'processor_input_frame': frame,
                        'query_tokens_sha256': source['query_tokens_sha256'], 'query_geometry': qmeta,
                        'candidate_physical_rows': list(source['candidate_physical_rows']),
                        'candidate_raw_scores': source['candidate_raw_scores'].clone(), 'candidates': candidates})
        print(json.dumps({'event': 'ROMA_QUERY_COMPLETE', 'execution_ordinal': source['execution_ordinal'],
                          'candidate_count': 128, 'query_id': source['query_id']}), flush=True)
    reused = C.require_reused_engineering_bridges()
    parity = legacy_full32_parity(token, records)
    ready = 'RC_ORIGINAL7_TRAIN128_ROMA_SHARD_READY' if all(r['all_bits_exact'] for r in parity) else 'RC_ORIGINAL7_TRAIN128_ROMA_ENGINEERING_DRIFT'
    payload = {'status': ready, 'kind': kind, 'shard': shard, 'records': records,
               'token_source': token_bindings, 'authority': C.bind(C.AUTHORITY), 'source_profile': C.bind(PROFILE),
               'program': C.bind(__file__), 'RAW_aggregate': C.bind(RAW_AGGREGATE), 'reused_engineering_bridges': reused, 'legacy_full32_parity': parity,
               'target_role_read_count': 0, 'target_insertion_count': 0, 'model_update_count': 0}
    save_tensor_payload(folder / 'payload.pt', payload)
    C.write_json(folder / 'receipt.json', {'status': ready, 'query_count': len(records),
                 'candidate_occurrence_count': len(records) * 128, 'payload': C.bind(folder / 'payload.pt'),
                 'authority': C.bind(C.AUTHORITY), 'source_profile': C.bind(PROFILE), 'program': C.bind(__file__),
                 'token_source': token_bindings, 'reused_engineering_bridges': reused, 'legacy_full32_parity': parity,
                 'runtime_guard': check_runtime(profile),
                 'runtime': {'torch': str(torch.__version__), 'cpu_threads': torch.get_num_threads(),
                             'GPU_name': torch.cuda.get_device_name(0), 'seed': 17, 'matmul_precision': 'highest'},
                 'target_role_read_count': 0, 'model_update_count': 0})
    C.need(ready == 'RC_ORIGINAL7_TRAIN128_ROMA_SHARD_READY', 'OLD_FULL32_ROMA_MAP_C4_INTERFACE_DRIFT_NO_TRAIN_INPUT_QUALIFICATION')
    nonce = uuid.uuid4().hex
    command = [sys.executable, str(Path(__file__).resolve()), '--phase', 'validate', '--kind', kind, '--nonce', nonce]
    if shard is not None:
        command += ['--shard', str(shard)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', RC_ROMA_VALIDATOR_NONCE=nonce)
    subprocess.run(command, env=environment, check=True)
    validation = C.read(folder / 'validation.json')
    C.need(validation['fresh_child_nonce'] == nonce and validation['status'] == SHARD_PASS, 'FRESH_CPU_REPLAY_REQUIRED')


def validate(kind, shard, nonce):
    C.need(nonce and os.environ.get('RC_ROMA_VALIDATOR_NONCE') == nonce, 'FRESH_VALIDATOR_NONCE')
    C.need(kind == 'shard', 'NO_NEW_ENGINEERING_BRIDGE_RUN')
    C.require_authority('roma_shard', __file__)
    profile = source_profile()
    token, token_bindings = source_token(kind, shard)
    folder = OUT / stage_name(kind, shard)
    receipt = C.read(folder / 'receipt.json')
    check_binding(receipt['payload'], folder / 'payload.pt', 'ROMA_PAYLOAD_BINDING')
    check_binding(receipt['authority'], C.AUTHORITY, 'ROMA_AUTHORITY_BINDING')
    check_binding(receipt['program'], __file__, 'ROMA_PROGRAM_BINDING')
    C.need(receipt['token_source'] == token_bindings, 'ROMA_TOKEN_SOURCE')
    payload = torch.load(folder / 'payload.pt', map_location='cpu', mmap=True, weights_only=True)
    C.need(payload['kind'] == kind and payload['shard'] == shard and payload['token_source'] == token_bindings, 'ROMA_PAYLOAD_SCHEMA')
    C.need(payload['status'] == receipt['status'] and payload['authority'] == C.bind(C.AUTHORITY) and
           payload['program'] == C.bind(__file__) and payload['source_profile'] == C.bind(PROFILE) and
           receipt['source_profile'] == C.bind(PROFILE), 'ROMA_PAYLOAD_SOURCE_CLOSURE')
    C.need(len(payload['records']) == len(token['records']), 'QUERY_COVERAGE')
    scalar_count = 0
    for row, source in zip(payload['records'], token['records'], strict=True):
        C.need(row['query_id'] == source['query_id'] and row['execution_ordinal'] == source['execution_ordinal'] and
               row['query_tokens_sha256'] == source['query_tokens_sha256'] and
               row['query_source_sha256'] == source['query_source_sha256'] and
               row['processor_input_frame'] == source['processor_input_frame'] and
               row['candidate_physical_rows'] == list(source['candidate_physical_rows']) and
               bit_equal(row['candidate_raw_scores'], source['candidate_raw_scores']), 'ROMA_QUERY_RAW_BINDING')
        C.need(len(row['candidates']) == 128, 'ALL128_CANDIDATES_REQUIRED')
        for position, candidate in enumerate(row['candidates']):
            physical = source['candidate_physical_rows'][position]
            reference = token['references'][int(physical)]
            C.need(candidate['candidate_position'] == position and candidate['physical_row'] == physical and
                   candidate['reference_tokens_sha256'] == reference['tokens_sha256'] and
                   candidate['reference_image_sha256'] == reference['source_image_sha256'], 'REFERENCE_AXIS_BINDING')
            wq, wr = candidate['query_visibility'], candidate['reference_visibility']
            for weights, tokens in ((wq, source['query_tokens']), (wr, reference['tokens'])):
                C.need(weights.dtype == torch.float64 and weights.shape == (tokens.shape[0],) and
                       bool(torch.isfinite(weights).all()) and bool(((weights >= 0) & (weights <= 1)).all()), 'CPU_MAP_DOMAIN')
            C.need(hten(wq) == candidate['query_map_sha256'] == candidate['old_scores']['query_map_sha256'] and
                   hten(wr) == candidate['reference_map_sha256'] == candidate['old_scores']['reference_map_sha256'], 'CPU_MAP_HASH')
            C.need(candidate['old_scores']['candidate_position'] == position and
                   candidate['old_scores']['physical_row'] == physical and
                   candidate['control_shifts'] == {'query': max(1, len(wq) // 2), 'reference': max(1, len(wr) // 2)}, 'CPU_CONTROL_AXIS_METADATA')
            expected = replay_c4(source['query_tokens'], reference['tokens'], wq, wr)
            for key in SCORE_KEYS:
                C.need(math.isfinite(expected[key]) and expected[key].hex() == float(candidate['old_scores'][key]).hex(), 'INDEPENDENT_C4_BIT_REPLAY:' + key)
                scalar_count += 1
            C.need(float(candidate['old_scores']['raw_score']).hex() == float(source['candidate_raw_scores'][position]).hex(), 'RAW_SCORE_REPLAY')
    reused = C.require_reused_engineering_bridges()
    C.need(payload['reused_engineering_bridges'] == receipt['reused_engineering_bridges'] == reused, 'REUSED_BRIDGE_PROVENANCE_UNCHANGED')
    check_binding(payload['RAW_aggregate'], RAW_AGGREGATE, 'TRAIN_RAW_AGGREGATE_PAYLOAD_BINDING')
    parity = legacy_full32_parity(token, payload['records'])
    C.need(parity == payload['legacy_full32_parity'] == receipt['legacy_full32_parity'], 'ORIGINAL_FULL32_NUMERICAL_PARITY_REPLAY')
    passed = all(r['all_bits_exact'] for r in parity)
    status = SHARD_PASS if passed else 'RC_ORIGINAL7_TRAIN128_ROMA_ENGINEERING_DRIFT'
    C.write_json(folder / 'validation.json', {'status': status, 'payload': C.bind(folder / 'payload.pt'),
                 'receipt': C.bind(folder / 'receipt.json'), 'authority': C.bind(C.AUTHORITY),
                 'program': C.bind(__file__), 'source_profile': C.bind(PROFILE), 'token_source': token_bindings,
                 'query_count': len(payload['records']), 'candidate_occurrence_count': scalar_count // 4,
                 'independently_recomputed_C4_scalars': scalar_count, 'all_C4_bits_exact': True,
                 'legacy_full32_parity_count': len(parity), 'all_original_TRAIN_maps_C4_bits_exact': passed,
                 'reused_engineering_bridges': reused, 'new_engineering_bridge_runs': 0,
                 'fresh_child_nonce': nonce, 'target_role_read_count': 0, 'model_update_count': 0})
    C.need(passed, 'ENGINEERING_DRIFT_NO_EXPANDED_ROMA_AUTHORIZED')
    print(json.dumps({'status': status, 'candidate_count': scalar_count // 4}), flush=True)


def numerical_source_parity():
    names = ('hten', 'token_sha', 'bit_equal', 'legacy_core', 'replay_c4', 'geometry', 'check_runtime', 'gpu_model', 'save_tensor_payload')
    old = {n.name: n for n in ast.parse(PARENT_PROGRAM.read_text()).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    for name in names:
        C.need(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False), 'FROZEN_NUMERICAL_OR_RUNTIME_FUNCTION_AST:' + name)
    return {'parent_program': C.bind(PARENT_PROGRAM), 'function_names': list(names), 'all_function_AST_identical': True}


def preflight():
    profile = source_profile()
    runtime_guard = check_runtime(profile)
    core = legacy_core(profile)
    # No natural token/image read, model construction, GPU operation, or fit.
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(9, 7, generator=generator, dtype=torch.float16)
    r = torch.randn(11, 7, generator=generator, dtype=torch.float16)
    wq = torch.rand(9, generator=generator, dtype=torch.float64)
    wr = torch.rand(11, generator=generator, dtype=torch.float64)
    cases = ((wq, wr), (torch.zeros_like(wq), wr), (wq, torch.zeros_like(wr)))
    for a, b in cases:
        expected = replay_c4(q, r, a, b)
        score, mass, _ = core.score(q, r, a, b)
        values = (score, mass, core.score(q, r, a.roll(max(1, len(a) // 2)), b)[0],
                  core.score(q, r, a, b.roll(max(1, len(b) // 2)))[0])
        C.need(all(expected[key].hex() == float(value).hex() for key, value in zip(SCORE_KEYS, values, strict=True)), 'SYNTHETIC_LITERAL_C4')
    toy_geometry = types.SimpleNamespace(cell_boxes_xyxy=torch.tensor([[0., 0., .5, .5], [.5, .5, 1., 1.]], dtype=torch.float64),
                                         valid_patch_mask=torch.tensor([True, False]))
    means = core.cell_means(torch.arange(16, dtype=torch.float64).reshape(4, 4), toy_geometry)
    C.need(means.tolist() == [2.5, 0.], 'CELL_MEAN_INVALID_PADDING')
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2
    for frame in ('DECODED_RAW_BEFORE_EXIF', 'EXIF_ORIENTED_BEFORE_RESIZE'):
        synthetic_geometry = build_colnomic_canonical_geometry_v2(
            source_image_sha256='0' * 64, source_key='synthetic',
            processor_config_sha256=profile['sources']['processor']['sha256'],
            raw_size_hw=(224, 224), exif_orientation=6, merged_grid_shape=(8, 8),
            processor_input_frame=frame)
        C.need(synthetic_geometry.cell_boxes_xyxy.shape == (64, 4), 'SYNTHETIC_BOTH_GEOMETRY_FRAMES')
    value = {'status': 'RC_ORIGINAL7_TRAIN128_ROMA_SOURCE_SYNTHETIC_PREFLIGHT_PASS',
             'program': C.bind(__file__), 'source_profile': C.bind(PROFILE),
             'runtime_guard': runtime_guard,
             'public_source_bindings': {
                 'roma_program': C.bind(__file__), 'roma_source_profile': C.bind(PROFILE),
                 'roma_shards_launcher': C.bind(LAUNCHER),
                 'roma_plan': C.bind(PLAN), 'roma_common_program': C.bind(Path(C.__file__)),
                 'roma_worker_manifest': C.bind(WORKER), 'roma_metadata_validation': C.bind(METADATA),
                 'roma_legacy_full32_parity_index': C.bind(PARITY_INDEX), 'roma_parent_qualified_program': C.bind(PARENT_PROGRAM),
             },
             'synthetic_C4_cases': 3, 'synthetic_C4_scalars_bit_exact': 12,
             'shard_launcher_enforces_absolute_research_wall_deadline': True,
             'existing_bridges_reused_without_new_bridge_execution': True,
             'qualified_parent_function_AST_parity': numerical_source_parity(),
             'natural_image_read_count': 0, 'natural_token_load_count': 0,
             'model_constructor_count': 0, 'optimizer_updates': 0}
    C.write_json(OUT / 'preflight' / C.sha(__file__) / 'result.json', value)
    print(json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('preflight', 'produce', 'validate'))
    parser.add_argument('--kind', choices=('shard',), default='shard')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--nonce')
    args = parser.parse_args()
    deny_protected_reads(allow_expanded_images=args.phase == 'produce' and args.kind == 'shard')
    torch.set_num_threads(8)
    if args.phase == 'preflight':
        C.need(args.shard is None, 'SYNTHETIC_PREFLIGHT_HAS_NO_NATURAL_SHARD')
        preflight()
    elif args.phase == 'produce':
        produce(args.kind, args.shard)
    else:
        validate(args.kind, args.shard, args.nonce)


if __name__ == '__main__':
    main()
