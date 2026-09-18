#!/usr/bin/env python3
"""Frozen ORIGINAL7 evaluation: append-only V2 runtime compatibility repair.

GPU computation is admitted only by the shared execution authority.  CPU
preflight is synthetic; CPU validation independently recomputes saved C4 without
constructing RoMa.  The old TRAIN56 bridge must qualify before expanded shards.
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
import rc_original7_eval128_execution_common_v2_compat as C

PROFILE = ROOT / 'registry/rc_original7_eval128_roma_source_profile_v1_20260910.json'
TOKEN = ROOT / 'results/rc_original7_eval128_token_raw_v1'
OUT = ROOT / 'results/rc_original7_eval128_roma_v1'
RAW_AGGREGATE = ROOT / 'results/rc_original7_eval128_full_evidence_v1/raw_aggregate.json'
WORKER = ROOT / 'results/rc_original7_expanded_eval128_manifest_v1/worker_manifest.json'
WORKER_SHA = 'f2c2e6f9181e103a61a181edf43e3309aa1de72743cfa6cf6097a9deb108e4e7'
CORE_SHA = 'fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c'
CHECKPOINT_SHA = '1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7'
TRAIN_EXPORT_SHA = 'c70f75ba1023ab0366c3509e4597c5289dfacc274dfa58b535e9ac3b9fc7b8fc'
SCORE_KEYS = ('real_score', 'visibility_mass', 'query_control_score', 'reference_control_score')
BRIDGE_PASS = 'RC_ORIGINAL7_EVAL128_ROMA_BRIDGE_PASS'
SHARD_PASS = 'RC_ORIGINAL7_EVAL128_ROMA_CPU_REPLAY_PASS'


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
    C.need(C.sha(WORKER) == WORKER_SHA, 'FROZEN_WORKER_MANIFEST')
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
    C.need(kind in ('bridge', 'shard'), 'STAGE_KIND')
    if kind == 'bridge':
        C.need(shard is None, 'BRIDGE_HAS_NO_SHARD_INDEX')
        return 'bridge'
    C.need(shard in range(16), 'SHARD_RANGE')
    return f'shard{shard:02d}'


def check_binding(value, path, label):
    C.need(C.bound_path(value) == Path(path).resolve() and value['sha256'] == C.sha(path), label)


def source_token(kind, shard, *, gates=True):
    name = stage_name(kind, shard)
    folder = TOKEN / name
    receipt, validation = C.read(folder / 'receipt.json'), C.read(folder / 'validation.json')
    expected = 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_BRIDGE_PASS' if kind == 'bridge' else 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_CPU_REPLAY_PASS'
    C.need(validation['status'] == expected, 'TOKEN_QUALIFICATION_REQUIRED')
    check_binding(validation['payload'], folder / 'payload.pt', 'TOKEN_PAYLOAD_VALIDATION_BINDING')
    check_binding(validation['receipt'], folder / 'receipt.json', 'TOKEN_RECEIPT_VALIDATION_BINDING')
    if kind == 'bridge':
        reused = C.require_reused_token_bridge()
        for name, filename in (('payload', 'payload.pt'), ('receipt', 'receipt.json'), ('validation', 'validation.json')):
            check_binding(reused[name], folder / filename, 'EXPLICIT_REUSED_TOKEN_BRIDGE:' + name)
    else:
        check_binding(validation['authority'], C.AUTHORITY, 'TOKEN_AUTHORITY_BINDING')
    suffix = 'BRIDGE_READY' if kind == 'bridge' else 'SHARD_READY'
    C.need(receipt['status'] == 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_' + suffix, 'TOKEN_RECEIPT_STATUS')
    payload = torch.load(folder / 'payload.pt', map_location='cpu', weights_only=True, mmap=True)
    C.need(payload['status'] == 'RC_ORIGINAL7_EVAL128_TOKEN_RAW_' + suffix, 'TOKEN_PAYLOAD_STATUS')
    records = payload['records']
    C.need(len(records) == (1 if kind == 'bridge' else 8), 'TOKEN_QUERY_COUNT')
    if kind == 'bridge':
        C.need(records[0]['execution_ordinal'] == 56, 'ONLY_KNOWN_TRAIN56_ENGINEERING')
    else:
        selected = C.read(WORKER)['records'][8 * shard:8 * (shard + 1)]
        C.need([r['query_id'] for r in records] == [r['query_id'] for r in selected], 'OPAQUE_QUERY_AXIS')
        for row, worker in zip(records, selected, strict=True):
            C.need(row['execution_ordinal'] == worker['execution_ordinal'] and
                   row['query_source_sha256'] == worker['source_image_sha256'] and
                   row['query_source_path'] == worker['query_image_path'], 'WORKER_IMAGE_BINDING')
        if gates:
            bridge = C.read(OUT / 'bridge/validation.json')
            C.need(bridge['status'] == BRIDGE_PASS, 'ROMA_ENGINEERING_BRIDGE_REQUIRED')
            check_binding(bridge['authority'], C.AUTHORITY, 'ROMA_BRIDGE_AUTHORITY')
            check_binding(bridge['payload'], OUT / 'bridge/payload.pt', 'ROMA_BRIDGE_PAYLOAD')
            check_binding(bridge['receipt'], OUT / 'bridge/receipt.json', 'ROMA_BRIDGE_RECEIPT')
            aggregate = C.read(RAW_AGGREGATE)
            C.need(aggregate['status'] == 'ORIGINAL7_EVAL128_RAW_AGGREGATE_PASS' and
                   aggregate['query_count'] == 128, 'ALL128_RAW_AGGREGATE_REQUIRED')
            check_binding(aggregate['authority'], C.AUTHORITY, 'RAW_AGGREGATE_AUTHORITY')
            check_binding(aggregate['worker_manifest'], WORKER, 'RAW_AGGREGATE_COHORT')
            C.need(len(aggregate['shards']) == 16 and
                   [entry['shard'] for entry in aggregate['shards']] == list(range(16)), 'RAW_AGGREGATE_ALL16_SHARDS')
            aggregate_shard = aggregate['shards'][shard]
            for item, filename in (('payload', 'payload.pt'), ('receipt', 'receipt.json'), ('validation', 'validation.json')):
                check_binding(aggregate_shard[item], folder / filename, 'RAW_AGGREGATE_SHARD_SOURCE:' + item)
    for row in records:
        axis = list(row['candidate_physical_rows'])
        C.need(axis == sorted(set(axis)) and len(axis) == 128, 'SORTED_UNFILTERED_C128')
        C.need(row['query_tokens'].dtype == torch.float16 and
               row['query_tokens'].shape[0] == math.prod(row['query_grid_shape']), 'QUERY_TOKEN_LAYOUT')
        C.need(token_sha(row['query_tokens']) == row['query_tokens_sha256'], 'QUERY_TOKEN_SHA')
        for physical in axis:
            ref = payload['references'][int(physical)]
            C.need(ref['physical_row'] == physical and ref['tokens'].dtype == torch.float16 and
                   ref['tokens'].shape[0] == math.prod(ref['grid_shape']) and
                   token_sha(ref['tokens']) == ref['tokens_sha256'], 'REFERENCE_TOKEN_LAYOUT_SHA')
    return payload, {'payload': C.bind(folder / 'payload.pt'), 'receipt': C.bind(folder / 'receipt.json'),
                     'validation': C.bind(folder / 'validation.json')}


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


def old_bridge_check(profile, token, records):
    old = torch.load(profile['sources']['old_train56_export']['path'], map_location='cpu', weights_only=True, mmap=True)
    originals = [r for r in old['records'] if r['execution_ordinal'] == 56]
    C.need(len(originals) == 1 and originals[0]['role'] == 'TRAIN', 'KNOWN_TRAIN56_ONLY')
    original, fresh, source = originals[0], records[0], token['records'][0]
    C.need(source['query_id'] == original['query_id'] and
           source['query_tokens_sha256'] == original['query_tokens_sha256'] and
           fresh['candidate_physical_rows'] == original['candidate_physical_rows'], 'BRIDGE_QUERY_TOKENS_AND_RAW_AXIS')
    rows = []
    for current, previous in zip(fresh['candidates'], original['candidates'], strict=True):
        C.need(current['physical_row'] == previous['physical_row'] and
               current['reference_tokens_sha256'] == previous['reference_tokens_sha256'], 'BRIDGE_REFERENCE_TOKENS')
        map_exact = bit_equal(current['query_visibility'], previous['query_visibility']) and bit_equal(current['reference_visibility'], previous['reference_visibility'])
        scores_exact = all(float(current['old_scores'][key]).hex() == float(previous['old_scores'][key]).hex() for key in SCORE_KEYS)
        raw_exact = float(current['old_scores']['raw_score']).hex() == float(previous['old_scores']['raw_score']).hex()
        rows.append({'physical_row': current['physical_row'], 'maps_bit_exact': map_exact,
                     'C4_bit_exact': scores_exact, 'RAW_bit_exact': raw_exact,
                     'query_map_max_abs': float((current['query_visibility'] - previous['query_visibility']).abs().max()),
                     'reference_map_max_abs': float((current['reference_visibility'] - previous['reference_visibility']).abs().max())})
    return {'execution_ordinal': 56, 'candidate_count': 128,
            'all_maps_bit_exact': all(r['maps_bit_exact'] for r in rows),
            'all_C4_bit_exact': all(r['C4_bit_exact'] for r in rows),
            'all_RAW_bit_exact': all(r['RAW_bit_exact'] for r in rows), 'rows': rows,
            'original_export': profile['sources']['old_train56_export']}


def produce(kind, shard):
    C.require_authority('roma_bridge' if kind == 'bridge' else 'roma_shard', __file__)
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
    bridge = old_bridge_check(profile, token, records) if kind == 'bridge' else None
    ready = 'RC_ORIGINAL7_EVAL128_ROMA_' + ('BRIDGE_READY' if kind == 'bridge' else 'SHARD_READY')
    payload = {'status': ready, 'kind': kind, 'shard': shard, 'records': records,
               'token_source': token_bindings, 'authority': C.bind(C.AUTHORITY), 'source_profile': C.bind(PROFILE),
               'program': C.bind(__file__), 'RAW_aggregate': C.bind(RAW_AGGREGATE) if kind == 'shard' else None,
               'target_role_read_count': 0, 'target_insertion_count': 0, 'model_update_count': 0}
    save_tensor_payload(folder / 'payload.pt', payload)
    C.write_json(folder / 'receipt.json', {'status': ready, 'query_count': len(records),
                 'candidate_occurrence_count': len(records) * 128, 'payload': C.bind(folder / 'payload.pt'),
                 'authority': C.bind(C.AUTHORITY), 'source_profile': C.bind(PROFILE), 'program': C.bind(__file__),
                 'token_source': token_bindings, 'engineering_bridge': bridge,
                 'runtime_guard': check_runtime(profile),
                 'runtime': {'torch': str(torch.__version__), 'cpu_threads': torch.get_num_threads(),
                             'GPU_name': torch.cuda.get_device_name(0), 'seed': 17, 'matmul_precision': 'highest'},
                 'target_role_read_count': 0, 'model_update_count': 0})
    nonce = uuid.uuid4().hex
    command = [sys.executable, str(Path(__file__).resolve()), '--phase', 'validate', '--kind', kind, '--nonce', nonce]
    if shard is not None:
        command += ['--shard', str(shard)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', RC_ROMA_VALIDATOR_NONCE=nonce)
    subprocess.run(command, env=environment, check=True)
    validation = C.read(folder / 'validation.json')
    C.need(validation['fresh_child_nonce'] == nonce and validation['status'] == (BRIDGE_PASS if kind == 'bridge' else SHARD_PASS), 'FRESH_CPU_REPLAY_REQUIRED')


def validate(kind, shard, nonce):
    C.need(nonce and os.environ.get('RC_ROMA_VALIDATOR_NONCE') == nonce, 'FRESH_VALIDATOR_NONCE')
    C.require_authority('roma_bridge' if kind == 'bridge' else 'roma_shard', __file__)
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
    bridge = old_bridge_check(profile, token, payload['records']) if kind == 'bridge' else None
    if bridge is not None:
        C.need(bridge == receipt['engineering_bridge'], 'ENGINEERING_RECHECK')
    passed = bridge is None or all(bridge[k] for k in ('all_maps_bit_exact', 'all_C4_bit_exact', 'all_RAW_bit_exact'))
    status = (BRIDGE_PASS if kind == 'bridge' else SHARD_PASS) if passed else 'RC_ORIGINAL7_EVAL128_ROMA_ENGINEERING_DRIFT'
    C.write_json(folder / 'validation.json', {'status': status, 'payload': C.bind(folder / 'payload.pt'),
                 'receipt': C.bind(folder / 'receipt.json'), 'authority': C.bind(C.AUTHORITY),
                 'program': C.bind(__file__), 'source_profile': C.bind(PROFILE), 'token_source': token_bindings,
                 'query_count': len(payload['records']), 'candidate_occurrence_count': scalar_count // 4,
                 'independently_recomputed_C4_scalars': scalar_count, 'all_C4_bits_exact': True,
                 'engineering_original_maps_C4_exact': passed if kind == 'bridge' else None,
                 'fresh_child_nonce': nonce, 'target_role_read_count': 0, 'model_update_count': 0})
    C.need(passed, 'ENGINEERING_DRIFT_NO_EXPANDED_ROMA_AUTHORIZED')
    print(json.dumps({'status': status, 'candidate_count': scalar_count // 4}), flush=True)


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
    value = {'status': 'RC_ORIGINAL7_EVAL128_ROMA_SOURCE_SYNTHETIC_PREFLIGHT_PASS',
             'program': C.bind(__file__), 'source_profile': C.bind(PROFILE),
             'runtime_guard': runtime_guard,
             'public_source_bindings': {
                 'roma_program': C.bind(__file__), 'roma_source_profile': C.bind(PROFILE),
                 'roma_bridge_launcher': C.bind(ROOT / 'slurm/rc_original7_eval128_roma_bridge_v2_compat_accelerated_30m.sbatch'),
                 'roma_shards_launcher': C.bind(ROOT / 'slurm/rc_original7_eval128_roma_shards_v2_compat_accelerated_30m.sbatch'),
             },
             'synthetic_C4_cases': 3, 'synthetic_C4_scalars_bit_exact': 12,
             'both_launchers_enforce_absolute_research_wall_deadline': True,
             'natural_image_read_count': 0, 'natural_token_load_count': 0,
             'model_constructor_count': 0, 'optimizer_updates': 0}
    C.write_json(OUT / 'preflight' / C.sha(__file__) / 'result.json', value)
    print(json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('preflight', 'produce', 'validate'))
    parser.add_argument('--kind', choices=('bridge', 'shard'))
    parser.add_argument('--shard', type=int)
    parser.add_argument('--nonce')
    args = parser.parse_args()
    deny_protected_reads(allow_expanded_images=args.phase == 'produce' and args.kind == 'shard')
    torch.set_num_threads(8)
    if args.phase == 'preflight':
        preflight()
    elif args.phase == 'produce':
        produce(args.kind, args.shard)
    else:
        validate(args.kind, args.shard, args.nonce)


if __name__ == '__main__':
    main()
