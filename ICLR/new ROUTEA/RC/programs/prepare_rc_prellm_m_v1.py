#!/usr/bin/env python3
"""Pin a small identity-supervised real-M adapter pilot from original H593 C128.

The outer held rows contain no target labels.  Selection never reads prior
new-model actions, held roles, or private/external evaluation data. TRAIN-only
sampling is explicitly stratified by the original RAW winner's correctness:
eight correct and eight incorrect, with four original engineering rows first.
Within each stratum SHA order and distinct components take priority. This is
training-set design and does not select held evaluation examples or tune on
held outcomes. All 128 original candidates remain bound to every selected row.
This program only prepares and checks files; it submits no compute jobs.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
OUT = ROOT / 'results/rc_prellm_m_adapter_v1'
MANIFEST = OUT / 'input_manifest.json'
ENGINEERING_MANIFEST = OUT / 'engineering_manifest4.json'
ENGINEERING_VALIDATION = OUT / 'engineering_manifest_validation4.json'
ENGINEERING_PROGRAM = OUT / 'engineering_prepare4.py'
PARENT = ROOT / 'registry/rc_h593_quality_operator_eval_authority_v1_20260922.json'
CACHE = ROOT / 'results/rc_h593_feature_fusion_cache_v1'
HEAD = ROOT / 'results/rc_h593_quality_operator_eval_v1/fold0/payload.json'
HEAD_VALIDATION = HEAD.with_name('validation.json')
ORIGINAL_PREFLIGHT = ROOT / ('results/rc_original7_eval128_token_raw_v2_compat_preflight/'
    '1ce5aaffabf17d5816a31181517535f0305820bcfac553fea586c82999393ce7.json')
SELECTION_SALT = 'RC_PRELLM_M_ADAPTER_V1_20260924'
TRAIN_COUNT = 16
TRAIN_PER_STRATUM = 8
PROBE_COUNT = 8


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    # Preserve input aliases: dereferencing an image symlink changes provenance.
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def checked(binding):
    need(bind(binding['path']) == binding, 'SOURCE_SHA_DRIFT:' + binding['path'])
    return Path(binding['path'])


def write(path, value, replace_from=None):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() == text:
            return
        need(replace_from is not None and path.read_bytes() == Path(replace_from).read_bytes(),
             'IMMUTABLE_FILE:' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    meta = str(tensor.dtype).encode('ascii')
    meta += json.dumps(list(tensor.shape), separators=(',', ':')).encode('ascii')
    return hashlib.sha256(meta + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def key(query_id):
    return hashlib.sha256((SELECTION_SALT + '|' + query_id).encode()).hexdigest()


def guard():
    def audit(event, args):
        if event == 'socket.connect':
            need(not isinstance(args[1], tuple), 'OFFLINE_PREPARATION')
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).absolute()).lower()
        need(not any(word in path for word in (
            'curator_roles', '/target_join/', 'd1-mi', 'd1_mi', 'formal392',
            '/grozi/', '/isic/', 'h593_sources_by_query.csv')),
            'PROTECTED_OR_HELD_LABEL_READ:' + path)
    sys.addaudithook(audit)


def feature_vector(raw, winner, mass, content):
    mean = sum(raw) / len(raw)
    std = math.sqrt(sum((x - mean) ** 2 for x in raw) / len(raw))
    def sym(a, b):
        return (a - b) / (abs(a) + abs(b) + 1e-12)
    return [[(raw[i] - raw[winner]) / max(std, 1e-12),
             sym(mass[i] * content[i], mass[winner] * content[winner]),
             sym(mass[i], mass[winner]), sym(content[i], content[winner])]
            for i in range(128) if i != winner]


def build_manifest():
    parent = read(PARENT)
    split = read(checked(parent['public_sources']['split']))['folds'][0]
    roles_binding = parent['fold_sources']['0']['train_roles']
    roles = {x['query_id']: x for x in read(checked(roles_binding))['records']}
    need(set(roles) == set(split['train_query_ids']), 'EXACT_FOLD0_TRAIN_ROLES')
    held_ids = set(split['heldout_query_ids'])
    need(len(roles) == 474 and len(held_ids) == 119 and not held_ids & set(roles),
         'ORIGINAL_OUTER_FOLD0_SPLIT')
    gallery_binding = parent['public_sources']['gallery']
    gallery = {x['physical_row']: x for x in read(checked(gallery_binding))['records']}
    cv_path = CACHE / 'catalog_validation.json'
    cv = read(cv_path)
    need(cv['status'] == 'FUSION_ALL593_CATALOG_PASS', 'ORIGINAL_TOKEN_CATALOG_PASS')
    catalog = read(checked(cv['catalog']))
    query_meta = {x['query_id']: x for x in catalog['queries']}
    cache_authority = read(checked(catalog['authority']))
    workers_binding = cache_authority['workers']
    workers = {x['query_id']: x for x in read(checked(workers_binding))['records']}
    @lru_cache(maxsize=24)
    def payload(query_id):
        return torch.load(checked(query_meta[query_id]['payload']),
                          map_location='cpu', weights_only=True)

    engineering = read(ENGINEERING_MANIFEST)
    need(bind(ENGINEERING_PROGRAM)['sha256'] == engineering['provenance']['program']['sha256'],
         'ARCHIVED_ENGINEERING_SOURCE_BITS')
    need(read(ENGINEERING_VALIDATION)['manifest']['sha256'] == bind(ENGINEERING_MANIFEST)['sha256'],
         'ARCHIVED_ENGINEERING_MANIFEST_BITS')
    selected_train = [r['query_id'] for r in engineering['train_rows']]
    need(len(selected_train) == 4 and all(q in roles for q in selected_train), 'ORIGINAL_FOUR_TRAIN')
    components = {roles[q]['component'] for q in selected_train}
    identities = {roles[q]['identity'] for q in selected_train}
    skipped_absent = []
    strata = {True: [], False: []}
    for query_id in sorted(roles, key=key):
        role = roles[query_id]
        p = payload(query_id)
        target = [i for i, physical in enumerate(p['candidate_physical_rows'])
                  if gallery[physical]['identity'] == role['identity']]
        need(len(target) <= 1, 'ONE_CORRECTED_IDENTITY_REPRESENTATIVE')
        if not target:
            skipped_absent.append(query_id)
            continue
        strata[target[0] == p['winner']].append(query_id)
    need(all(q in strata[True] for q in selected_train), 'ORIGINAL_ENGINEERING_FOUR_RAW_CORRECT')
    duplicate_components = []
    for correct in (True, False):
        count = sum(q in strata[correct] for q in selected_train)
        # Distinct components are preferred globally, including the four
        # archived engineering examples, before allowing repeated components.
        for allow_repeat in (False, True):
            for query_id in strata[correct]:
                if count == TRAIN_PER_STRATUM:
                    break
                if query_id in selected_train:
                    continue
                role = roles[query_id]
                if not allow_repeat and role['component'] in components:
                    continue
                if role['component'] in components:
                    duplicate_components.append(query_id)
                selected_train.append(query_id)
                components.add(role['component'])
                identities.add(role['identity'])
                count += 1
            if count == TRAIN_PER_STRATUM:
                break
        need(count == TRAIN_PER_STRATUM, 'EIGHT_TRAIN_PER_RAW_STRATUM')
    need(len(selected_train) == TRAIN_COUNT, 'SIXTEEN_ELIGIBLE_TRAIN_QUERIES')
    selected_probe = sorted(held_ids, key=key)[:PROBE_COUNT]
    need(selected_probe == [r['query_id'] for r in engineering['probe_rows']], 'SAME_EIGHT_PROBE_IDS')
    token_sources = {}

    def tokens(image_key):
        if image_key in token_sources:
            return token_sources[image_key]
        entry = catalog['images'][image_key]
        item = entry['item']
        data = torch.load(checked(entry['input']), map_location='cpu', weights_only=True)
        t = data['tokens']
        need(data['item'] == item, 'TOKEN_IMAGE_ITEM')
        need(tensor_sha(t) == item['tokens_sha256'], 'ORIGINAL_TOKEN_BITS')
        need(t.dtype == torch.float16 and list(t.shape) == [math.prod(item['grid']), 128],
             'ORIGINAL_IMAGE_TOKEN_LAYOUT')
        need(bool(torch.isfinite(t).all()), 'FINITE_ORIGINAL_TOKENS')
        value = dict(image_key=image_key, image_path=item['path'],
            image_sha256=item['image_sha256'], grid_hw=item['grid'], frame=item['frame'],
            tokens_sha256=item['tokens_sha256'], token_file=entry['input'],
            token_field='tokens', token_shape=list(t.shape), token_dtype=str(t.dtype))
        token_sources[image_key] = value
        return value

    def row(query_id, role):
        meta, p = query_meta[query_id], payload(query_id)
        need(p['query_id'] == query_id and len(p['pairs']) == 128, 'FULL_QUERY_C128')
        axis = p['candidate_physical_rows']
        need(len(axis) == len(set(axis)) == 128 and axis == sorted(axis), 'PHYSICAL_C128_AXIS')
        need(len({gallery[x]['identity'] for x in axis}) == 128, 'C128_IDENTITY_DEDUP')
        raw = list(map(float, p['candidate_raw_scores']))
        winner = max(range(128), key=lambda i: (raw[i], -axis[i]))
        need(winner == p['winner'], 'RAW_WINNER_FIXED')
        query_tokens = tokens(p['query_image_key'])
        need(bind(query_tokens['image_path'])['sha256'] == p['source_image_sha256']
             == query_tokens['image_sha256'], 'QUERY_IMAGE_BYTES')
        op_validation_path = ROOT / f'results/rc_h593_quality_operator_v1/query{p["execution_ordinal"]:03d}/validation.json'
        opv = read(op_validation_path)
        need(opv['status'] == 'QUALITY_OPERATOR_QUERY_PASS', 'QUALITY_OPERATOR_VALIDATED')
        op = read(checked(opv['payload']))
        need(op['query_id'] == query_id and op['candidate_physical_rows'] == axis
             and op['winner'] == winner, 'OPERATOR_C128_BINDING')
        mass = [float(pair['native_c4'][1]) for pair in p['pairs']]
        content = [float(s['real_score']) for s in op['modes']['M0Q0R0']['scores']]
        expected_mass = [float(s['visibility_mass']) for s in op['modes']['M1Q0R0']['scores']]
        need(mass == expected_mass, 'EXACT_TRUE_ROMA_M')
        need(all(0 <= v <= 1 and math.isfinite(v) for v in mass), 'M_PROBABILITY_RANGE')
        expected = feature_vector(raw, winner, mass, content)
        old_x = op['modes']['M1Q0R0']['X']
        err = max(abs(a-b) for new, old in zip(expected, old_x) for a,b in zip(new, old[:4]))
        need(err < 2e-10 and max(abs(v) for x in old_x for v in x[4:]) < 2e-10,
             'FROZEN_FIVE_PARAMETER_HEAD_FEATURES')
        references = []
        for i, pair in enumerate(p['pairs']):
            physical = axis[i]
            need(pair['position'] == i and pair['physical_row'] == physical, 'PAIR_AXIS')
            ref = tokens(pair['image_key'])
            # The original catalog deduplicates byte-identical images with
            # different gallery filenames. Preserve the physical-row path and
            # validate bytes for such aliases instead of rejecting deduplication.
            if Path(ref['image_path']) != Path(gallery[physical]['image_path']):
                need(bind(gallery[physical]['image_path'])['sha256'] == ref['image_sha256'],
                     'GALLERY_IMAGE_ALIAS_BYTES')
            references.append(dict(position=i, physical_row=physical,
                identity=gallery[physical]['identity'], gallery_image_path=gallery[physical]['image_path'], **ref))
        out = dict(query_id=query_id, execution_ordinal=meta['execution_ordinal'], fold=0,
            split='train' if role is not None else 'outer_held_probe',
            image_path=query_tokens['image_path'], source_image_sha256=p['source_image_sha256'],
            frame=query_tokens['frame'], query_tokens=query_tokens, candidate_ids=axis,
            candidate_identities=[gallery[x]['identity'] for x in axis],
            raw_ranked_candidate_ids=p['raw_ranked_physical_rows'],
            raw_scores=raw, M=mass, L0=content, winner_index=winner,
            challenger_positions=p['challenger_positions'], references=references,
            reference_tokens=[r['token_file'] for r in references],
            sources=dict(query_payload=meta['payload'], operator_validation=bind(op_validation_path),
                operator_payload=opv['payload'], original_worker=workers[query_id]),
            feature_parity_max_abs_error=err)
        if role is not None:
            target = [i for i, physical in enumerate(axis) if gallery[physical]['identity'] == role['identity']]
            need(len(target) == 1, 'TRAIN_TARGET_PRESENT')
            out.update(target_position=target[0], target_id=role['identity'],
                target_positions=target, target_physical_row=axis[target[0]],
                component=role['component'], group=role['group'], raw_correct=target[0] == winner)
        return out

    train_rows = [row(q, roles[q]) for q in selected_train]
    probe_rows = [row(q, None) for q in selected_probe]
    for old, new in zip(engineering['probe_rows'], probe_rows):
        need(all(new[k] == v for k, v in old.items()), 'ORIGINAL_PROBE_ROW_CONTENT_PRESERVED')
    for old, new in zip(engineering['train_rows'], train_rows[:4]):
        need(all(new[k] == v for k, v in old.items()), 'ORIGINAL_ENGINEERING_ROW_CONTENT_PRESERVED')
    need(sum(r['raw_correct'] for r in train_rows) == TRAIN_PER_STRATUM, 'EIGHT_RAW_CORRECT_EIGHT_WRONG')
    need(not {r['source_image_sha256'] for r in train_rows} &
         {r['source_image_sha256'] for r in probe_rows}, 'TRAIN_PROBE_IMAGE_DISJOINT')
    hv = read(HEAD_VALIDATION)
    need(hv['status'] == 'QUALITY_OPERATOR_EVAL_FRESH_NUMPY_PASS' and hv['payload'] == bind(HEAD),
         'FROZEN_HEAD_VALIDATED')
    head = read(HEAD)
    need(head['fold'] == 0 and set(head['train_query_ids']) == set(roles)
         and head['heldout_label_reads'] == 0, 'HEAD_FOLD0_TRAIN_ONLY')
    hex7 = head['parameters']['COST1_REFIT_M1Q0R0']['theta_hex']
    need(float.fromhex(hex7[4]) == float.fromhex(hex7[5]) == 0., 'NO_LOCAL_HEAD_TERMS')
    hex5 = [hex7[i] for i in (0, 1, 2, 3, 6)]
    preflight = read(ORIGINAL_PREFLIGHT)
    model_sources = {k:v for k,v in preflight['public_source_bindings'].items()
                     if k.startswith(('token_raw_base_', 'token_raw_model_', 'token_raw_library_'))
                     or k in ('token_raw_adapter_weights', 'token_raw_processor_class_source',
                              'token_raw_encoder_class_source')}
    for name, binding in model_sources.items():
        if 'weights' not in name:
            checked(binding)
        else:
            need(Path(binding['path']).is_file(), 'MODEL_WEIGHT_FILE_EXISTS')
    processor = read(checked(model_sources['token_raw_model_preprocessor_config_json']))
    log_mass = [math.log(max(v, 1e-8)) for r in train_rows for v in r['M']]
    log_mean = sum(log_mass) / len(log_mass)
    log_std = math.sqrt(sum((v - log_mean) ** 2 for v in log_mass) / len(log_mass))
    need(log_std > 0 and math.isfinite(log_std), 'TRAIN_ONLY_M_NORMALIZATION')
    return dict(schema_version=2, status='PRELLM_M_PILOT_INPUTS_FROZEN',
        evidence_scope='Original four-query engineering subset plus stratified sixteen-query TRAIN pilot; eight opened outer-held probes; not full-fold or external evidence',
        fold=0, candidates_per_query=128, train_rows=train_rows, probe_rows=probe_rows,
        mass_normalization=dict(log_mean=log_mean, log_std=log_std, epsilon=1e-8,
            definition='log(max(M,epsilon)); population mean/std over sixteen TRAIN x original C128 only',
            fit_queries=TRAIN_COUNT, fit_pairs=len(log_mass), held_used=False),
        engineering_subset=dict(manifest=bind(ENGINEERING_MANIFEST),
            validation=bind(ENGINEERING_VALIDATION), program=bind(ENGINEERING_PROGRAM),
            query_ids=selected_train[:4], train_row_indices=list(range(4)),
            note='Original outcome-blind four-query selection retained unchanged; all RAW correct'),
        training_schedule=dict(updates=16, query_order=selected_train, updates_per_query=1,
            selection='Fixed final checkpoint; no held-label or probe-based tuning'),
        selection=dict(salt=SELECTION_SALT, algorithm='Ascending SHA256(salt|query_id)',
            train='Preserve four engineering rows first; then fill eight RAW-correct and eight RAW-incorrect TRAIN rows in SHA order, globally distinct components preferred',
            train_count=TRAIN_COUNT, train_target_absent_skipped=skipped_absent,
            train_raw_correct=TRAIN_PER_STRATUM, train_raw_incorrect=TRAIN_PER_STRATUM,
            train_components=len(components), repeated_component_query_ids=duplicate_components,
            available_target_present_strata={str(k):len(v) for k,v in strata.items()},
            probe='First eight original outer-held query IDs by SHA; no label, target-presence, or group filtering',
            probe_count=PROBE_COUNT, probe_component_count='Only report at post-prediction label join',
            target_insertion=False, train_raw_correctness_stratification=True,
            new_model_result_selection=False, probe_correctness_selection=False, held_identity_label_reads=0),
        frozen_head=dict(name='COST1_REFIT_M1Q0R0', parent=bind(HEAD), validation=bind(HEAD_VALIDATION),
            theta_hex=hex5, theta=[float.fromhex(v) for v in hex5], original_theta_hex=hex7,
            original_parameter_indices=[0,1,2,3,6], trained_on_fold=0,
            feature_order=['standardized_raw_gap','symmetric_M_times_L_gap','symmetric_M_gap','symmetric_L_gap'],
            feature_definition='raw gap uses population std over C128; sym(a,b)=(a-b)/(|a|+|b|+1e-12)',
            content_definition='FP64 normalized original image tokens; mean over query tokens of max reference cosine',
            action='Highest challenger logit > 0 SWITCH, otherwise original RAW winner HOLD; challengers in original physical-row order',
            inference_true_M_all_arms=True, retrain_head=False),
        encoder=dict(model_path=str(WORKSPACE/'models/downloaded_models/colnomic-embed-multimodal-7b'),
            base_model_path=str(WORKSPACE/'models/downloaded_models/colqwen2.5-7B-base'),
            original_encoding='ColQwen2_5 BF16 eval; processor.process_images RGB with original per-image frame; image_token_id mask; float then FP16 cache',
            min_pixels=processor['min_pixels'], max_pixels=processor['max_pixels'],
            patch_size=processor['patch_size'], merge_size=processor['merge_size'],
            processor_config=processor, original_preflight=bind(ORIGINAL_PREFLIGHT),
            model_sources=model_sources, weight_hashes='Pinned original preflight hashes; presence checked here; runtime must verify before training'),
        provenance=dict(program=bind(__file__), parent=bind(PARENT), split=parent['public_sources']['split'],
            train_roles=roles_binding, gallery=gallery_binding, catalog=cv['catalog'],
            catalog_validation=bind(cv_path), workers=workers_binding,
            original_token_materializer=bind(ROOT/'programs/materialize_rc_original7_train128_token_raw_v1.py'),
            operator_program=bind(ROOT/'programs/rc_quality_content_operator_v1.py')),
        audit=dict(unique_token_files_verified=len(token_sources), token_files_sha_checked=True,
            original_token_tensor_sha_checked=True, train_components=len(components),
            query_image_hashes_checked=True, fresh_encoder_parity='Required at runtime before first optimizer step',
            held_target_labels_in_manifest=False, held_identity_label_reads=0,
            new_roma_forwards=0, new_encoder_forwards=0, training_updates=0))


def verify_manifest(manifest):
    need(manifest['status'] == 'PRELLM_M_PILOT_INPUTS_FROZEN', 'MANIFEST_STATUS')
    need(len(manifest['train_rows']) == TRAIN_COUNT and len(manifest['probe_rows']) == PROBE_COUNT,
         'PILOT_COUNTS')
    for row in manifest['probe_rows']:
        need(not any(k.startswith('target') or k in ('identity', 'component', 'group') for k in row),
             'PROBE_LABEL_FIELDS_FORBIDDEN')
    for row in manifest['train_rows'] + manifest['probe_rows']:
        need(all(len(row[k]) == 128 for k in ('candidate_ids', 'candidate_identities', 'raw_scores', 'M', 'L0', 'references')),
             'COMPLETE_C128')
        need(all(math.isfinite(v) for k in ('M', 'L0', 'raw_scores') for v in row[k]), 'FINITE_INPUTS')
        need(row['challenger_positions'] == [i for i in range(128) if i != row['winner_index']],
             'CHALLENGER_AXIS')
    return dict(status='PRELLM_M_PILOT_MANIFEST_PASS', manifest=bind(MANIFEST),
        train_queries=TRAIN_COUNT, probe_queries=PROBE_COUNT, candidates_per_query=128,
        train_components=manifest['selection']['train_components'],
        train_raw_correct=sum(r['raw_correct'] for r in manifest['train_rows']),
        train_raw_incorrect=sum(not r['raw_correct'] for r in manifest['train_rows']),
        original_engineering_queries_preserved=4,
        held_identity_label_reads=0, outer_held_labels_included=False,
        fresh_encoder_parity_pending=True, new_gpu_forwards=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify'), nargs='?', default='prepare')
    args = parser.parse_args()
    torch.set_num_threads(2)
    guard()
    if args.stage == 'prepare':
        write(MANIFEST, build_manifest(), replace_from=ENGINEERING_MANIFEST)
    manifest = read(MANIFEST)
    checked(manifest['provenance']['program'])
    if args.stage == 'verify':
        need(build_manifest() == manifest, 'FRESH_SOURCE_REBUILD_EQUALS_MANIFEST')
    result = verify_manifest(manifest)
    write(OUT/'input_manifest_validation.json', result, replace_from=ENGINEERING_VALIDATION)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
