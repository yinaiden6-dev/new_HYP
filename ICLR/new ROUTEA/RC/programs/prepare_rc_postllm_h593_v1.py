#!/usr/bin/env python3
"""Freeze H593 post-LLM inputs while keeping fold labels out of shared rows.

This is file-only intake.  It neither trains nor runs ColNomic/RoMa.  Original
natural C128 physical ordering, RAW scores, M, and free-content scores are
preserved.  Existing label-free hidden states may be reused across folds;
pilot-trained adapters, heads, normalization, and optimizer states may not.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_postllm_h593_v1'
PARENT = ROOT / 'registry/rc_h593_quality_operator_eval_authority_v1_20260922.json'
CATALOG_VALIDATION = ROOT / 'results/rc_h593_feature_fusion_cache_v1/catalog_validation.json'
SIMPLE_VALIDATION = ROOT / 'results/rc_h593_simple_explanations_v1/cache_validation.json'
PILOT = ROOT / 'results/rc_prellm_m_adapter_v2'
PILOT_AUTH = ROOT / 'registry/rc_prellm_m_adapter_authority_v2_20260924.json'
REPORT = ROOT / 'reports/REPORT_POSTLLM_H593_CACHE_INTAKE_20260924.md'


def need(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).absolute()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return dict(path=str(path), sha256=h.hexdigest())


def checked(binding):
    need(bind(binding['path']) == binding, 'SOURCE_SHA_DRIFT:' + binding['path'])
    return Path(binding['path'])


def write(path, data):
    path = Path(path)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        need(path.read_text() == text, 'IMMUTABLE_INTAKE:' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    temp.write_text(text)
    os.replace(temp, path)


def prepare():
    import torch
    torch.set_num_threads(2)
    parent = read(PARENT)
    split_binding = parent['public_sources']['split']
    split = read(checked(split_binding))
    gallery_binding = parent['public_sources']['gallery']
    gallery = {r['physical_row']: r for r in read(checked(gallery_binding))['records']}
    cv = read(CATALOG_VALIDATION)
    need(cv['status'] == 'FUSION_ALL593_CATALOG_PASS', 'TOKEN_CATALOG_PASS')
    catalog = read(checked(cv['catalog']))
    sv = read(SIMPLE_VALIDATION)
    need(sv['status'] == 'CACHE_PASS' and sv['queries'] == 593, 'SIMPLE_CACHE_PASS')
    simple = read(checked(sv['payload']))
    scores = {r['query_id']: r for r in simple['rows']}
    old_manifest = read(PILOT / 'input_manifest.json')
    old_rows = {r['query_id']: r for r in old_manifest['train_rows'] + old_manifest['probe_rows']}
    old_authority = bind(PILOT_AUTH)
    meta = sorted(catalog['queries'], key=lambda r: r['execution_ordinal'])
    need(len(meta) == 593 and [r['execution_ordinal'] for r in meta] == list(range(593)), 'ALL593_QUERY_AXIS')
    need(set(scores) == {r['query_id'] for r in meta}, 'SCORE_CATALOG_AXIS')

    token_meta = {}
    def tokens(key):
        if key not in token_meta:
            entry = catalog['images'][key]
            item = entry['item']
            checked(entry['input'])
            need(Path(item['path']).is_file(), 'ORIGINAL_IMAGE_PRESENT')
            token_meta[key] = dict(image_key=key, image_path=item['path'],
                image_sha256=item['image_sha256'], grid_hw=item['grid'], frame=item['frame'],
                tokens_sha256=item['tokens_sha256'], token_file=entry['input'], token_field='tokens',
                token_shape=[math.prod(item['grid']), 128], token_dtype='torch.float16')
        return token_meta[key]

    rows, reused, missing = [], [], []
    source_bytes = 0
    for qm in meta:
        p = torch.load(checked(qm['payload']), map_location='cpu', weights_only=True)
        q = qm['query_id']; i = qm['execution_ordinal']; s = scores[q]
        axis = p['candidate_physical_rows']; raw = list(map(float, p['candidate_raw_scores']))
        need(p['query_id'] == q and p['execution_ordinal'] == i and p['label_reads'] == 0, 'QUERY_PAYLOAD_LINEAGE')
        need(len(axis) == len(set(axis)) == 128 and axis == sorted(axis) == s['axis'], 'ORIGINAL_PHYSICAL_C128')
        need(p['raw_ranked_physical_rows'] == s['raw_ranked'], 'RAW_RANK_ORDER')
        winner = max(range(128), key=lambda j: (raw[j], -axis[j]))
        need(winner == p['winner'] == s['winner'], 'RAW_WINNER')
        need(p['challenger_positions'] == s['challengers'] == [j for j in range(128) if j != winner], 'CHALLENGER_ORDER')
        mass = [float(pair['native_c4'][1]) for pair in p['pairs']]
        need(mass == s['mass'] and all(0 <= x <= 1 and math.isfinite(x) for x in mass), 'EXACT_ORIGINAL_M')
        qt = tokens(p['query_image_key'])
        need(qt['image_sha256'] == p['source_image_sha256'] == s['source_image_sha256'], 'QUERY_IMAGE_BINDING')
        need(bind(qt['image_path'])['sha256'] == qt['image_sha256'], 'QUERY_IMAGE_BYTES')
        references = []
        for j, pair in enumerate(p['pairs']):
            physical = axis[j]
            need(pair['position'] == j and pair['physical_row'] == physical, 'REFERENCE_PAIR_AXIS')
            t = tokens(pair['image_key'])
            references.append(dict(position=j, physical_row=physical,
                identity=gallery[physical]['identity'], gallery_image_path=gallery[physical]['image_path'],
                **t))
        identities = [gallery[v]['identity'] for v in axis]
        need(len(set(identities)) == 128, 'IDENTITY_DEDUP_C128')
        legacy_dir = PILOT / 'encoder_cache' / q
        legacy = None
        if (legacy_dir / 'validation.json').is_file():
            seal = read(legacy_dir / 'validation.json')
            need(seal['status'] == 'QUERY_ENCODER_CACHE_PASS' and seal['authority'] == old_authority,
                 'LEGACY_HIDDEN_SEAL')
            checked(seal['payload']); checked(seal['parity'])
            need(seal['image_sha256'] == qt['image_sha256'] and not seal['held_labels_read'], 'LEGACY_LABEL_FREE_IMAGE')
            legacy = dict(validation=bind(legacy_dir / 'validation.json'), payload=seal['payload'], parity=seal['parity'])
            source_bytes += Path(seal['payload']['path']).stat().st_size
            reused.append(q)
        else:
            missing.append(q)
        row = dict(query_id=q, execution_ordinal=i, image_path=qt['image_path'],
            source_image_sha256=qt['image_sha256'], frame=qt['frame'], query_tokens=qt,
            candidate_ids=axis, candidate_identities=identities,
            raw_ranked_candidate_ids=p['raw_ranked_physical_rows'], raw_scores=raw,
            M=mass, L0=s['free_content'], winner_index=winner,
            challenger_positions=p['challenger_positions'], references=references,
            reference_tokens=[r['token_file'] for r in references],
            hidden_cache=legacy, hidden_validation=legacy['validation'] if legacy else None,
            hidden_cache_dir=str(OUT / 'encoder_cache' / q),
            legacy_cache_dir=str(legacy_dir) if legacy else None,
            native_content_source='Original image-only FP16 ColNomic token bank; FP64 normalization and free MaxSim. New training must use validated cache fresh_L0 for zero-adapter consistency.',
            sources=dict(query_payload=qm['payload'], score_cache=sv['payload']))
        need(all(math.isfinite(x) for k in ('raw_scores', 'M', 'L0') for x in row[k]), 'FINITE_ROW')
        if q in old_rows:
            for key in ('candidate_ids', 'candidate_identities', 'raw_scores', 'M', 'L0', 'winner_index', 'challenger_positions', 'reference_tokens', 'query_tokens'):
                need(row[key] == old_rows[q][key], 'EXACT_OLD24_INPUTS:' + key)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(stage='intake', rows=i + 1, reusable_hidden=len(reused), missing_hidden=len(missing))), flush=True)
    need(set(reused) == set(old_rows) and len(reused) == 24, 'EXACT24_LEGACY_HIDDEN')
    byid = {r['query_id']: r for r in rows}
    folds, fold_counts, all_held = {}, [], []
    for f, original in enumerate(split['folds']):
        need(original['fold'] == f, 'ORIGINAL_FOLD_INDEX')
        train_all = original['train_query_ids']; held = original['heldout_query_ids']
        need(set(train_all).isdisjoint(held) and set(train_all) | set(held) == set(byid), 'FULL_ORIGINAL_FOLD')
        need(not {byid[q]['source_image_sha256'] for q in train_all} &
             {byid[q]['source_image_sha256'] for q in held}, 'IMAGE_DISJOINT_FOLD')
        binding = parent['fold_sources'][str(f)]['train_roles']
        role_rows = read(checked(binding))['records']; roles = {r['query_id']: r for r in role_rows}
        need(set(roles) == set(train_all), 'ONLY_ORIGINAL_FOLD_TRAIN_LABELS')
        labels, eligible, excluded = [], [], []
        for q in train_all:
            role = roles[q]; row = byid[q]
            target = [j for j, identity in enumerate(row['candidate_identities']) if identity == role['identity']]
            need(len(target) <= 1, 'AT_MOST_ONE_TARGET')
            lab = dict(query_id=q, target_id=role['identity'], target_positions=target,
                target_position=target[0] if target else None, component=role['component'],
                group=role['group'], target_in_C128=bool(target), eligible=bool(target),
                raw_correct=bool(target and target[0] == row['winner_index']))
            labels.append(lab)
            (eligible if target else excluded).append(q)
        label_path = OUT / 'fold_inputs' / f'fold{f}_train_labels.json'
        write(label_path, dict(status='ORIGINAL_FOLD_TRAIN_LABELS_ONLY', fold=f, source=binding,
              records=labels, held_query_labels_included=False))
        logs = [math.log(max(m, 1e-8)) for q in eligible for m in byid[q]['M']]
        mean = sum(logs) / len(logs); std = math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs))
        need(std > 0 and math.isfinite(std), 'TRAIN_ONLY_MASS_NORMALIZATION')
        normalization = dict(log_mean=mean, log_std=std, epsilon=1e-8,
            definition='Population mean/std of log(max(M,1e-8)) over eligible original TRAIN x natural C128 only',
            fit_queries=len(eligible), fit_pairs=len(logs), held_used=False)
        folds[str(f)] = dict(fold=f, train_query_ids=eligible, train_all_query_ids=train_all,
            excluded_target_absent_train_query_ids=excluded, held_query_ids=held,
            train_labels=bind(label_path), mass_normalization=normalization,
            train_order='Original split_manifest TRAIN query order filtered by target presence',
            held_policy='Every original outer-held query retained, including target-absent C128 rows',
            initialization='Fresh zero-output adapter and fresh fold-TRAIN-only INTERNAL3 warm head; never pilot16 weights or normalization')
        fold_counts.append(dict(fold=f, train_all=len(train_all), eligible_train=len(eligible),
            train_target_absent=len(excluded), held=len(held), train_components=len({r['component'] for r in role_rows})))
        all_held.extend(held)
    need(len(all_held) == len(set(all_held)) == 593, 'ONE_OOF_PREDICTION_PER_QUERY')
    manifest = dict(schema_version=1, status='POSTLLM_H593_LABEL_FREE_INPUTS_FROZEN',
        query_count=593, candidates_per_query=128, rows=rows, folds=folds,
        encoder=old_manifest['encoder'], model_source_validation=bind(PILOT / 'model_source_validation.json'),
        held_join_source=parent['join_sources']['curator'],
        held_join_policy='Read only after every arm/fold prediction has been sealed; preparer did not read this file',
        old_probe_status='Opened development evidence; H593 OOF extension is not a new untouched test set',
        cache_plan=dict(reuse_query_ids=reused, missing_query_ids=missing, reused=len(reused), missing=len(missing),
            shards=50, shard_assignment='execution_ordinal % 50', new_roma_forwards=0,
            new_reference_encoder_forwards=0, missing_query_encoder_forwards=len(missing)),
        initialization_prohibitions=['pilot-trained adapter', 'pilot-trained head', 'pilot optimizer state', 'pilot mass normalization'],
        provenance=dict(program=bind(__file__), parent=bind(PARENT), split=split_binding,
            gallery=gallery_binding, catalog=cv['catalog'], catalog_validation=bind(CATALOG_VALIDATION),
            score_cache=sv['payload'], score_cache_validation=bind(SIMPLE_VALIDATION),
            reusable_hidden_authority=old_authority, reusable_manifest=bind(PILOT / 'input_manifest.json')))
    path = OUT / 'input_manifest.json'; write(path, manifest)
    audit = dict(status='H593_POSTLLM_INTAKE_PASS_GPU_QUERY_CACHE_INCOMPLETE', manifest=bind(path),
        query_count=593, candidate_pairs=593 * 128, token_files=len(token_meta),
        all_required_token_files_sha_verified=True, query_image_bytes_sha_verified=593,
        legacy_hidden_reusable=24, legacy_hidden_payload_bytes=source_bytes, missing_query_hidden=569,
        no_new_roma_or_reference_encoding_required=True, original_folds=fold_counts,
        source_training_labels_read='Five separate original TRAIN role files only; no curator opened',
        held_label_reads_from_curator=0, labels_in_shared_rows=False,
        reused_pilot_learned_parameters=False, no_training_or_forward_in_preparation=True,
        agents_md_search='No AGENTS.md found beneath benchmark or home by rg --files --hidden; .git/cache excluded',
        hidden_search_scope='RC result encoder_cache paths and post/pre/internal-M implementations; only validated V2 24 hidden payloads reusable',
        validation=dict(all_old24_inputs_exact=True, candidate_axes_preserved=True, target_insertion=False,
            reference_tokens_and_RoMa_M_reused=True, oof_held_queries=593, held_absent_not_filtered=True))
    write(OUT / 'intake_audit.json', audit)
    write(OUT / 'input_manifest_validation.json', dict(status='H593_MANIFEST_593_C128_FIVEFOLD_PASS',
        manifest=bind(path), intake=bind(OUT / 'intake_audit.json'),
        fold_labels=[folds[str(f)]['train_labels'] for f in range(5)],
        rows=593, candidate_pairs=75904, train_and_held_images_disjoint=True,
        runtime_hidden_parity_pending=True))
    lines = ['# 后 LLM 内部改造：H593 缓存与划分盘点', '',
        '本次扩展采用原 H593 分组五折和原 ColNomic 自然 C128。只复用冻结编码结果；不复用旧 TRAIN16 的适配器、小头或 M 归一化。', '',
        '| 输入 | 已有 | 需补 |', '|---|---:|---:|',
        '| Query 原始检索 tokens | 593 | 0 |',
        '| 原 C128 与 RoMa M | 593 × 128 | 0 |',
        '| Reference 检索 tokens | 全部所需候选 | 0 |',
        '| Query 后 LLM hidden | 24 | 569 |', '',
        f'共核对 {len(token_meta)} 个去重图像 token 文件的 SHA；核对全部 593 张 query 原图 SHA。旧24张 hidden 的封存、payload 和 parity 均逐一核对。旧 V1 只有失败的 parity，未当作可复用缓存。', '',
        '| 折 | 原 TRAIN | 可训练（target 在 C128） | TRAIN target 缺失 | 全部 held |', '|---|---:|---:|---:|---:|']
    lines += [f"| {v['fold']} | {v['train_all']} | {v['eligible_train']} | {v['train_target_absent']} | {v['held']} |" for v in fold_counts]
    lines += ['', '缺失 target 的 TRAIN 行仍记录但不进入该 COST1 身份训练；held 一张不删，最终每张仅接收所属折的 OOF 预测。M 的均值、标准差在每折实际可训练行上分别计算。', '',
        '共享 manifest 不含 query 身份标签。每折 TRAIN 标签独立存储；curator 仅保留封存路径，准备阶段没有打开。零适配器校验、实际 fresh_L0 和 projection 一致性仍需 GPU 缓存采集/校验完成；本盘点不意味着训练已完成。', '',
        '后 LLM hidden 不能由最终 128 维检索 tokens 无损倒推，故仅569张 query 需要补冻结编码前向。RoMa 与 reference 不重算；原图/token/hidden 均采用只读指针，不复制大文件。', '',
        '[输入 manifest](../results/rc_postllm_h593_v1/input_manifest.json) · [盘点 JSON](../results/rc_postllm_h593_v1/intake_audit.json) · [准备程序](../programs/prepare_rc_postllm_h593_v1.py)', '']
    report_text = '\n'.join(lines)
    if REPORT.exists():
        need(REPORT.read_text() == report_text, 'IMMUTABLE_INTAKE_REPORT')
    else:
        REPORT.write_text(report_text)
    print(json.dumps(dict(status=audit['status'], manifest=bind(path), fold_counts=fold_counts,
                         legacy_hidden=24, missing_hidden=569)), flush=True)


def verify():
    manifest = read(OUT / 'input_manifest.json')
    audit = read(OUT / 'intake_audit.json')
    checked(audit['manifest']); checked(manifest['provenance']['program'])
    for f in manifest['folds'].values():
        labels = read(checked(f['train_labels']))
        need({r['query_id'] for r in labels['records']} == set(f['train_all_query_ids']), 'FOLD_TRAIN_LABEL_AXIS')
        need(set(f['train_all_query_ids']).isdisjoint(f['held_query_ids']), 'FOLD_DISJOINT')
    for row in manifest['rows']:
        need(not any(k in row for k in ('target_id', 'target_positions', 'identity', 'component', 'group')), 'SHARED_ROWS_LABEL_FREE')
        need(all(len(row[k]) == 128 for k in ('candidate_ids', 'candidate_identities', 'raw_scores', 'M', 'L0', 'reference_tokens')), 'C128_COMPLETE')
    print(json.dumps(dict(status='H593_FROZEN_MANIFEST_VERIFIED', rows=len(manifest['rows']), folds=len(manifest['folds']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify'), nargs='?', default='prepare')
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else verify()
