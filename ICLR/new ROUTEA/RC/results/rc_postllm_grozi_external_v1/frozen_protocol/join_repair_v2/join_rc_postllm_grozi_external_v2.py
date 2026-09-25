#!/usr/bin/env python3
"""Join repair: respect the original identity-deduplicated gallery ranking."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import join_rc_postllm_h593_v1 as J
from run_rc_simple_external_replay_v1 import compare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_postllm_grozi_external_v1'
AUTH = ROOT / 'registry/rc_postllm_grozi_external_authority_v1_20260925.json'
REPORT = ROOT / 'reports/REPORT_POSTLLM_GROZI_EXTERNAL_V1_20260925.md'
REPAIR = ROOT / 'registry/rc_postllm_grozi_join_repair_v2_20260925.json'
MODELS = ('POST_REAL', 'POST_REAL_CONSTANT', 'POST_REAL_SHUFFLED',
          'NO_ADAPTER_INTERNAL3', 'EXTERNAL_ADDITIVE4', 'EXTERNAL_PRODUCT5')
need, read, bind, write = J.need, J.read, J.bind, J.write


def bindings(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from bindings(child)
    elif isinstance(value, list):
        for child in value:
            yield from bindings(child)



def validate_identity_axis(order, axis, identities, scores=None):
    """Each identity keeps its highest-scoring physical row, stable on row ties."""
    need(len(order) == len(set(order)) == len(set(identities)), 'COMPLETE_IDENTITY_RANK')
    need(all(type(i) is int and 0 <= i < len(identities) for i in order), 'PHYSICAL_ROW_RANGE')
    need(len({identities[i] for i in order}) == len(order)
         and {identities[i] for i in order} == set(identities), 'IDENTITY_COVERAGE')
    need(len(axis) == len(set(axis)) == 128 and axis == sorted(order[:128]), 'NATURAL_C128')
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        need(scores.shape == (len(identities),) and np.isfinite(scores).all(), 'FULL_PHYSICAL_SCORES')
        expected, seen = [], set()
        for i in np.argsort(-scores, kind='stable').tolist():
            if identities[i] not in seen:
                seen.add(identities[i]); expected.append(i)
        need(order == expected, 'INDEPENDENT_IDENTITY_RANK_FROM_FULL_SCORES')


def prelabel(authority_path, out):
    """Read source parameters, metadata and predictions, never query identities."""
    check = J.Checker(); a = read(authority_path); ab = bind(authority_path)
    repair = read(REPAIR)
    need(repair['original_authority'] == ab and repair['replacement_join'] == bind(__file__), 'JOIN_REPAIR_BINDING')
    check(repair['original_join']); check(repair['regression_tests'])
    need(repair['inference_or_model_changes'] is False, 'REPAIR_SCOPE')
    for source in a['code_sources']:
        check(source)
    for key in ('source_authority', 'source_snapshot', 'source_fit', 'external_authority',
                'workers', 'legacy_gallery', 'gallery_append'):
        check(a[key])
    source = read(a['source_authority']['path'])
    for item in source['code_sources']:
        check(item)
    fit = read(a['source_fit']['path'])
    state = torch.load(a['source_snapshot']['path'], map_location='cpu', weights_only=True)
    need(fit['status'] == 'H593_FIXED_EIGHT_PASSES_COMPLETE' and
         fit['snapshot'] == a['source_snapshot'] and fit['authority'] == a['source_authority'], 'SOURCE_FIT')
    need(state['authority'] == a['source_authority'] and state['arm'] == fit['arm'] == 'POST_REAL'
         and state['step'] == fit['steps'] == 3656 and fit['train_queries'] == 457
         and fit['held_label_reads'] == 0 and not fit['direct_M_in_head'], 'FIXED_FOLD0_MODEL')
    heads = {kind: read(check(b)) for kind, b in a['source_heads'].items()}
    need(set(heads) == {'INTERNAL3', 'ADDITIVE4', 'PRODUCT5'} and
         state['warmstart'] == a['source_heads']['INTERNAL3'], 'MATCHED_SOURCE_HEADS')
    for kind, head in heads.items():
        need(head['kind'] == kind and head['authority'] == a['source_authority']
             and head['input_scope'] == fit['input_scope'] and head['held_label_reads'] == 0, 'HEAD_LINEAGE')
    original = read(a['external_authority']['path'])
    need(original['sources']['worker'] == a['workers'] and
         original['sources']['gallery_append'] == a['gallery_append'], 'EXTERNAL_LINEAGE')
    workers = read(a['workers']['path'])['records']
    gallery = read(a['legacy_gallery']['path'])['records'] + read(a['gallery_append']['path'])['records']
    need(len(workers) == 480 and len(gallery) == 5533 and
         [r['physical_row'] for r in gallery] == list(range(5533)), 'FROZEN_DATA_AXES')
    identities = [r['identity'] for r in gallery]
    need(len(set(identities)) == 5532, 'ORIGINAL_IDENTITY_COUNT')
    records, seals, seen = [], [], set(); maximum = 0.
    for shard in range(60):
        path = out / 'shards' / f'{shard:02d}' / 'validation.json'; seal = read(path)
        need(seal['status'] == 'GROZI_POSTLLM_SHARD_PASS' and seal['authority'] == ab
             and seal['shard'] == shard and seal['query_count'] == len(seal['queries']) == 8, 'SHARD_SEAL')
        raw = roma = None; raw_binding = roma_binding = None
        for ordinal, query_binding in enumerate(seal['queries']):
            p = read(check(query_binding)); row = p['row']; qid = p['query_id']
            need(p['status'] == 'GROZI_POSTLLM_FROZEN_QUERY_PASS' and p['authority'] == ab
                 and p['source_snapshot'] == a['source_snapshot'] and p['shard'] == shard
                 and p['held_label_reads'] == p['external_training_updates'] == 0, 'QUERY_SEAL')
            need(qid == row['query_id'] == workers[8 * shard + ordinal]['query_id'] and qid not in seen, 'QUERY_ORDER')
            seen.add(qid)
            need(not any(k in row for k in ('target_id', 'target_positions', 'raw_correct')), 'LABEL_FREE_PREDICTION')
            axis = row['candidate_ids']; order = p['raw_ranked_physical_rows']
            validate_identity_axis(order, axis, identities)
            need(row['candidate_identities'] == [identities[i] for i in axis] and
                 axis[row['winner_index']] == order[0], 'CANDIDATE_METADATA_AND_RAW_WINNER')
            mass = np.asarray(row['M'], dtype=np.float64)
            need(mass.shape == (128,) and np.isfinite(mass).all() and
                 bool(((mass >= 0) & (mass <= 1)).all()), 'MASS_DOMAIN')
            for b in bindings(p['input_sources']):
                check(b)
            b = p['input_sources']['raw']['payload']
            if raw is None:
                raw_binding = b
                payload = torch.load(check(b), map_location='cpu', mmap=True, weights_only=True)
                need(payload['authority'] == a['external_authority'], 'ORIGINAL_RAW_AUTHORITY')
                raw = {r['query_id']: r for r in payload['records']}
                roma_binding = p['input_sources']['roma']['payload']
                rp = torch.load(check(roma_binding), map_location='cpu', mmap=True, weights_only=True)
                need(rp['authority'] == a['external_authority'] and rp['raw'] == raw_binding, 'ORIGINAL_ROMA_LINEAGE')
                roma = {r['query_id']: r for r in rp['records']}
            need(b == raw_binding and p['input_sources']['roma']['payload'] == roma_binding
                 and qid in raw and qid in roma, 'ORIGINAL_RAW_ROMA_SHARD')
            old = raw[qid]
            validate_identity_axis(order, axis, identities, old['raw_physical_scores'].double().numpy())
            need(old['candidate_physical_rows'] == axis and old['raw_ranked_physical_rows'] == order and
                 np.array_equal(np.asarray(row['raw_scores']), old['candidate_raw_scores'].double().numpy()), 'EXACT_OLD_RAW_QUERY')
            ro = roma[qid]
            need(ro['candidate_physical_rows'] == axis and len(ro['candidates']) == 128, 'ORIGINAL_ROMA_AXIS')
            for position, candidate in enumerate(ro['candidates']):
                original_mass = float(torch.sqrt(candidate['query_visibility'].mean() * candidate['reference_visibility'].mean()))
                need(candidate['physical_row'] == axis[position] and original_mass.hex() == float(mass[position]).hex()
                     == float(candidate['old_scores']['visibility_mass']).hex(), 'EXACT_ORIGINAL_PAIR_M')
            cv = read(check(p['cache_validation']))
            need(cv['status'] == 'GROZI_POSTLLM_QUERY_CACHE_PASS' and cv['authority'] == ab
                 and cv['query_id'] == qid and cv['candidate_ids'] == axis
                 and cv['image_sha256'] == old['query_source_sha256']
                 and cv['old_query_tokens_sha256'] == old['query_tokens_sha256']
                 and cv['external_training_updates'] == cv['held_label_reads'] == 0, 'CACHE_QUERY_LINEAGE')
            parity = read(check(cv['parity']))
            for actual in (parity, p['projection_parity']):
                need(actual['status'] == 'POSTLLM_NATIVE_TOKEN_PARITY_PASS' and actual['query_id'] == qid
                     and actual['exact_equal'] and actual['max_abs_error'] == actual['atol'] == 0., 'EXACT_NATIVE_PROJECTOR_PARITY')
            need(parity['historical_fp16_exact'] and parity['source_query_tokens_sha256'] == old['query_tokens_sha256']
                 and parity['content_max_drift'] <= a['source_parity']['max_content_error'], 'HISTORICAL_CACHE_PARITY')
            cache = torch.load(check(cv['payload']), map_location='cpu', mmap=True, weights_only=True)
            need(cache['authority'] == ab and cache['query_id'] == qid, 'CACHE_PAYLOAD_LINEAGE')
            fresh = cache['fresh_L0'].double().numpy().copy()
            need(fresh.shape == (128,) and np.isfinite(fresh).all(), 'FRESH_CONTENT_AXIS')
            del cache
            need(set(p['models']) == set(MODELS), 'ALL_FROZEN_CONTROLS')
            for model, result in p['models'].items():
                kind = 'INTERNAL3' if model.startswith('POST_') else model.removeprefix('NO_ADAPTER_').removeprefix('EXTERNAL_')
                need(result['kind'] == kind, 'HEAD_FEATURE_KIND')
                if not model.startswith('POST_'):
                    need(np.array_equal(np.asarray(result['L'], dtype=np.float64), fresh), 'UNADAPTED_CONTENT_EXACT_CACHE')
                theta = state['head'].double().tolist() if model.startswith('POST_') else heads[kind]['theta']
                expected = J.numpy_decision(row, result['L'], theta, kind)
                maximum = max(maximum, J.compare_decision(result['decision'], expected))
            records.append(p)
        seals.append(bind(path))
        print(json.dumps(dict(stage='prelabel_shard_pass', shard=shard, queries=len(records))), flush=True)
    need(len(records) == len(seen) == 480, 'ALL480_PREDICTIONS')
    preseal = dict(status='GROZI_POSTLLM_ALL480_PRELABEL_PASS', authority=ab, join_repair=bind(REPAIR),
                   source_snapshot=a['source_snapshot'], shard_validations=seals,
                   query_count=480, numpy_max_logit_error=maximum, held_label_reads=0,
                   external_training_updates=0, raw_axis_and_scores_exact_legacy_parity=True)
    write(out / 'all_predictions_prelabel_seal.json', preseal)
    return a, records, identities, check, maximum


def join(authority_path=AUTH, out=OUT, report=REPORT):
    a, predictions, identities, check, maximum = prelabel(authority_path, out)
    olda = read(a['external_authority']['path'])
    role_binding = olda['curator_after_all_seals']
    roles = {r['query_id']: r for r in read(check(role_binding))['records']}
    need(set(roles) == {p['query_id'] for p in predictions}, 'EXTERNAL_ROLE_AXIS')
    oldroot = Path(a['workers']['path']).parent
    oldval = read(oldroot / 'result_validation.json'); oldresult = read(check(oldval['result']))
    oldrows = {r['query_id']: r for r in oldresult['rows']}
    rows = []
    for p in predictions:
        qid = p['query_id']; role = roles[qid]; order = p['raw_ranked_physical_rows']
        target = role['identity']; rank = next(i+1 for i, v in enumerate(order) if identities[v] == target)
        selected = {'RAW': order[0], **{m: d['decision']['prediction_id'] for m, d in p['models'].items()}}
        correct = {m: identities[v] == target for m, v in selected.items()}
        ranks = {m: next(i+1 for i, v in enumerate([chosen] + [v for v in order if v != chosen])
                         if identities[v] == target) for m, chosen in selected.items()}
        rr = dict(query_id=qid, identity=target, component=role['component'],
                  target_in_C128=rank <= 128, selected=selected, correct=correct, ranks=ranks)
        old = oldrows[qid]
        need(selected['RAW'] == old['selected']['RAW'] and correct['RAW'] == old['correct']['RAW']
             and ranks['RAW'] == old['ranks']['RAW'] and rr['target_in_C128'] == old['target_in_C128']
             and rr['component'] == old['component'], 'LEGACY_RAW_PER_QUERY_PARITY')
        rows.append(rr)
    need(len(rows) == 480 and len({r['component'] for r in rows}) == 27, 'ALL480_AND27_VIDEO_GROUPS')
    counts = {m: sum(r['correct'][m] for r in rows) for m in ('RAW',) + MODELS}
    recall = sum(r['target_in_C128'] for r in rows)
    need(counts['RAW'] == oldresult['counts']['RAW'] == 321 and recall == 410, 'ORIGINAL_RAW321_RECALL410')
    comparisons = {b + '__to__POST_REAL': compare(rows, b, 'POST_REAL')
                   for b in ('RAW', 'NO_ADAPTER_INTERNAL3', 'POST_REAL_CONSTANT', 'POST_REAL_SHUFFLED', 'EXTERNAL_ADDITIVE4', 'EXTERNAL_PRODUCT5')}
    for model in MODELS:
        if 'RAW__to__' + model not in comparisons:
            comparisons['RAW__to__' + model] = compare(rows, 'RAW', model)
    mrr = {m: float(np.mean([1. / r['ranks'][m] for r in rows])) for m in counts}
    summary = {}
    for model in counts:
        rescue = sum(not r['correct']['RAW'] and r['correct'][model] for r in rows)
        breaks = sum(r['correct']['RAW'] and not r['correct'][model] for r in rows)
        summary[model] = dict(correct=counts[model], MRR=mrr[model], rescue=rescue, breaks=breaks,
                             net=rescue-breaks, RAW_correct_loss_rate=breaks/counts['RAW'])
        need(counts[model] == counts['RAW'] + rescue - breaks, 'PAIRED_ACCOUNTING')
    result = dict(status='GROZI_POSTLLM_FROZEN480_COMPLETE', authority=bind(authority_path),
        source_snapshot=a['source_snapshot'], join_repair=bind(REPAIR), predictions_seal=bind(out / 'all_predictions_prelabel_seal.json'),
        source_fit='Pre-fixed H593 fold0 POST_REAL, 457 TRAIN queries, eight passes; no external model selection',
        primary='POST_REAL', query_count=480, group_count=27, group_unit='source_video',
        candidate_source='Original ColNomic full-gallery natural C128; physical gallery 5533 / deduplicated identities 5532; all127 challengers; HOLD=0',
        counts=counts, MRR=mrr, summary=summary, ranking_rule='Selected physical candidate first; remaining full-gallery RAW order unchanged',
        target_recall_C128=recall, target_absent=480-recall, comparisons=comparisons, rows=rows,
        external_training_updates=0, external_threshold_selection=False, source_models_frozen=True,
        scope='Single source-fold frozen transfer on the previously opened GroZi480 product-crop panel; not H593 fivefold OOF',
        untouched_confirmation_claimed=False, localization_IoU_claimed=False, ownership_claimed=False,
        bootstrap=dict(draws=100000, seed=20260924, estimand='Equally weighted source-video paired accuracy difference'),
        legacy_result=oldval['result'])
    write(out / 'result.json', result)
    c = comparisons['RAW__to__POST_REAL']
    lines = ['# POST-LLM 内部模型：GroZi480 冻结外部迁移', '',
        '固定 H593 fold0 POST_REAL 及其配套头与 TRAIN 标准化；无外部训练、调参或挑折。',
        '完整480张、27个视频组；自然C128包含target410张，另外70张仍计入分母。', '',
        '| 模型 | 正确/480 | MRR | 救回 | 误伤 | 净增 | RAW正确误伤率 |', '|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {m} | {counts[m]} | {mrr[m]:.6f} | {summary[m]['rescue']} | {summary[m]['breaks']} | "
              f"{summary[m]['net']:+d} | {summary[m]['RAW_correct_loss_rate']:.2%} |" for m in ('RAW',) + MODELS]
    lines += ['', f"相对RAW：救回 {c['rescue']}、误伤 {c['loss']}、净增 {c['net']}。",
              f"等视频权重增益95% bootstrap区间：{c['group_bootstrap95']}。", '',
        '这项结果检验一个预先固定的内部模型能否跨数据集使用真实M；是否有效按本次真实结果解释，不预先宣称GO。',
        '恒定/错绑是同一冻结模型的输入干预；无适配器和外部头均来自同一个源域折。',
        'MRR按最终选中候选置首、其余图库候选保持原RAW次序计算；误伤率分母为RAW原正确321张。',
        '此前已使用该外部面板；本次不称全新未触碰确认，也不建立区域IoU、分割精度或ownership结论。', '',
        f'[完整结果]({out}/result.json) · [独立核算]({out}/validation.json)', '']
    text = '\n'.join(lines)
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        need(report.read_text() == text, 'IMMUTABLE_REPORT')
    else:
        report.write_text(text)
    write(out / 'validation.json', dict(status='GROZI_POSTLLM_NUMPY_SOURCE_COUNTS_GROUPS_PASS',
        authority=bind(authority_path), join_repair=bind(REPAIR), result=bind(out / 'result.json'), report=bind(report),
        predictions_seal=bind(out / 'all_predictions_prelabel_seal.json'), numpy_max_logit_error=maximum,
        query_count=480, video_groups=27, raw_correct=321, target_recall_C128=410,
        source_snapshot=a['source_snapshot'], exact_legacy_RAW_per_query=True,
        closure_pointers=dict(result=str(out/'result.json'), report=str(report)),
        external_training_updates=0, source_checkpoint_selection='Fixed fold0 before external predictions'))
    print(json.dumps(dict(status=result['status'], counts=counts, paired_vs_RAW=c), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, default=AUTH)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    join(args.authority, args.out, args.report)
