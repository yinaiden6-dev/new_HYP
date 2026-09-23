#!/usr/bin/env python3
"""Export 127 sealed challenger logits plus the original zero-score HOLD action."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import render_rc_cost1_medicine_visibility_v1 as H
import numpy as np
import torch

sys.path.insert(0, str(H.ROOT / 'src'))
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as F

OUT = H.OUT / 'candidate_scores'
FEATURES = list(F.FEATURE_NAMES)
SHORT = ['RAW', 'S', 'M', 'L', 'Q', 'R']
WEIGHT_KEYS = ['weight_' + s for s in SHORT]
SOURCE_FILES = {}


def source(path):
    path = Path(path)
    SOURCE_FILES[str(path)] = H.bind(path)
    return path


def read(path):
    return H.read(source(path))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_case(case, location):
    qid = case['query_id']
    head_path = source(case['head_source']['path'])
    assert H.V.sha(head_path) == case['head_source']['sha256']
    head = read(head_path)
    head_val = read(head_path.parent / 'validation.json')
    assert head_val['payload'] == case['head_source'] and 'PASS' in head_val['status']
    assert qid not in head['train_query_ids']
    pred = next(p for p in head['predictions'] if p['query_id'] == qid)['models']['COST1']
    weights_hex = head['parameters']['COST1']
    assert weights_hex == case['head_parameters_hex']
    weights = np.array([float.fromhex(s) for s in weights_hex], dtype=np.float64)
    lane, shard = location
    folder = H.DATA / 'features' / lane / f'shard{shard:02d}'
    feature = next(r for r in H.cache(folder)['records'] if r['query_id'] == qid)
    feature_binding = read(folder / 'receipt.json')['payload']
    source(feature_binding['path'])
    raw = H.cache(Path(case['raw_payload']['path']).parent)
    roma = H.cache(Path(case['roma_payload']['path']).parent)
    for kind in ('raw_payload', 'roma_payload'):
        p = source(case[kind]['path'])
        assert H.V.sha(p) == case[kind]['sha256']
    q = next(r for r in raw['records'] if r['query_id'] == case['source_query_id'])
    m = next(r for r in roma['records'] if r['query_id'] == case['source_query_id'])
    axis = feature['candidate_physical_rows']
    assert len(axis) == len(set(axis)) == len(m['candidates']) == 128
    assert axis == q['candidate_physical_rows'] == m['candidate_physical_rows']
    challenger_positions = feature['challenger_positions']
    winner = feature['winner']
    assert len(challenger_positions) == 127
    assert sorted(challenger_positions + [winner]) == list(range(128))
    sealed = np.array([float.fromhex(v) for v in pred['logits_hex']], dtype=np.float64)
    X = feature['modes']['REAL']['X'].numpy()
    assert sealed.shape == (127,) and X.shape == (127, 6) and X.dtype == np.float64
    assert np.isfinite(sealed).all() and np.isfinite(X).all()
    reproduced = np.sum(X * weights[:-1], axis=1) + weights[-1]
    error = float(np.max(np.abs(reproduced - sealed)))
    assert error < 2e-10
    selected = axis[challenger_positions[int(sealed.argmax())]] if sealed.max() > 0 else axis[winner]
    assert selected == pred['selected'] == case['final_selected']
    assert axis[winner] == case['raw_selected']
    assert float(sealed.max()).hex() == case['selected_logit'].hex()
    raw_scores = [float(v) for v in q['candidate_raw_scores']]
    raw_order = q['candidate_ranked_physical_rows']
    assert len(raw_order) == len(set(raw_order)) == 128 and set(raw_order) == set(axis)
    raw_rank = {physical: i + 1 for i, physical in enumerate(raw_order)}
    evidence = {c['candidate_position']: c['old_scores'] for c in m['candidates']}
    ch_index = {pos: i for i, pos in enumerate(challenger_positions)}
    rows = []
    for pos, physical in enumerate(axis):
        is_hold = pos == winner
        c = m['candidates'][pos]
        assert c['candidate_position'] == pos and c['physical_row'] == physical
        ev = evidence[pos]
        assert float(ev['raw_score']).hex() == raw_scores[pos].hex()
        features = terms = None
        logit = logit_hex = residual = None
        if not is_hold:
            j = ch_index[pos]
            # Rebuild each 6-vector from the same raw and C4 scalar evidence.
            independent = F.candidate_feature(raw_scores, evidence, pos, winner).numpy()
            assert independent.tobytes() == X[j].tobytes()
            features = [float(v) for v in X[j]]
            terms = [float(v) for v in X[j] * weights[:-1]]
            logit_hex = pred['logits_hex'][j]
            logit = float.fromhex(logit_hex)
            residual = logit - math.fsum(terms + [float(weights[-1])])
            assert abs(residual) < 2e-10
        score = 0.0 if is_hold else logit
        r = dict(query_id=qid, display_id=case['display_id'], candidate_position=pos,
                 reference_physical_row=physical, reference_image_path=raw['references'][physical]['source_path'],
                 raw_rank=raw_rank[physical], raw_score=raw_scores[pos], raw_score_hex=raw_scores[pos].hex(),
                 is_target=physical == case['target_physical'], is_raw_winner=is_hold,
                 is_selected=physical == selected, action_kind='HOLD' if is_hold else 'CHALLENGER',
                 challenger_index=None if is_hold else ch_index[pos],
                 head_logit=logit, head_logit_hex=logit_hex,
                 policy_score=score, policy_score_hex=score.hex(),
                 local_score_S=float(ev['real_score']), visibility_mass_M=float(ev['visibility_mass']),
                 normalized_similarity_L=float(ev['real_score']) / max(float(ev['visibility_mass']), 1e-12),
                 query_control_score=float(ev['query_control_score']),
                 reference_control_score=float(ev['reference_control_score']),
                 query_robustness_Q=float(ev['real_score']) - float(ev['query_control_score']),
                 reference_robustness_R=float(ev['real_score']) - float(ev['reference_control_score']),
                 feature_values=features,
                 feature_values_hex=None if is_hold else [v.hex() for v in features],
                 weighted_feature_terms=terms,
                 weighted_feature_terms_hex=None if is_hold else [v.hex() for v in terms],
                 bias_contribution=None if is_hold else float(weights[-1]),
                 fp64_sum_residual=residual,
                 plot_raw_component=0.0 if is_hold else terms[0],
                 plot_other_component=0.0 if is_hold else logit - terms[0],
                 plot_is_hold_anchor=is_hold)
        assert abs(r['plot_raw_component'] + r['plot_other_component'] - score) < 2e-14
        rows.append(r)
    ranked = sorted(rows, key=lambda r: (-r['policy_score'], 0 if r['is_raw_winner'] else 1,
                                          r['challenger_index'] if r['challenger_index'] is not None else -1))
    for rank, r in enumerate(ranked, 1):
        r['policy_rank'] = rank
    assert ranked[0]['reference_physical_row'] == selected
    assert sum(r['is_target'] for r in rows) == sum(r['is_raw_winner'] for r in rows) == sum(r['is_selected'] for r in rows) == 1
    metadata = dict(schema='COST1_C128_COMPLETE_DECISION_SCORES_V1', query_id=qid,
                    display_id=case['display_id'], candidate_count=128, challenger_count=127,
                    feature_order=FEATURES, feature_short_names=SHORT,
                    head_parameters_hex=weights_hex, head_parameters=[float(v) for v in weights],
                    head_source=case['head_source'], source_fold=case['source_fold'],
                    features_source=feature_binding, raw_source=case['raw_payload'], roma_source=case['roma_payload'],
                    source_query_id=case['source_query_id'], selected_physical_row=selected,
                    target_physical_row=case['target_physical'], raw_winner_physical_row=axis[winner],
                    threshold=0.0,
                    score_definition='127 original sealed COST1 challenger logits; RAW winner HOLD policy score exactly zero. HOLD has no head evaluation; its feature/contribution fields are null.',
                    decision_rule='First maximum in frozen challenger order; SWITCH iff maximum logit > 0; otherwise HOLD.',
                    joint_plot_definition='x = weighted RAW feature; y = sealed challenger logit - x. Thus x+y is the original decision score. HOLD is a policy anchor at (0,0), not a head-computed feature point.',
                    maximum_logit_replay_error=error, new_training_updates=0,
                    new_encoder_or_matcher_calls=0, candidates=rows)
    dest = OUT / 'data' / (case['stem'] + '_C128.json')
    H.dump(dest, metadata)
    reread = json.loads(dest.read_text())
    assert len(reread['candidates']) == 128
    for r in reread['candidates']:
        assert r['policy_score'].hex() == r['policy_score_hex']
        if r['action_kind'] != 'HOLD':
            assert r['head_logit'].hex() == pred['logits_hex'][r['challenger_index']]
            assert [v.hex() for v in r['feature_values']] == r['feature_values_hex']
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if not isinstance(v, list) and k not in
             ['feature_values', 'feature_values_hex', 'weighted_feature_terms', 'weighted_feature_terms_hex']}
        for i, short in enumerate(SHORT):
            for prefix, name in [('feature_', 'feature_values'), ('contribution_', 'weighted_feature_terms')]:
                d[prefix + short] = None if r[name] is None else r[name][i]
                d[prefix + short + '_hex'] = None if r[name] is None else r[name][i].hex()
            d['weight_' + short] = float(weights[i])
        d['head_bias'] = float(weights[-1])
        flat.append(d)
    write_csv(OUT / 'data' / (case['stem'] + '_C128.csv'), flat)
    return metadata, flat


def documentation(summaries):
    return """# COST1：完整 128 候选决策数据

[全部 1536 行 CSV](data/all_12_cases_1536_candidates.csv) · [数据 ZIP](../COST1_medicine_C128_scores.zip)

每个案例包含自然 C128 的完整 128 行：127 个 challenger 直接使用原封存 COST1 logit；RAW 首选对应 HOLD，其 **policy_score 固定为 0**，head_logit、六项头特征和贡献均为 null。HOLD 不是把零向量输入头后得到的偏置。

CSV/JSON 提供原始 RAW 分数与名次、原 C128 位置、图库物理行、参考图路径、目标/RAW首选/最终选择标记、六项特征、六项加权贡献、偏置、封存 logit、policy_score 和各自十六进制 FP64 表示。决策分数不是校准身份概率。

特征顺序为 RAW / S / M / L / Q / R；完整名称和头参数保存在每例 JSON。candidate_position 从 0 开始，raw_rank 和 policy_rank 从 1 开始。HOLD 与 challenger 同为 0 时保持 HOLD；正分并列采用原 challenger 顺序。

为方便自行绘图，数据另附 plot_raw_component 与 plot_other_component，分别为 RAW 项加权贡献和封存分数减去该贡献，二者之和等于原决策分数。HOLD 对应的 (0,0) 只是决策基准，其真实头特征仍为 null。此包仅提供数据，不含新生成点图。

仍为同一批 H593 分组留出案例，参数来源记录在 JSON。无重训、无新编码器或 RoMa 前向。

| 案例 | 候选行数 | CSV | JSON |
|---|---:|---|---|
""" + '\n'.join(f"| {s['display_id']} | 128 | [CSV](data/{s['stem']}.csv) | [JSON](data/{s['stem']}.json) |" for s in summaries) + '\n'


def main():
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    original_cases = read(H.OUT / 'data/cases_manifest.json')
    assert len(original_cases) == 12
    _, _, _, locations = H.select()
    combined, summaries = [], []
    for case in original_cases:
        metadata, flat = export_case(case, locations[case['query_id']])
        combined.extend(flat)
        stem = case['stem'] + '_C128'
        selected = next(r for r in flat if r['is_selected'])
        summaries.append(dict(stem=stem, display_id=case['display_id'], query_id=case['query_id'],
                              candidate_count=128, selected_logit=selected['policy_score'],
                              target_raw_rank=selected['raw_rank'],
                              replay_error=metadata['maximum_logit_replay_error']))
        print('EXPORTED', stem, '128 candidates', flush=True)
    assert len(combined) == 1536
    write_csv(OUT / 'data/all_12_cases_1536_candidates.csv', combined)
    H.dump(OUT / 'data/cases_index.json', summaries)
    source(__file__); source(F.__file__)
    H.dump(OUT / 'data/source_manifest.json', SOURCE_FILES)
    validation = dict(status='COST1_ALL12_C128_SCORES_VERIFIED', cases=12,
                      candidates=1536, sealed_challenger_scores=1524, zero_HOLD_scores=12,
                      sealed_scores_hex_exact=True, feature_reconstruction_bit_exact=True,
                      selected_actions_match_original=True, data_only=True,
                      maximum_logit_replay_error=max(s['replay_error'] for s in summaries),
                      new_training_updates=0, new_encoder_or_matcher_calls=0)
    H.dump(OUT / 'data/validation.json', validation)
    (OUT / 'README.md').write_text(documentation(summaries))
    archive = H.OUT / 'COST1_medicine_C128_scores.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and (p.suffix in ('.csv', '.json') or p.name == 'README.md'):
                z.write(p, str(p.relative_to(H.OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert hashlib.sha256(z.read(name)).hexdigest() == H.V.sha(H.OUT / name)
    print(json.dumps(validation), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
