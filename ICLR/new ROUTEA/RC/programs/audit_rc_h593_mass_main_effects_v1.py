#!/usr/bin/env python3
"""Independent scalar replay and pre-fixed constant/degree sidecar.

Does not import the producer, its feature/decision routines or saved feature
matrices.  The preserved RAW standardized difference is read from the bound
original scalar cache, as in the previously independently checked head replay.
All labels are already opened; no fitting, threshold selection or model search.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rc_h593_mass_main_effects_v1'
OUT = SOURCE / 'independent_constant_degree_audit'
ARMS = ('ORIGINAL_M', 'REFERENCE_MAIN_EFFECT', 'REMOVE_REFERENCE_MAIN_EFFECT', 'CONSTANT_M')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def checked(bound):
    need(bind(bound['path']) == bound, 'SOURCE_SHA:' + bound['path'])
    return Path(bound['path'])


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def symmetric(x, y):
    return (x-y)/(abs(x)+abs(y)+1e-12)


def degree_bin(n):
    return '0' if n == 0 else '1' if n == 1 else '2-5' if n <= 5 else '>5'


def summary(rows, arm, baseline='RAW'):
    correct = sum(r['correct'][arm] for r in rows)
    base = sum(r['correct'][baseline] for r in rows)
    rescues = sum(r['correct'][arm] and not r['correct'][baseline] for r in rows)
    breaks = sum(not r['correct'][arm] and r['correct'][baseline] for r in rows)
    original_rescues = [r for r in rows if r['correct']['ORIGINAL_M'] and not r['correct']['RAW']]
    return dict(count=len(rows), correct=correct, baseline_correct=base, rescue=rescues,
        breaks=breaks, net=correct-base, original_mass_rescues=len(original_rescues),
        original_mass_rescues_retained=sum(r['correct'][arm] for r in original_rescues),
        new_rescues_beyond_original_mass=sum(r['correct'][arm] and not r['correct']['RAW'] and not r['correct']['ORIGINAL_M'] for r in rows),
        changed_decisions_vs_original_mass=sum(r['selected'][arm] != r['selected']['ORIGINAL_M'] for r in rows),
        switches_from_raw=sum(r['selected'][arm] != r['raw_winner'] for r in rows))


def table(rows):
    return {a: summary(rows, a) for a in ('RAW',)+ARMS}


def run():
    start = time.monotonic()
    need(not (OUT / 'result.json').exists(), 'FRESH_SIDECAR_REQUIRED')
    source_protocol = read(SOURCE / 'protocol.json')
    seal = read(SOURCE / 'all_predictions_prelabel_seal.json')
    need(seal['queries'] == 593 and seal['label_reads'] == 0, 'PRELABEL_SEAL')
    need(seal['protocol'] == bind(SOURCE / 'protocol.json'), 'SEAL_PROTOCOL')
    for v in seal['fold_validations']:
        checked(v)
    source_result = read(SOURCE / 'result.json')
    need(source_result['prelabel_seal'] == bind(SOURCE / 'all_predictions_prelabel_seal.json'), 'RESULT_SEAL')
    protocol = {
        'program': bind(__file__), 'source_protocol': bind(SOURCE / 'protocol.json'),
        'source_result': bind(SOURCE / 'result.json'), 'source_seal': bind(SOURCE / 'all_predictions_prelabel_seal.json'),
        'constant_M': 1.0, 'head': 'Identical frozen original per-fold COST1_PRODUCT5',
        'threshold': 0.0, 'tie': 'Original candidate axis argmax first; strict positive SWITCH else HOLD',
        'target_degree': 'Minimum TRAIN observations over correct-identity reference physical rows inside this natural C128',
        'degree_bins': ['0', '1', '2-5', '>5'],
        'stable_subset': 'Target present, all target physical rows degree>=5, RAW winner degree>=5, original-M top proposed challenger degree>=5',
        'predictors_fixed_before_sidecar': True, 'new_fits': 0, 'new_gpu_forwards': 0,
        'evidence': 'Post-hoc diagnostic on already opened development OOF predictions; not confirmatory selection',
    }
    write(OUT / 'protocol.json', protocol)
    p = source_protocol['sources']
    cache = read(checked(p['cache']))['rows']
    split = read(checked(p['split']))['folds']
    gallery = {r['physical_row']: r['identity'] for r in read(checked(p['gallery']))['records']}
    curator = {r['query_id']: r for r in read(checked(p['curator']))['records']}
    historical = {r['query_id']: r for r in read(checked(p['historical_simple_result']))['rows']}
    oldrows = {r['query_id']: r for r in source_result['rows']}
    rows = []
    maxerr = 0.
    comparisons = 0
    fitchecks = []
    for fold in range(5):
        effects = read(SOURCE / f'fold{fold}/effects.json')
        val = read(SOURCE / f'fold{fold}/validation.json')
        need(val['effects'] == bind(SOURCE / f'fold{fold}/effects.json'), 'EFFECTS_SEAL')
        checked(val['predictions'])
        trainids = set(split[fold]['train_query_ids'])
        heldids = set(split[fold]['heldout_query_ids'])
        need(not trainids & heldids, 'TRAIN_HELD_DISJOINT')
        train = [r for r in cache if r['query_id'] in trainids]
        need(set(effects['fit']['train_query_ids']) == trainids, 'TRAIN_AXIS')
        qeff = {x['query_id']: x['a'] for x in effects['query_effects']}
        reff = {x['physical_row']: x for x in effects['reference_effects']}
        mu = effects['mu']
        counts = np.zeros(5413, dtype=np.int64)
        ref_resid = np.zeros(5413, dtype=np.float64)
        query_resid = []
        for r in train:
            residual = []
            for g, mass in zip(r['axis'], r['mass']):
                e = math.log(max(mass, 1e-12))-mu-qeff[r['query_id']]-reff[g]['b']
                counts[g] += 1
                ref_resid[g] += e
                residual.append(e)
            query_resid.append(math.fsum(residual)/128)
        need(all(counts[g] == reff[g]['observations'] for g in range(5413)), 'DEGREE_RECOUNT')
        need(all(reff[g]['b'] == 0 for g in range(5413) if not counts[g]), 'UNKNOWN_REFERENCE_FALLBACK')
        maxq = max(map(abs, query_resid))
        maxref = float(np.max(np.abs(ref_resid[counts > 0]/counts[counts > 0])))
        need(max(maxq, maxref) < 1e-9, 'NORMAL_EQUATIONS_INDEPENDENT')
        fitchecks.append(dict(fold=fold, train_queries=len(train), train_pairs=int(counts.sum()),
            query_equation_error=maxq, reference_equation_error=maxref,
            unknown_ref_count=int(sum(counts == 0))))
        head = read(checked(source_protocol['heads'][str(fold)]))
        theta = [float.fromhex(x) for x in head['theta_hex']]
        for r in cache:
            if r['query_id'] not in heldids:
                continue
            identity = curator[r['query_id']]['identity']
            targetpos = [i for i, g in enumerate(r['axis']) if gallery[g] == identity]
            w = r['winner']
            need(r['axis'][w] == r['raw_ranked'][0], 'ORIGINAL_RAW_WINNER')
            degrees = [int(counts[g]) for g in r['axis']]
            original = oldrows[r['query_id']]
            need(degrees == original['reference_train_observations'], 'SOURCE_RESULT_DEGREES')
            ms = {
                'ORIGINAL_M': list(r['mass']),
                'REFERENCE_MAIN_EFFECT': [math.exp(mu+reff[g]['b']) for g in r['axis']],
                'REMOVE_REFERENCE_MAIN_EFFECT': [m/math.exp(reff[g]['b']) for m, g in zip(r['mass'], r['axis'])],
                'CONSTANT_M': [1.0]*128,
            }
            logits = {}
            selected = {'RAW': r['axis'][w]}
            tops = {}
            components = {}
            for arm, mass in ms.items():
                values = []
                terms = []
                for j, c in enumerate(r['challengers']):
                    free = r['free_content']
                    terms_i = [theta[0]*r['X'][j][0],
                        theta[1]*symmetric(mass[c]*free[c], mass[w]*free[w]),
                        theta[2]*symmetric(mass[c], mass[w]),
                        theta[3]*symmetric(free[c], free[w]), theta[4]]
                    values.append(math.fsum(terms_i))
                    terms.append(terms_i)
                topidx = max(range(127), key=lambda j: values[j])
                c = r['challengers'][topidx]
                pos = c if values[topidx] > 0 else w
                logits[arm] = values
                components[arm] = terms
                selected[arm] = r['axis'][pos]
                tops[arm] = dict(position=c, physical_row=r['axis'][c], degree=degrees[c],
                    logit=values[topidx], selected_position=pos, selected_degree=degrees[pos])
                if arm != 'CONSTANT_M':
                    old = original['models'][arm]
                    err = max(abs(x-y) for x, y in zip(values, old['logits']))
                    need(err < 2e-10 and selected[arm] == old['selected_physical_row'], 'INDEPENDENT_SCORE_ACTION_REPLAY')
                    comparisons += 127
                    maxerr = max(maxerr, err)
            correct = {arm: gallery[g] == identity for arm, g in selected.items()}
            need(correct['ORIGINAL_M'] == historical[r['query_id']]['correct']['COST1_PRODUCT5@zero'], 'ORIGINAL_CORRECT')
            mintarget = min((degrees[i] for i in targetpos), default=None)
            stable = bool(targetpos) and mintarget >= 5 and degrees[w] >= 5 and tops['ORIGINAL_M']['degree'] >= 5
            rows.append(dict(query_id=r['query_id'], execution_ordinal=r['execution_ordinal'], fold=fold,
                component=curator[r['query_id']]['component'], group=curator[r['query_id']]['group'],
                original_query_id=curator[r['query_id']]['original_query_id'],
                axis=r['axis'], challengers=r['challengers'], winner_position=w,
                raw_winner=r['axis'][w], target_identity=identity, target_in_C128=bool(targetpos),
                target_positions=targetpos, reference_degrees=degrees,
                target_degrees=[degrees[i] for i in targetpos], minimum_target_degree=mintarget,
                target_degree_bin='ABSENT' if mintarget is None else degree_bin(mintarget),
                winner_degree=degrees[w], winner_degree_bin=degree_bin(degrees[w]),
                original_top_challenger_degree=tops['ORIGINAL_M']['degree'],
                original_top_challenger_degree_bin=degree_bin(tops['ORIGINAL_M']['degree']),
                stable_target_winner_challenger_ge5=stable,
                all_128_references_ge5=all(x >= 5 for x in degrees),
                selected=selected, correct=correct, top_challengers=tops, logits=logits,
                logit_term_order=['dRAW','sym_M_times_L','sym_M','sym_L','bias'],
                logit_terms=components))
    need(len(rows) == len({r['query_id'] for r in rows}) == 593, 'OOF593')
    strata = {}
    for name in ('target', 'winner', 'original_top_challenger'):
        for degree in ('0', '1', '2-5', '>5'):
            strata[name + '_degree_' + degree] = [r for r in rows if r[name+'_degree_bin'] == degree]
    strata['target_absent'] = [r for r in rows if not r['target_in_C128']]
    strata['target_winner_original_top_challenger_ge5'] = [r for r in rows if r['stable_target_winner_challenger_ge5']]
    strata['all_128_references_ge5'] = [r for r in rows if r['all_128_references_ge5']]
    summary_all = table(rows)
    result = dict(status='INDEPENDENT_MASS_MAIN_EFFECTS_CONSTANT_AND_DEGREE_AUDIT_PASS',
        protocol=bind(OUT/'protocol.json'), sources=source_protocol['sources'],
        query_count=593, compared_source_logits=comparisons, maximum_logit_error=maxerr,
        constant_new_logits=593*127, summary=summary_all,
        comparisons_vs_original_mass={a: summary(rows,a,'ORIGINAL_M') for a in ARMS[1:]},
        strata={k: table(v) for k,v in strata.items()},
        strata_vs_original={k:{a:summary(v,a,'ORIGINAL_M') for a in ARMS[1:]} for k,v in strata.items()},
        fold_summary={str(f):table([r for r in rows if r['fold']==f]) for f in range(5)},
        group_summary={g:table([r for r in rows if r['component']==g]) for g in sorted({r['component'] for r in rows})},
        independent_train_fit_checks=fitchecks, seconds=time.monotonic()-start,
        limits=[
            'Fixed reference main effect is conditional on TRAIN natural-C128 exposures, not all reference-image predictable quality.',
            'Dividing M by this prior leaves query effects, pair variation, unmodelled structure and estimation error; it is not pure interaction.',
            'Held target appearances in TRAIN as other queries candidates do not mean target-label leakage; only unlabeled masses fit effects.',
            'Frozen-head comparison establishes behavior of this calibration; refitting could change conclusions.',
            'Degree strata are descriptive and overlapping; no significance claim or tuning from subgroup results.',
            'At constant M=1, product and content differences coincide: the frozen PRODUCT5 becomes a particular combined-content head, not an optimal trained content-only model.',
        ], rows=sorted(rows,key=lambda r:r['execution_ordinal']))
    write(OUT/'result.json',result)
    lines = [
        '# H593 M主效应分解：独立回放、恒定质量与参考观测数对照',
        '', '这是既有H593开发面板的事后机制诊断。原五折、自然C128、原冻结COST1_PRODUCT5、严格零阈值不变；不训练小头，不运行GPU。',
        '', f'独立从原缓存M/L/dRAW、TRAIN拟合主效应及原头参数重算 {comparisons:,} 个已有logit，最大差 {maxerr:.3g}；动作逐图一致。另补固定M=1的75,311个logit。',
        '', '|质量路径|正确/593|对RAW救回/损失|原57次救回保留|相对原M救回/损失|',
        '|---|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        v=summary_all[arm]; c=summary(rows,arm,'ORIGINAL_M')
        lines.append(f"|{arm}|{v['correct']}|{v['rescue']}/{v['breaks']}|{v['original_mass_rescues_retained']}/{v['original_mass_rescues']}|{c['rescue']}/{c['breaks']}|")
    lines += ['', '参考单位是图库physical row。每折仅用TRAIN全部query的无标签natural-C128质量拟合logM=mu+a_query+b_reference，2000次固定ALS。未知reference固定b=0；各折都通过独立观测数和最小二乘正规方程核查。',
        '', '## 排除低观测数主导的描述性分层',
        '', 'target采用该正确身份在C128内全部physical rows中的最小TRAIN观测数；top challenger是原M在127个挑战者中提出的最高分者，即使最终HOLD也保留。>=5子集同时要求target、RAW winner和原top challenger都达到5次，分箱2–5与>5的边界按原约定保留。',
        '', '|子集|n|RAW|原M|reference主效应|去reference主效应|恒定M|原纠错保留（ref / 去ref / 恒定）|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, sub in strata.items():
        t=table(sub)
        kept=' / '.join(f"{t[a]['original_mass_rescues_retained']}/{t[a]['original_mass_rescues']}" for a in ARMS[1:])
        values='|'.join(str(t[a]['correct']) for a in ('RAW',)+ARMS)
        lines.append(f'|{name}|{len(sub)}|{values}|{kept}|')
    lines += ['', '## 可支持的范围', '',
        '这些对照检查的是已估计的参考主效应是否解释原纠错，而不是证明所有单图质量先验均无效。natural-C128选择造成条件采样；未建模的单图属性仍可能留在去主效应后的信号里。', '',
        '去reference主效应后的M保留query整体尺度、未解释的配对变化及估计误差，不能直接命名为纯交互质量。query统一正比例尺度在epsilon=0时从相对M和相对M×L中抵消；原实验已量化实际epsilon误差及其动作影响。', '',
        '恒定M下，PRODUCT5中的乘积列和内容列合并成某个既定内容系数。这是冻结头消融，不是重新训练到最优的纯内容模型。', '',
        '同为481不意味着同一正确集合。任何下降均只说明冻结原头在该干预下如何变化，不能据此宣称该信息在任何可重训模型中不可替代。分组和观测数分层是描述性诊断，不产生新的独立确认或最优模型。', '',
        '完整逐候选分数、五项logit贡献、参考degree、动作和分组统计见同目录result.json。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'validation.json',dict(status=result['status'],result=bind(OUT/'result.json'),
        report=bind(OUT/'report_zh.md'),protocol=bind(OUT/'protocol.json'),
        maximum_logit_error=maxerr,compared_source_logits=comparisons,seconds=time.monotonic()-start))
    print(json.dumps(dict(status=result['status'],summary=summary_all,
        stable_subset=result['strata']['target_winner_original_top_challenger_ge5'],
        maximum_logit_error=maxerr,seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    run()
