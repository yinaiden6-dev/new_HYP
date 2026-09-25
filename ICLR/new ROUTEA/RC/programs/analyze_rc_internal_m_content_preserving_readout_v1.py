#!/usr/bin/env python3
"""Post-hoc score decomposition of the sealed content-preserving readout."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rc_internal_m_content_preserving_readout_v1'
REPORT = ROOT / 'reports/REPORT_INTERNAL_M_CONTENT_PRESERVING_READOUT_ANALYSIS_20260924.md'


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    authority = read(ROOT / 'registry/rc_internal_m_content_preserving_readout_v1_authority_20260924.json')
    manifest = read(authority['manifest']['path'])
    data = read(OUT / 'input_data.json')['rows']
    result = read(OUT / 'result.json')
    labels = {r['query_id']: r for r in result['rows']}
    theta = np.array(read(OUT / 'heads/PRESERVE_REAL.json')['theta'])
    rows = []
    for split in ('train', 'probe'):
        for row in manifest[split + '_rows']:
            q = row['query_id']
            d = read(OUT / 'predictions' / split / (q + '.json'))['predictions']['PRESERVE_REAL']
            winner = row['winner_index']
            ix = [j for j in range(128) if j != winner]
            raw = np.array(row['raw_scores'])
            old, new = (np.array(data[q][k]) for k in ('L0', 'REAL'))
            def sym(v):
                return (v[ix] - v[winner]) / (np.abs(v[ix]) + abs(v[winner]) + 1e-12)
            x = np.stack([(raw[ix] - raw[winner]) / max(float(raw.std()), 1e-12),
                          sym(old), sym(new) - sym(old), np.ones(127)], axis=1)
            scores = np.zeros(128)
            scores[ix] = x @ theta
            assert np.max(np.abs(scores - d['scores128'])) < 1e-12
            terms = np.zeros((128, 4))
            terms[ix] = x * theta
            best = ix[int(np.argmax(scores[ix]))]
            identity = labels[q]['identity']
            targets = [j for j, value in enumerate(row['candidate_identities']) if value == identity]
            item = dict(query_id=q, split=split, target_identity=identity,
                        raw_correct=labels[q]['correct']['RAW'], new_correct=labels[q]['correct']['PRESERVE_REAL'],
                        best_challenger_identity=row['candidate_identities'][best],
                        best_challenger_score=float(scores[best]), target_present=bool(targets))
            if targets:
                assert len(targets) == 1
                t = targets[0]
                wrong = max((j for j in range(128) if j != t), key=lambda j: scores[j])
                item.update(target_score=float(scores[t]), target_score_terms=terms[t].tolist(),
                            target_score_rank=1 + int((scores > scores[t]).sum()),
                            strongest_wrong_identity=row['candidate_identities'][wrong],
                            target_minus_strongest_wrong_terms=(terms[t] - terms[wrong]).tolist(),
                            target_minus_strongest_wrong=float(scores[t] - scores[wrong]),
                            original_target_minus_raw_winner=float(old[t] - old[winner]),
                            target_is_best_challenger=t == best,
                            L0_target_rank=1 + int((old > old[t]).sum()),
                            Ltheta_target_rank=1 + int((new > new[t]).sum()))
                assert abs(sum(item['target_score_terms']) - item['target_score']) < 1e-12
                assert abs(sum(item['target_minus_strongest_wrong_terms']) - item['target_minus_strongest_wrong']) < 1e-12
            rows.append(item)
    train_wrong = [r for r in rows if r['split'] == 'train' and not r['raw_correct']]
    assert len(train_wrong) == 8
    diagnostic = dict(status='SEALED_SCORE_DECOMPOSITION_PASS', exploratory=True,
                      feature_order=['dRAW', 'sym_L0', 'sym_Ltheta_minus_sym_L0', 'bias'], theta=theta.tolist(),
                      equivalent_basis=['dRAW', 'sym_L0', 'sym_Ltheta', 'bias'],
                      equivalent_theta=[float(theta[0]), float(theta[1] - theta[2]), float(theta[2]), float(theta[3])],
                      training_raw_wrong_queries=len(train_wrong),
                      training_raw_wrong_target_below_winner_on_L0=sum(r['original_target_minus_raw_winner'] < 0 for r in train_wrong),
                      training_raw_wrong_target_best_challenger=sum(r['target_is_best_challenger'] for r in train_wrong),
                      rows=rows, sources=[bind(OUT / p) for p in ('result.json', 'input_data.json', 'heads/PRESERVE_REAL.json')]
                      + [bind(authority['manifest']['path']), bind(__file__)])
    destination = OUT / 'score_decomposition.json'
    destination.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + '\n')
    infant = next(r for r in rows if r['query_id'] == 'H593-f522efe671aecfa850b10989')
    cef = next(r for r in rows if r['query_id'] == 'H593-90acde9b0567a472f4232120')
    lines = [
        '# 保留原内容后的读出：结果与仍存在的瓶颈', '',
        '本轮固定V4适配器、TRAIN16和ColNomic自然C128，只用CPU缓存训练4个小头。'
        '所有新头同warm起点、同2000次完整TRAIN16 COST1更新。PROBE8已经在历史开发中打开，'
        '本轮是事后开发检验，不是独立确认；没有新GPU前向。', '',
        '| 方法 | TRAIN16 | PROBE8 | probe救回/误伤 |', '|---|---:|---:|---:|',
    ]
    for arm in ('RAW', 'ORIGINAL3_CONTINUE', 'ADAPTED3_FROZEN_REFIT', 'PRESERVE_REAL',
                'PRESERVE_TRAIN_CONSTANT', 'PRESERVE_TRAIN_SHUFFLED',
                'PRESERVE_REAL_INFER_REAL_CONSTANT', 'PRESERVE_REAL_INFER_REAL_SHUFFLED', 'EXTERNAL_ADDITIVE4'):
        t, p = result['summary']['TRAIN16'][arm], result['summary']['PROBE8'][arm]
        lines.append(f"| {arm} | {t['correct']}/16 | {p['correct']}/8 | {len(p['rescues'])}/{len(p['breaks'])} |")
    lines += ['', '## 这次改变了什么', '',
        '原读出只看适配后的Ltheta；本轮分别读取L0和内部变化，末端仍不直接读取M。'
        '新头训练集3次纠错在推理时把M设常量或错绑后全部消失，因此真实内部M影响了实际动作。'
        '但probe各内部路径仍为4/8，尚无留出纠错。'
        '新头与旧ADAPTED3虽同为11/16，正确集合有一救一失，并未保住旧适配头全部正确。', '',
        '保留输入不等于保留其决策贡献。学习结果是：', '',
        f'`score = {theta[0]:.6f} dRAW + {theta[1]:.6f} sym(L0) + {theta[2]:.6f} [sym(Ltheta)-sym(L0)] {theta[3]:+.6f}`', '',
        f'等价于 `{theta[0]:.6f} dRAW {theta[1]-theta[2]:+.6f} sym(L0) + {theta[2]:.6f} sym(Ltheta) {theta[3]:+.6f}`。'
        '这只是同一线性函数的代数展开，不是另一个训练结果；不同参数化下AdamW正则与轨迹并不等价。'
        '负的展开系数不能单独证明机制错误，但说明“加回L0就能保护原内容”的保证并不存在。', '',
        '## 两个关键probe的分数分解', '',
        'Infants：正确候选与错误ChildsIbuprofen的竞争中，原内容项支持正确答案，'
        '内部变化项却更支持错误答案。正确减错误：', '',
        f"`dRAW {infant['target_minus_strongest_wrong_terms'][0]:+.6f} + L0 {infant['target_minus_strongest_wrong_terms'][1]:+.6f} + delta {infant['target_minus_strongest_wrong_terms'][2]:+.6f} = {infant['target_minus_strongest_wrong']:+.6f}`。",
        '因此仍选择错误挑战者。阈值只决定是否切换，不能修复这个候选次序。', '',
        'Cefdinir：正确候选已是最高挑战者，仍未跨过HOLD=0。', '',
        f"`dRAW {cef['target_score_terms'][0]:+.6f} + L0 {cef['target_score_terms'][1]:+.6f} + delta {cef['target_score_terms'][2]:+.6f} + bias {cef['target_score_terms'][3]:+.6f} = {cef['target_score']:+.6f}`。",
        '这是接受证据不足，与Infants的候选排序问题不同。不能依据这张已打开的probe手动下调阈值并称为泛化修复。', '',
        '## 能得出的结论', '',
        f"TRAIN的8张RAW错误中，{diagnostic['training_raw_wrong_target_below_winner_on_L0']}/8张的原始L0也把目标放在RAW winner之后；"
        f"新头把{diagnostic['training_raw_wrong_target_best_challenger']}/8张目标排成最高挑战者，最终接受其中3张。"
        '这符合训练在利用“调制带来的改善量”的观察；不足以证明数据分布是唯一根因。', '',
        '本轮排除了“只需把原始内容重新提供给末端，固定训练配方就能产生probe纠错”这一具体修复。'
        '它没有证明内部利用M不可能，也没有证明直接M不可替代。'
        '当前内部路径主要产生质量相关的内容分数变化，但其候选区分与接受尺度尚未在这8张开发probe上复现外部校准的2次纠错。', '',
        '如继续模型改造，需要改变内部调制的学习约束或训练覆盖，并在新的组上验证；'
        '本轮没有依据probe再选系数、阈值或启动新的GPU训练。', '',
        '独立复核从原始RAW/L0/L重新构造24张×8臂×127个分数、严格HOLD动作和身份计数，'
        '共24,384个挑战者分数全部通过；最大分数误差5.33e-15。'
        '验证器没有导入生产特征/决策函数，也没有用保存的X作为输入。'
        '验收见同目录`independent_audit/validation.json`。', '',
        f'数据：`{destination.relative_to(ROOT)}`；逐候选输入、特征、127个logits和HOLD=0均在同一结果目录。', '',
    ]
    REPORT.write_text('\n'.join(lines))
    (OUT / 'score_decomposition_validation.json').write_text(json.dumps(
        dict(status='SEALED_SCORE_DECOMPOSITION_PASS', data=bind(destination), report=bind(REPORT)), indent=2) + '\n')
    print(REPORT)


if __name__ == '__main__':
    main()
