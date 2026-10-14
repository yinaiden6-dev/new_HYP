"""Count success/failure by current capture labels for all saved configurations.

Failure types always use automatic E0--E7 rules, before human calibration.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset/tools'))
from error_analysis import Analysis, MAIN, file_hash, write_csv, write_json
from photo_metadata import PhotoStore

DIMENSIONS = {
    'background': [('multi_object', '多物体'), ('single_object', '单体')],
    'placement': [('handheld', '手持'), ('tabletop', '桌面')],
    'occlusion': [('foreground', '有前景遮挡'), ('none', '无前景遮挡')],
}
DIMENSION_NAMES = {'background': '物体数量', 'placement': '拍摄方式', 'occlusion': '前景遮挡'}
OTHER_NAMES = {'other': '其他', 'unclear': '不明确', 'unlabeled': '未标注'}
STATUS = 'automatic_rules_before_human_calibration'


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def group_statistics(cid, cfg, dimension, label, label_zh, rows, total_photos, total_errors):
    successes = sum(r['correct'] for r in rows)
    errors = len(rows) - successes
    counts = Counter(r['error_type'] for r in rows if not r['correct'])
    if sum(counts.values()) != errors or not counts.keys() <= MAIN.keys():
        raise ValueError('Every failed prediction must have an automatic main error type')
    result = dict(config_id=cid, model=cfg['model'], mode=cfg['mode'], dimension=dimension,
        label=label, label_zh=label_zh, photos=len(rows), successes=successes, errors=errors,
        success_rate=ratio(successes, len(rows)), error_rate=ratio(errors, len(rows)),
        error_share_of_all_errors=ratio(errors, total_errors),
        scene_share_of_all_photos=ratio(len(rows), total_photos),
        all_photos=total_photos, all_errors=total_errors, failure_label_status=STATUS)
    failure_types = [dict(config_id=cid, model=cfg['model'], mode=cfg['mode'],
        dimension=dimension, label=label, label_zh=label_zh, error_type=code,
        error_type_zh=title, type_errors=counts[code], scene_photos=len(rows),
        scene_errors=errors, all_errors=total_errors,
        type_share_of_scene_errors=ratio(counts[code], errors),
        type_rate_among_scene_photos=ratio(counts[code], len(rows)),
        type_share_of_all_errors=ratio(counts[code], total_errors), failure_label_status=STATUS)
        for code, title in MAIN.items()]
    return result, failure_types


def write_table(lines, headers, rows):
    lines.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines.append('')


def fraction(count, denominator):
    return f'{count}/{denominator}（{100 * count / denominator:.2f}%）' if denominator else '—（分母为 0）'


def write_report(output, catalog, scene_counts, groups, type_rows, reviewed):
    n = len(catalog.photos)
    lookup = {(r['config_id'], r['dimension'], r['label']): r for r in groups}
    failure_lookup = {(r['config_id'], r['dimension'], r['label'], r['error_type']): r for r in type_rows}
    lines = [f'# {len(catalog.configs)} 个配置的场景成功率、失败率与自动失败标签', '',
        f'统计正式队列的全部 {n:,} 张图片，成功指 Top-1 命中真值包装设计。'
        '参考图按原评测的 SHA 合并规则处理。失败主类型使用规则建议，不使用人工校准后的错误类型。', '',
        '三个场景维度分别统计：同一张图片在物体数量、拍摄方式、前景遮挡三个维度各出现一次，不能把六类数量相加当作图片总数。', '',
        '## 图片数量与分母', '']
    write_table(lines, ['维度', '标签', '图片数', '占全部图片'], [
        [DIMENSION_NAMES[d], name, scene_counts[d][label], f'{100 * scene_counts[d][label] / n:.2f}%']
        for d, labels in DIMENSIONS.items() for label, name in labels])
    lines += ['- 类内成功率 = 该类 Top-1 正确数 / 该类全部图片数。',
        '- 类内失败率 = 该类 Top-1 错误数 / 该类全部图片数，等于 1 − 类内成功率。',
        '- 错误构成比例 = 该类错误数 / 该配置全部错误数。',
        '- 失败类型在类内错误的占比 = 该类某 E 类型错误数 / 该类全部错误数。',
        '- 失败类型在类内图片的发生率 = 该类某 E 类型错误数 / 该类全部图片数。',
        '- 分母为 0 时记为空值，表示不适用。', '',
        '## 多物体与单体：各自内部的成功率、失败率', '']
    for d, labels in DIMENSIONS.items():
        if d != 'background':
            lines += [f'## {DIMENSION_NAMES[d]}：各自内部的成功率、失败率', '']
        headers = ['配置']
        for label, name in labels:
            headers += [f'{name}成功 / {scene_counts[d][label]}', f'{name}失败 / {scene_counts[d][label]}']
        values = []
        for cid in catalog.configs:
            values.append([cid] + [fraction(lookup[cid, d, label][kind], lookup[cid, d, label]['photos'])
                for label, _ in labels for kind in ['successes', 'errors']])
        write_table(lines, headers, values)
    lines += ['## 失败内部的场景构成', '',
        '本表每个百分比的分母都是该配置的全部错误数；它与上面的类内失败率不同。', '']
    headers = ['配置', '全部错误数'] + [name + '占全部错误' for pairs in DIMENSIONS.values() for _, name in pairs]
    values = []
    for cid in catalog.configs:
        overall = lookup[cid, 'overall', 'all']
        values.append([cid, overall['errors']] + [fraction(lookup[cid, d, label]['errors'], overall['errors'])
            for d, labels in DIMENSIONS.items() for label, _ in labels])
    write_table(lines, headers, values)
    lines += ['## 未人工校准的主失败标签', '',
        '优先级为 E0/E1 → E4/E5/E6 → E2/E3 → E7。融合类型按对应图像、文字基础通道的正确性判断，'
        '并优先于排名型错误；E2 表示全图库排名在 Top-10 之外。', '',
        'E0/E1 等数据问题的规则建议不等于人工审查结论。包装混淆关系 C0–C7、可见面、表观原因尚未在本轮自动推断。', '']
    write_table(lines, ['代码', '主失败标签'], [[code, name] for code, name in MAIN.items()])
    headers = ['配置', '全部错误数'] + list(MAIN)
    write_table(lines, headers, [[cid, lookup[cid, 'overall', 'all']['errors']] + [
        failure_lookup[cid, 'overall', 'all', e]['type_errors'] for e in MAIN] for cid in catalog.configs])
    lines += ['## 各场景失败内部的类型分布', '',
        '以下分母为该场景的错误数。各行 E0–E7 计数之和等于该场景错误数；详细比例见 `failure_types.csv`。', '']
    for d, labels in DIMENSIONS.items():
        for label, name in labels:
            lines += [f'### {name}错误：E0–E7 计数及该类错误内部占比', '']
            write_table(lines, ['配置', f'{name}错误数'] + list(MAIN), [[cid, lookup[cid, d, label]['errors']] + [
                fraction(failure_lookup[cid, d, label, e]['type_errors'], lookup[cid, d, label]['errors']) for e in MAIN] for cid in catalog.configs])
    lines += ['## 输出与复现', '',
        '- `summary.csv`：六类场景及整体的计数、类内成功／失败率、错误构成比例。',
        '- `background.csv`、`placement.csv`、`occlusion.csv`：每个配置一行的宽表。',
        '- `failure_types.csv`：每个场景失败内部 E0–E7 的数量和比例。',
        '- `failed_photos.csv`：全部失败预测的图片 ID、场景标签、自动主失败类型和去重错误对 ID。',
        '- `joint_statistics.csv`：拍摄方式 × 物体数量 × 遮挡的交叉统计。',
        '- `photo_labels.csv`：本轮使用的场景标签快照。',
        '- `manifest.json`：输入文件哈希、标签数量与分类来源。', '',
        '```bash', '/home/zyin/envs/labelstudio/bin/python -B scripts/summarize_scene_statistics.py', '```', '',
        f'场景标签使用当前正式元数据，不过滤人工审核状态；场景 reviewed=true 为 {reviewed} 张。'
        '失败标签全部为未人工校准的规则标签。统计不修改场景元数据、真值或人工错误标注。', '',
        '## 配置含义', '']
    write_table(lines, ['配置', '模型', '检索形式'], [[cid, c['model'], c['mode']] for cid, c in catalog.configs.items()])
    (output / 'README.md').write_text('\n'.join(lines))


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    catalog = Analysis(root)
    store = PhotoStore(root / 'dataset')
    with store.locked():
        records = store.records()
        if records.keys() != catalog.photos.keys():
            raise ValueError('Capture labels and predictions must cover the same photo inventory')
        scene_counts = {d: Counter(p['type'].get(d) or 'unlabeled' for p in records.values()) for d in DIMENSIONS}
        inputs = dict(catalog.inputs)
        for name in ['dataset/lists/packages.csv'] + ['dataset/' + r['metadata_path'] for r in store.packages.values()]:
            inputs[name] = file_hash(root / name)
        groups, types, joint, failures = [], [], [], []
        label_differences = 0
        for cid, cfg in catalog.configs.items():
            rows = []
            for pid, p in catalog.predictions[cid].items():
                if type(p['correct']) is not bool:
                    raise ValueError('Invalid correctness field')
                tags = {d: records[pid]['type'].get(d) or 'unlabeled' for d in DIMENSIONS}
                label_differences += any(p.get('type', {}).get(d) != records[pid]['type'].get(d) for d in DIMENSIONS)
                code = catalog.auto_main(cid, pid)
                row = dict(photo_id=pid, correct=p['correct'], error_type=code, **tags)
                rows.append(row)
                if not p['correct']:
                    if code not in MAIN:
                        raise ValueError('A failure needs an automatic error type')
                    failures.append(dict(config_id=cid, model=cfg['model'], mode=cfg['mode'], photo_id=pid,
                        package_id=p['package_id'], correct_rank=p.get('rank'),
                        wrong_reference_id=p.get('selected_reference_id'), wrong_design_id=p['selected_design_id'],
                        correct_reference_ids='|'.join(p.get('positive_reference_ids', [])),
                        case_id=pid + '__' + (p['selected_design_id'] or 'missing_reference'), **tags,
                        error_type=code, error_type_zh=MAIN[code], failure_label_status=STATUS))
            total_errors = sum(not r['correct'] for r in rows)
            g, t = group_statistics(cid, cfg, 'overall', 'all', '全部', rows, len(rows), total_errors)
            groups.append(g); types.extend(t)
            for d, pairs in DIMENSIONS.items():
                names = dict(pairs)
                for label in sorted(scene_counts[d].keys() - names.keys()):
                    names[label] = OTHER_NAMES.get(label, label)
                partition = []
                for label, name in names.items():
                    g, t = group_statistics(cid, cfg, d, label, name, [r for r in rows if r[d] == label], len(rows), total_errors)
                    groups.append(g); types.extend(t); partition.append(g)
                if sum(g['photos'] for g in partition) != len(rows) or sum(g['errors'] for g in partition) != total_errors:
                    raise ValueError('Scene partition does not cover all predictions')
            combinations = itertools.product(*[list(dict(DIMENSIONS[d])) + sorted(scene_counts[d].keys() - dict(DIMENSIONS[d]).keys())
                                               for d in ['placement', 'background', 'occlusion']])
            for values in combinations:
                selected = [r for r in rows if tuple(r[d] for d in ['placement', 'background', 'occlusion']) == values]
                g, _ = group_statistics(cid, cfg, 'joint', '/'.join(values), '/'.join(values), selected, len(rows), total_errors)
                joint.append(g)
        if any(file_hash(root / name) != expected for name, expected in inputs.items()):
            raise ValueError('Inputs changed while computing scene statistics; rerun for one consistent snapshot')
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'summary.csv', groups, list(groups[0]))
        write_csv(output / 'failure_types.csv', types, list(types[0]))
        write_csv(output / 'joint_statistics.csv', joint, list(joint[0]))
        if failures:
            write_csv(output / 'failed_photos.csv', failures, list(failures[0]))
        labels = [dict(photo_id=pid, package_id=p['package_id'], **{d: p['type'].get(d) or 'unlabeled' for d in DIMENSIONS},
            scene_reviewed=p['reviewed'], excluded=p['excluded']) for pid, p in sorted(records.items())]
        write_csv(output / 'photo_labels.csv', labels, list(labels[0]))
        for d, pairs in DIMENSIONS.items():
            wide = []
            index = {(g['config_id'], g['dimension'], g['label']): g for g in groups}
            for cid, cfg in catalog.configs.items():
                overall = index[cid, 'overall', 'all']
                row = dict(config_id=cid, model=cfg['model'], mode=cfg['mode'], all_photos=len(records), all_errors=overall['errors'])
                for label, _ in pairs:
                    g = index[cid, d, label]
                    for key in ['photos', 'successes', 'errors', 'success_rate', 'error_rate', 'error_share_of_all_errors']:
                        row[label + '_' + key] = g[key]
                wide.append(row)
            write_csv(output / f'{d}.csv', wide, list(wide[0]))
        reviewed = sum(p['reviewed'] for p in records.values())
        manifest = dict(schema='drugpack.scene_success_failure.v1', created_utc=datetime.now(timezone.utc).isoformat(),
            configurations=len(catalog.configs), photos=len(records), scene_counts={d: dict(c) for d, c in scene_counts.items()},
            failure_occurrences=len(failures), failure_label_status=STATUS, scene_reviewed_photos=reviewed,
            source_prediction_label_differences=label_differences, main_type_priority='E0/E1 → E4/E5/E6 → E2/E3 → E7',
            inputs=inputs, output_sha256={p.name: file_hash(p) for p in output.glob('*.csv')},
            denominators=dict(success_rate='successes / scene_photos', error_rate='errors / scene_photos',
                error_share_of_all_errors='scene_errors / all_errors', type_share_of_scene_errors='type_errors / scene_errors',
                type_rate_among_scene_photos='type_errors / scene_photos', type_share_of_all_errors='type_errors / all_errors'))
        write_json(output / 'manifest.json', manifest)
        write_report(output, catalog, scene_counts, groups, types, reviewed)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.root, args.output or args.root / 'results/scene_statistics')
    print(json.dumps({k: result[k] for k in ['configurations', 'photos', 'scene_counts', 'failure_occurrences',
        'failure_label_status', 'source_prediction_label_differences']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
