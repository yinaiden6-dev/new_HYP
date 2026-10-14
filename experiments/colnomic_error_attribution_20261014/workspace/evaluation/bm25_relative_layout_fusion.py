"""Add an explicit pairwise relative-coordinate graph bonus to ColNomic 7B + BM25."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .bm25_image_fusion import load_source
from .cache import save_tensor
from .data import Dataset, digest, file_hash, read_csv, read_json, write_json
from .metrics import evaluate, paired_report
from .ocr_relative_layout import extract_relative_lines, relative_layout_score as layout_score, reference_idf
from .scoring import fuse, tokenize

ROOT = Path(__file__).resolve().parents[1]


def candidate_rows(scores, references, k):
    """Take k byte-distinct candidates; copy any bonus to their duplicate rows."""
    groups = {}
    for i, row in enumerate(references):
        groups.setdefault(row['sha256'], []).append(i)
    candidates = []
    for ranking in np.argsort(-scores.numpy(), axis=1, kind='stable'):
        selected, seen = [], set()
        for i in ranking:
            sha = references[i]['sha256']
            if sha not in seen:
                seen.add(sha); selected.append(int(i))
                if len(selected) == k:
                    break
        candidates.append(selected)
    return candidates, groups


def apply_layout(base, candidates, values, references, weight):
    result = base.clone()
    groups = {}
    for i, ref in enumerate(references):
        groups.setdefault(ref['sha256'], []).append(i)
    for qi, (indices, bonuses) in enumerate(zip(candidates, values, strict=True)):
        for ri, bonus in zip(indices, bonuses, strict=True):
            result[qi, groups[references[ri]['sha256']]] += weight * float(bonus)
    return result


def register_result(root, config, metrics):
    path = root / 'results/all_results.csv'
    rows = read_csv(path)
    fields = list(rows[0])
    row = dict(id=config['id'], family='bm25_layout_fusion', model=config['model'],
               mode=config['mode'], queries=config['queries'], references=config['references'],
               result=config['result'], **metrics['photo_average'])
    rows = [r for r in rows if r['id'] != config['id']] + [row]
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def verify_result(root, dataset, folder):
    """Check reconstruction, hashes and independent NumPy byte-deduplicated ranks."""
    from .foreground import verify_rankings
    config = read_json(folder / 'config.json')
    recipe = config['scoring']
    assert config['dataset_sha256'] == dataset.hashes
    assert config['annotation_sha256'] == dataset.annotation_sha256
    assert config['active_difference'] is None
    assert config['id'] == recipe['settings']['id']
    for path, sha in recipe['input_sha256'].items():
        assert file_hash(root / path) == sha, path
    for path, sha in recipe['code_sha256'].items():
        assert file_hash(root / path) == sha, path
    value = torch.load(root / config['score_path'], map_location='cpu', weights_only=True)
    base = torch.load(root / recipe['base_score_path'], map_location='cpu', weights_only=True)
    assert value['signature'] == config['score_signature'] == digest(recipe)
    assert value['recipe'] == recipe and base['signature'] == recipe['base_signature']
    assert value['query_ids'] == [r['photo_id'] for r in dataset.photos]
    assert value['reference_ids'] == [r['reference_id'] for r in dataset.references]
    candidates, groups = candidate_rows(base['scores'], dataset.references, recipe['settings']['candidate_k'])
    assert candidates == value['candidate_indices'].tolist()
    features = value['layout_scores'].numpy()
    assert np.isfinite(features).all() and (features >= 0).all() and (features <= 1).all()
    edges = value['relative_relation_edges'].numpy()
    assert recipe['absolute_coordinates_used'] is False and recipe['relative_coordinates_required'] is True
    assert edges.shape == features.shape and (edges >= 0).all()
    assert np.array_equal(features > 0, edges > 0)
    assert (value['matched_lines'].numpy()[features > 0] >= recipe['settings']['minimum_matched_lines']).all()
    # Reconstruct in NumPy, separately from the torch update used by the runner.
    expected = base['scores'].numpy().copy()
    for qi, indices in enumerate(candidates):
        for ci, ri in enumerate(indices):
            for duplicate in groups[dataset.references[ri]['sha256']]:
                expected[qi, duplicate] += recipe['settings']['layout_weight'] * float(features[qi, ci])
    np.testing.assert_allclose(value['scores'].numpy(), expected, rtol=0, atol=1e-6)
    rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    verify_rankings(value['scores'], rows, dataset)
    reranked, metrics, by_type = evaluate(value['scores'], dataset)
    assert reranked == rows and metrics == read_json(folder / 'metrics.json')
    assert by_type == read_json(folder / 'by_type.json')
    return dict(status='passed', independently_verified_ranks=True,
                independently_reconstructed_scores=True, queries=len(rows),
                references=len(dataset.references), coordinate_feature_range=[float(features.min()), float(features.max())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    settings_path = root / 'configs/bm25_relative_layout_fusion.json'
    settings = read_json(settings_path)
    dataset = Dataset(root)
    folder = root / settings['result']
    torch.set_num_threads(2)
    if args.verify_only:
        print(json.dumps(verify_result(root, dataset, folder), ensure_ascii=False), flush=True)
        return
    if dataset.active_difference is not None:
        raise ValueError('Resolve the active/formal inventory difference before running')
    if args.workers < 1 or settings['candidate_k'] < 1 or settings['layout_weight'] < 0:
        raise ValueError('Invalid workers, candidate count or layout weight')
    base_folder = root / settings['base_result']
    base_config = read_json(base_folder / 'config.json')
    assert base_config['id'] == settings['base_id']
    assert base_config['model'] == settings['model']
    assert dataset.matches_full_inventory(base_config['dataset_sha256'])
    assert base_config['annotation_sha256'] == dataset.annotation_sha256
    base_path = root / base_config['score_path']
    base = torch.load(base_path, map_location='cpu', weights_only=True)
    assert base['signature'] == base_config['score_signature'] == digest(base_config['scoring'])
    assert base['query_ids'] == [r['photo_id'] for r in dataset.photos]
    assert base['reference_ids'] == [r['reference_id'] for r in dataset.references]
    image, image_predictions, image_source = load_source(dataset, settings['model'], 'image')
    bm25, bm_predictions, bm_source = load_source(dataset, 'bm25', 'bm25')
    assert base_config['sources'] == dict(image=image_source, bm25=bm_source)
    photos = read_json(root / 'cache/ocr/photos.json')
    references = read_json(root / 'cache/ocr/references.json')
    torch.testing.assert_close(base['scores'], fuse(image['scores'], bm25['scores'],
                               [bool(tokenize(photos[r['photo_id']]['text'])) for r in dataset.photos], .5),
                               rtol=0, atol=0)
    for population, records, key in [(photos, dataset.photos, 'photo_id'),
                                      (references, dataset.references, 'reference_id')]:
        for row in records:
            assert population[row[key]]['image_sha256'] == row['sha256']
    input_paths = [settings['base_result'] + '/' + name for name in
                   ('config.json', 'predictions.jsonl', 'metrics.json')]
    input_paths += [str(base_path.relative_to(root)), 'cache/ocr/photos.json',
                    'cache/ocr/references.json', 'configs/bm25_relative_layout_fusion.json']
    recipe = dict(method='pairwise_relative_ocr_graph_bonus_v1', settings=settings,
                  base_score_path=str(base_path.relative_to(root)), base_signature=base['signature'],
                  input_sha256={p: file_hash(root / p) for p in input_paths},
                  code_sha256={p: file_hash(root / p) for p in
                               ('evaluation/ocr_layout.py', 'evaluation/ocr_relative_layout.py', 'evaluation/bm25_relative_layout_fusion.py',
                                'evaluation/scoring.py', 'evaluation/metrics.py')},
                  geometry='explicit directed neighbor graphs: relative dx/dy, above/below/left/right, relative width/height and direction; one-to-one text-line mapping',
                  formula='F03 + layout_weight * IDF_weighted_consistent_relative_edges_fraction',
                  coordinates='pairwise displacements in the source line reading frame divided by geometric-mean line height; absolute coordinates discarded',
                  absolute_coordinates_used=False, relative_coordinates_required=True,
                  labels='used only for evaluation; no subject boxes or identity labels in scoring',
                  tokenizer='unchanged BM25 tokenizer; no new numeric normalization',
                  missing_geometry='zero bonus; retain the F03 score',
                  candidate_selection='top 50 unique image SHA-256 candidates from F03; duplicate rows share bonus')
    signature = digest(recipe)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'recipe.json', recipe)
    started = time.monotonic()
    idf = reference_idf([references[r['reference_id']] for r in dataset.references])
    reference_lines = [extract_relative_lines(references[r['reference_id']], idf, settings) for r in dataset.references]
    query_lines = [extract_relative_lines(photos[r['photo_id']], idf, settings) for r in dataset.photos]
    candidates, _ = candidate_rows(base['scores'], dataset.references, settings['candidate_k'])
    preprocessing_seconds = time.monotonic() - started
    print(json.dumps(dict(stage='prepared', queries=len(dataset.photos), candidates=settings['candidate_k'],
                          layout_weight=settings['layout_weight'], recipe_signature=signature)), flush=True)
    started = time.monotonic()

    def score_query(qi):
        return [layout_score(query_lines[qi], reference_lines[ri], idf, settings) for ri in candidates[qi]]

    layouts, matched, relative_edges = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for qi, result in enumerate(executor.map(score_query, range(len(query_lines)))):
            layouts.append([r[0] for r in result]); matched.append([r[1] for r in result]); relative_edges.append([r[2] for r in result])
            if (qi + 1) % 25 == 0 or qi + 1 == len(query_lines):
                progress = dict(stage='relative_layout', done=qi + 1, total=len(query_lines),
                                elapsed_seconds=round(time.monotonic() - started, 1))
                write_json(folder / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    scores = apply_layout(base['scores'], candidates, layouts, dataset.references, settings['layout_weight'])
    scoring_seconds = time.monotonic() - started
    score_path = root / settings['score_path']
    save_tensor(score_path, dict(scores=scores, signature=signature, recipe=recipe,
                                query_ids=[r['photo_id'] for r in dataset.photos],
                                reference_ids=[r['reference_id'] for r in dataset.references],
                                candidate_indices=torch.tensor(candidates, dtype=torch.int64),
                                layout_scores=torch.tensor(layouts, dtype=torch.float64),
                                matched_lines=torch.tensor(matched, dtype=torch.int32),
                                relative_relation_edges=torch.tensor(relative_edges, dtype=torch.int32)))
    started = time.monotonic()
    predictions, metrics, by_type = evaluate(scores, dataset)
    base_predictions = [json.loads(line) for line in (base_folder / 'predictions.jsonl').read_text().splitlines()]
    previous_predictions = [json.loads(line) for line in (root / 'results/bm25_layout_fusion/colnomic7b/fixed_v1/predictions.jsonl').read_text().splitlines()]
    comparisons = dict(versus_l01=paired_report(predictions, previous_predictions), versus_f03=paired_report(predictions, base_predictions),
                       versus_bm25=paired_report(predictions, bm_predictions),
                       versus_image=paired_report(predictions, image_predictions))
    evaluation_seconds = time.monotonic() - started
    config = dict(id=settings['id'], model=settings['model'], mode='bm25_relative_layout_fusion',
                  protocol=dataset.full_protocol, queries=len(dataset.photos), references=len(dataset.references),
                  dataset_sha256=dataset.hashes, annotation_sha256=dataset.annotation_sha256,
                  active_difference=None, result=settings['result'], score_path=settings['score_path'],
                  scoring=recipe, score_signature=signature, labels_used_only_for_evaluation=True,
                  parameter_selection=settings['selection'], model_initializations=0,
                  generated_embeddings=0, generated_ocr=0,
                  created_utc=datetime.now(timezone.utc).isoformat(), workers=args.workers,
                  preprocessing_seconds=preprocessing_seconds, scoring_seconds=scoring_seconds,
                  evaluation_seconds=evaluation_seconds,
                  runner_sha256=file_hash(Path(__file__)),
                  timing_note='Cached inputs only; excludes OCR, image encoding and base scoring')
    for name, value in [('config', config), ('metrics', metrics), ('by_type', by_type),
                        ('paired_comparisons', comparisons)]:
        write_json(folder / (name + '.json'), value)
    (folder / 'predictions.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in predictions))
    changed = []
    for qi, (new, old) in enumerate(zip(predictions, base_predictions, strict=True)):
        if new['selected_reference_id'] != old['selected_reference_id']:
            changed.append(dict(photo_id=new['photo_id'], package_id=new['package_id'],
                old_reference_id=old['selected_reference_id'], new_reference_id=new['selected_reference_id'],
                old_rank=old['rank'], new_rank=new['rank'],
                transition='rescued' if new['correct'] else 'harmed' if old['correct'] else 'still_wrong',
                candidate_layout={dataset.references[ri]['reference_id']: dict(score=layouts[qi][ci], matched_lines=matched[qi][ci], relative_relation_edges=relative_edges[qi][ci])
                                  for ci, ri in enumerate(candidates[qi]) if dataset.references[ri]['reference_id'] in
                                  (old['selected_reference_id'], new['selected_reference_id'])}))
    write_json(folder / 'changed_top1.json', changed)
    verification = verify_result(root, dataset, folder)
    write_json(folder / 'verification.json', verification)
    register_result(root, config, metrics)
    r1 = metrics['photo_average']['recall_at_1']
    before = read_json(base_folder / 'metrics.json')['photo_average']['recall_at_1']
    pair = comparisons['versus_f03']; ci = pair['intervals']['photo_average']['recall_at_1']
    lines = ['# L02：ColNomic 7B + BM25 + 显式相对坐标', '',
             f"Full-{len(dataset.photos)}，{len(dataset.references)} 条参考图；固定参数、无训练、无权重搜索。", '',
             '| 配置 | R@1 | R@5 | R@10 | MRR |', '|---|---:|---:|---:|---:|']
    previous_metrics = read_json(root / 'results/bm25_layout_fusion/colnomic7b/fixed_v1/metrics.json')
    for name, report in [('F03', read_json(base_folder / 'metrics.json')), ('L01', previous_metrics), ('L02', metrics)]:
        v = report['photo_average']
        lines.append(f"| {name} | {v['recall_at_1']:.2%} | {v['recall_at_5']:.2%} | {v['recall_at_10']:.2%} | {v['mrr']:.4f} |")
    lines += ['', f"相对 F03：R@1 {100*(r1-before):+.2f} 个百分点；救回 {pair['rescued']} 张，新增错误 {pair['harmed']} 张。",
              f"按来源组重采样的差值 95% CI：[{100*ci['low']:+.2f}, {100*ci['high']:+.2f}] 个百分点。", '',
              '每条文字行连接最近的八条文字行，建立有向关系图。对于文字行 i、j：', '',
              '- 相对位置：把中心差 c_j-c_i 投影到 i 的阅读方向与垂直方向，再除以 sqrt(h_i*h_j)。',
              '- 相对尺寸：log(w_j/w_i)、log(h_j/h_i)。',
              '- 相对方向：两行阅读方向之间的夹角；同时约束左右和上下顺序。', '',
              '构图后丢弃绝对中心、绝对宽高和全图方向；评分只读取相对关系。',
              '查询和参考分别整体平移、均匀缩放或旋转，不改变关系特征。',
              '先建立一对一文字行对应，最终只给对应文字行之间一致的相对关系计分。',
              '至少需要三行组成相对关系连通片段；文字匹配而没有相对关系支持时，坐标分为零。',
              '坐标分 = 一致关系的 IDF 加权得分 / 查询全部邻接关系的 IDF 权重；范围 0–1。',
              '最终分数 = F03 + 0.5 × 相对关系分，候选仍为 F03 前 50 个去重参考图。', '',
              'OCR、分词、ColNomic 7B 编码和 F03 的图文融合权重保持原样。未使用人工主体框。',
              '关系归一化不完整处理透视、文字分行变化和多盒面折叠；多物体文字仍可能参与匹配。',
              '完整队列包含已查看的开发错例，结果是探索性对照，不是独立测试结论。', '',
              '[参数](../../../../configs/bm25_relative_layout_fusion.json) · [逐图 Top-1 变化](changed_top1.json) · [独立核验](verification.json)', '',
              '复现：`python -B -m drugpack relative-layout-fusion`；核验：`python -B -m drugpack relative-layout-fusion --verify-only`', '']
    (folder / 'REPORT.md').write_text('\n'.join(lines))
    write_json(folder / 'progress.json', dict(stage='complete', queries=len(predictions), recall_at_1=r1))
    print(json.dumps(dict(id=settings['id'], recall_at_1=r1, delta_vs_f03=r1-before,
                          rescued=pair['rescued'], harmed=pair['harmed'], verification=verification)), flush=True)


if __name__ == '__main__':
    main()
