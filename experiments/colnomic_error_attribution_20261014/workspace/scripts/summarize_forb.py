#!/usr/bin/env python3
"""Export FORB results and distinguish summed worker time from elapsed wall time."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import file_hash, read_json

DATA = ROOT / 'other_datasets/FORB'
MODELS = ('dinov3', 'siglip2', 'Qwen3-VL-Embedding-2B', 'colnomic7b', 'bm25')
LABELS = {'dinov3': 'DINOv3 ViT-L/16', 'siglip2': 'SigLIP2 Large/16-256',
          'Qwen3-VL-Embedding-2B': 'Qwen3-VL-Embedding-2B',
          'colnomic7b': 'ColNomic 7B + mean MaxSim', 'bm25': 'PP-OCRv6 + BM25'}


def result_path(data, model):
    return Path(data) / 'results' / model / ('bm25' if model == 'bm25' else 'image')


def ready(data, model):
    directory = result_path(data, model)
    paths = [directory / name for name in ('config.json', 'metrics.json', 'predictions.jsonl')]
    paths.extend(Path(data) / 'cache/scores' / model / name for name in ('scores.npy', 'metadata.json'))
    return all(path.is_file() for path in paths)


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def verified(data, model, metadata):
    directory = result_path(data, model)
    path = directory / 'verification.json'
    if not path.exists():
        return False
    value = read_json(path)
    return bool(value.get('passed') and value.get('score_sha256') == metadata['scores_sha256']
        and value.get('score_metadata_sha256') == file_hash(data / 'cache/scores' / model / 'metadata.json')
        and value.get('verifier_sha256') == file_hash(ROOT / 'scripts/verify_forb.py')
        and all(value.get('result_files', {}).get(name) == file_hash(directory / name)
                for name in ('config.json', 'metrics.json', 'predictions.jsonl')))


def timing(data, model, metadata):
    score_seconds = float(metadata['scoring_seconds'])
    hosts = sorted({chunk['worker']['host'] for chunk in metadata['chunks']})
    common = dict(model=model, unique_encoded_images=None, encoding_worker_seconds_sum=None,
        ocr_worker_inference_seconds_sum=None, model_loading_seconds_observed_sum=None,
        processing_wall_span_seconds_observed=None, processing_wall_span_complete=False,
        score_worker_seconds_sum=score_seconds, score_wall_seconds=None,
        processing_plus_score_worker_seconds_sum=None, amortized_worker_ms_per_query=None,
        scoring_hosts=';'.join(hosts), processing_hosts='', hardware='',
        timing_scope='sum of worker stage wall times; not elapsed wall time or request latency',
        unmeasured='score wall-clock span; BM25 index construction; orchestration, cache merge and verification')
    if model == 'bm25':
        source = read_json(data / 'cache/ocr/metadata.json')
        common.update(unique_encoded_images=source['unique_images'],
            ocr_worker_inference_seconds_sum=float(source['inference_seconds']),
            model_loading_seconds_observed_sum=float(source['loading_seconds']))
        sessions = [session for task in source['tasks'] for session in task['sessions']]
        starts = [datetime.fromisoformat(session['started_at']).timestamp() for session in sessions]
        ends = [datetime.fromisoformat(session['finished_at']).timestamp() for session in sessions
                if session.get('finished_at')]
        if starts and len(ends) == len(starts):
            common['processing_wall_span_seconds_observed'] = max(ends) - min(starts)
            common['processing_wall_span_complete'] = True
        common['processing_hosts'] = ';'.join(sorted({session['host'] for session in sessions}))
        processing_seconds = source['inference_seconds']
    else:
        indexes = [read_json(data / 'cache/image_embeddings' / model / side / 'index.json')
                   for side in ('photo', 'ref')]
        if (indexes[0]['unique_images'] != indexes[1]['unique_images']
                or indexes[0]['unique_encoding_seconds'] != indexes[1]['unique_encoding_seconds']):
            raise ValueError(f'Photo/reference shared encoding cost differs: {model}')
        # Both indexes describe the same deduplicated inference work: count once.
        processing_seconds = float(indexes[0]['unique_encoding_seconds'])
        common.update(unique_encoded_images=indexes[0]['unique_images'],
                      encoding_worker_seconds_sum=processing_seconds)
        runtimes = []
        for path in sorted((data / 'cache/intermediate/workers').glob('*.json')):
            value = read_json(path)
            if value.get('model') == model and value.get('model_loading_seconds') is not None:
                runtimes.append(value)
        if runtimes:
            common['model_loading_seconds_observed_sum'] = sum(v['model_loading_seconds'] for v in runtimes)
            starts = [v['started_unix'] for v in runtimes]
            ends = [v['finished_unix'] for v in runtimes if v.get('finished_unix') is not None]
            if ends:
                common['processing_wall_span_seconds_observed'] = max(ends) - min(starts)
            # Worker records can be overwritten by restarts, so never claim a
            # complete experiment wall time from the remaining runtime files.
            common['processing_hosts'] = ';'.join(sorted({v['hostname'] for v in runtimes}))
            common['hardware'] = ';'.join(sorted({gpu for v in runtimes for gpu in v.get('hardware', [])}))
    common['processing_plus_score_worker_seconds_sum'] = processing_seconds + score_seconds
    common['amortized_worker_ms_per_query'] = 1000 * (processing_seconds + score_seconds) / metadata['shape'][0]
    return common


def percentage(value):
    return f'{100 * value:.3f}%'


def run(data, models=None, ready_only=False):
    data = Path(data).resolve()
    requested = models or list(MODELS)
    selected = [model for model in requested if ready(data, model)] if ready_only else requested
    if not selected:
        print('No complete FORB result is ready for summarization.', flush=True)
        return []
    missing = [model for model in selected if not ready(data, model)]
    if missing:
        raise ValueError(f'Incomplete FORB results: {missing}; use --ready to skip unfinished models')
    manifest = read_json(data / 'cache/manifest.json')
    nq, nr = len(manifest['photos']), len(manifest['references'])
    if (nq, nr) != (8995, 43585):
        raise ValueError('Expected all 8,995 official queries and 43,585 gallery images')
    reports, rows, costs = {}, [], []
    for model in selected:
        directory = result_path(data, model)
        report = read_json(directory / 'metrics.json')
        config = read_json(directory / 'config.json')
        metadata = read_json(data / 'cache/scores' / model / 'metadata.json')
        if (report['queries'] != nq or report['references'] != nr
                or config['dataset_sha256'] != file_hash(data / 'cache/manifest.json')
                or config['score_metadata'] != metadata or metadata.get('status') != 'complete'):
            raise ValueError(f'Result/count/cache provenance mismatch: {model}')
        row = dict(model=model, mode='bm25' if model == 'bm25' else 'image', queries=nq, references=nr)
        for name in ('mAP@1', 'mAP@5', 'R@1', 'R@5', 'R@10', 'MRR', 'mAP@10', 'mAP@100', 'mAP'):
            row[name] = report[name]
        row.update(correct_at_1=report['correct_at_1'], failures=report['failures'],
                   independent_verification_passed=verified(data, model, metadata))
        rows.append(row)
        reports[model] = report
        costs.append(timing(data, model, metadata))
    pending = [model for model in MODELS if model not in selected]
    lines = ['# FORB 官方完整 Subset 基线实验', '',
        f'协议：`{manifest["protocol"]}`。全部 **8,995 张 query** 检索全部 **43,585 张图库**；八个领域共用完整图库。',
        '模型冻结，不训练、不用测试标签选参；使用官方图片和原始图库顺序。重复图片及 query/gallery 字节重复均保留，计算缓存可按图片 SHA-256 复用。',
        '官方指标为 mAP@1 和 mAP@5。每条 query 的 AP@k 使用全部 ground truth 数作为分母：`precision_sum / max(number_of_all_groundtruths, precision_sum, 1)`；不以 `min(positives, k)` 替代。',
        '分数相同时按官方图库原始行号排序。其余 Recall、MRR 和 mAP 为补充指标。所有均值保留全部 query。', '',
        '| 模型 | mAP@1 | mAP@5 | R@1 | R@5 | R@10 | MRR | 独立校验 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        model = row['model']
        mode = row['mode']
        status = f'[通过]({model}/{mode}/verification.json)' if row['independent_verification_passed'] else '待校验'
        lines.append(f'| {LABELS[model]} | {percentage(row["mAP@1"])} | {percentage(row["mAP@5"])} | '
                     f'{percentage(row["R@1"])} | {percentage(row["R@5"])} | {percentage(row["R@10"])} | '
                     f'{row["MRR"]:.6f} | {status} |')
    if pending:
        lines.extend(['', '本报告尚未纳入：' + '、'.join(LABELS[model] for model in pending) + '。'])
    lines.extend(['', 'DINOv3、SigLIP2、Qwen3-VL 使用已归一化向量的 FP32 点积；ColNomic 7B 使用全部有效 token 的 FP32 mean MaxSim。所有方法保存完整 query × gallery 分数，无候选筛选或 token 下采样。',
        'OCR 使用项目统一 PP-OCRv6 配置；BM25 固定 k1=1.2、b=0.75，以完整图库统计 IDF。空 OCR query 记失败；无法识别文字的正例不给检索命中分，仍保留在 AP 分母中。', '',
        '## 分领域', '',
        '| 模型 | 领域 | Query 数 | mAP@1 | mAP@5 | R@1 | MRR |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for model in selected:
        for domain, report in reports[model]['by_domain'].items():
            lines.append(f'| {LABELS[model]} | {domain} | {report["queries"]} | {percentage(report["mAP@1"])} | '
                         f'{percentage(report["mAP@5"])} | {percentage(report["R@1"])} | {report["MRR"]:.6f} |')
    lines.extend(['', '分领域仅分组统计 query；每条 query 仍检索全部 43,585 张图库。', '',
        '## 分难度', '', '| 模型 | 难度 | Query 数 | mAP@1 | mAP@5 | R@1 | MRR |',
        '|---|---|---:|---:|---:|---:|---:|'])
    for model in selected:
        for difficulty, report in reports[model]['by_difficulty'].items():
            lines.append(f'| {LABELS[model]} | {difficulty} | {report["queries"]} | {percentage(report["mAP@1"])} | '
                         f'{percentage(report["mAP@5"])} | {percentage(report["R@1"])} | {report["MRR"]:.6f} |')
    lines.extend(['', '## 计时与缓存', '',
        '详细计时见 [timing.csv](timing.csv)。编码/OCR 和评分时间是各 worker 实测阶段耗时之和；多个 GPU 同时运行时，该总和可以大于实际经过的墙钟时间。',
        '相同图片字节的编码/OCR 计算只计一次；photo/ref 两份 bank 内记录的相同 unique encoding 时间不重复相加。除单独列出的模型加载时间外，合计处理时间不含模型加载、任务调度、缓存合并及验证。',
        '批量摊销毫秒仅描述工作量，不能视为单次请求延迟。当前 score chunk 未记录绝对开始/结束时间，评分墙钟时间保留为空；BM25 建索引耗时未单独采集，不包含在 chunk 评分耗时中。',
        '编码 worker 运行记录的开始/结束跨度只是已保存记录覆盖的观察窗口；重启可能覆盖记录，不据此宣称完整实验墙钟时间。OCR 全部任务 session 齐全时可报告其开始到结束的跨度。', '',
        '| 模型 | 编码/OCR worker 秒之和 | 评分 worker 秒之和 | 合计 worker 秒 |',
        '|---|---:|---:|---:|'])
    for cost in costs:
        processing = (cost['encoding_worker_seconds_sum'] if cost['encoding_worker_seconds_sum'] is not None
                      else cost['ocr_worker_inference_seconds_sum'])
        lines.append(f'| {LABELS[cost["model"]]} | {processing:.3f} | {cost["score_worker_seconds_sum"]:.3f} | '
                     f'{cost["processing_plus_score_worker_seconds_sum"]:.3f} |')
    lines.extend(['', '完整缓存位置：`../cache/image_embeddings/<model>/{photo,ref}/index.json` 与分片；'
        '`../cache/ocr/{photo,ref}.json`；`../cache/scores/<model>/scores.npy`、metadata、manifest 与 query 分块。',
        '逐模型配置、全量 positive ranks 和 top-10 见 `<model>/<mode>/{config.json,metrics.json,predictions.jsonl}`。'
        '独立 CPU 校验见 [verification.json](verification.json) 与逐模型 `verification.json`；dense 全量分数用 FP64 重算，'
        'MaxSim 固定 16 对使用 FP64 验算，BM25 固定 query 针对完整图库 IDF 验算。全部模型均独立检查每条 query 的排序、所有正例排名和指标。'])
    verification = data / 'results/verification.json'
    if verification.exists():
        value = read_json(verification)
        duplicates = value.get('duplicates', {})
        if duplicates:
            lines.extend(['', f'重复审计：官方共 {duplicates["official_rows"]:,} 行，'
                f'{duplicates["unique_image_sha256"]:,} 个不同图片 SHA-256；'
                f'query/gallery 间共有 {duplicates["cross_query_gallery_sha256_groups"]} 组相同字节。'
                '全部官方行均保留；完整 ID 对照记录于 verification.json。'])
    lines.extend(['', '生成时间：' + datetime.now(timezone.utc).isoformat(), ''])
    result = data / 'results'
    result.mkdir(parents=True, exist_ok=True)
    with (result / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_csv(result / 'summary.csv', rows)
        write_csv(result / 'timing.csv', costs)
        atomic_text(result / 'REPORT.md', '\n'.join(lines))
    print(f'Exported {len(rows)} complete FORB baselines to {result}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', action='append', choices=MODELS, help='Repeat to select several baselines')
    parser.add_argument('--ready', action='store_true', help='Include only selected models with complete final results')
    args = parser.parse_args()
    run(args.data, args.model, args.ready)


if __name__ == '__main__':
    main()
