#!/usr/bin/env python3
"""Read-only FORB progress monitor; no tensor loads, GPU calls or SSH.

Use --watch 30 for JSON Lines snapshots. Unchanged JSON files are cached in
memory by mtime/size; OCR task payloads are reduced immediately to small summaries.
Reported completion is metadata/file presence, not a replacement for final audits.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'other_datasets/FORB'
MODELS = ('dinov3', 'siglip2', 'Qwen3-VL-Embedding-2B', 'colnomic7b', 'bm25')


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value else None


def epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() if value else None


def infer_host(worker):
    match = re.search(r'(?:i14s|host|h|s|141\.3\.14\.)(43|45|46)(?:\D|$)', worker)
    return 'i14s' + match[1] if match else 'unknown'


def inventory(value):
    return dict(queries=len(value['photos']), references=len(value['references']),
                unique_images=len({r['sha256'] for key in ('photos', 'references') for r in value[key]}))


def ocr_task(value):
    sessions = value.get('sessions', [])
    worker_hosts = {s['worker']: s['host'] for s in sessions}
    workers = {}
    for entry in value.get('entries', {}).values():
        name = entry['worker']
        target = workers.setdefault(name, dict(host=worker_hosts.get(name, infer_host(name)),
            images=0, empty_images=0, lines=0, inference_seconds=0., domains={}, last_image_unix=0.))
        target['images'] += 1
        target['empty_images'] += not bool(entry['text'].strip())
        target['lines'] += len(entry['lines'])
        target['inference_seconds'] += entry['inference_seconds']
        target['last_image_unix'] = max(target['last_image_unix'], epoch(entry.get('completed_at')) or 0.)
        domain = entry['image_path'].split('/')[0]
        domain_stats = target['domains'].setdefault(domain, dict(images=0, inference_seconds=0.))
        domain_stats['images'] += 1
        domain_stats['inference_seconds'] += entry['inference_seconds']
    return dict(complete=value.get('complete', False), signature=value.get('signature'),
                images=len(value.get('entries', {})), total=len(value['image_hashes']),
                workers=workers, sessions=sessions, updated_unix=epoch(value.get('updated_at')))


class JSONCache:
    def __init__(self):
        self.values = {}
        self.reads = self.hits = 0
        self.errors = []

    def read(self, path, reduce=None):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        stamp = (stat.st_mtime_ns, stat.st_size)
        previous = self.values.get(path)
        if previous and previous[0] == stamp:
            self.hits += 1
            return previous[1]
        try:
            value = json.loads(path.read_text())
            value = reduce(value) if reduce else value
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.errors.append(dict(file=str(path), error=str(error)))
            return None
        self.values[path] = (stamp, value)
        self.reads += 1
        return value


def stage_stats():
    return dict(completed_images=0, inference_seconds_sum=0., latest_task_update_unix=0.)


def add_stage(hosts, host, stage, images, seconds, updated):
    value = hosts.setdefault(host, dict(stages={}, workers=[]))
    target = value['stages'].setdefault(stage, stage_stats())
    target['completed_images'] += images
    target['inference_seconds_sum'] += seconds
    target['latest_task_update_unix'] = max(target['latest_task_update_unix'], updated or 0.)
    return target


class Monitor:
    def __init__(self, data):
        self.data = data.resolve()
        self.cache = JSONCache()
        self.windows = {}

    def eta(self, name, done, total, seconds, now):
        previous = self.windows.get(name)
        result = dict(remaining_images=max(0, total - done),
                      remaining_inference_seconds_estimate=round(seconds / done * (total - done), 1) if done else None,
                      rough_wall_eta_seconds=None, observed_wall_images_per_second=None)
        if done == total:
            result['rough_wall_eta_seconds'] = 0.
        if previous and done >= previous[1] and now - previous[0] >= 10:
            elapsed, progress = now - previous[0], done - previous[1]
            result.update(observed_window_seconds=round(elapsed, 1),
                          observed_window_images=progress,
                          observed_wall_images_per_second=round(progress / elapsed, 4))
            if progress > 0:
                result['rough_wall_eta_seconds'] = round((total - done) * elapsed / progress, 1)
            # Keep a useful rolling interval even when JSON is polled frequently.
            if elapsed >= 300:
                self.windows[name] = (now, done)
        elif previous is None or done < previous[1]:
            self.windows[name] = (now, done)
        return result

    def scores(self, model, total_queries):
        base = self.data / 'cache/scores' / model
        manifest = self.cache.read(base / 'manifest.json')
        result = dict(query_rows_done=0, query_rows_total=total_queries, chunks_done=0,
                      chunks_total=None, merged=False, latest_chunk_update_utc=None)
        if manifest is None:
            return result
        total, step = manifest['shape'][0], manifest['chunk_size']
        expected = {f'{i:06d}-{min(i + step, total):06d}.pt': min(step, total - i)
                    for i in range(0, total, step)}
        files = list((base / 'chunks').glob('*.pt'))
        found = [p for p in files if p.name in expected]
        metadata = self.cache.read(base / 'metadata.json')
        result.update(query_rows_done=sum(expected[p.name] for p in found), query_rows_total=total,
                      chunks_done=len(found), chunks_total=len(expected),
                      unexpected_chunk_files=len(files) - len(found),
                      merged=bool(metadata and metadata.get('status') == 'complete' and (base / 'scores.npy').exists()),
                      latest_chunk_update_utc=utc(max((p.stat().st_mtime for p in found), default=0.)))
        return result

    def result(self, model):
        base = self.data / 'results' / model / ('bm25' if model == 'bm25' else 'image')
        metrics = self.cache.read(base / 'metrics.json')
        complete = bool(metrics and (base / 'config.json').exists() and (base / 'predictions.jsonl').exists())
        output = dict(complete=complete)
        if metrics:
            values = metrics.get('query_average', metrics)
            output['metrics'] = {k: values.get(k, metrics.get(k)) for k in ('R@1', 'R@5', 'R@10', 'MRR', 'mAP@100')}
        return output

    def snapshot(self):
        start = now = time.time()
        before_reads, before_hits = self.cache.reads, self.cache.hits
        self.cache.errors = []
        totals = self.cache.read(self.data / 'cache/manifest.json', inventory)
        if totals is None:
            raise ValueError('A readable cache/manifest.json is required')
        unique_total = totals['unique_images']
        intermediate = self.data / 'cache/intermediate'
        hosts, runtimes, models = {}, {}, {}
        for path in sorted((intermediate / 'workers').glob('*.json')):
            value = self.cache.read(path)
            if not value:
                continue
            name, host = value['worker'], value['hostname']
            value = dict(value, runtime_updated_unix=path.stat().st_mtime)
            runtimes[name] = value
            row = {key: value.get(key) for key in ('worker', 'model', 'pid', 'visible_devices', 'status',
                    'current_task', 'tasks_completed', 'hardware', 'error') if key in value}
            row.update(started_utc=utc(value.get('started_unix')),
                       runtime_updated_utc=utc(value['runtime_updated_unix']),
                       runtime_age_seconds=round(now - value['runtime_updated_unix'], 1),
                       reported_wall_seconds=round((value.get('finished_unix') or now) - value['started_unix'], 1))
            hosts.setdefault(host, dict(stages={}, workers=[]))['workers'].append(row)
        for model in MODELS:
            status = dict(scores=self.scores(model, totals['queries']), result=self.result(model))
            models[model] = status
            if model == 'bm25':
                continue
            directory = intermediate / 'encoding' / model
            markers = list(directory.glob('*.done.json'))
            done = tasks = bytes_total = 0
            seconds = latest = 0.
            signatures = set()
            shared = dict(images=0, inference_seconds_sum=0., host_groups={})
            for path in markers:
                marker = self.cache.read(path)
                if marker is None:
                    continue
                vector_path = path.with_name(path.name.removesuffix('.done.json') + '.pt')
                if not vector_path.exists() or vector_path.stat().st_size != marker['bytes']:
                    self.cache.errors.append(dict(file=str(path), error='Missing/wrong-size vector file'))
                    continue
                count, duration = len(marker['image_sha256']), marker['encoding_seconds']
                done += count
                tasks += 1
                seconds += duration
                bytes_total += marker['bytes']
                modified = path.stat().st_mtime
                latest = max(latest, modified)
                signatures.add(marker['signature'])
                owners = sorted({runtimes.get(w, {}).get('hostname', infer_host(w)) for w in marker['workers']})
                if len(owners) == 1:
                    add_stage(hosts, owners[0], model, count, duration, modified)
                else:
                    shared['images'] += count
                    shared['inference_seconds_sum'] += duration
                    key = '+'.join(owners) or 'unknown'
                    shared['host_groups'][key] = shared['host_groups'].get(key, 0) + count
            status.update(unique_done=done, unique_total=unique_total, completed_tasks=tasks,
                          partial_tensor_files=max(0, len(list(directory.glob('*.pt'))) - tasks),
                          encoding_inference_seconds_sum=round(seconds, 2), completed_tensor_bytes=bytes_total,
                          latest_task_update_utc=utc(latest), signatures_seen=len(signatures),
                          merged_banks={side: (self.data / 'cache/image_embeddings' / model / side / 'index.json').exists()
                                        for side in ('photo', 'ref')},
                          shared_host_tasks=shared, eta=self.eta(model, done, unique_total, seconds, now))
            if len(signatures) > 1:
                self.cache.errors.append(dict(model=model, error='Multiple encoding signatures; counts need audit'))
        ocr = dict(unique_done=0, unique_total=unique_total, empty_images=0, lines=0,
                   completed_tasks=0, partial_tasks=0,
                   inference_seconds_sum=0., latest_task_update_utc=None, workers={})
        plan = self.cache.read(intermediate / 'ocr/plan.json',
                               lambda p: dict(signature=p['signature'], tasks=len(p['tasks']), task_size=p['task_size']))
        ocr['tasks_total'] = plan['tasks'] if plan else None
        latest_ocr = 0.
        for path in sorted((intermediate / 'ocr/tasks').glob('*.json')):
            value = self.cache.read(path, ocr_task)
            if value is None:
                continue
            if plan and value['signature'] != plan['signature']:
                self.cache.errors.append(dict(file=str(path), error='OCR signature differs from plan'))
                continue
            ocr['unique_done'] += value['images']
            ocr['completed_tasks' if value['complete'] else 'partial_tasks'] += 1
            updated = value['updated_unix'] or path.stat().st_mtime
            latest_ocr = max(latest_ocr, updated)
            for worker, stats in value['workers'].items():
                target = ocr['workers'].setdefault(worker, dict(host=stats['host'], images=0, empty_images=0, lines=0,
                    inference_seconds_sum=0., latest_image_unix=0., domains={}))
                target['images'] += stats['images']
                target['empty_images'] += stats['empty_images']
                target['lines'] += stats['lines']
                target['inference_seconds_sum'] += stats['inference_seconds']
                target['latest_image_unix'] = max(target['latest_image_unix'], stats['last_image_unix'])
                ocr['inference_seconds_sum'] += stats['inference_seconds']
                ocr['empty_images'] += stats['empty_images']
                ocr['lines'] += stats['lines']
                # A resumed task can contain old entries from a different host;
                # attribute recency to that host's last image, not a later save.
                host_stage = add_stage(hosts, stats['host'], 'ocr', stats['images'], stats['inference_seconds'],
                                       stats['last_image_unix'] or updated)
                host_stage['empty_images'] = host_stage.get('empty_images', 0) + stats['empty_images']
                host_stage['lines'] = host_stage.get('lines', 0) + stats['lines']
                for domain, domain_stats in stats['domains'].items():
                    for destination in (target['domains'], host_stage.setdefault('domains', {})):
                        domain_target = destination.setdefault(domain, dict(images=0, inference_seconds_sum=0.))
                        domain_target['images'] += domain_stats['images']
                        domain_target['inference_seconds_sum'] += domain_stats['inference_seconds']
            for session in value['sessions']:
                target = ocr['workers'].setdefault(session['worker'], dict(host=session['host'], images=0, empty_images=0, lines=0,
                    inference_seconds_sum=0., latest_image_unix=0., domains={}))
                started = epoch(session['started_at'])
                ended = epoch(session.get('finished_at')) or updated
                target['observed_session_wall_seconds_sum'] = target.get('observed_session_wall_seconds_sum', 0.) + max(0., ended - started)
                target['loading_seconds_sum'] = target.get('loading_seconds_sum', 0.) + session.get('loading_seconds', 0.)
                if started > target.get('latest_session_started_unix', 0.):
                    target.update(latest_session_started_unix=started,
                                  latest_session_finished=bool(session.get('finished_at')), device=session.get('device'))
        ocr['latest_task_update_utc'] = utc(latest_ocr)
        ocr['eta'] = self.eta('ocr', ocr['unique_done'], unique_total, ocr['inference_seconds_sum'], now)
        ocr['inference_seconds_sum'] = round(ocr['inference_seconds_sum'], 2)
        ocr['merged'] = all((self.data / 'cache/ocr' / f'{side}.json').exists() for side in ('photo', 'ref', 'metadata'))
        for worker in ocr['workers'].values():
            worker['seconds_per_image'] = round(worker['inference_seconds_sum'] / worker['images'], 3) if worker['images'] else None
            worker['latest_image_utc'] = utc(worker.pop('latest_image_unix'))
            worker['latest_session_started_utc'] = utc(worker.pop('latest_session_started_unix', None))
            for key in ('inference_seconds_sum', 'observed_session_wall_seconds_sum', 'loading_seconds_sum'):
                if key in worker:
                    worker[key] = round(worker[key], 2)
            for domain in worker['domains'].values():
                domain['seconds_per_image'] = round(domain['inference_seconds_sum'] / domain['images'], 3)
                domain['inference_seconds_sum'] = round(domain['inference_seconds_sum'], 2)
        for host in hosts.values():
            for stats in host['stages'].values():
                stats['seconds_per_image'] = round(stats['inference_seconds_sum'] / stats['completed_images'], 3) if stats['completed_images'] else None
                stats['inference_seconds_sum'] = round(stats['inference_seconds_sum'], 2)
                stats['latest_task_update_utc'] = utc(stats.pop('latest_task_update_unix'))
                for domain in stats.get('domains', {}).values():
                    domain['seconds_per_image'] = round(domain['inference_seconds_sum'] / domain['images'], 3)
                    domain['inference_seconds_sum'] = round(domain['inference_seconds_sum'], 2)
        return dict(updated_utc=utc(now), totals=totals, models=models, ocr=ocr, hosts=hosts,
                    timing_note='Inference seconds are summed worker compute time, not elapsed wall time. Wall ETA uses progress between watch snapshots; excludes partial encoding tasks. Runtime status is last reported state, not a liveness check.',
                    monitoring=dict(json_files_read=self.cache.reads - before_reads,
                                    unchanged_json_cache_hits=self.cache.hits - before_hits,
                                    snapshot_seconds=round(time.time() - start, 3),
                                    errors=self.cache.errors, validation='metadata and file presence/size only'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--watch', type=float, help='Poll interval in seconds; emit one JSON object per line')
    parser.add_argument('--samples', type=int, help='Optional number of watch snapshots before exiting')
    parser.add_argument('--compact', action='store_true', help='Single-line JSON also for one-shot mode')
    args = parser.parse_args()
    if args.watch is not None and args.watch < 1:
        parser.error('--watch must be at least one second')
    if args.samples is not None and (args.samples < 1 or args.watch is None):
        parser.error('--samples requires --watch and a positive count')
    monitor = Monitor(args.data)
    count = 0
    try:
        while True:
            print(json.dumps(monitor.snapshot(), ensure_ascii=False,
                             indent=None if args.watch or args.compact else 2), flush=True)
            count += 1
            if args.watch is None or (args.samples is not None and count >= args.samples):
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
