"""Export original query photos whose top-1 prediction is incorrect, by configuration."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'wrong')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    verification = json.loads((ROOT/'results/verification.json').read_text())
    assert verification['status'] == 'passed' and verification['configurations'] == 35
    queries = verification['queries_per_configuration']
    configurations = read_csv(ROOT/'results/all_results.csv')
    assert len(configurations) == 35 and len({r['id'] for r in configurations}) == 35
    photos = {r['photo_id']: r for r in read_csv(ROOT/'dataset/lists/photos.csv')}
    references = {r['reference_id']: r for r in read_csv(ROOT/'dataset/lists/references.csv')}
    targets = defaultdict(list)
    summaries = []
    expected = {}
    inputs = {'results/all_results.csv': sha256(ROOT/'results/all_results.csv'),
              'results/verification.json': sha256(ROOT/'results/verification.json'),
              'dataset/lists/photos.csv': sha256(ROOT/'dataset/lists/photos.csv')}
    fields = ['photo_id', 'filename', 'package_id', 'identity', 'category', 'correct_reference_ids',
              'correct_reference_paths', 'selected_reference_id', 'selected_reference_path',
              'rank', 'failure', 'placement', 'background', 'occlusion', 'source_path', 'sha256']
    for config in configurations:
        result = ROOT/config['result']
        spec = json.loads((result/'config.json').read_text())
        assert spec['id'] == config['id'] and spec['queries'] == queries
        assert spec['protocol'] == verification['methods'][config['id']].get('protocol', 'Full-1117')
        predictions = [json.loads(line) for line in (result/'predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == queries and {r['photo_id'] for r in predictions} == set(photos)
        errors = [r for r in predictions if r['correct'] is False]
        assert len(errors) == round(queries*(1-float(config['recall_at_1'])))
        name = '__'.join([config['id'], config['model'], config['mode']])
        if config['mode'] in ('fusion', 'ocr_fusion', 'bm25_image_fusion'):
            name += '__alpha_0.5'
        assert re.fullmatch(r'[A-Za-z0-9._-]+', name)
        folder = output/name
        folder.mkdir(exist_ok=True)
        expected[name] = set()
        rows = []
        for row in errors:
            photo = photos[row['photo_id']]
            assert row['package_id'] == photo['package_id'] and row['rank'] != 1
            source = ROOT/'dataset'/photo['path']
            assert source.is_file()
            filename = source.name
            assert filename not in expected[name]
            expected[name].add(filename)
            targets[row['photo_id']].append(folder/filename)
            selected = row['selected_reference_id']
            positive = row['positive_reference_ids']
            rows.append(dict(photo_id=row['photo_id'], filename=filename, package_id=photo['package_id'],
                identity=photo['identity'], category=photo['category'],
                correct_reference_ids=';'.join(positive),
                correct_reference_paths=';'.join('dataset/'+references[r]['path'] for r in positive),
                selected_reference_id=selected, selected_reference_path='dataset/'+references[selected]['path'] if selected else '',
                rank=row['rank'], failure=row['failure'], placement=row['type'].get('placement'),
                background=row['type'].get('background'), occlusion=row['type'].get('occlusion'),
                source_path='dataset/'+photo['path'], sha256=photo['sha256']))
        write_csv(folder/'errors.csv', rows, fields)
        (folder/'config.json').write_bytes((result/'config.json').read_bytes())
        inputs[str((result/'predictions.jsonl').relative_to(ROOT))] = sha256(result/'predictions.jsonl')
        summaries.append(dict(id=config['id'], model=config['model'], mode=config['mode'], folder=name,
            queries=queries, errors=len(errors), recall_at_1=config['recall_at_1'], result=config['result']))
    assert {p.name for p in output.iterdir() if p.is_dir()} == set(expected)
    copied = reused = written_bytes = completed = 0
    started = time.monotonic()

    def copy_photo(photo_id, paths):
        photo = photos[photo_id]
        data = (ROOT/'dataset'/photo['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == photo['sha256'], photo_id
        new = old = size = 0
        for path in paths:
            if path.exists():
                assert not path.is_symlink() and sha256(path) == photo['sha256'], str(path)
                old += 1
                continue
            temporary = path.with_name(path.name+'.partial')
            written = temporary.write_bytes(data)
            assert written == len(data)
            temporary.replace(path)
            new += 1
            size += len(data)
        return new, old, size

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(copy_photo, photo_id, paths) for photo_id, paths in targets.items()]
        for future in as_completed(futures):
            new, old, size = future.result()
            copied += new; reused += old; written_bytes += size; completed += 1
            if completed % 50 == 0 or completed == len(targets):
                print(json.dumps(dict(unique_photos_done=completed, unique_photos=len(targets),
                    copies_done=copied+reused, seconds=round(time.monotonic()-started, 1))), flush=True)
    for name, names in expected.items():
        folder = output/name
        actual = {p.name for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')}
        assert actual == names, name
        for filename in names:
            path = folder/filename
            assert path.is_file() and not path.is_symlink()
        assert len(read_csv(folder/'errors.csv')) == len(names)
    assert copied+reused == sum(r['errors'] for r in summaries)
    write_csv(output/'summary.csv', summaries, list(summaries[0]))
    report = dict(status='complete', created_at=datetime.now(timezone.utc).isoformat(), output=str(output),
        criterion='top-1 exact-package prediction is incorrect (correct=false)',
        configurations=len(summaries), queries_per_configuration=queries,
        exported_photo_copies=copied+reused, unique_error_photos=len(targets),
        newly_written_bytes=written_bytes, copied=copied, reused=reused,
        source_sha256_verified=True, regular_files=True, inputs_sha256=inputs,
        elapsed_seconds=time.monotonic()-started, groups=summaries)
    (output/'export.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 35 组实验的错误照片', '',
        '依据已核验的 Full-1117 结果导出；检索第一名未匹配正确包装（correct=false）即列为错误。',
        '每组复制原始完整查询照片；同一照片可能在不同实验中重复出现。图片保留原文件名。',
        '文件夹名包含实验 ID、模型和检索形式，融合配置附 alpha_0.5。',
        '每组 errors.csv 记录真实包装、正确参考图、检索第一名、正确参考图的排名及原照片路径；路径相对仓库根目录。config.json 为对应实验配置。',
        f'共 {len(summaries)} 个实验文件夹，{copied+reused:,} 份图片，涉及 {len(targets):,} 张不同照片。', '',
        '| ID | 文件夹 | 错误照片数 |', '|---|---|---:|']
    lines += [f"| {r['id']} | [{r['folder']}]({r['folder']}/) | {r['errors']} |" for r in summaries]
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256', 'groups')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
