"""Verify disk checkpoints without restarting encoders or modifying prior results."""
import argparse
import csv
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import torch

from scripts.colnomic_mechanism.common import ROOT, read, write, sha, csvout
from scripts.encode_colnomic_images import token_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'checkpoint_validation.json').exists():
        raise FileExistsError('Keep prior validation: choose a new output directory')
    torch.set_num_threads(2)
    folder = ROOT / 'results/token_mechanism_experiments/20261013_v2/stage4_extended_photos_v3'
    manifest = read(folder / 'encoding_manifest.json')
    sources = read(folder / 'sources.json')
    completion = read(folder / 'encoding_complete.json')
    progress = read(folder / 'encoding_progress.json')
    bank = torch.load(ROOT / 'cache/image_embeddings/colnomic7b/photos.pt', weights_only=True, mmap=True, map_location='cpu')
    query_index = {qid: i for i, qid in enumerate(bank['ids'])}
    ids = [r['id'] for r in manifest]
    checks = {'unique_manifest_ids': len(ids) == len(set(ids)),
              'completion_matches_manifest': completion['records'] == len(manifest),
              'progress_matches_manifest': progress['completed'] == progress['total'] == len(manifest),
              'last_progress_matches_manifest': progress['last'] == ids[-1],
              'output_ids_exact': {p.stem for p in (folder / 'embeddings').glob('*.pt')} == set(ids)}
    evidence = []
    for row in manifest:
        path = folder / 'embeddings' / (row['id'] + '.pt')
        record = torch.load(path, weights_only=True, map_location='cpu')
        z = record['embedding']
        source = bank['embeddings'][query_index[row['source_id']]]
        item = dict(id=row['id'], arm=row['arm'], source_id=row['source_id'], package_id=row['package_id'], split=row['split'],
                    output_sha256=sha(path), input_sha256=sha(row['path']),
                    image_sha_matches=sha(row['path']) == record['image_sha256'],
                    tensor_sha_matches=token_hash(z) == record['tensor_sha256'],
                    shape=list(z.shape), dtype=str(z.dtype), shape_matches_source=z.shape == source.shape,
                    finite=bool(torch.isfinite(z).all()), nonzero_tokens=int(z.ne(0).any(1).sum()),
                    manifest_fields_match=all(record.get(k) == v for k, v in row.items()),
                    processor_matches=record.get('processor_backend') == 'torchvision')
        if row['arm'] == 'base':
            item['cache_row'] = query_index[row['source_id']]
            item['cache_image_sha_matches'] = item['input_sha256'] == bank['sha256'][item['cache_row']]
            item['original_cache_exact'] = torch.equal(z, source)
            item['max_absolute_cache_error'] = float((z.float() - source.float()).abs().max())
        item['passed'] = all(item[k] for k in ['image_sha_matches', 'tensor_sha_matches', 'shape_matches_source', 'finite', 'manifest_fields_match', 'processor_matches']) and item.get('original_cache_exact', True) and item.get('cache_image_sha_matches', True)
        evidence.append(item)
    scopes = {field: {split: sorted({r[field] for r in sources if r['split'] == split}) for split in ['train', 'val', 'test']}
              for field in ['package_id', 'group_id', 'image_sha256', 'source_sha256']}
    intersections = {field: {a + '__' + b: sorted(set(values[a]) & set(values[b])) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
                     for field, values in scopes.items()}
    checks['source_split_disjoint'] = all(not v for fields in intersections.values() for v in fields.values())
    checks['all_records_valid'] = all(r['passed'] for r in evidence)
    checks['all_sources_have_six_arms'] = all(sum(r['source_id'] == source['photo_id'] for r in manifest) == 6 for source in sources)
    checks['base_coverage_matches_sources'] = {r['source_id'] for r in manifest if r['arm'] == 'base'} == {r['photo_id'] for r in sources}
    processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,lstart,etime,stat,args'], text=True)
    (output / 'process_snapshot.txt').write_text(processes)
    write(output / 'checkpoint_validation.json', dict(status='passed' if all(checks.values()) else 'failed',
          created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checks=checks,
          records=len(evidence), original_cache_exact_count=sum(r.get('original_cache_exact', False) for r in evidence),
          source_identities_per_split={s: len(scopes['package_id'][s]) for s in ['train', 'val', 'test']},
          source_split_intersections=intersections, source_folder=str(folder), evidence='encoding_records.csv',
          resume_at='analyze_layout_photos.py; encoder already complete', gpu_encoding_restarted=False))
    csvout(output / 'encoding_records.csv', evidence)
    attachment = Path('/home/zyin/.codex/attachments/67ce07df-ae1d-4c84-946f-79d8a7171cc7/已粘贴的文本.txt')
    if attachment.exists():
        shutil.copyfile(attachment, output / 'TASK_SPEC.md')
    assert all(checks.values()), checks
    print(json.dumps(dict(status='passed', records=len(evidence), exact_originals=sum(r.get('original_cache_exact', False) for r in evidence), identities={s: len(scopes['package_id'][s]) for s in ['train', 'val', 'test']}), ensure_ascii=False))


if __name__ == '__main__':
    main()
