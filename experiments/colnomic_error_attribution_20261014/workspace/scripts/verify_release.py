"""Audit the formal dataset, OCR and configured embedding banks."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.data import file_hash,read_csv,read_json,write_json,validate_encoding_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--skip-embedding-checks',action='store_true')
    parser.add_argument('--skip-split-reconstruction', action='store_true',
                        help='Keep frozen split checks; omit reconstruction from historical backup inputs')
    parser.add_argument('--ready',action='store_true',help='Verify completed banks and report pending models')
    parser.add_argument('--forbid-photo',action='append',default=[])
    args=parser.parse_args();root=args.root
    command = [sys.executable, '-B', str(root/'dataset/tools/check.py'),
               '--root', str(root/'dataset'), 'verify']
    if args.skip_split_reconstruction:
        command.append('--skip-split-reconstruction')
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError('Dataset audit failed:\n' + result.stdout + result.stderr)
    dataset = json.loads(result.stdout)
    assert dataset['status'] == 'passed', dataset
    photos=read_csv(root/'dataset/lists/photos.csv');refs=read_csv(root/'dataset/lists/references.csv')
    positives=read_csv(root/'dataset/lists/positives.csv')
    assert len(positives)==len({r['package_id'] for r in positives})==len(read_csv(root/'dataset/lists/packages.csv'))
    assert len({r['reference_id'] for r in positives})==len(positives)
    assert [r['reference_id'] for r in refs]==[f'reference_{i:06d}' for i in range(len(refs))]
    assert [r['photo_id'] for r in photos]==sorted(r['photo_id'] for r in photos)
    forbidden=set(args.forbid_photo)
    assert not forbidden.intersection(r['photo_id'] for r in photos)
    ocr_meta=read_json(root/'cache/ocr/metadata.json');ocr_report={}
    for pop,rows,key in [('photos',photos,'photo_id'),('references',refs,'reference_id')]:
        recorded=validate_encoding_manifest(root,pop,root/'cache/image_embeddings/manifests'/f'{pop}.csv')
        encoded_paths={r[key]:'dataset/'+r['path'] for r in recorded}
        path=root/f'cache/ocr/{pop}.json';ocr=read_json(path)
        assert list(ocr)==[r[key] for r in rows],f'OCR order differs: {pop}'
        assert file_hash(path)==ocr_meta['populations'][pop]['sha256']
        empty=[];line_count=0
        for row in rows:
            entry=ocr[row[key]]
            assert entry['image_sha256']==row['sha256']
            assert entry['image_path'] in (encoded_paths[row[key]],'dataset/'+row['path'])
            assert entry['text']=='\n'.join(x['text'] for x in entry['lines'])
            assert entry['normalized_text']=='\n'.join(x['normalized_text'] for x in entry['lines'])
            if not entry['text'].strip():empty.append(row[key])
            for line in entry['lines']:
                assert 0<=line['confidence']<=1
                for x,y in line['polygon']:
                    assert -1e-3<=x<=entry['width']+1e-3 and -1e-3<=y<=entry['height']+1e-3
            line_count+=len(entry['lines'])
        assert ocr_meta['populations'][pop]['images']==len(rows)
        assert ocr_meta['populations'][pop]['lines']==line_count
        assert ocr_meta['populations'][pop]['empty_ids']==empty
        ocr_report[pop]=dict(records=len(rows),lines=line_count,empty_ids=empty,paths_and_hashes_verified=True)
    sys.path.insert(0,str(root/'dataset/tools'))
    from photo_metadata import PhotoStore
    store=PhotoStore(root/'dataset');store.verify_active();labels=store.records()
    counts={field:dict(Counter(p['type'][field] for p in labels.values())) for field in ('placement','background')}
    active_differences = {}
    for name in ('photos.csv','train.csv','val.csv','test.csv','review.csv','train_subsets/25.csv','train_subsets/50.csv'):
        formal = read_csv(root/'dataset/lists'/name)
        active = read_csv(root/'dataset/lists/active'/name)
        if formal != active:
            active_differences[name] = dict(formal=len(formal), active=len(active))
    if not args.skip_embedding_checks:
        for name in ('image','ocr'):
            command=[sys.executable,'-B',str(root/f'scripts/verify_{name}_embeddings.py')]
            subprocess.run(command+(['--ready'] if args.ready else []),check=True)
    verification={name:read_json(root/f'cache/{name}_embeddings/verification.json') for name in ('image','ocr')}
    assert all(v['status']=='complete' or (args.ready and v['status']=='building') for v in verification.values())
    for identifier in forbidden:
        assert not list((root/'dataset/packages').rglob(identifier+'.*'))
        for directory in ('dataset','cache/ocr','cache/image_embeddings','cache/ocr_embeddings'):
            for path in (root/directory).rglob('*'):
                if path.is_file() and path.suffix in ('.json','.csv','.html','.md'):
                    assert identifier not in path.read_text(),str(path)
    pending=verification['image'].get('pending_models',[])
    report=dict(status='partial' if pending else 'passed',pending_models=pending,dataset=dataset,ocr=ocr_report,labels=counts,
        package_reference_bijection=True,reference_ids_consecutive=True,
        formal_and_active_equal=not active_differences,active_differences=active_differences,
        embedding_verification=verification,
        absent_photo_ids_checked=len(forbidden))
    write_json(root/'dataset/verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('dataset','embedding_verification')},indent=2))


if __name__=='__main__':main()
