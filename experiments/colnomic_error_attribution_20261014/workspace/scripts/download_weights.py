"""Download pinned DINOv3/SigLIP2 weights into the project and record hashes."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data import file_hash, read_json, write_json


def main():
    os.environ.setdefault('HF_HUB_CACHE',str(ROOT/'cache/huggingface'))
    os.environ.setdefault('HF_XET_CACHE',str(ROOT/'cache/huggingface/xet'))
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('all', 'dinov3', 'siglip2'), default='all')
    args = parser.parse_args()
    settings = read_json(ROOT / 'configs/baselines.json')
    manifest = read_json(ROOT / 'weights/manifest.json')
    failed = []
    for name in ('dinov3', 'siglip2') if args.model == 'all' else (args.model,):
        spec = settings['models'][name]
        repo, revision = (spec['image_model'][key] for key in ('repo', 'revision'))
        destination = ROOT / spec['checkpoint']
        try:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            files = {item.rfilename: item for item in info.siblings}
            records = []
            for filename in ('config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors'):
                print(f'Preparing {name}/{filename}', flush=True)
                path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=destination))
                checksum = file_hash(path)
                remote = files[filename]
                if remote.size is not None and path.stat().st_size != remote.size:
                    raise ValueError(f'Size mismatch: {path}')
                if remote.lfs and checksum != remote.lfs.sha256:
                    raise ValueError(f'Weight checksum mismatch: {path}')
                records.append(dict(path=str(path.relative_to(ROOT / 'weights')), bytes=path.stat().st_size,
                                    sha256=checksum, method='local_or_download; verified against pinned upstream'))
            entry = dict(model=name, role='checkpoint', source=f'https://huggingface.co/{repo}/tree/{revision}',
                         destination=spec['checkpoint'], revision=revision, files=records)
            old=next((r for r in manifest['checkpoints'] if r['model']==name),{})
            if 'reused_local_source' in old:
                entry['reused_local_source']=old['reused_local_source']
            manifest['checkpoints'] = [r for r in manifest['checkpoints'] if r['model'] != name] + [entry]
            write_json(ROOT / 'weights/manifest.json', manifest)
            write_json(destination / 'provenance.json', entry)
            print(f'{name}: verified local checkpoint ready', flush=True)
        except Exception as error:
            failed.append(name)
            print(f'{name}: {type(error).__name__}: {str(error).splitlines()[0]}', file=sys.stderr)
    if failed:
        raise SystemExit('Unavailable weights: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
