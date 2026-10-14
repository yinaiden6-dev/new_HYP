"""Create an offline, relocatable inference transfer without re-encoding."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREFIX='content_box_ce_polish18_resume'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--all-photos',action='store_true',help='Include all formal photos and the full reference gallery')
    args=parser.parse_args()
    deployment=Path(read(ROOT/'cache/content_box_ce_polish18/deployment/active.json')['path'])
    manifest=read(deployment/'manifest.json')
    done=[q for q in manifest['queries'] if (deployment/'queries'/q['id']/'prediction.json').exists()]
    pending=[q for q in manifest['queries'] if q not in done]
    needed=sorted({row for q in pending for row in q['candidate_physical_rows']})
    with (ROOT/'dataset/lists/photos.csv').open(newline='') as stream:photo_inventory=list(csv.DictReader(stream))
    with (ROOT/'dataset/lists/references.csv').open(newline='') as stream:reference_inventory=list(csv.DictReader(stream))
    photo_images=[ROOT/'dataset'/q['path'] for q in photo_inventory] if args.all_photos else [Path(q['path']) for q in pending]
    reference_images=[ROOT/'dataset'/r['path'] for r in reference_inventory] if args.all_photos else [Path(manifest['references'][str(row)]['path']) for row in needed]
    for process in read(deployment/'processes.json'):
        proc=Path('/proc')/str(process['pid'])/'cmdline'
        if proc.exists() and b'run_content_box_ce_polish18.py' in proc.read_bytes():
            raise RuntimeError('Stop the inference workers before sealing a transfer')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    staging=ROOT/'cache/content_box_ce_polish18/transfer'/stamp
    staging.mkdir(parents=True,exist_ok=False)
    kind='all1117' if args.all_photos else 'resume'
    output=args.output or ROOT/'exports'/f'content_box_ce_polish18_{kind}_{stamp}.zip'
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    sources={}
    def add(source,relative=None):
        source=Path(source);relative=str(relative or source.relative_to(ROOT))
        if source.is_symlink():raise ValueError('Unsealed symbolic link: '+str(source))
        if not source.is_file():raise FileNotFoundError(source)
        if relative.startswith('/') or '..' in Path(relative).parts:raise ValueError('Unsafe archive path')
        if relative in sources and sources[relative]!=source:raise ValueError('Duplicate transfer path')
        sources[relative]=source
    def tree(path):
        for source in sorted((ROOT/path).rglob('*')):
            if source.is_file() and not any(p in {'.git','__pycache__','.pytest_cache'} for p in source.relative_to(ROOT/path).parts):add(source)
    for relative in ['scripts/run_content_box_ce_polish18.py','scripts/resume_content_box_ce_polish18.py',
            'scripts/package_content_box_ce_polish18.py','tests/test_polish18_pipeline.py',
            'evaluation/__init__.py','evaluation/data.py','drugpack/__init__.py','drugpack/method/__init__.py','drugpack/method/token_layout.py',
            'cache/image_embeddings/colnomic7b/photos.pt','cache/image_embeddings/colnomic7b/references.pt','cache/scores/colnomic7b/image.pt',
            'weights/romav2/manifest.json','weights/romav2/torchhub/hub/checkpoints/romav2.0.1.pt',
            'weights/romav2/torchhub/hub/trusted_list']:
        if (ROOT/relative).exists():add(ROOT/relative)
    for relative in ['weights/content_box_ce_polish18','weights/romav2/source',
            'weights/romav2/torchhub/hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa',
            'cache/method/intermediate/runtime','cache/content_box_ce_polish18/source_reference']:
        tree(relative)
    for source in sorted((ROOT/'weights/colnomic7b/adapter').iterdir()):
        if source.suffix in {'.json','.txt','.md'}:add(source)
    for source in sorted((ROOT/'dataset/lists').glob('*')):
        if source.is_file() and source.suffix in {'.csv','.json','.md'}:add(source)
    if args.all_photos:
        with (ROOT/'dataset/lists/packages.csv').open(newline='') as stream:
            for package in csv.DictReader(stream):add(ROOT/'dataset'/package['metadata_path'])
        add(ROOT/'dataset/references/metadata.json')
        for source in sorted((ROOT/'dataset/annotations').glob('*')):
            if source.is_file() and source.suffix in {'.csv','.json','.md'}:add(source)
        for name in ['release.json','verification.json','finalization.json']:
            if (ROOT/'dataset'/name).exists():add(ROOT/'dataset'/name)
    rc=ROOT/'cache/content_box_ce_polish18/upstream/ICLR/new ROUTEA/RC'
    for program in ['run_romav2_colnomic_visibility_xf_six_case_v1.py','run_rc_h593_learned_colnomic_only_v1.py',
            'run_rc_h593_ce_continuation_v1.py','validate_rc_h593_content_complement_v1.py']:
        add(rc/'programs'/program)
    tree(str((rc/'src').relative_to(ROOT)))
    add(rc/'results/rc_h593_box_ce_polish_v1/fold0/payload.json')
    for path in photo_images:add(path)
    for path in reference_images:add(path)
    pairs=[];complete_checkpoints=0
    for q in pending:
        for path in sorted((deployment/'pairs'/q['id']).glob('*.npz')):add(path);pairs.append(path)
    for q in done:
        rows=sorted(set([q['candidate_physical_rows'][0],q['ranked_physical_rows'][0],q['candidate_physical_rows'][-1]]))
        for row in rows:add(deployment/'pairs'/q['id']/f'{row:06d}.npz');complete_checkpoints+=1
        for name in ['input.json','features.npz','prediction.json','receipt.json']:add(deployment/'queries'/q['id']/name)
    for name in ['verification.json','summary.json','predictions.csv','README.md']:
        if (deployment/name).exists():add(deployment/name)
    deploy_relative=deployment.relative_to(ROOT)
    transfer=dict(created_utc=stamp,source_root=str(ROOT),source_deployment=str(deployment),
        original_manifest_sha256=hashlib.sha256((deployment/'manifest.json').read_bytes()).hexdigest(),
        completed_query_ids=[q['id'] for q in done],pending_query_ids=[q['id'] for q in pending],
        pending_query_images=len(pending),pending_reference_images=len(needed),
        all_formal_photos_included=args.all_photos,query_images_included=len(photo_images),reference_images_included=len(reference_images),
        partial_pairs_reused=len(pairs),completed_roma_checkpoints=complete_checkpoints,
        completed_full_roma_maps_preserved_only_at_source=len(done)*128-complete_checkpoints,
        candidate_pairs_remaining=len(pending)*128-len(pairs),
        omitted='Completed queries retain all X11/C4/content, scores and decisions; only fixed native verification maps are transferred. Other completed native maps remain unchanged at the source.',
        colnomic_encoder_weights_included=False,encoder_forwards=0,fold=0,threshold=0)
    for relative,value in [(deploy_relative/'manifest.json',dict(manifest,transfer=transfer)),
            (Path('cache/content_box_ce_polish18/deployment/active.json'),dict(path=deployment.name,signature=manifest['signature'])),
            (Path('transfer_state.json'),transfer)]:
        path=staging/relative;write(path,value);add(path,relative)
    with (ROOT/'dataset/lists/test.csv').open(newline='') as stream:records=list(csv.DictReader(stream))
    pending_ids=set(transfer['pending_query_ids']);remaining=[r for r in records if r['photo_id'] in pending_ids]
    path=staging/'pending_queries.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(remaining)
    add(path,'pending_queries.csv')
    requirements='numpy==2.2.6\npillow==12.3.0\nrich==15.0.0\ntqdm==4.70.1\neinops==0.8.1\nopencv-python-headless==4.14.0.94\n'
    path=staging/'requirements.txt';path.write_text(requirements);add(path,'requirements.txt')
    checker='''"""Verify every transferred file against its sealed SHA256."""\nimport hashlib,json\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nm=json.loads((root/'bundle_files.json').read_text())\nfor index,record in enumerate(m['files']):\n p=root/record['path']\n if not p.is_file() or p.stat().st_size!=record['bytes']:raise ValueError('Missing/truncated: '+str(p))\n with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()\n if actual!=record['sha256']:raise ValueError('Changed: '+str(p))\n if (index+1)%250==0:print(str(index+1)+' files checked',flush=True)\nprint('PASS: '+str(len(m['files']))+' files; fixed fold0 transfer intact')\n'''
    path=staging/'check_bundle.py';path.write_text(checker);add(path,'check_bundle.py')
    readme=f'''# CONTENT_BOX_CE_POLISH18 续跑包

包内有 {len(photo_images)} 张实拍原图、{len(reference_images)} 张参考原图，以及完整检索缓存和数据清单。
默认续跑清单仍为原固定测试集 161 张，本机已完成 {len(done)} 张，剩余 {len(pending)} 张。
保留部分 query 的 {len(pairs)} 个已完成配对；还需计算 {transfer['candidate_pairs_remaining']} 个配对。
已完成 query 的全部 X11、C4、内容五维、127 个挑战者分数及最终答案保留。
为控制体积，这些完整 query 只附带 {complete_checkpoints} 个固定核验配对的原生 RoMa 图；其余原生图在原机保留。
完整 query 不会重算。`pending_queries.csv` 是图片清单；执行入口仍用原 161 张清单和固定 ordinal。

解压后可放在任意目录；请先进入 `{PREFIX}`。Python >=3.11，CUDA GPU，原机 11 GB 显存已验证，每 worker 建议预留至少 16 GB 主机内存。
本机验证环境为 Python 3.12、PyTorch 2.9.1+cu126、torchvision 0.24.1+cu126。原 BF16 精度、precise800/1280 和双向配置保持一致。

```bash
python3 -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -r requirements.txt
python3 check_bundle.py
python3 scripts/run_content_box_ce_polish18.py status
python3 scripts/resume_content_box_ce_polish18.py --gpus 0 1 2 3 --dry-run
python3 scripts/resume_content_box_ce_polish18.py --gpus 0 1 2 3
```

只有一张 GPU 时使用 `--gpus 0`。入口按原固定 ordinal 分片，并自动跳过已完成 query 和配对。
运行日志写入 deployment 下新的 `logs/resume_*/`。Ctrl+C 可停止，之后运行同一命令可恢复。
完成后自动运行 `verify` 和 `report`；结果在 `{deploy_relative}`：`summary.json`、`predictions.csv`、`verification.json`。

也可独立执行：

```bash
python3 scripts/run_content_box_ce_polish18.py run --device cuda:0
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 scripts/run_content_box_ce_polish18.py verify
python3 scripts/run_content_box_ce_polish18.py report
```

默认直接续跑已有 deployment，不需要执行 `prepare` 或重新构造测试清单。
原始完整 token 银行和图库分数保留，顺序、SHA 身份去重和自然 C128 不变，不注入 target。
不需要下载或运行 ColNomic 编码器。ColNomic 的 processor 配置文件用于审计；已有 image tokens 直接使用。
RoMa checkpoint 和 DINOv3 TorchHub 代码已附带，无需联网下载权重；einops 0.8.1 也已原样附带。
固定 `models/fold0.json`、127×11 特征和严格 0 阈值；原 `infer.py` 未修改。
496/593 是历史五折成绩，不能作为这次本机固定单头测试成绩。
配对、缓存和原图使用 SHA 绑定；迁移只解析文件位置，不修改原 recipe 签名或 candidate/challenger 顺序。
上游源码 commit：`{manifest['recipe']['source_commit']}`；源训练图片重叠结果在 summary.json（当前报告为 partial）。
'''
    path=staging/'README.md';path.write_text(readme);add(path,'README.md')
    total=sum(source.stat().st_size for source in sources.values())
    print(json.dumps(dict(event='archive_start',path=str(output),files=len(sources),bytes=total,
        query_images=len(photo_images),reference_images=len(reference_images),pending_queries=len(pending),reused_partial_pairs=len(pairs))),flush=True)
    temp=output.with_suffix(output.suffix+'.partial');inventory=[];copied=0
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for index,(relative,source) in enumerate(sorted(sources.items())):
            hasher=hashlib.sha256();size=0
            with source.open('rb') as origin,archive.open(PREFIX+'/'+relative,'w',force_zip64=True) as target:
                while data:=origin.read(4<<20):target.write(data);hasher.update(data);size+=len(data)
            inventory.append(dict(path=relative,bytes=size,sha256=hasher.hexdigest()));copied+=size
            if (index+1)%250==0 or size>100<<20:
                print(json.dumps(dict(event='archive_progress',files=index+1,total_files=len(sources),bytes=copied,total_bytes=total)),flush=True)
        files_manifest=dict(schema='polish18_transfer_files.v1',files=inventory,transfer=transfer)
        archive.writestr(PREFIX+'/bundle_files.json',json.dumps(files_manifest,ensure_ascii=False,indent=2)+'\n')
    print('Checking archive file SHA256 values...',flush=True)
    with zipfile.ZipFile(temp) as archive:
        for index,record in enumerate(inventory):
            hasher=hashlib.sha256();size=0
            with archive.open(PREFIX+'/'+record['path']) as stream:
                while data:=stream.read(4<<20):hasher.update(data);size+=len(data)
            if hasher.hexdigest()!=record['sha256'] or size!=record['bytes']:raise ValueError('Archive verification failed: '+record['path'])
            if (index+1)%500==0:print(json.dumps(dict(event='archive_verified',files=index+1,total_files=len(inventory))),flush=True)
    temp.replace(output)
    with output.open('rb') as stream:archive_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(archive_sha+'  '+output.name+'\n')
    write(staging/'package_receipt.json',dict(archive=str(output),archive_sha256=archive_sha,archive_bytes=output.stat().st_size,
        files_verified=len(inventory),status='passed',transfer=transfer))
    print(json.dumps(dict(event='package_complete',path=str(output),bytes=output.stat().st_size,sha256=archive_sha,
        files_verified=len(inventory),pending_queries=len(pending),pending_references=len(needed),reused_partial_pairs=len(pairs))),flush=True)


if __name__=='__main__':main()
