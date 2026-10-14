"""One resumable encoder worker for committed deterministic control manifests."""
import argparse
import fcntl
import importlib.metadata
import math
from pathlib import Path
import time
import torch

from scripts.colnomic_mechanism.common import ROOT,read,write,csvout,sha
from scripts.colnomic_mechanism.optimized_encoder import load
from scripts.encode_colnomic_images import token_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-directory',type=Path)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(args.manifest)
    assert len(manifest)==len({r['id'] for r in manifest})
    if args.limit is not None:manifest=manifest[:args.limit]
    runtime={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','colpali-engine','Pillow','torchvision']}
    sources={}
    if args.prior_directory:
        for path in args.prior_directory.glob('*.pt'):
            old=torch.load(path,map_location='cpu',weights_only=True)
            if old.get('processor_backend')=='torchvision' and old.get('runtime')==runtime:
                assert token_hash(old['embedding'])==old['tensor_sha256']
                sources[old['image_sha256']]=path
    qb=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',mmap=True,weights_only=True,map_location='cpu')
    qi={q:i for i,q in enumerate(qb['ids'])}
    parity=[];encoder=None;residency=None;reused=0;encoded=0
    try:
        for i,row in enumerate(manifest):
            target=output/(row['id']+'.pt');digest=sha(row['path'])
            assert row.get('image_sha256',digest)==digest,'manifest image SHA differs from input'
            if target.exists():
                record=torch.load(target,map_location='cpu',weights_only=True)
                assert record['image_sha256']==digest and record['runtime']==runtime and token_hash(record['embedding'])==record['tensor_sha256']
                assert all(record.get(k)==v for k,v in row.items())
                print('Resume',i+1,'/',len(manifest),row['id'],flush=True)
            elif digest in sources:
                old=torch.load(sources[digest],map_location='cpu',weights_only=True)
                record=dict(row)
                record.update(embedding=old['embedding'],image_sha256=digest,tensor_sha256=old['tensor_sha256'],runtime=runtime,
                    processor_backend='torchvision',reused_from=str(sources[digest]),reused_source_sha256=sha(sources[digest]),
                    original_cached_tensors_untouched=True,elapsed_seconds=0.)
                reused+=1
                torch.save(record,target.with_suffix('.pt.tmp'));target.with_suffix('.pt.tmp').replace(target)
                print('Reuse',i+1,'/',len(manifest),row['id'],flush=True)
            else:
                if encoder is None:
                    assert torch.cuda.device_count()>=2
                    free=[torch.cuda.mem_get_info(j)[0] for j in [0,1]]
                    fractions=[.15,.16]
                    needed=[math.ceil(torch.cuda.get_device_properties(j).total_memory*fractions[j])+64*2**20 for j in [0,1]]
                    if any(a<b for a,b in zip(free,needed)):
                        write(output/'resource_wait.json',dict(reason='insufficient unallocated memory for bounded worker',free_bytes=free,required_bytes=needed,next_id=row['id']))
                        raise RuntimeError('Wait for existing GPU jobs: allocator cap plus64MiB headroom unavailable; no other task was stopped')
                    write(output/'preflight.json',dict(free_after_cuda_context_bytes=free,allocator_caps_fraction=fractions,minimum_bytes=needed,
                        basis='same validated optimized encoder caps; compare after creating CUDA context, not nvidia-smi pre-context free memory'))
                    config=read(ROOT/'configs/baselines.json')['models']['colnomic7b']
                    encoder,residency=load(config,ROOT)
                    from transformers import AutoImageProcessor
                    encoder.processor.image_processor=AutoImageProcessor.from_pretrained(ROOT/config['checkpoint'],local_files_only=True,backend='torchvision')
                    write(output/'runtime.json',dict(packages=runtime,model_config=config,residency=residency,manifest_sha256=sha(args.manifest),
                        worker_sha256=sha(Path(__file__)),optimized_encoder_sha256=sha(Path(__file__).with_name('optimized_encoder.py')),
                        numeric_encoder_sha256=sha(Path(__file__).with_name('offloaded_encoder.py')),processor_backend='torchvision',
                        GPU_memory_caps_fraction=[.15,.16],other_processes_not_stopped=True))
                start=time.monotonic();embedding=encoder.encode_image(Path(row['path']))
                assert embedding.ndim==2 and embedding.shape[1]==128 and torch.isfinite(embedding).all()
                record=dict(row)
                record.update(embedding=embedding,image_sha256=digest,tensor_sha256=token_hash(embedding),runtime=runtime,
                    processor_backend='torchvision',original_cached_tensors_untouched=True,elapsed_seconds=time.monotonic()-start)
                torch.save(record,target.with_suffix('.pt.tmp'));target.with_suffix('.pt.tmp').replace(target)
                encoded+=1
                print('Encode',i+1,'/',len(manifest),row['id'],round(record['elapsed_seconds'],2),flush=True)
            if row['arm']=='base':
                cached=qb['embeddings'][qi[row['source_id']]];fresh=record['embedding']
                assert digest==qb['sha256'][qi[row['source_id']]] and cached.shape==fresh.shape
                parity.append(dict(photo_id=row['source_id'],package_id=row['package_id'],split=row['split'],cache_row=qi[row['source_id']],
                    cache_provenance=qb['provenance'][qi[row['source_id']]]['kind'],exact=torch.equal(cached,fresh),
                    max_absolute_error=float((cached.float()-fresh.float()).abs().max()),
                    cosine_mean=float(torch.nn.functional.cosine_similarity(cached.float(),fresh.float(),dim=1).mean()),
                    interpretation='same cached runtime if exact; otherwise separately labeled new-runtime control only'))
                csvout(output/'original_cache_parity.csv',parity)
            write(output/'progress.json',dict(completed=i+1,total=len(manifest),manifest_total=len(read(args.manifest)),last=row['id'],newly_encoded=encoded,reused_compatible=reused))
        write(output/'complete.json',dict(status='complete' if args.limit is None else 'partial_limit',records=len(manifest),unique_sources=len({r['source_id'] for r in manifest}),
              original_cache_exact=sum(r['exact'] for r in parity),original_cache_not_exact=sum(not r['exact'] for r in parity),
              not_exact_originals_are_new_runtime_not_cache_reproductions=True,manifest_sha256=sha(args.manifest)))
    finally:
        if encoder is not None:encoder.release()


if __name__=='__main__':main()
