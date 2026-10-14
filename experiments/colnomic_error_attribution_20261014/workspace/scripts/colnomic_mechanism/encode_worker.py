"""Frozen GPU computation with CPU weight storage; fresh originals are always encoded."""
import argparse,time,sys,platform,importlib.metadata
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,sha
from scripts.colnomic_mechanism.offloaded_encoder import Encoder
from scripts.encode_colnomic_images import token_hash

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--processor-backend',choices=['pil','torchvision','legacy'],default='pil');args=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.16,0)
 runtime=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ['torch','transformers','colpali-engine','peft','accelerate','Pillow','numpy','torchvision']},
  gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,processor_backend=args.processor_backend)
 encoder_sha=sha(Path(__file__).with_name('offloaded_encoder.py'));worker_sha=sha(Path(__file__))
 config=read(ROOT/'configs/baselines.json')['models']['colnomic7b'];start=time.monotonic()
 model=Encoder('colnomic7b',config,ROOT,'cuda:0',device_map_override={'':'cpu'},offload_weights=True)
 from transformers import AutoImageProcessor
 if args.processor_backend!='legacy':model.processor.image_processor=AutoImageProcessor.from_pretrained(ROOT/config['checkpoint'],local_files_only=True,backend=args.processor_backend)
 print('Processor backend',getattr(model.processor.image_processor,'backend',type(model.processor.image_processor).__name__),flush=True)
 print('Frozen model loaded with CPU weight offload',time.monotonic()-start,'seconds',flush=True)
 records=read(args.manifest);args.output.mkdir(parents=True,exist_ok=True)
 for i,record in enumerate(records):
  path=Path(record['path'])
  if not path.is_absolute():path=args.manifest.parent/path
  target=args.output/(record['id']+'.pt')
  if target.exists():
   old=torch.load(target,map_location='cpu',weights_only=True)
   if (old['image_sha256']==sha(path) and old.get('processor_backend')==args.processor_backend
       and old.get('runtime')==runtime and old.get('experimental_encoder_sha256')==encoder_sha
       and old.get('worker_sha256')==worker_sha):print('Reused',record['id'],flush=True);continue
  started=time.monotonic();value=model.encode_image(path)
  metadata=dict(record,processor_backend=args.processor_backend,image_sha256=sha(path),all_tokens_sha256=token_hash(value),embedding=value,
   seconds=time.monotonic()-started,compute_dtype='torch.bfloat16',storage_dtype=str(value.dtype),
   encoder='original ShardedColQwen forward with CPU weight offload to one 2080Ti; no new training',
   source_encoder_sha256=sha(ROOT/'evaluation/encoders.py'),experimental_encoder_sha256=encoder_sha,
   worker_sha256=worker_sha,runtime=runtime)
  temporary=target.with_name(target.name+'.tmp');torch.save(metadata,temporary);temporary.replace(target)
  print('Encoded',i+1,'/',len(records),record['id'],'seconds',round(metadata['seconds'],2),'shape',tuple(value.shape),flush=True)
  write(args.output/'progress.json',dict(completed=i+1,total=len(records),last=record['id']))
 model.release()
 write(args.output/'complete.json',dict(status='complete',records=len(records),model_config=config,
  no_trainable_encoder_parameters=True,CPU_weights_GPU_computation=True,runtime=runtime,
  experimental_encoder_sha256=encoder_sha,worker_sha256=worker_sha))

if __name__=='__main__':main()
