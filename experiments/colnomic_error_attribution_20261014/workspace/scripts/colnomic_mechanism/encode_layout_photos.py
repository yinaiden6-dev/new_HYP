"""Encode only new control images; verify originals against their original cache rows."""
import sys,time,importlib.metadata
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,csvout,sha
from scripts.colnomic_mechanism.optimized_encoder import load
from scripts.encode_colnomic_images import token_hash


def main():
 from transformers import AutoImageProcessor
 folder=OUT/'stage4_extended_photos_v3';manifest=read(folder/'encoding_manifest.json');output=folder/'embeddings';output.mkdir(exist_ok=True)
 config=read(ROOT/'configs/baselines.json')['models']['colnomic7b'];runtime={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','colpali-engine','Pillow','torchvision']}
 encoder,resident=load(config,ROOT);encoder.processor.image_processor=AutoImageProcessor.from_pretrained(ROOT/config['checkpoint'],local_files_only=True,backend='torchvision')
 write(folder/'encoder_runtime.json',dict(packages=runtime,residency=resident,model_config=config,
  numeric_source_sha256=sha(Path(__file__).with_name('offloaded_encoder.py')),worker_sha256=sha(Path(__file__)),optimized_source_sha256=sha(Path(__file__).with_name('optimized_encoder.py')),
  arithmetic_unchanged=True,new_encoder_training=False,original_cache_not_modified=True,processor_backend='torchvision'))
 bank=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True);indices={qid:i for i,qid in enumerate(bank['ids'])};parity=[]
 for i,row in enumerate(manifest):
  path=Path(row['path']);target=output/(row['id']+'.pt');digest=sha(path)
  if target.exists():
   old=torch.load(target,map_location='cpu',weights_only=True)
   if old['image_sha256']==digest and old.get('runtime')==runtime and old.get('processor_backend')=='torchvision':
    value=old;print('Reused',row['id'],flush=True)
   else:raise ValueError('Existing incompatible output: '+str(target))
  else:
   start=time.monotonic();z=encoder.encode_image(path);value=dict(**row,embedding=z,image_sha256=digest,tensor_sha256=token_hash(z),runtime=runtime,
    processor_backend='torchvision',seconds=time.monotonic()-start,original_cache_unchanged=True)
   tmp=target.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(target)
   print('Encoded',i+1,'/',len(manifest),row['id'],'seconds',round(value['seconds'],2),flush=True)
  if row['arm']=='base':
   index=indices[row['source_id']];cached=bank['embeddings'][index];fresh=value['embedding'];assert fresh.shape==cached.shape and digest==bank['sha256'][index]
   record=dict(photo_id=row['source_id'],split=row['split'],package_id=row['package_id'],cache_row=index,
    exact=torch.equal(fresh,cached),max_absolute_error=float((fresh.float()-cached.float()).abs().max()),
    mean_cosine=float(torch.nn.functional.cosine_similarity(fresh.float(),cached.float(),dim=1).mean()))
   parity.append(record);csvout(folder/'original_cache_replay.csv',parity)
  write(folder/'encoding_progress.json',dict(completed=i+1,total=len(manifest),last=row['id']))
 encoder.release();write(folder/'encoding_complete.json',dict(status='executed',records=len(manifest),unique_sources=len(parity),
  all_original_query_cache_replays_exact=all(r['exact'] for r in parity),
  unchanged_gallery='cache/image_embeddings/colnomic7b/references.pt',original_cache_replay='original_cache_replay.csv'))

if __name__=='__main__':main()
