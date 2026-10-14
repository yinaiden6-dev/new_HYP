"""Label-free tests of historical gallery attention, channel order and resize choices."""
import sys,time,importlib.metadata
from pathlib import Path
import torch,numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,csvout,sha
from scripts.colnomic_mechanism.optimized_encoder import load


def main():
 from transformers import AutoImageProcessor
 folder=OUT/'cache_diagnosis';folder.mkdir(parents=True,exist_ok=True);config=read(ROOT/'configs/baselines.json')['models']['colnomic7b'];encoder,resident=load(config,ROOT);write(folder/'resident_execution.json',resident)
 banks={p:torch.load(ROOT/f'cache/image_embeddings/colnomic7b/{p}.pt',weights_only=True,map_location='cpu',mmap=True) for p in ['photos','references']}
 # Focus photos are used ONLY as verification anchors; no probe-selection labels are read.
 ids=['photo_41de3eb383934ec3','reference_000229','reference_000233'];records=[]
 def run(name,backend='torchvision',resample=None,bgr=False,gpu_preprocess=False,siglip_normalization=False):
  encoder.processor.image_processor=AutoImageProcessor.from_pretrained(ROOT/config['checkpoint'],local_files_only=True,backend=backend)
  if resample is not None:encoder.processor.image_processor.resample=resample
  if siglip_normalization:
   encoder.processor.image_processor.image_mean=[.5,.5,.5];encoder.processor.image_processor.image_std=[.5,.5,.5]
  for id in ids:
   b=banks['photos' if id.startswith('photo') else 'references'];index=b['ids'].index(id);path=ROOT/b['paths'][index];old=b['embeddings'][index].float()
   with Image.open(path) as im:rgb=im.convert('RGB')
   if bgr:rgb=Image.fromarray(np.asarray(rgb)[:,:,::-1].copy())
   inputs=encoder.processor.process_images([rgb])
   if gpu_preprocess:
    pixels=encoder.processor.image_processor(images=[rgb],device='cuda:1',return_tensors='pt')
    assert torch.equal(pixels['image_grid_thw'].cpu(),inputs['image_grid_thw'].cpu())
    inputs['pixel_values']=pixels['pixel_values'].reshape(inputs['pixel_values'].shape)
   start=time.monotonic();fresh=encoder.encode_prepared(inputs)[0][0];cos=torch.nn.functional.cosine_similarity(old,fresh.float(),dim=1)
   row=dict(recipe=name,id=id,max_absolute_error=float((old-fresh.float()).abs().max()),mean_cosine=float(cos.mean()),min_cosine=float(cos.min()),seconds=time.monotonic()-start)
   records.append(row);csvout(folder/'extra_recipe_metrics.csv',records);torch.save(dict(embedding=fresh,id=id,recipe=name,image_sha256=sha(path)),folder/(name+'_'+id+'.pt'));print(row,flush=True)
 if '--remaining' not in sys.argv:run('resident_exact_control')
 else:run('resident_exact_control_remaining')
 assert records[0]['max_absolute_error']==0.,'Residency optimization must replay recent cached query exactly'
 if '--remaining' not in sys.argv:
  run('channel_BGR_verification',bgr=True)
  run('bilinear_resize_verification',resample=Image.Resampling.BILINEAR)
 else:
  run('gpu_resize_verification',gpu_preprocess=True)
  run('siglip_normalization_verification',siglip_normalization=True)
 # Eager uses its own QK/softmax arithmetic instead of the registered FP32 SDPA.
 # Same eager algebra with query-row blocking to fit shared GPUs. No key-axis approximation.
 import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
 original_eager=qwen.eager_attention_forward
 def chunked_eager(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
  groups=query.shape[1]//key.shape[1]
  if groups!=1:key=key.repeat_interleave(groups,dim=1);value=value.repeat_interleave(groups,dim=1)
  outputs=[]
  for begin in range(0,query.shape[2],128):
   end=min(begin+128,query.shape[2]);weights=query[:,:,begin:end]@key.transpose(2,3);weights=weights*scaling
   if attention_mask is not None:weights=weights+attention_mask[:,:,begin:end,:key.shape[2]]
   weights=torch.nn.functional.softmax(weights,dim=-1,dtype=torch.float32).to(query.dtype)
   outputs.append(weights@value)
  return torch.cat(outputs,dim=2).transpose(1,2).contiguous(),None
 class Toy:
  num_key_value_groups=1
  training=False
 toy=Toy();q=torch.randn(1,2,257,16,device='cuda:1',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q)
 expected=original_eager(toy,q,k,v,None,.25)[0];observed=chunked_eager(toy,q,k,v,None,.25)[0]
 assert torch.equal(expected,observed),'blocked eager must match full eager on fixed synthetic input'
 qwen.eager_attention_forward=chunked_eager
 for module in encoder.model.modules():
  if hasattr(module,'config') and hasattr(module.config,'_attn_implementation'):module.config._attn_implementation='eager'
 run('eager_attention_chunked_verification')
 encoder.release();write(folder/'extra_complete.json',dict(status='executed',selection_by_embedding_parity_only=True,identity_labels_used=False,
  original_banks_modified=False,resident_recent_photo_exact=True,versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','accelerate']}))

if __name__=='__main__':main()
