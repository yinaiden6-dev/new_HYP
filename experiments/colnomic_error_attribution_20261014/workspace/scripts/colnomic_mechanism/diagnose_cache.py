"""Audit actual LoRA tensor loading and isolate encoding recipe factors."""
import sys,time,importlib.metadata
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,sha,csvout
from scripts.colnomic_mechanism.diagnostic_encoder import Encoder


def adapter_audit(model,source):
 from peft.utils.save_and_load import get_peft_model_state_dict
 actual=get_peft_model_state_dict(model,save_embedding_layers=False)
 converted={k.replace('base_model.model.model.','base_model.model.language_model.',1):v for k,v in source.items()}
 common=set(actual)&set(converted);table=[]
 for key in sorted(common):
  value=actual[key].detach().cpu();expected=converted[key].to(value.dtype)
  table.append(dict(key=key,dtype=str(value.dtype),source_dtype=str(converted[key].dtype),
   exact=torch.equal(value,expected),max_absolute_error=float((value.float()-expected.float()).abs().max()),
   source_norm=float(expected.float().norm()),actual_norm=float(value.float().norm())))
 return dict(actual_keys=len(actual),source_keys=len(source),matched_keys=len(common),
  missing_after_legacy_name_conversion=sorted(set(converted)-set(actual)),extra_keys=sorted(set(actual)-set(converted)),
  exact_keys=sum(r['exact'] for r in table),mismatched_keys=sum(not r['exact'] for r in table),records=table)


def main():
 from safetensors.torch import load_file
 from accelerate import cpu_offload
 from accelerate.hooks import remove_hook_from_module
 from peft.utils.save_and_load import set_peft_model_state_dict
 from transformers import AutoImageProcessor
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261013)
 torch.cuda.set_per_process_memory_fraction(.17,0)
 folder=OUT/'cache_diagnosis';folder.mkdir(parents=True,exist_ok=True)
 config=read(ROOT/'configs/baselines.json')['models']['colnomic7b'];source=load_file(str(ROOT/config['checkpoint']/'adapter_model.safetensors'))
 captured={}
 def callback(model):
  captured.update(adapter_audit(model,source));write(folder/'automatic_adapter_loading.json',captured)
  print('Adapter audit:',{k:captured[k] for k in ['source_keys','actual_keys','matched_keys','exact_keys','mismatched_keys']},flush=True)
 encoder=Encoder('colnomic7b',config,ROOT,'cuda:0',device_map_override={'':'cpu'},offload_weights=True,before_offload=callback)
 banks={p:torch.load(ROOT/f'cache/image_embeddings/colnomic7b/{p}.pt',map_location='cpu',mmap=True,weights_only=True) for p in ['photos','references']}
 ids=['photo_41de3eb383934ec3','photo_21bd7df9ed6795e7','reference_000229','reference_000233'];targets=[]
 for id in ids:
  population='photos' if id.startswith('photo') else 'references';bank=banks[population];i=bank['ids'].index(id)
  targets.append((id,ROOT/bank['paths'][i],bank['embeddings'][i],bank['provenance'][i]))
 metrics=[]
 def run(name,backend):
  encoder.processor.image_processor=AutoImageProcessor.from_pretrained(ROOT/config['checkpoint'],local_files_only=True,backend=backend)
  for id,path,old,provenance in targets:
   start=time.monotonic();fresh=encoder.encode_image(path);assert fresh.shape==old.shape
   cosine=torch.nn.functional.cosine_similarity(old.float(),fresh.float(),dim=1)
   spec=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json');population='photos' if id.startswith('photo') else 'references'
   layout=next(r['layout'] for r in spec[population] if r['id']==id);visual=cosine[layout['start']:layout['start']+layout['count']]
   record=dict(recipe=name,backend=backend,id=id,provenance_kind=provenance['kind'],max_absolute_error=float((old.float()-fresh.float()).abs().max()),
    mean_cosine=float(cosine.mean()),visual_mean_cosine=float(visual.mean()),min_cosine=float(cosine.min()),seconds=time.monotonic()-start)
   metrics.append(record);torch.save(dict(embedding=fresh,recipe=name,backend=backend,id=id,image_sha256=sha(path)),folder/(name+'_'+id+'.pt'))
   csvout(folder/'recipe_metrics.csv',metrics);print(record,flush=True)
 run('automatic_adapter_fp32','torchvision')
 remove_hook_from_module(encoder.model,recurse=True)
 assert all(p.device.type!='meta' for p in encoder.model.parameters())
 converted={k.replace('base_model.model.model.','base_model.model.language_model.',1):v for k,v in source.items()}
 loaded=set_peft_model_state_dict(encoder.model,converted,adapter_name='default')
 strict=adapter_audit(encoder.model,source);write(folder/'explicit_adapter_loading.json',dict(**strict,load_unexpected=loaded.unexpected_keys))
 assert strict['mismatched_keys']==0 and not strict['missing_after_legacy_name_conversion'] and not strict['extra_keys']
 cpu_offload(encoder.model,execution_device=torch.device('cuda:0'),offload_buffers=True)
 run('explicit_adapter_fp32','torchvision')
 run('explicit_adapter_fp32_pil','pil')
 remove_hook_from_module(encoder.model,recurse=True)
 for name,param in encoder.model.named_parameters():
  if 'lora_' in name:param.data=param.data.to(torch.bfloat16)
 write(folder/'bf16_adapter_loading.json',adapter_audit(encoder.model,source))
 cpu_offload(encoder.model,execution_device=torch.device('cuda:0'),offload_buffers=True)
 run('explicit_adapter_bf16','torchvision')
 run('explicit_adapter_bf16_pil','pil')
 encoder.release()
 write(folder/'complete.json',dict(status='executed',automatic_adapter_tensor_parity=captured['mismatched_keys']==0,
  recipe_count=5,anchor_ids=ids,identity_labels_used=False,versions={n:importlib.metadata.version(n) for n in ['torch','transformers','colpali-engine','peft','accelerate']},
  automatic_loading_audit='automatic_adapter_loading.json',numeric_source_sha256=sha(Path(__file__).with_name('diagnostic_encoder.py'))))

if __name__=='__main__':main()
