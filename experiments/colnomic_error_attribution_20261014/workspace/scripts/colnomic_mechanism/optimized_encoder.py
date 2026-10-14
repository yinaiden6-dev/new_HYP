"""Identical unmerged arithmetic; keep small LoRA tensors and embedding lookup resident."""
import torch
from accelerate.hooks import remove_hook_from_module,add_hook_to_module,AlignDevicesHook
from scripts.colnomic_mechanism.offloaded_encoder import Encoder


def load(config,root):
 assert torch.cuda.device_count()>=2,'Two visible GPUs with ~1.4GB / ~1.8GB free required'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.15,0);torch.cuda.set_per_process_memory_fraction(.16,1)
 encoder=Encoder('colnomic7b',config,root,'cuda:1',device_map_override={'':'cpu'},offload_weights=True)
 base=encoder.model.get_base_model();embedding=base.language_model.embed_tokens
 remove_hook_from_module(embedding,recurse=True);embedding.to('cuda:0')
 add_hook_to_module(embedding,AlignDevicesHook(execution_device=torch.device('cuda:0'),offload=False,io_same_device=True))
 resident=[]
 for name,module in list(encoder.model.named_modules()):
  if isinstance(module,torch.nn.Linear) and ('.lora_A.' in name or '.lora_B.' in name):
   remove_hook_from_module(module,recurse=True);module.to('cuda:1');resident.append(dict(name=name,bytes=sum(p.numel()*p.element_size() for p in module.parameters()),dtype=str(module.weight.dtype)))
 report=dict(embedding_lookup_device='cuda:0',embedding_bytes=sum(p.numel()*p.element_size() for p in embedding.parameters()),
  lora_device='cuda:1',resident_lora_bytes=sum(r['bytes'] for r in resident),lora_modules=len(resident),
  arithmetic='unmerged LoRA; original BF16 base/FP32 LoRA/FP32 pre-Ampere SDPA; residency changes only',resident=resident)
 return encoder,report
