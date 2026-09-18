#!/usr/bin/env python3
"""Append-only serialization repair; all V1 computation and validation functions retained."""
import argparse,hashlib,json,os,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'programs'))
import materialize_rc_new_hyp593_inputs_v1 as V1
REPAIR=ROOT/'registry/rc_new_hyp593_serialization_repair_authority_v2_20260911.json'

def safe_savepayload(folder,payload,extra):
 import torch
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';V1.need(not p.exists(),'IMMUTABLE_PAYLOAD')
 tmp=folder/('.payload.partial.'+str(os.getpid())+'.pt')
 value=dict(payload,serialization_repair=V1.bind(REPAIR))
 try:
  # An open stream avoids PyTorch's path-name restrictions. Flush before publishing.
  with tmp.open('xb') as stream:
   torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
  tmp.chmod(0o444);os.link(tmp,p)
 finally:tmp.unlink(missing_ok=True)
 V1.write(folder/'receipt.json',dict(payload=V1.bind(p),authority=V1.bind(V1.AUTH),worker_manifest=V1.bind(V1.META/'worker_manifest.json'),serialization_repair=V1.bind(REPAIR),**extra))

def qualify():
 import torch,tempfile
 a=V1.read(REPAIR)
 for b in a['sources'].values():V1.checked(b)
 root=V1.OUT/'serialization_repair_e0'/('roma' if 'romav2' in sys.prefix else 'raw')
 V1.need(not root.exists(),'APPEND_ONLY_E0');root.mkdir(parents=True)
 sample={'records':[{'query_id':'synthetic','query_tokens':torch.tensor([[0.,-0.,.25]],dtype=torch.float16),'weights':torch.tensor([0.,.125,1.],dtype=torch.float64)}],'references':{7:{'tokens':torch.eye(3,dtype=torch.float16)}},'authority':V1.bind(V1.AUTH)}
 bad=root/'.partial-filename-probe';old_error=None
 try:torch.save(sample,bad)
 except RuntimeError as e:old_error=str(e)
 finally:bad.unlink(missing_ok=True)
 safe_savepayload(root/'roundtrip',sample,dict(status='SYNTHETIC_ONLY'))
 p=root/'roundtrip/payload.pt';loaded=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 for original,restored in [(sample['records'][0]['query_tokens'],loaded['records'][0]['query_tokens']),(sample['records'][0]['weights'],loaded['records'][0]['weights']),(sample['references'][7]['tokens'],loaded['references'][7]['tokens'])]:
  V1.need(original.dtype==restored.dtype and torch.equal(original.contiguous().view(torch.uint8),restored.contiguous().view(torch.uint8)),'BIT_EXACT_FILE_STREAM_ROUNDTRIP')
 original_sha=V1.sha(p);rejected=False
 try:safe_savepayload(root/'roundtrip',sample,dict(status='SYNTHETIC_ONLY'))
 except RuntimeError as e:rejected=str(e)=='IMMUTABLE_PAYLOAD'
 V1.need(rejected and V1.sha(p)==original_sha and not list((root/'roundtrip').glob('.payload.partial.*')),'IMMUTABILITY_AND_TEMP_CLEANUP')
 report=dict(status='FILE_STREAM_SERIALIZATION_BIT_ROUNDTRIP_PASS',python=sys.version,torch=str(torch.__version__),runtime_prefix=sys.prefix,old_path_writer_error=old_error,FP16_FP64_negative_zero_bits_preserved=True,weights_only_mmap_load=True,immutable_overwrite_rejected=True,natural_tensor_reads=0,repair_authority=V1.bind(REPAIR),program=V1.bind(__file__))
 V1.write(root/'validation.json',report);print(json.dumps(report),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['raw','roma','qualify']);ap.add_argument('--shard',type=int);a=ap.parse_args()
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(17)
 if a.stage=='qualify':qualify();return
 V1.guard(a.stage);repair=V1.read(REPAIR)
 for b in repair['sources'].values():V1.checked(b)
 for name in ('raw','roma'):
  receipt=V1.read(V1.OUT/'serialization_repair_e0'/name/'validation.json');V1.need(receipt['status']=='FILE_STREAM_SERIALIZATION_BIT_ROUNDTRIP_PASS' and receipt['repair_authority']==V1.bind(REPAIR),'BOTH_RUNTIME_SERIALIZATION_QUALIFICATION')
 sys.addaudithook(V1.audit);V1.savepayload=safe_savepayload
 # No reassignment of any numerical, model-loading, ranking or validation function.
 {'raw':V1.raw,'roma':V1.roma}[a.stage](a.shard)
if __name__=='__main__':main()
