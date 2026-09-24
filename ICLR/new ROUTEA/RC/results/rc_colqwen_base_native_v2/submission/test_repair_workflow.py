import importlib.util,sys,tempfile,os,subprocess,json
from pathlib import Path
r=Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC');p=r/'programs/run_rc_colqwen_base_native_v2.py'
spec=importlib.util.spec_from_file_location('cqrepair',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch
x=torch.eye(4);assert m.compare_tokens(x,x)['passed'];assert not m.compare_tokens(x,-x)['passed'];assert not m.compare_tokens(x,x[:2])['passed']
checks=[]
for mode in ['reuse_verified','reencode_gallery']:
 with tempfile.TemporaryDirectory() as d:
  m.OUT=Path(d);m.write(m.OUT/'compatibility.json',dict(mode=mode));calls=[]
  m.submit=lambda stage,dep=None,array=None: calls.append([stage,dep,array]) or str(9000000+len(calls))
  os.environ['SLURM_JOB_ID']='8000000';m.advance('compat',0)
  assert calls==[['score' if mode=='reuse_verified' else 'gallery','8000000','0']];checks.append([mode,calls])
with tempfile.TemporaryDirectory() as d:
 m.OUT=Path(d);m.write(m.OUT/'gallery/shards/00/validation.json',dict(status='BASE_GALLERY_SHARD_PASS'));calls=[]
 m.submit=lambda stage,dep=None,array=None: calls.append([stage,dep,array]) or str(9000000+len(calls))
 m.advance('gallery',0);assert calls==[['gallery','8000000','1-49%50'],['gallery-join','9000001',None]];checks.append(['gallery_chain',calls])
for kind in ['gpu','cpu']:
 text=(r/f'slurm/rc_colqwen_base_native_{kind}_v2.sbatch').read_text();block=text[text.index('set +e'):]
 for code in [0,75,124,1]:
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);(d/'timeout').write_text('#!/bin/sh\nexit "${TEST_CODE}"\n');(d/'scontrol').write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_TRACE"\n');(d/'pythonstub').write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_TRACE"\n')
   for name in ['timeout','scontrol','pythonstub']:(d/name).chmod(0o700)
   env=dict(os.environ,PATH=str(d)+':/usr/bin:/bin',PY=str(d/'pythonstub'),TEST_TRACE=str(d/'trace'),TEST_CODE=str(code),SLURM_JOB_ID='9000009',SLURM_ARRAY_JOB_ID='9000000',SLURM_ARRAY_TASK_ID='9',SLURM_RESTART_COUNT='0')
   cp=subprocess.run(['bash','-c','set -euo pipefail\n'+block,'mock','fit' if kind=='cpu' else 'score'],env=env,capture_output=True,text=True)
   trace=(d/'trace').read_text() if (d/'trace').exists() else ''
   assert cp.returncode==(1 if code==1 else 0),(kind,code,cp.stderr)
   if code in [75,124]:assert trace.strip()=='requeue 9000000_9',trace
   elif code==0:assert 'advance-' in trace,trace
   else:assert trace=='',trace
   checks.append([kind,code,'PASS'])
print(json.dumps(dict(status='PASS',checks=checks),indent=2))
