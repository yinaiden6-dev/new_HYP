import importlib.util,sys,copy,tempfile,os,subprocess,json
from pathlib import Path
import torch
r=Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC');sys.path.insert(0,str(r/'programs'));p=r/'programs/run_rc_m_distill_full_repair_v1.py'
spec=importlib.util.spec_from_file_location('fullrepair',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);torch.set_num_threads(2)
g=torch.Generator().manual_seed(99);q=dict(z=torch.randn(5,128,generator=g,dtype=torch.float64),xy=torch.rand(5,2,generator=g,dtype=torch.float64));refs=[dict(z=torch.randn(7+i,128,generator=g,dtype=torch.float64),xy=torch.rand(7+i,2,generator=g,dtype=torch.float64)) for i in range(3)]
original=[];rebuilt=[]
for ref in refs:
 for z,x in zip([q,ref],m.D.C.relation(q['z'],ref['z'],q['xy'],ref['xy'],'PAIR')):
  y=m.expand(z,m.compact(x));assert torch.equal(x,y);original.append(x);rebuilt.append(y)
a=m.D.C.Quality();a.up.weight.data.normal_(generator=g,std=.01);b=copy.deepcopy(a)
x=dict(x=torch.cat(original),lengths=list(map(len,original)),teacher=torch.tensor([.01,.03,.1],dtype=torch.float64),query_id='synthetic');y={**x,'x':torch.cat(rebuilt)}
l1=m.S.objective(m.D.packed_mass(a,x)[0],x['teacher'],1);l2=m.S.objective(m.D.packed_mass(b,y)[0],y['teacher'],1);l1.backward();l2.backward();assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
assert m.evaluate(a,[x],0.)==m.B.evaluate(a,[x],0.)
# Exercise dependency release with fake scheduler calls and pinned gate files.
checks=[]
for passed in [False,True]:
 with tempfile.TemporaryDirectory() as tmp:
  m.OUT=Path(tmp);m.write(m.OUT/'n64/result.json',dict(status='FIT_GATE_PASS' if passed else 'FIT_GATE_NOT_MET'))
  os.environ['SLURM_JOB_ID']='8001';calls=[];m.submit=lambda stage,n=0,dep=None,array=None:calls.append([stage,n,dep,array]) or str(9000+len(calls))
  m.advance(dict(max_chunks=16),'run',64)
  assert bool(calls)==passed
  if passed:assert calls==[['cache',0,'8001','0-7%8'],['cache-join',0,'9001',None]]
  checks.append(['gate',passed,'PASS'])
with tempfile.TemporaryDirectory() as tmp:
 m.OUT=Path(tmp);calls=[];m.submit=lambda stage,n=0,dep=None,array=None:calls.append([stage,n,dep,array]) or '9003';m.advance({},'cache-join',0);assert calls==[['run',457,'8001',None]]
print(json.dumps(dict(status='COMPACT_BIT_EXACT_FORWARD_GRADIENT_EVALUATION_AND_DEPENDENCY_PASS',checks=checks)))
