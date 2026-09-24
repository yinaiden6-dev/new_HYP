import importlib.util,copy,tempfile,subprocess,os,json
from pathlib import Path
import torch
r=Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC');path=r/'programs/run_rc_m_distill_optimizer_repair_v1.py'
s=importlib.util.spec_from_file_location('repair',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(2);g=torch.Generator().manual_seed(77)
data=[dict(x=torch.randn(38,520,generator=g,dtype=torch.float64),lengths=[5,7,6,8,5,7],teacher=torch.tensor([.01,.03,.1],dtype=torch.float64)) for _ in range(3)]
a=m.D.C.Quality();a.up.bias.data.fill_(-3.);b=copy.deepcopy(a)
loss=sum(m.S.objective(m.D.packed_mass(a,d)[0],d['teacher'],1) for d in data)/3;loss.backward()
for d in data:(m.S.objective(m.D.packed_mass(b,d)[0],d['teacher'],1)/3).backward()
error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()));assert error<1e-12,error
opt=torch.optim.LBFGS(b.parameters(),lr=1.,max_iter=1,max_eval=8,line_search_fn='strong_wolfe')
def closure():
 opt.zero_grad();v=0.
 for d in data:
  y=m.S.objective(m.D.packed_mass(b,d)[0],d['teacher'],1)/3;y.backward();v+=float(y.detach())
 return torch.tensor(v,dtype=torch.float64)
before=float(closure());opt.step(closure);after=float(closure());assert after<before
# Resume reproduces next optimizer update exactly.
c=copy.deepcopy(b);op2=torch.optim.LBFGS(c.parameters(),lr=1.,max_iter=1,max_eval=8,line_search_fn='strong_wolfe');op2.load_state_dict(copy.deepcopy(opt.state_dict()))
def closure2():
 op2.zero_grad();v=0.
 for d in data:
  y=m.S.objective(m.D.packed_mass(c,d)[0],d['teacher'],1)/3;y.backward();v+=float(y.detach())
 return torch.tensor(v,dtype=torch.float64)
opt.step(closure);op2.step(closure2);assert all(torch.equal(x,y) for x,y in zip(b.parameters(),c.parameters()))
text=(r/'slurm/rc_m_distill_optimizer_repair_v1.sbatch').read_text();block=text[text.index('set +e'):]
for code in [0,124,1]:
 with tempfile.TemporaryDirectory() as name:
  d=Path(name)
  for f,t in {'timeout':'exit "$TEST_CODE"','scontrol':'echo "$*" >> "$TRACE"','pythonstub':'echo "$*" >> "$TRACE"'}.items():
   p=d/f;p.write_text('#!/bin/sh\n'+t+'\n');p.chmod(0o700)
  env=dict(os.environ,PATH=str(d)+':/usr/bin:/bin',PY=str(d/'pythonstub'),TEST_CODE=str(code),TRACE=str(d/'trace'),SLURM_JOB_ID='90001',SLURM_RESTART_COUNT='0')
  cp=subprocess.run(['bash','-c','set -euo pipefail\n'+block,'mock','run','16'],env=env,capture_output=True,text=True)
  assert cp.returncode==(1 if code==1 else 0),(code,cp.stderr)
  trace=(d/'trace').read_text() if (d/'trace').exists() else ''
  assert (code!=124 or trace.strip()=='requeue 90001') and (code!=0 or 'advance 16' in trace)
print(json.dumps(dict(status='FULL_BATCH_GRADIENT_OPTIMIZER_RESUME_AND_LAUNCHER_PASS',gradient_error=error,synthetic_before=before,synthetic_after=after)))
