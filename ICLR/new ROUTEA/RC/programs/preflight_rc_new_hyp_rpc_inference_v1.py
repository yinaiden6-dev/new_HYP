#!/usr/bin/env python3
"""Synthetic-only RPC adapter and bootstrap checks; no external outcome reads."""
import ast,collections,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_new_hyp_rpc_inference_v1 as I
import analyze_rc_new_hyp_rpc_external_v1 as J
OUT=ROOT/'results/rc_new_hyp_rpc_transfer_v1/submission'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def function_ast(path,name):return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
def main():
    checks={}
    old=ROOT/'programs/run_rc_new_hyp_grozi120_inference_v1.py';new=ROOT/'programs/run_rc_new_hyp_rpc_inference_v1.py'
    assert function_ast(old,'predict').replace('GROZI','RPC')==function_ast(new,'predict')
    checks['head_feature_logits_action_ast_unchanged']=True
    for scores in [torch.arange(200,dtype=torch.float64),torch.zeros(200,dtype=torch.float64),torch.tensor([i%11 for i in range(200)],dtype=torch.float64)]:
        labels=['SYNTH-'+str(i) for i in range(200)]
        expected=sorted(range(200),key=lambda i:(-float(scores[i]),i))
        assert I.rank(scores,labels)==expected
        assert I.S.top128_rows(scores,labels)==expected[:128]
    checks['full200_rank_and_natural128_ties']=True
    for scenario in ('equal','all_gain','heterogeneous'):
        rows=[]
        for i in range(200):
            for camera in range(3):
                b=False if scenario=='all_gain' else (i+camera)%5==0
                m=b if scenario=='equal' else True if scenario=='all_gain' else (b and i%11!=0) or i%7==0
                rows.append(dict(identity=f'SYNTH-{i:03}',stratum=f'STRATUM-{i%17:02}',correct={'B':b,'M':m}))
        result=J.comparison(rows,'B','M')
        delta=np.array([sum(int(rows[3*i+c]['correct']['M'])-int(rows[3*i+c]['correct']['B']) for c in range(3))/3 for i in range(200)])
        rng=np.random.default_rng(20260913);sampled=[]
        for g in range(17):
            ids=np.array([i for i in range(200) if i%17==g])
            sampled.append(ids[rng.integers(len(ids),size=(100000,len(ids)))])
        direct=delta[np.concatenate(sampled,axis=1)].mean(axis=1)
        assert np.allclose(np.quantile(direct,[.025,.975]),result['sku_stratified_bootstrap95'],rtol=0,atol=1e-12)
        assert abs(delta.mean()-result['equal_sku_difference'])<1e-12
        if scenario=='equal':assert result['sku_stratified_bootstrap95']==[0,0] and not result['reliable_positive']
        if scenario=='all_gain':assert np.allclose(result['sku_stratified_bootstrap95'],[1,1]) and result['reliable_positive']
        checks['independent_stratified_bootstrap_'+scenario]=True
    branches=[]
    with tempfile.TemporaryDirectory(prefix='rpc-infer-stub-') as temp:
        stub=Path(temp)/'timeout';stub.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');stub.chmod(0o755)
        environment=dict(os.environ,PATH=temp+':/usr/bin:/bin',rc_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python',rc_roma_python='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-romav2/bin/python',SLURM_ARRAY_TASK_ID='74')
        script=ROOT/'slurm/rc_new_hyp_rpc_inference_v1.sbatch';body=script.read_text();body=body[body.index('case "${1:?stage}" in'):]
        for stage,seconds,key in [('raw','540s','rc_python'),('roma','840s','rc_roma_python')]:
            got=json.loads(subprocess.run(['bash','-c',body,'fixture',stage],env=environment,check=True,capture_output=True,text=True).stdout)
            expected=['--signal=TERM','--kill-after=10s',seconds,environment[key],'programs/run_rc_new_hyp_rpc_inference_v1.py',stage,'--shard','74']
            assert got==expected;branches.append(dict(stage=stage,argv=got))
        join=ROOT/'slurm/rc_new_hyp_rpc_join_v1.sbatch';body=join.read_text();body=body[body.index('exec timeout'):]
        got=json.loads(subprocess.run(['bash','-c',body],env=environment,check=True,capture_output=True,text=True).stdout)
        assert got==['--signal=TERM','--kill-after=10s','540s',environment['rc_python'],'programs/analyze_rc_new_hyp_rpc_external_v1.py']
        branches.append(dict(stage='join',argv=got))
    for name in ('rc_new_hyp_rpc_reference_encoding_v1.sbatch','rc_new_hyp_rpc_inference_v1.sbatch','rc_new_hyp_rpc_join_v1.sbatch'):subprocess.run(['bash','-n',str(ROOT/'slurm'/name)],check=True)
    runtime_script="""import sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'programs'),str(Path.cwd()/'src')]
import run_rc_new_hyp_rpc_inference_v1 as I
I.torch.set_num_threads(8);I.torch.set_num_interop_threads(1);I.torch.set_float32_matmul_precision('highest');I.torch.manual_seed(17)
if sys.argv[1]=='raw':
 p=I.read(I.M.RAW_PREFLIGHT)
 assert I.R.runtime_libraries()==p['runtime_libraries']
 print(json.dumps(dict(stage='raw',worker_import=True,runtime_libraries=I.R.runtime_libraries())))
else:
 m=I.M.loadmodule(I.M.ROMA_SOURCE,'rpc_preflight_runtime');p=I.read(I.M.PROFILE)
 runtime=m.check_runtime(p);_,digest=I.M.compile_roma(m)
 assert digest=='2ee24877651fb928bcfeb4f193beab28dbcda3cf4ef686e26fdc02f393a62cbb'
 print(json.dumps(dict(stage='roma',worker_import=True,runtime=runtime,numeric_loop_sha256=digest)))
"""
    environments=[]
    for stage,venv in [('raw','.venv-colpali'),('roma','.venv-romav2')]:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TORCH_HOME='/hkfs/work/workspace/scratch/ap7811-benchmark/third_party/model_cache/torch')
        p=subprocess.run(['/hkfs/work/workspace/scratch/ap7811-benchmark/'+venv+'/bin/python','-c',runtime_script,stage],env=env,capture_output=True,text=True)
        assert p.returncode==0, p.stdout+p.stderr
        environments.append(json.loads(p.stdout))
    result=dict(status='RPC_SYNTHETIC_ADAPTER_STATISTICS_LAUNCH_RUNTIME_PASS',checks=checks,branches=branches,runtimes=environments,program=bind(__file__),inference_program=bind(new),join_program=bind(ROOT/'programs/analyze_rc_new_hyp_rpc_external_v1.py'),model_forwards=0,external_outcome_reads=0)
    I.write(OUT/'inference_preflight_v1.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks,runtimes=[r['stage'] for r in environments])))
if __name__=='__main__':main()
