#!/usr/bin/env python3
"""Device-only CPU execution supplement for frozen ColPali native scoring."""
import argparse
import ast
import copy
import inspect
import os
from pathlib import Path
import sys
import time
import run_rc_colpali_native_mass_head_v1 as H

ROOT=H.ROOT
NATIVE=H.OUT
OUT=ROOT/'results/rc_colpali_native_mass_cpu_execution_v1'
AUTH=ROOT/'registry/rc_colpali_native_mass_cpu_execution_v1_20260924.json'
PLAN=ROOT/'plan/RC_COLPALI_NATIVE_MASS_CPU_EXECUTION_20260924.md'
LAUNCH=ROOT/'slurm/rc_colpali_native_mass_cpu_score_v1.sbatch'


def cpu_function():
    tree=ast.parse(inspect.getsource(H.score))
    class DeviceOnly(ast.NodeTransformer):
        assertions=0
        devices=0
        def visit_Assert(self,node):
            if ast.unparse(node.test)=='torch.cuda.is_available()':
                self.assertions+=1;node.test=ast.UnaryOp(op=ast.Not(),operand=node.test)
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='cuda':
                self.devices+=1;node.args[0]=ast.Constant(value='cpu')
            return self.generic_visit(node)
    converter=DeviceOnly();converted=converter.visit(copy.deepcopy(tree));ast.fix_missing_locations(converted)
    assert converter.assertions==1 and converter.devices==2
    namespace={};exec(compile(converted,str(Path(H.__file__))+' [CPU device-only]','exec'),H.__dict__,namespace)
    return namespace['score'],dict(assertions=converter.assertions,device_literals=converter.devices)


def prepare():
    assert not AUTH.exists()
    _,changes=cpu_function();parent=H.read(H.AUTH)
    for b in parent['sources']:H.checked(b)
    H.write(AUTH,dict(status='CPU_DEVICE_ONLY_EXECUTION_AUTHORIZED',parent=H.bind(H.AUTH),
        sources=[H.bind(p) for p in (Path(__file__),PLAN,LAUNCH)],
        transformed_source=H.bind(Path(H.__file__)),ast_changes=changes,qualification_ordinals=[0,50],
        pilot_gpu_validation=H.bind(NATIVE/'shards/00/validation.json'),
        float_tolerance=2e-9,precision='FP64',cpu_only=True,identity_reads=0))
    print('CPU_EXECUTION_PREPARED',flush=True)


def guard():
    a=H.read(AUTH)
    for b in a['sources']+[a['parent'],a['transformed_source'],a['pilot_gpu_validation']]:H.checked(b)
    parent=H.read(H.AUTH)
    for b in parent['sources']:H.checked(b)
    allowed={Path(parent['workers']['path']).resolve()}
    def audit(event,args):
        if event=='socket.connect':assert not isinstance(args[1],tuple),'OFFLINE'
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix=='.json' and ROOT/'results' in p.parents:
            assert not any(x in p.parts for x in ('fold0','fold1','fold2','fold3','fold4')),'NO_FOLD_LABELS'
            assert NATIVE in p.parents or OUT in p.parents or H.TOK in p.parents or p in allowed,('UNAUTHORIZED_RESULT',str(p))
    sys.addaudithook(audit)
    return a,parent,H.read(H.checked(parent['workers']))['records']


def run(stage,index,budget):
    assert os.environ.get('SLURM_JOB_ID')
    import numpy as np
    import torch
    a,parent,rows=guard();score,changes=cpu_function();assert changes==a['ast_changes']
    start=time.monotonic()
    if stage=='qualify':
        if (OUT/'qualification.json').exists():return 0
        subset=[r for r in rows if r['execution_ordinal'] in a['qualification_ordinals']]
        assert [r['execution_ordinal'] for r in subset]==a['qualification_ordinals']
        H.OUT=OUT/'qualification_replay'
        code=score(parent,subset,0,budget)
        if code:return code
        comparisons=[]
        for row in subset:
            suffix=f"queries/query{row['execution_ordinal']:03d}"
            cv=H.read(H.OUT/suffix/'validation.json');gv=H.read(NATIVE/suffix/'validation.json')
            cpu=H.read(H.checked(cv['payload']));gpu=H.read(H.checked(gv['payload']))
            assert cpu['candidate_physical_rows']==gpu['candidate_physical_rows'] and cpu['winner']==gpu['winner']
            errors={k:float(np.max(abs(np.asarray(cpu[k])-np.asarray(gpu[k])))) for k in ('raw_scores','statistics','mass')}
            ct=torch.load(H.checked(cv['intermediate']),weights_only=True);gt=torch.load(H.checked(gv['intermediate']),weights_only=True)
            trace_errors={k:max(float((x[k]-y[k]).abs().max()) for x,y in zip(ct['pairs'],gt['pairs']))
                          for k in ('raw_max','image_max','image_reverse_max')}
            index_differences={k:sum(int((x[k]!=y[k]).sum()) for x,y in zip(ct['pairs'],gt['pairs']))
                               for k in ('raw_reference_token','image_reference_token')}
            assert max([*errors.values(),*trace_errors.values()])<a['float_tolerance']
            comparisons.append(dict(query_id=row['query_id'],index=row['execution_ordinal'],errors=errors,
                trace_errors=trace_errors,index_differences=index_differences,candidates=128,cpu=cv,gpu=gv))
        H.write(OUT/'qualification.json',dict(status='CPU_FULL128_SCORE_TRACE_AND_WINNER_PARITY_PASS',
            execution_authority=H.bind(AUTH),comparisons=comparisons,seconds=time.monotonic()-start,
            cpus=8,job_id=os.environ['SLURM_JOB_ID'],original_results_overwritten=False))
        print(dict(event='CPU_QUALIFIED',seconds=time.monotonic()-start,comparisons=comparisons),flush=True)
        return 0
    q=H.read(OUT/'qualification.json');assert q['status']=='CPU_FULL128_SCORE_TRACE_AND_WINNER_PARITY_PASS' and q['execution_authority']==H.bind(AUTH)
    assert 1<=index<=49
    code=score(parent,rows,index,budget)
    done=[H.bind(NATIVE/f"queries/query{r['execution_ordinal']:03d}/validation.json") for r in rows
          if r['execution_ordinal']%50==index and (NATIVE/f"queries/query{r['execution_ordinal']:03d}/validation.json").exists()]
    H.write(OUT/'runs'/f"{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json",
        dict(execution_authority=H.bind(AUTH),qualification=H.bind(OUT/'qualification.json'),shard=index,
             exit_code=code,seconds=time.monotonic()-start,completed_queries=done,device='cpu'))
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','qualify','score'));p.add_argument('--index',type=int,default=0);p.add_argument('--budget',type=float,default=180)
    args=p.parse_args()
    if args.stage=='prepare':prepare()
    else:sys.exit(run(args.stage,args.index,args.budget))
