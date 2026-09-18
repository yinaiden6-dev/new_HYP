#!/usr/bin/env python3
"""Verify dataset adaptation, uneven patient statistics and both runtime paths."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import run_rc_new_hyp_isic_inference_v1 as I
import analyze_rc_new_hyp_isic_transfer_v1 as J
OUT = ROOT / 'results/rc_new_hyp_isic_transfer_v1/submission'


def bind(p):
    return dict(path=str(Path(p).resolve()), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())


def function_ast(text, name):
    return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def main():
    checks = {}
    old = (ROOT / 'programs/run_rc_new_hyp_rpc_inference_v1.py').read_text()
    new = (ROOT / 'programs/run_rc_new_hyp_isic_inference_v1.py').read_text()
    normalized = old.replace('RPC', 'ISIC').replace('queries=8', 'queries=len(workers(shard))')
    assert function_ast(normalized, 'predict') == function_ast(new, 'predict')
    checks['original_feature_logits_and_action_ast_identical'] = True
    for scores in (torch.arange(390, dtype=torch.float64), torch.zeros(390, dtype=torch.float64),
                   torch.tensor([i % 11 for i in range(390)], dtype=torch.float64)):
        labels = ['SYNTH-' + str(i) for i in range(390)]
        expected = sorted(range(390), key=lambda i: (-float(scores[i]), i))
        assert I.rank(scores, labels) == expected
        assert I.S.top128_rows(scores, labels) == expected[:128]
    checks['full390_natural128_and_ties'] = True
    all_workers = [r for s in range(68) for r in I.workers(s)]
    assert len(all_workers) == 537 and len(I.workers(67)) == 1
    assert len({r['query_id'] for r in all_workers}) == 537
    assert all(set(r) == {'query_id', 'image_path', 'execution_ordinal'} for r in all_workers)
    checks['last_shard_one_query_and_anonymous_worker_schema'] = True
    for scenario in ('equal', 'all_gain', 'unequal_cluster_sizes'):
        rows = []
        for g in range(19):
            for j in range(1 + g % 6):
                b = False if scenario == 'all_gain' else (g+j) % 4 == 0
                m = b if scenario == 'equal' else True if scenario == 'all_gain' else (b and g % 7 != 0) or g % 5 == 0
                rows.append(dict(component=f'P-{g:03d}', correct={'B': b, 'M': m}))
        got = J.comparison(rows, 'B', 'M')
        groups = sorted({r['component'] for r in rows})
        delta = np.array([sum(int(r['correct']['M']) - int(r['correct']['B']) for r in rows if r['component'] == g)
                          / sum(r['component'] == g for r in rows) for g in groups])
        draws = np.random.default_rng(20260914).integers(len(groups), size=(100000, len(groups)))
        direct = np.quantile(delta[draws].sum(axis=1) / len(groups), [.025, .975])
        assert np.allclose(direct, got['patient_bootstrap95'], rtol=0, atol=1e-12)
        assert abs(delta.mean() - got['equal_patient_difference']) < 1e-12
        if scenario == 'equal': assert got['patient_bootstrap95'] == [0, 0] and not got['reliable_positive']
        if scenario == 'all_gain': assert got['patient_bootstrap95'] == [1, 1] and got['reliable_positive']
        checks['independent_patient_bootstrap_' + scenario] = True
    branches = []
    base = '/hkfs/work/workspace/scratch/ap7811-benchmark/'
    with tempfile.TemporaryDirectory(prefix='isic-launcher-') as temp:
        stub = Path(temp) / 'timeout'
        stub.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n'); stub.chmod(0o755)
        env = dict(os.environ, PATH=temp + ':/usr/bin:/bin', rc_python=base+'.venv-colpali/bin/python',
                   rc_roma_python=base+'.venv-romav2/bin/python', SLURM_ARRAY_TASK_ID='67')
        body = (ROOT / 'slurm/rc_new_hyp_isic_inference_v1.sbatch').read_text()
        body = body[body.index('case "${1:?stage}" in'):]
        for stage, seconds, key in [('raw', '540s', 'rc_python'), ('roma', '840s', 'rc_roma_python')]:
            got = json.loads(subprocess.run(['bash', '-c', body, 'fixture', stage], env=env, check=True, capture_output=True, text=True).stdout)
            assert got == ['--signal=TERM', '--kill-after=10s', seconds, env[key],
                           'programs/run_rc_new_hyp_isic_inference_v1.py', stage, '--shard', '67']
            branches.append(dict(stage=stage, argv=got))
        for name, seconds, program, tail in [
                ('reference_encoding', '840s', 'encode_rc_new_hyp_isic_references_v1.py', ['encode']),
                ('join', '540s', 'analyze_rc_new_hyp_isic_transfer_v1.py', [])]:
            body = (ROOT / f'slurm/rc_new_hyp_isic_{name}_v1.sbatch').read_text()
            body = body[body.index('exec timeout'):]
            got = json.loads(subprocess.run(['bash', '-c', body], env=env, check=True, capture_output=True, text=True).stdout)
            assert got == ['--signal=TERM', '--kill-after=10s', seconds, env['rc_python'], 'programs/'+program] + tail
            branches.append(dict(stage=name, argv=got))
    for name in ('reference_encoding', 'inference', 'join'):
        subprocess.run(['bash', '-n', str(ROOT / f'slurm/rc_new_hyp_isic_{name}_v1.sbatch')], check=True)
    script = '''import sys,json
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'programs'),str(Path.cwd()/'src')]
import run_rc_new_hyp_isic_inference_v1 as I
I.torch.set_num_threads(8);I.torch.set_num_interop_threads(1);I.torch.set_float32_matmul_precision('highest');I.torch.manual_seed(17)
if sys.argv[1]=='raw':
 p=I.read(I.M.RAW_PREFLIGHT);assert I.R.runtime_libraries()==p['runtime_libraries']
 print(json.dumps(dict(stage='raw',runtime=I.R.runtime_libraries())))
else:
 m=I.M.loadmodule(I.M.ROMA_SOURCE,'isic_preflight_runtime');p=I.read(I.M.PROFILE)
 runtime=m.check_runtime(p);_,digest=I.M.compile_roma(m)
 assert digest=='2ee24877651fb928bcfeb4f193beab28dbcda3cf4ef686e26fdc02f393a62cbb'
 print(json.dumps(dict(stage='roma',runtime=runtime,numeric_loop_sha256=digest)))
'''
    runtimes = []
    for stage, venv in [('raw', '.venv-colpali'), ('roma', '.venv-romav2')]:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_HOME=base+'third_party/model_cache/torch')
        p = subprocess.run([base+venv+'/bin/python', '-c', script, stage], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        runtimes.append(json.loads(p.stdout))
    result = dict(status='ISIC_ADAPTER_PATIENT_STATISTICS_LAUNCH_RUNTIME_PASS', checks=checks,
                  branches=branches, runtimes=runtimes, program=bind(__file__), model_forwards=0, target_reads=0)
    I.write(OUT / 'inference_preflight_v1.json', result)
    print(json.dumps(dict(status=result['status'], checks=checks)), flush=True)


if __name__ == '__main__': main()
