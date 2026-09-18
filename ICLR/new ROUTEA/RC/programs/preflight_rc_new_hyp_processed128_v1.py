#!/usr/bin/env python3
"""Independent subset, numerical-loop, statistic and launcher checks."""
import ast
import collections
import csv
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
sys.path[:0] = [str(ROOT/'programs'), str(ROOT/'src')]
import run_rc_new_hyp_processed128_v1 as I
import analyze_rc_new_hyp_processed128_v1 as J
import prepare_rc_new_hyp_processed128_v1 as P


def function_ast(path, name):
    return ast.dump(next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def main():
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest'); torch.manual_seed(17)
    a = I.read(I.AUTH)
    for b in a['sources'].values():
        I.checked(b)
    checks = {}
    old = ROOT/'programs/run_rc_new_hyp_rpc_inference_v1.py'
    assert function_ast(old, 'predict').replace('RPC_', 'PROCESSED_') == function_ast(Path(I.__file__), 'predict')
    checks['fixed_head_feature_logits_action_AST_unchanged'] = True
    _, raw_hash = I.M.compile_raw(I.R)
    assert raw_hash == a['raw_numeric_loop_sha256']
    body = I.M.segment(I.M.ROMA_SOURCE, 'produce', '    core = legacy_core(profile)', '    reused = C.require_reused_engineering_bridges()')
    assert hashlib.sha256(body.encode()).hexdigest() == a['roma_numeric_loop_sha256']
    checks['RAW_and_ROMA_original_numeric_loops_exact'] = True
    pre = I.read(I.M.RAW_PREFLIGHT)
    assert I.R.runtime_libraries() == pre['runtime_libraries']
    checks['frozen_RAW_runtime'] = True
    checks['stable_full5413_identity_ranking'] = I.R.rank_probe()['status']
    selection = I.read(I.OUT/'selection.json')
    with I.checked(selection['mapping']).open(newline='') as f:
        source = list(csv.DictReader(f))
    gallery = I.read(I.OUT/'gallery_manifest.json')['records']
    by_name = collections.Counter(Path(r['image_path']).stem for r in gallery)
    grouped = collections.defaultdict(list)
    for r in source:
        name = r['file_name'].rsplit('__', 1)[0]
        if by_name[name] == 1:
            grouped[name].append(r['file_name'])
    expected = [sorted(grouped[n], key=P.key)[0] for n in sorted(grouped, key=P.key)[:128]]
    assert expected == [r['file_name'] for r in selection['selected']]
    worker = I.read(I.OUT/'worker_manifest.json')['records']
    assert [r['query_id'] for r in worker] == [f'PROC-Q-{i:04d}' for i in range(128)]
    assert sum((I.workers(i) for i in range(16)), []) == worker
    for w, name in zip(worker, expected):
        assert Path(w['query_image_path']).resolve() == P.WS/'1/processed'/name
        assert I.M.sha(w['query_image_path']) == w['source_image_sha256']
        assert set(w) == {'query_id','execution_ordinal','query_image_path','source_image_sha256','track'}
    checks['independent_outcome_blind_subset_and_bytes'] = True
    # Unequal source sizes exercise the query-weighted cluster estimator.
    for scenario in ('equal', 'gain', 'mixed'):
        rows = []
        for group, n in enumerate((1, 2, 3)):
            for j in range(n):
                b = scenario == 'equal' or (scenario == 'mixed' and group == 1)
                m = b if scenario == 'equal' else scenario == 'gain' or group == 2
                rows.append(dict(component=str(group), correct={'RAW':b, 'M':m}))
        c = J.comparison(rows, 'M', draws=4000)
        ds = np.array([sum(int(r['correct']['M'])-int(r['correct']['RAW']) for r in rows if r['component'] == str(g)) for g in range(3)])
        ns = np.array([1, 2, 3]); rng = np.random.default_rng(20260916)
        ix = rng.integers(3, size=(4000, 3)); direct = ds[ix].sum(1)/ns[ix].sum(1)
        assert np.allclose(c['component_bootstrap95'], np.quantile(direct, [.025,.975]), rtol=0, atol=1e-12)
        assert c['net'] == int(ds.sum())
    checks['independent_cluster_bootstrap_and_rescue_loss_counts'] = True
    launcher = ROOT/'slurm/rc_new_hyp_processed128_v1.sbatch'
    subprocess.run(['bash','-n',str(launcher)], check=True)
    branches = []
    with tempfile.TemporaryDirectory(prefix='processed-launch-check-') as tmp:
        stub = Path(tmp)/'timeout'
        stub.write_text('#!/usr/bin/python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n'); stub.chmod(0o755)
        env = dict(os.environ, PATH=tmp+':/usr/bin:/bin', rc_python=str(P.WS/'.venv-colpali/bin/python'),
                   rc_roma_python=str(P.WS/'.venv-romav2/bin/python'), SLURM_ARRAY_TASK_ID='15')
        body = launcher.read_text().split('case "${1:?stage}" in', 1)[1]
        body = 'case "${1:?stage}" in' + body
        for stage, seconds, py in [('raw','540s','rc_python'),('roma','840s','rc_roma_python'),('join','540s','rc_python')]:
            got = json.loads(subprocess.run(['bash','-c',body,'fixture',stage], env=env, check=True, capture_output=True, text=True).stdout)
            suffix = ['programs/analyze_rc_new_hyp_processed128_v1.py'] if stage == 'join' else ['programs/run_rc_new_hyp_processed128_v1.py',stage,'--shard','15']
            assert got == ['--signal=TERM','--kill-after=10s',seconds,env[py]] + suffix
            branches.append(dict(stage=stage, argv=got))
    checks['actual_launcher_branches_and_arguments'] = True
    script = """import json,sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'programs'),str(Path.cwd()/'src')]
import run_rc_new_hyp_processed128_v1 as I
I.torch.set_num_threads(8);I.torch.set_num_interop_threads(1);I.torch.set_float32_matmul_precision('highest');I.torch.manual_seed(17)
m=I.M.loadmodule(I.M.ROMA_SOURCE,'processed_preflight_runtime');p=I.read(I.M.PROFILE)
print(json.dumps(m.check_runtime(p)))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TORCH_HOME=str(P.WS/'third_party/model_cache/torch'))
    proc = subprocess.run([str(P.WS/'.venv-romav2/bin/python'),'-c',script], cwd=ROOT, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runtime = json.loads(proc.stdout)
    checks['frozen_ROMA_runtime_import'] = True
    result = dict(status='PROCESSED128_PREFLIGHT_PASS', authority=I.bind(I.AUTH), checks=checks,
                  branches=branches, roma_runtime=runtime, model_forwards=0, query_outcome_reads=0)
    I.write(I.OUT/'submission/preflight.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
