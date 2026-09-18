#!/bin/bash
#SBATCH --job-name=head_time4_doc
#SBATCH --account=hk-project-p0025545
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --no-requeue
#SBATCH --output="/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/logs/head-time4-doc-%j.out"
#SBATCH --error="/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/logs/head-time4-doc-%j.err"
set -euo pipefail
cd '/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC'
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR='/tmp/rc-cost4-publication-mpl'
PYTHON='/hkfs/work/workspace/scratch/ap7811-benchmark/.venv-colpali/bin/python'
"$PYTHON" - <<'PY'
import hashlib,json
from pathlib import Path
a=json.loads(Path('registry/rc_head_training_time_cost4_publish_authority_v1_20260918.json').read_text())
for path,expected in a['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
p=Path('results/rc_head_training_time_v2_cost4_20260918')
v=json.loads((p/'validation.json').read_text())
assert v['status']=='TIMING_REPEATS_AND_FROZEN_HEAD_PARITY_PASS'
assert v['timed_fits']==v['expected_fits']==27
assert v['old_head_files_unchanged'] and v['within_config_parameter_bits_equal']
assert v['result']['sha256']==hashlib.sha256((p/'result.json').read_bytes()).hexdigest()
PY
"$PYTHON" programs/update_rc_results_training_time_v2_cost4.py
"$PYTHON" - <<'PY'
import csv,hashlib,json,os,zipfile
from pathlib import Path
import numpy as np
root=Path.cwd()
out=root/'reports/new_hyp_complete_results_20260916_v1'
bench=root/'results/rc_head_training_time_v2_cost4_20260918'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=out/'new_HYP_complete_results_20260916.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist()
    assert set(names)=={p.name for p in out.iterdir() if p.is_file() and p!=archive}
    for name in names:assert hashlib.sha256(z.read(name)).hexdigest()==sha(out/name),name
update=json.loads((out/'training_time_update_validation.json').read_text())
for name,expected in update['artifact_sha256'].items():assert sha(out/name)==expected,name
d=json.loads((bench/'result.json').read_text())
with (out/'training_time_summary.csv').open(encoding='utf-8-sig') as f:summary=list(csv.DictReader(f))
assert len(summary)==9
for row in summary:
    records=[r for r in d['records'] if r['model']==row['model'] and r['pool_size']==int(row['pool_size'])]
    assert len(records)==3
    assert float(row['median_seconds'])==float(np.median([r['fit_seconds'] for r in records]))
with (out/'training_time_fits.csv').open(encoding='utf-8-sig') as f:
    for fit in csv.DictReader(f):
        rows=[r for r in summary if r['model']==fit['model']]
        x=np.array([float(r['effective_queries']) for r in rows]); y=np.array([float(r['median_seconds']) for r in rows])
        slope=((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum()
        intercept=y.mean()-slope*x.mean()
        assert abs(float(fit['slope_seconds_per_effective_query'])-slope)<1e-12
        assert abs(float(fit['intercept_seconds'])-intercept)<1e-12
auth=json.loads((root/'registry/rc_head_training_time_authority_v2_cost4_20260918.json').read_text())
for model in ('COST4','COST1','CE'):
    bound=auth['sources'][model+'_frozen_head']
    assert sha(Path(bound['path']))==bound['sha256']
readout=dict(status='COST4_TIMING_AND_PUBLISHED_ARCHIVE_INDEPENDENTLY_VERIFIED',
    job_id=os.environ['SLURM_JOB_ID'],partition=os.environ.get('SLURM_JOB_PARTITION'),
    archive=str(archive),archive_sha256=sha(archive),archive_members=len(names),
    measured_fits=27,summary_rows=9,OLS_recomputed=True,original_head_hashes_unchanged=True,
    RPC_formal_external_confirmation=False)
(bench/'publication_validation.json').write_text(json.dumps(readout,indent=2)+'\n')
print(json.dumps(readout),flush=True)
PY
