#!/usr/bin/env python3
"""Recover document-only publication using the already installed user-site packages."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'results/rc_head_training_time_v2_cost4_20260918'
os.chdir(ROOT)
assert not os.environ.get('PYTHONNOUSERSITE'), 'Report-only recovery needs the installed openpyxl package'
assert not (BENCH / 'publication_validation.json').exists(), 'Publication already validated'

packages = {}
for name in ('numpy', 'matplotlib', 'openpyxl', 'PIL'):
    module = importlib.import_module(name)
    packages[name] = dict(version=module.__version__, path=module.__file__)

authority = json.loads((ROOT / 'registry/rc_head_training_time_cost4_publish_authority_v1_20260918.json').read_text())
for path, expected in authority['source_sha256'].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path

launcher = ROOT / 'slurm/rc_head_training_time_v2_cost4_publish.sbatch'
blocks = [part.split('\nPY\n', 1)[0] for part in launcher.read_text().split("<<'PY'\n")[1:]]
assert len(blocks) == 2
exec(compile(blocks[0], str(launcher) + ':preflight', 'exec'), {})

record = dict(status='DOCUMENT_ONLY_RECOVERY_ENVIRONMENT_VERIFIED',
    failed_publication_job='5150702', successful_timing_job='5150701',
    cause='PYTHONNOUSERSITE=1 hid the installed user-site openpyxl package',
    repair='Use the installed report packages for document generation only; retain original source SHA checks',
    training_rerun=False, original_builder_unchanged=True,
    python=sys.executable, packages=packages,
    recovery_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(BENCH / 'publication_recovery.json').write_text(json.dumps(record, indent=2) + '\n')
subprocess.run([sys.executable, str(ROOT / 'programs/update_rc_results_training_time_v2_cost4.py')], check=True)

verification = blocks[1]
old = "job_id=os.environ['SLURM_JOB_ID'],partition=os.environ.get('SLURM_JOB_PARTITION'),"
assert verification.count(old) == 1
verification = verification.replace(old,
    "job_id=None,partition=None,execution='local_document_recovery',failed_publication_job='5150702',timing_job='5150701',")
exec(compile(verification, str(launcher) + ':independent_verification', 'exec'), {})
