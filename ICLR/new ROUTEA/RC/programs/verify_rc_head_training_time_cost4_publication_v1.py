#!/usr/bin/env python3
"""Verify the published artifact set without deleting an unrelated historical ZIP copy."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/new_hyp_complete_results_20260916_v1'
BENCH = ROOT / 'results/rc_head_training_time_v2_cost4_20260918'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
archive = OUT / 'new_HYP_complete_results_20260916.zip'
update = read(OUT / 'training_time_update_validation.json')
marker = read(OUT / 'training_time_cost4_update_validation.json')
assert marker['current_validation_sha256'] == sha(OUT / 'training_time_update_validation.json')
assert marker['timed_fits'] == update['measured_fits'] == 27
expected = set(update['artifact_sha256']) | {
    'training_time_update_validation.json', 'training_time_cost4_update_validation.json'}
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(set(names)) == 43
    assert set(names) == expected
    for name in names:
        assert hashlib.sha256(z.read(name)).hexdigest() == sha(OUT / name), name
for name, checksum in update['artifact_sha256'].items():
    assert sha(OUT / name) == checksum, name

# An independently copied previous ZIP is outside the declared artifact set.
# Preserve it and verify its exact historical identity instead of packaging it recursively.
extras = sorted(p for p in OUT.iterdir() if p.is_file() and p != archive and p.name not in expected)
extra_records = []
for p in extras:
    assert p.suffix == '.zip' and sha(p) == update['previous_published_archive']['sha256'], p
    extra_records.append(dict(name=p.name, sha256=sha(p), scope='Unchanged copy of the previous published archive; not a current package member'))

d = read(BENCH / 'result.json')
v = read(BENCH / 'validation.json')
assert v['result']['sha256'] == sha(BENCH / 'result.json')
assert len(d['records']) == v['timed_fits'] == 27
with (OUT / 'training_time_summary.csv').open(encoding='utf-8-sig') as f:
    summary = list(csv.DictReader(f))
assert len(summary) == 9
for row in summary:
    records = [r for r in d['records'] if r['model'] == row['model'] and r['pool_size'] == int(row['pool_size'])]
    assert len(records) == 3 and len({r['theta_sha256'] for r in records}) == 1
    assert float(row['median_seconds']) == float(np.median([r['fit_seconds'] for r in records]))
    if int(row['pool_size']) == 593:
        assert all(r['historical_head_max_abs_error'] == 0 for r in records)
with (OUT / 'training_time_fits.csv').open(encoding='utf-8-sig') as f:
    fits = list(csv.DictReader(f))
assert len(fits) == 3
for fit in fits:
    rows = [r for r in summary if r['model'] == fit['model']]
    x = np.array([float(r['effective_queries']) for r in rows])
    y = np.array([float(r['median_seconds']) for r in rows])
    slope = ((x-x.mean())*(y-y.mean())).sum() / ((x-x.mean())**2).sum()
    intercept = y.mean()-slope*x.mean()
    assert abs(float(fit['slope_seconds_per_effective_query'])-slope) < 1e-12
    assert abs(float(fit['intercept_seconds'])-intercept) < 1e-12
auth = read(ROOT / 'registry/rc_head_training_time_authority_v2_cost4_20260918.json')
for model in ('COST4', 'COST1', 'CE'):
    bound = auth['sources'][model+'_frozen_head']
    assert sha(Path(bound['path'])) == bound['sha256']
for name in ('external_all_models', 'external_comparisons'):
    with (OUT / (name+'.csv')).open(encoding='utf-8-sig') as f:
        assert all(r['panel'] != 'RPC600' for r in csv.DictReader(f))

result = dict(status='COST4_TIMING_AND_PUBLISHED_ARCHIVE_INDEPENDENTLY_VERIFIED',
    timing_job='5150701', failed_publication_job='5150702', execution='local_document_recovery',
    archive=str(archive), archive_sha256=sha(archive), archive_members=len(names),
    measured_fits=27, summary_rows=9, OLS_recomputed=True, original_head_hashes_unchanged=True,
    RPC_formal_external_confirmation=False, historical_copies_preserved=extra_records,
    verification_source_sha256=sha(Path(__file__)))
(BENCH / 'publication_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
