"""Check bilingual report counts against existing artifacts, then package outputs."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

OUT = Path(__file__).resolve().parent
RC = OUT.parents[1]
sources = {}
def read(rel):
    b = (RC / rel).read_bytes()
    sources[rel] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    return json.loads(b)

h = read('results/rc_h593_group_risk_strong_base_v1/result.json')
e = read('results/rc_original7_eval128_full_evidence_v1/result.json')
d = read('results/romav2_colnomic_difficult90_frozen_regression_v1/result.json')
old = read('results/rc_original_mixed96_group_risk_v1/result.json')
inc = read('results/rc_h593_increment_attribution_v1/result.json')
read('results/rc_h593_group_risk_strong_base_v1/result_validation.json')
read('results/rc_h593_increment_attribution_v1/result_validation.json')
read('results/roma_rgh_reference_first_natural_s0_v1/aggregate/postjoin_result.json')
for rel in ['reports/NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md',
            'reports/NEW_HYP_UNIFIED_DECISION_HYPOTHESIS_V3_20260911.md',
            'reports/REPORT_H593_UNIFIED_HYPOTHESIS_PREDICTION_READOUT_V1_20260911.md',
            'reports/REPORT_ROMA_RGH_S0_JOB5139368_FINAL_DECISION_AND_MULTIPLICITY_BOTTLENECK_20260910.md']:
    b = (RC/rel).read_bytes()
    sources[rel] = {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}

rows=h['rows']
assert len(rows)==593 and len({r['query_id'] for r in rows})==593
def transition(a,b):
    return [sum(not r['correct'][a] and r['correct'][b] for r in rows),
            sum(r['correct'][a] and not r['correct'][b] for r in rows)]
assert h['scores']['RAW']['correct']==426
assert h['scores']['ALL_COND']['correct']==445 and transition('RAW','ALL_COND')==[19,0]
assert h['scores']['GROUP_BASE']['correct']==447 and transition('RAW','GROUP_BASE')==[22,1]
assert transition('ALL_BASE','GROUP_BASE')==[8,1]
assert transition('SMALL_CONST','GROUP_BASE')==[4,3]
assert h['recall_C128']==sum(r['target_in_C128'] for r in rows)==570
assert sum(r['target_in_C128'] and not r['correct']['GROUP_BASE'] for r in rows)==123
assert sum(not r['target_in_C128'] for r in rows)==23
assert d['summaries']['REAL']['final_top1']==69 and d['summaries']['REAL']['rescue']==8
assert e['metrics_all128']['REAL']['final_top1']==99
assert e['metrics_all128']['REAL']['rescue']==12 and e['metrics_all128']['REAL']['break']==1
r32=old['panels']['EVAL32']['rows']
assert sum(r['correct']['RAW'] for r in r32)==25
assert sum(r['correct']['ORIGINAL7'] for r in r32)==28
assert inc['binding']['CONDITIONAL4']['INCREMENT_BIND_rescues_retained']==0

table_rows=[]
logs={}
for lang in ['zh','en']:
    tex=(OUT/f'main_{lang}.tex').read_text()
    assert tex.count(r'\begin{frame}')==13 and tex.count(r'\end{frame}')==13
    extracted=[]
    for line in tex.splitlines():
        if line.startswith(r'\model{') and line.count('&') == 5:
            extracted.append([c.strip() for c in line.split('&')][2:])
    table_rows.append(extracted)
    log=(OUT/f'main_{lang}.log').read_text(errors='replace')
    assert not re.search(r'^!|Missing character:|Overfull', log, re.M)
    assert re.search(r'Output written on .*?\(13 pages\)',log)
    assert (OUT/f'main_{lang}.pdf').stat().st_size>10000
    logs[lang]={'pages':13,'compile_errors':0,'missing_glyphs':0,'overfull_boxes':0,
                'note':'Chinese upstream ctexhook release-date warning may occur with TeX Live 2020.'}
assert len(table_rows[0])==5 and table_rows[0]==table_rows[1]
receipt={'status':'BILINGUAL_REPORT_DATA_AND_COMPILE_AUDIT_PASS',
         'date':'2026-09-12','evidence_snapshot':'2026-09-11',
         'sources':sources,'compile_checks':logs,
         'performance_table_values_identical':True,
         'training_or_threshold_updates':0,'slurm_jobs_submitted':0,
         'not_an_independent_revalidation_of_all_experiments':True,
         'ctex_package':{'origin':'https://mirrors.ctan.org/install/language/chinese/ctex.tds.zip',
                         'sha256':hashlib.sha256((OUT/'ctex.tds.zip').read_bytes()).hexdigest(),
                         'license':'LPPL; unmodified upstream archive and generated ctexhook.sty retained'}}
(OUT/'data_source_audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
names=['main_zh.tex','main_en.tex','preamble.tex','main_zh.pdf','main_en.pdf',
       'README.md','build.sh','data_source_audit.json','audit_and_package.py',
       'ctexhook.sty','ctex.tds.zip']
with zipfile.ZipFile(OUT/'bilingual_latex_and_pdf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names:
        z.write(OUT/name,name)
with zipfile.ZipFile(OUT/'bilingual_latex_and_pdf.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(names)
print(json.dumps({'status':receipt['status'],'pages_each':13,'archive_bytes':(OUT/'bilingual_latex_and_pdf.zip').stat().st_size}))
