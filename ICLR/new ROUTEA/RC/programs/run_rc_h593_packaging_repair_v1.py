#!/usr/bin/env python3
"""Repair report-only runtime; execute unchanged, hash-pinned publishers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'isolated/h593_packaging_runtime_20260920_v1'
AUTH=ROOT/'registry/rc_h593_packaging_runtime_repair_authority_v1_20260920.json'
OUT=ROOT/'results/rc_h593_packaging_runtime_repair_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(stage):
    authority=json.loads(AUTH.read_text())
    for b in authority['sources'].values():
        assert sha(b['path'])==b['sha256'],b['path']
    manifest=json.loads((RUNTIME/'manifest.json').read_text())
    for name,b in manifest['files'].items():
        p=RUNTIME/name;assert p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],name
    assert sys.flags.no_user_site==1, 'Explicit isolated dependency path required'
    import openpyxl
    import et_xmlfile
    for m in (openpyxl,et_xmlfile):
        assert RUNTIME/'site-packages' in Path(m.__file__).resolve().parents, m.__file__
    pandoc=Path(shutil.which('pandoc') or '').resolve()
    assert pandoc==(RUNTIME/'bin/pandoc').resolve(),str(pandoc)
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.smoke_',dir=OUT) as temp:
        folder=Path(temp);book=openpyxl.Workbook();book.active.append(['核验',593]);book.save(folder/'probe.xlsx')
        loaded=openpyxl.load_workbook(folder/'probe.xlsx',read_only=True)
        assert loaded.active['A1'].value=='核验' and loaded.active['B1'].value==593;loaded.close()
        (folder/'probe.md').write_text('# 运行时验证\n\n不修改任何实验预测。\n')
        for ext in ('html','docx'):
            subprocess.run([str(pandoc),'probe.md','--standalone','--metadata','title=Packaging check','-o','probe.'+ext],cwd=folder,check=True)
        with zipfile.ZipFile(folder/'probe.docx') as z:assert z.testzip() is None
    record=dict(status='ISOLATED_XLSX_HTML_DOCX_RUNTIME_PASS',stage=stage,job=os.environ.get('SLURM_JOB_ID'),
                python=sys.executable,openpyxl=openpyxl.__version__,openpyxl_path=openpyxl.__file__,
                pandoc=str(pandoc),runtime_manifest_sha256=sha(RUNTIME/'manifest.json'),scientific_inputs_changed=False)
    (OUT/f"environment_{stage}_{os.environ.get('SLURM_JOB_ID','local')}.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    if stage=='preflight':return
    source=authority['publishers'][stage]
    assert sha(source['path'])==source['sha256']
    sys.path.insert(0,str(ROOT/'programs'))
    spec=importlib.util.spec_from_file_location('qualified_report_publisher',source['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()
    (OUT/f"completion_{stage}_{os.environ.get('SLURM_JOB_ID','local')}.json").write_text(json.dumps(dict(status='REPAIRED_PUBLISHER_COMPLETE',stage=stage,publisher=source,runtime=record),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','ablation','content']);args=parser.parse_args();main(args.stage)
