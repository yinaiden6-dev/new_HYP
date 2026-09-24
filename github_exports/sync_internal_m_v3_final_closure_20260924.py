#!/usr/bin/env python3
"""Archive the completed internal-M v3 experiment and paper closure text."""
from pathlib import Path
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone

ROOT = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
RC = ROOT / 'ICLR/new ROUTEA/RC'
DEST = ROOT / 'github_exports/new_HYP_20260919'
INDEX = DEST / 'backup/internal_m_v3_final_closure_20260924'
RESULT = RC / 'results/rc_internal_m_learned_use_v3'
INCLUDE = [
    RC / 'reports/RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md',
    RC / 'reports/REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_FINAL_20260924.md',
    RC / 'reports/internal_m3_5161639_dev_migration_20260924.json',
    RC / 'reports/REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_SUBMITTED_20260924.md',
    RC / 'programs/run_rc_internal_m_learned_use_v3.py',
    RC / 'programs/submit_rc_internal_m_learned_use_v3.py',
    RC / 'programs/rc_prellm_m_adapter_v1.py',
    RC / 'tests/test_internal_m_learned_use_v3.py',
    RC / 'slurm/rc_internal_m_learned_use_v3.sbatch',
    RC / 'registry/rc_internal_m_learned_use_v3_authority_20260924.json',
    RC / 'plan/RC_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_DESIGN_20260924.md',
    RC / 'plan/RC_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_EXECUTION_20260924.md',
    Path(__file__).resolve(),
]

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def main():
    receipt = json.loads((RESULT / 'validation.json').read_text())
    assert receipt['status'] == 'TRAIN16_RESULT_ARTIFACTS_PASS'
    for k in ('result', 'report', 'authority'):
        b = receipt[k]
        assert digest(Path(b['path']).read_bytes()) == b['sha256']
    selected = set(INCLUDE)
    excluded = []
    for p in RESULT.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix in {'.json', '.md', '.csv'} and not any(s in p.name for s in ('.partial', '.tmp', '.lock')):
            selected.add(p)
        else:
            excluded.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                             'reason': 'binary checkpoint/token trace, runtime lock, or non-result file'})
    # Include directly referenced paper reports so its evidence links remain usable.
    paper = INCLUDE[0]
    for target in re.findall(r'\]\(([^)]+)\)', paper.read_text()):
        if target.startswith('http'):
            continue
        p = (paper.parent / target).resolve()
        assert p.is_file(), p
        if p.suffix in {'.md', '.json'}:
            selected.add(p)
    files = []
    for p in sorted(selected):
        assert not p.is_symlink(), p
        rel = p.relative_to(ROOT)
        blob = p.read_bytes()
        assert b'\0' not in blob, p
        if p.suffix == '.json':
            json.loads(blob)
        out = DEST / rel
        changed = not out.exists() or out.read_bytes() != blob
        out.parent.mkdir(parents=True, exist_ok=True)
        if changed:
            shutil.copyfile(p, out)
        assert digest(out.read_bytes()) == digest(blob)
        files.append({'path': str(rel), 'sha256': digest(blob), 'bytes': len(blob), 'changed': changed})
    INDEX.mkdir(parents=True, exist_ok=True)
    manifest = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), source_root=str(ROOT),
                    scope='Final internal M v3 TRAIN16 results and paper closure draft',
                    files=files, excluded=excluded, opened_probe_evaluations=0,
                    scientific_status='Both internal arms 8/16; external ADD4/PRODUCT5 12/16; no internal-M correction gain',
                    exclusions='No model/adapter binary weights, raw images, token caches, err/out logs or partial files')
    (INDEX / 'files.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (INDEX / 'README.md').write_text('''# 论文收口稿与内部M V3完整结果

2026-09-24。内部真实M和恒定M均完成128次更新及TRAIN16终点评估；本次补齐此前88步快照之后的全部文本结果。文件SHA及排除清单见[files.json](files.json)。

- [论文收口稿：摘要、贡献、近邻工作及主表](../../ICLR/new%20ROUTEA/RC/reports/RC_NEW_HYP_PAPER_CLOSURE_DRAFT_20260924.md)
- [内部M V3最终解释及独立复核](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_FINAL_20260924.md)
- [完整结果目录](../../ICLR/new%20ROUTEA/RC/results/rc_internal_m_learned_use_v3/)

| 同一TRAIN16，自然ColNomic C128 | 正确 | 对RAW救回/误伤 |
|---|---:|---:|
| RAW／无M共同读出 | 8/16 | 0/0 |
| 外部ADDITIVE4／PRODUCT5 | 12/16 | 4/0 |
| 内部PRE_CONSTANT／PRE_REAL，128更新 | 8/16 | 0/0 |
| PRE_REAL评估时换常量或错绑M | 8/16 | 0/0 |

参数、tokens和分数会变化，但内部没有产生纠错。内部128次逐query更新与外部2000次全批量更新的优化预算不同；这是TRAIN拟合诊断，没有probe/held结果，不证明内部注入原则上无效。外部模型的H593或外部面板成绩保持各自原口径。

保存完整C128内容分数、127挑战者logits、小头参数、逐步记录、配置、源码、报告及验收；二进制适配器/优化器checkpoint、token traces、原图和err/out日志未上传。绝对路径和原始封存SHA保留，恢复运行仍需要本地模型及缓存。
''')
    readme = DEST / 'README.md'
    text = readme.read_text()
    item = '- **[2026-09-24 论文收口稿与内部M V3最终结果](backup/internal_m_v3_final_closure_20260924/README.md)**：内部真实M／恒定M均8/16，外部加性／乘积均12/16；完整终点已封存，未获得内部纠错收益。\n\n'
    if item not in text:
        text = text.replace('## 从这里阅读\n\n', '## 从这里阅读\n\n' + item, 1)
    readme.write_text(text)
    print(json.dumps({'copied_files': len(files), 'changed_files': sum(x['changed'] for x in files),
                      'changed_bytes': sum(x['bytes'] for x in files if x['changed']),
                      'excluded_files': len(excluded), 'manifest': str(INDEX / 'files.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
