from pathlib import Path
import json,hashlib,re,shutil
from datetime import datetime,timezone
R=Path('/hkfs/work/workspace/scratch/ap7811-benchmark'); D=R/'github_exports/new_HYP_20260919'; RC=R/'ICLR/new ROUTEA/RC'
F=RC/'reports/figures/new_hyp_rescue90_en_20260924_v1'; V=RC/'results/rc_internal_m_condition_scale_v4'
files=[]; excluded=[]
for f in sorted(F.rglob('*')):
 if not f.is_file():continue
 if 'assets' in f.relative_to(F).parts or f.suffix=='.zip':
  excluded.append({'path':str(f.relative_to(R)),'reason':'Separate photo assets or local download ZIP; figures retain embedded display images'});continue
 files.append(f)
for f in sorted(V.rglob('*')):
 if f.is_file() and f.suffix in {'.json','.csv','.md'} and not any(s in f.name for s in ['.partial','.lock','.tmp']):files.append(f)
for f in (RC/'programs').glob('*rescue90*.py'):files.append(f)
files.append(Path('/tmp/export_rescue90_snapshot.py'))
manifest=[]; changed=[]
for f in files:
 rel=Path('github_exports/export_rescue90_snapshot_20260924.py') if f==Path('/tmp/export_rescue90_snapshot.py') else f.relative_to(R)
 b=f.read_bytes()
 if f.suffix=='.json':json.loads(b)
 if f.suffix in {'.json','.csv','.md','.html','.py'}:
  assert not re.search(rb'(?:ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',b),str(f)
 assert len(b)<100*1024*1024,str(f)
 out=D/rel;out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists() or out.read_bytes()!=b:out.write_bytes(b);changed.append(str(rel))
 assert hashlib.sha256(out.read_bytes()).digest()==hashlib.sha256(b).digest()
 manifest.append({'path':str(rel),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
base=D/'backup/rescue90_english_and_internal_m4_20260924';base.mkdir(parents=True,exist_ok=True)
progress={a: max([int(f.stem) for f in (V/a/'steps').glob('*.json')],default=0) for a in ['PRE_REAL','PRE_SHUFFLED']}
m={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'files':manifest,'excluded':excluded,'v4_captured_updates':progress,'v4_final_result_present':(V/'result.json').exists(),'note':'Stage snapshot; presence of step files is not scientific completion. No weights, token caches, execution logs or standalone original photos.'}
(base/'files.json').write_text(json.dumps(m,indent=2)+'\n')
(base/'README.md').write_text('''# English rescue figures and internal-M V4 snapshot

- [90 English correction figures, enlarged references](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_rescue90_en_20260924_v1/)
- [90-page PDF](../../ICLR/new%20ROUTEA/RC/reports/figures/new_hyp_rescue90_en_20260924_v1/rescue90.pdf)
- [Internal-M V4 stage records](../../ICLR/new%20ROUTEA/RC/results/rc_internal_m_condition_scale_v4/)

Medicine 30, GroZi 30, ISIC 30. Actual reference display dimensions are at least 2.1 times the previous layout. Selected successful corrections are illustrations, not a new accuracy estimate. PNG/SVG/PDF, native visualization grids, complete challenger scores, attribution and validator reports are included. No new model inference was used to draw them.

The figure README and validation receipts describe the complete local bundle. This Git export omits its 389 MB ZIP and standalone photo assets, in accordance with the separate original-image backup scope. Photos remain embedded in presentation panels; the HTML gallery uses these panels. Re-rendering requires the separately stored photo assets. The native NPZ files contain only small visualization weight grids, not encoder or model caches.

V4 is a timestamped snapshot, not a new completed experiment claim. Per-file hashes, omissions and captured updates are in files.json. Binary weights and err/out logs are excluded.
''')
for f in base.iterdir():changed.append(str(f.relative_to(D)))
f=D/'README.md';s=f.read_text();entry='- **[90张英文纠错图（放大 reference）及内部M V4最新快照](backup/rescue90_english_and_internal_m4_20260924/README.md)**：药盒／GroZi／ISIC各30张，附完整候选分数、原生热图与验证记录。\n\n'
if entry not in s:f.write_text(s.replace('## 从这里阅读\n\n','## 从这里阅读\n\n'+entry,1));changed.append('README.md')
Path('/tmp/rescue90_git_paths.json').write_text(json.dumps(sorted(set(changed))))
print(json.dumps({'files':len(manifest),'changed':len(set(changed)),'bytes':sum(x['bytes'] for x in manifest),'v4_updates':progress,'final_result':m['v4_final_result_present']}))
