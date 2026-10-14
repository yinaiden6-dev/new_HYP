import csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch

ROOT=Path(os.environ.get('COLNOMIC_WORKSPACE',str(Path(__file__).resolve().parents[2]))).resolve()
OUT=Path(os.environ.get('COLNOMIC_EXPERIMENT_OUTPUT',str(ROOT/'results/token_mechanism_experiments/20261013_v1'))).resolve()

def workspace_path(path, *, must_exist=False):
 """Resolve payload paths inside ROOT, explicitly rebasing the archived prefix.

 No fallback reads of the original workspace are allowed after extraction.
 Resolving symlinks or ``..`` outside the current workspace is rejected.
 """
 value=os.fspath(path)
 if '://' in value:raise ValueError('Workspace paths must be filesystem paths: '+value)
 candidate=Path(value)
 historical=Path('/mnt/lsdf_zyin')
 if candidate.is_absolute():
  if candidate==ROOT or ROOT in candidate.parents:
   target=candidate
  elif candidate==historical or historical in candidate.parents:
   target=ROOT/candidate.relative_to(historical)
  else:raise ValueError('Absolute path outside approved workspace prefixes: '+value)
 else:target=ROOT/candidate
 target=target.resolve(strict=must_exist)
 if target!=ROOT and ROOT not in target.parents:
  raise ValueError('Resolved path escapes current workspace: '+value)
 return target

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def csvout(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rs for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rs])
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return torch.load(ROOT/p,map_location='cpu',weights_only=True,mmap=True)

def verify_split_records(photos,split_records,group_records,query_bank=None,reference_bank=None,positives=None):
 """Return evidence from actual set intersections; reject overlaps and omissions."""
 checks={};assignments={};byid={r['photo_id']:r for r in photos}
 assert len(byid)==len(photos),'duplicate photo IDs'
 groups={r['package_id']:r for r in group_records}
 for split,records in split_records.items():
  for row in records:
   qid=row['photo_id'];assert qid in byid and qid not in assignments,'missing or duplicate split membership'
   assert all(row.get(k)==byid[qid].get(k) for k in ['package_id','design_id','identity','sha256','source_sha256']),'split CSV differs from master'
   assert groups[row['package_id']]['split']==split,'group split differs'
   assignments[qid]=split
 assert set(assignments)==set(byid),'split coverage incomplete'
 def check(name,values):
  scopes={split:{value for qid in assignments if assignments[qid]==split for value in values(qid)} for split in split_records}
  assert all('' not in scope and None not in scope for scope in scopes.values()),'empty '+name
  intersections={a+'__'+b:sorted(scopes[a]&scopes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
  result=dict(passed=all(not v for v in intersections.values()),unique_counts={k:len(v) for k,v in scopes.items()},intersections=intersections)
  checks[name]=result
  assert result['passed'],'split overlap in '+name+': '+str(intersections)
 for field in ['package_id','design_id','identity','sha256','source_sha256']:
  check(field,lambda qid,f=field:[byid[qid][f]])
 check('group_id',lambda qid:[groups[byid[qid]['package_id']]['group_id']])
 if query_bank is not None:
  assert set(query_bank['ids'])==set(byid) and len(query_bank['ids'])==len(byid),'query bank coverage'
  for i,qid in enumerate(query_bank['ids']):
   assert query_bank['sha256'][i]==byid[qid]['sha256'],'query bank SHA mismatch'
   assert query_bank['source_sha256'][i]==byid[qid]['source_sha256'],'query bank source SHA mismatch'
 if reference_bank is not None and positives is not None:
  rindex={rid:i for i,rid in enumerate(reference_bank['ids'])};positive_by_package={}
  for r in positives:positive_by_package.setdefault(r['package_id'],[]).append(r['reference_id'])
  for field in ['sha256','source_sha256']:
   check('positive_reference_'+field,lambda qid,f=field:[reference_bank[f][rindex[rid]] for rid in positive_by_package[byid[qid]['package_id']]])
 return dict(status='passed' if all(c['passed'] for c in checks.values()) else 'failed',checks=checks,
  photos_per_split={k:len(v) for k,v in split_records.items()},query_bank_sha_checked=query_bank is not None,
  positive_reference_hashes_checked=reference_bank is not None and positives is not None)

def data():
 qb=load('cache/image_embeddings/colnomic7b/photos.pt');rb=load('cache/image_embeddings/colnomic7b/references.pt')
 a=load('cache/scores/colnomic7b/image.pt');b=load('cache/scores/bm25/bm25.pt');f=load('cache/scores/bm25_image_fusion/colnomic7b/alpha_0.5.pt')
 for s in [a,b,f]:assert s['query_ids']==qb['ids'] and s['reference_ids']==rb['ids']
 metadata=read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')
 split_records={split:rows(ROOT/f'dataset/lists/{split}.csv') for split in ['train','val','test']}
 splits={r['photo_id']:split for split,records in split_records.items() for r in records}
 photos={r['photo_id']:r for r in rows(ROOT/'dataset/lists/photos.csv')}
 group_records=rows(ROOT/'dataset/lists/groups.csv');groups={r['package_id']:r['group_id'] for r in group_records}
 split_verification=verify_split_records(list(photos.values()),split_records,group_records,qb,rb,rows(ROOT/'dataset/lists/positives.csv'))
 return dict(qb=qb,rb=rb,image=a,bm25=b,fusion=f,metadata=metadata,splits=splits,photos=photos,groups=groups,split_verification=split_verification,
  qo=read(ROOT/'cache/ocr/photos.json'),ro=read(ROOT/'cache/ocr/references.json'),boxes=read(ROOT/'dataset/annotations/subject_boxes.json')['records'])

def preregister():
 OUT.mkdir(parents=True,exist_ok=True)
 config=dict(schema='colnomic.mechanism_experiments.v1',seed=20261013,
  scope='all 1117 photos, all 5405 reference rows; same candidate ledger then complete gallery',
  split='unchanged package_holdout_v4_supplement, identities/design/source groups verified disjoint',
  historical_exposure='baseline errors and historical test results have already been inspected; not a historically blind test',
  geometry=dict(alignment='unique exact normalized OCR-line anchors inside confirmed query subject box; RANSAC homography independent of embeddings and identity labels',
   minimum_line_characters=8,minimum_distinct_lines=4,ransac_normalized_reference_threshold=.025,
   minimum_inlier_lines=4,minimum_query_coverage=.15,no_reflection=True,
   primary='MaxSim with lambda*(1-clipped spatial distance) bonus on subject visual tokens when alignment accepted; other tokens unchanged',
   secondary='pure spatial penalty, rejected alignments fall back to original MaxSim',
   geometry_only='inlier OCR geometry support without ColNomic embeddings; lexical anchor selection is shared alignment input, not a text-free method',
   coordinate_only='nearest reference patch distance after the same alignment, without embeddings',
   lambda_grid=[0.,.01,.025,.05,.1,.2],selection='validation package-macro Recall@1 then MRR then smaller lambda; test labels never select alignment or parameters',
   full_gallery='every reference retains its score; no GT insertion or top-K truncation',fusion='same frozen population z-score alpha .5 with BM25'),
  probes=dict(features='frozen 128D features only; no row/col indices, explicit coordinates or image IDs in feature heads',
   ridge_grid=[.001,.01,.1,1.,10.,100.],logistic_C_grid=[.01,.1,1.,10.],shuffle_seeds=[17,31,53],
   position='raw normalized coordinates and subject-relative displayed coordinates; OCR-row rank separately',
   identity='pairwise same-instance prediction across held identities; train negatives restricted to training positive designs',
   controls=['train-mean/majority','label shuffle','text-only pair head','coordinate-only pair head'],
   layout='controlled same-text word-position classes balanced within each source identity; hold source identities out, label shuffle within identity',
   limitation='absolute position prediction is not proof of identity-layout discrimination; failed probe is not proof of no information'),
  intervention=dict(method='user-authorized deterministic pixel relocation, no generative editing',
   text='existing glyph pixels reused; no text replacement or font synthesis',
   arms=['original freshly encoded','sham common background editing, glyph at original position','same glyph shifted on its existing otherwise empty line'],
   controls='base/sham/edit use identical image size, colors, text and capacity; nonedited pixels and foreground glyph conservation verified',
   target='photo_41de3eb383934ec3; reference_000233 details. line; other source identities selected by OCR and whitespace eligibility before embeddings',
   encoder='same pinned checkpoints, BF16 outputs/FP32 pre-Ampere SDPA; CPU weight offload allowed, paired fresh baseline and cached parity reported',
   metrics='positive/negative margins, full-gallery rank, fresh-base drift, sham difference and edit-minus-sham; controlled-source pilot has limited generality'))
 path=OUT/'preregistration.json'
 if path.exists():assert read(path)==config
 else:write(path,config)
 return config
