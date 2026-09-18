#!/usr/bin/env python3
"""Outcome-blind complete nonformal population and identity/group component folds."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_new_hyp593_oof5_v1/metadata'
SEED='RC_NEW_HYP593_OOF5_V1_20260911'
SOURCES={'ledger':'cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json','index':'results/l0_natural_hardneg_v_representation_v2_identityfix/target_join/query_target_join_index.json','formal':'registry/rc_eval128_formal392_negative_exclusion_v1_20260910.json'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,sort_keys=True,indent=2);f.write('\n')
def key(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def population():
 ledger=read(ROOT/SOURCES['ledger'])['queries'];idx=read(ROOT/SOURCES['index']);fm=read(ROOT/SOURCES['formal'])
 lookup={r['query_id']:r for r in idx['records']};fq=set(fm['exclude_query_ids']);fh=set(fm['exclude_image_sha256']);records=[];label_sources=[]
 assert len(ledger)==len(lookup)==987
 for q in ledger:
  if q['query_id'] in fq or q['source_image_sha256'] in fh:continue
  e=lookup[q['query_id']];p=ROOT/Path(SOURCES['index']).parent/e['path'];r=read(p)
  assert r['query_id']==q['query_id'] and r['query_ordinal']==q['query_ordinal']
  h=hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert h==r['logical_sha256']==e['record_logical_sha256'];label_sources.append(bind(p))
  records.append(dict(original_query_id=q['query_id'],source_image_sha256=q['source_image_sha256'],original_path=q['path'],track=q['track'],identity=r['identity'],group=r['group_id']))
 byhash=defaultdict(list)
 for r in records:byhash[r['source_image_sha256']].append(r)
 unique=[];duplicates=[]
 for h,rs in sorted(byhash.items()):
  assert len({r['identity'] for r in rs})==len({r['group'] for r in rs})==1
  rs.sort(key=lambda r:r['original_query_id']);unique.append(rs[0])
  if len(rs)>1:duplicates.append(dict(image_sha=h,query_ids=[r['original_query_id'] for r in rs],kept=rs[0]['original_query_id']))
 assert len(records)==595 and len(unique)==593 and len({r['identity'] for r in unique})==68 and len({r['group'] for r in unique})==64
 return unique,duplicates,label_sources

def folds(rows):
 parent={}
 def find(a):
  parent.setdefault(a,a)
  if parent[a]!=a:parent[a]=find(parent[a])
  return parent[a]
 for r in rows:
  a,b=find('I:'+r['identity']),find('G:'+r['group']);parent[max(a,b)]=min(a,b)
 comps=defaultdict(list)
 for r in rows:comps[find('I:'+r['identity'])].append(r)
 cs={key('|'.join(sorted('I:'+r['identity']+'|G:'+r['group'] for r in rs))):rs for rs in comps.values()}
 counts=[0]*5;assignment={}
 for c in sorted(cs,key=lambda c:(-len(cs[c]),key(c))):
  f=min(range(5),key=lambda f:(counts[f],f));counts[f]+=len(cs[c])
  for r in cs[c]:assignment[r['source_image_sha256']]=(f,c)
 return assignment,counts,len(cs)

def cache_index():
 result={};bindings={}
 for tag in ('train','eval'):
  manifest=ROOT/f'results/rc_original7_{"train128" if tag=="train" else "expanded_eval128"}_manifest_v1/worker_manifest.json'
  wm=read(manifest);bindings[tag+'_worker']=bind(manifest)
  for shard in range(16):
   pair={}
   for kind in ('token_raw','roma'):
    folder=ROOT/f'results/rc_original7_{tag}128_{kind}_v1'/f'shard{shard:02d}'
    rec=read(folder/'receipt.json');val=read(folder/'validation.json')
    assert val['status'].endswith('CPU_REPLAY_PASS') and val['payload']==rec['payload'] and val['receipt']==bind(folder/'receipt.json')
    pair[kind]=dict(payload=rec['payload'],receipt=bind(folder/'receipt.json'),validation=bind(folder/'validation.json'))
   for j,w in enumerate(wm['records'][shard*8:(shard+1)*8]):
    h=w['source_image_sha256'];assert h not in result
    result[h]=dict(**pair,record_index=j,source_query_id=w['query_id'],source_execution_ordinal=w['execution_ordinal'])
 return result,bindings

def produce():
 assert not OUT.exists();rs,dup,labels=population();assignment,counts,ncomp=folds(rs);cache,cb=cache_index();workers=[];roles=[];missing=[]
 rs.sort(key=lambda r:(key(r['source_image_sha256']),r['source_image_sha256']))
 for i,r in enumerate(rs):
  q='H593-'+key(r['source_image_sha256'])[:24];path=OUT/'images'/(q+Path(r['original_path']).suffix.lower());path.parent.mkdir(parents=True,exist_ok=True);assert Path(r['original_path']).is_file();os.symlink(r['original_path'],path)
  f,c=assignment[r['source_image_sha256']]
  w=dict(query_id=q,execution_ordinal=i,track=r['track'],source_image_sha256=r['source_image_sha256'],query_image_path=str(path))
  if r['source_image_sha256'] in cache:w['reuse']=cache[r['source_image_sha256']]
  else:w['missing_ordinal']=len(missing);missing.append(q)
  workers.append(w);roles.append(dict(**r,query_id=q,execution_ordinal=i,outer_fold=f,component=c))
 assert len(workers)==593 and len(missing)==337
 sources={k:bind(ROOT/v) for k,v in SOURCES.items()};sources.update(cb);sources['program']=bind(__file__)
 write(OUT/'worker_manifest.json',dict(status='H593_OUTCOME_BLIND_WORKER_FROZEN',records=workers,query_count=593,missing_query_ids=missing,missing_shards=43,queries_per_shard=8,sources=sources))
 write(OUT/'curator_roles.json',dict(status='H593_GROUPED_ROLES_FROZEN',records=roles,identity_record_sources=labels,duplicate_resolution=dup))
 write(OUT/'split_manifest.json',dict(seed=SEED,algorithm='Identity/source-group connected components; descending component image count then SHA; assign least image-loaded fold, tie fold index',query_count=593,identities=68,groups=64,components=ncomp,fold_image_counts=counts,folds=[dict(fold=f,heldout_query_ids=[r['query_id'] for r in roles if r['outer_fold']==f],train_query_ids=[r['query_id'] for r in roles if r['outer_fold']!=f]) for f in range(5)],previous_EVAL_is_now_development_OOF_population=True,old_99_of_128_not_comparable=True))
 subprocess.run([sys.executable,__file__,'validate'],check=True)
 print(json.dumps(read(OUT/'validation.json')['counts']))
def validate():
 rs,dup,_=population();expected={r['source_image_sha256']:r for r in rs};w=read(OUT/'worker_manifest.json');roles=read(OUT/'curator_roles.json')['records'];sp=read(OUT/'split_manifest.json')
 assert len(roles)==len(w['records'])==593 and len({r['query_id'] for r in roles})==593
 assert {r['source_image_sha256'] for r in roles}==set(expected)
 for r,x in zip(roles,w['records']):
  assert r['query_id']==x['query_id'] and r['source_image_sha256']==x['source_image_sha256']
  assert all(r[k]==expected[r['source_image_sha256']][k] for k in ('identity','group','original_path'))
  assert Path(x['query_image_path']).is_symlink() and os.readlink(x['query_image_path'])==r['original_path']
 for f in sp['folds']:
  a=[r for r in roles if r['query_id'] in set(f['train_query_ids'])];b=[r for r in roles if r['query_id'] in set(f['heldout_query_ids'])]
  assert len(a)+len(b)==593
  for k in ('identity','group','source_image_sha256','component','query_id'):assert not {r[k] for r in a}&{r[k] for r in b}
 cache,_=cache_index()
 for x in w['records']:
  if 'reuse' in x:assert x['reuse']==cache[x['source_image_sha256']]
 assert sum('reuse' in x for x in w['records'])==256 and len(w['missing_query_ids'])==337
 write(OUT/'validation.json',dict(status='H593_METADATA_GROUP_SEPARATION_PASS',fresh_process=True,worker=bind(OUT/'worker_manifest.json'),curator=bind(OUT/'curator_roles.json'),split=bind(OUT/'split_manifest.json'),counts=dict(images=593,identities=68,groups=64,components=sp['components'],fold_images=sp['fold_image_counts'],reuse=256,missing=337),ranking_or_outcome_reads=0,formal_private_label_reads=0))
if __name__=='__main__':
 if sys.argv[1]=='produce':produce()
 else:validate()
