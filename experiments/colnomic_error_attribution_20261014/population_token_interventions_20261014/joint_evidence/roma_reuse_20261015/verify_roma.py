"""Independent local replay from the saved candidate inputs and human-box support."""
from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;A=D/'roma_reuse'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
c=read(P/'cohort.json');qids=[r['query'] for r in c];refs=read(V.parent/'reference_detail_v4/fullgallery/manifest.json')['rows'];ids=[x['id'] for x in refs];sha=[x['image_sha256'] for x in refs];ri={s:j for j,s in enumerate(ids)}
checks=0;hashes={};summaries={}
for folder,label in [(A,'fusion'),(A/'pure_selected_image','image')]:
 inputs=np.load(folder/'inputs.npz');s=read(folder/'SUMMARY.json');base=np.load(folder/'baseline.npz');top=inputs['top10_rows'];app=inputs['applicable'];z=inputs['baseline_scores'];result={}
 assert len(qids)==len(base['ranks'])==1117 and qids==base['query_ids'].tolist()
 for method,info in s['results'].items():
  b=info['beta'];key=info['feature'];saved=np.load(folder/(method+'.npz'));expected=[];winner=[]
  for i,row in enumerate(c):
   indices=top[i].tolist()
   if app[i] and b:
    vals=(z[i]+b*inputs[key][i]).tolist();indices=[j for j,v in sorted(zip(indices,vals),key=lambda x:(-x[1],x[0]))]
   pos=set(row['positive_sha256']);pr=next((k+1 for k,j in enumerate(indices) if sha[j] in pos),int(base['ranks'][i]));expected.append(pr);winner.append(ids[indices[0]])
  assert np.array_equal(expected,saved['ranks']),method
  assert winner==saved['top1'].tolist(),method
  rr=np.array(expected);bb=base['ranks'];allm=info['metrics']['all'];assert int((rr==1).sum())==allm['correct'];assert int(((rr==1)&(bb>1)).sum())==allm['repairs'];assert int(((rr>1)&(bb==1)).sum())==allm['harms'];checks+=len(expected);result[method]=allm['correct']
  hashes[str((folder/(method+'.npz')).relative_to(A))]=hashlib.sha256((folder/(method+'.npz')).read_bytes()).hexdigest()
 summaries[label]=result

support=read(A/'SUBJECT_GATE_SUPPORT.json');lookup={(x['query'],x['reference_sha256']):x for x in support.values()};gates=read(A/'SUBJECT_GATE_SUMMARY.json');gatechecks=0
for family,folder,propname in [('fusion',A,'S_previous_fixed_0.2'),('image',A/'pure_selected_image','selected')]:
 base=np.load(folder/'baseline.npz');prop=np.load(folder/(propname+'.npz'));info=gates['families'][family]
 for method,r in info['results'].items():
  field={'subject_overlap':'support','area_only':'area','rolled_overlap':'rolled'}[r['feature']];rr=[];tt=[]
  for i,q in enumerate(qids):
   change=str(base['top1'][i])!=str(prop['top1'][i]);accept=not change or lookup[q,sha[ri[str(prop['top1'][i])]]][field]>=r['threshold'];source=prop if accept else base;rr.append(int(source['ranks'][i]));tt.append(str(source['top1'][i]));assert (rr[-1]==1)==(sha[ri[tt[-1]]] in set(c[i]['positive_sha256']))
  saved=np.load(A/f'gate_{family}_{method}.npz');assert rr==saved['ranks'].tolist() and tt==saved['top1'].tolist();rr=np.array(rr);bb=base['ranks'];assert int((rr==1).sum())==r['metrics']['all']['correct'];assert int(((rr==1)&(bb>1)).sum())==r['metrics']['all']['repairs'];assert int(((rr>1)&(bb==1)).sum())==r['metrics']['all']['harms'];gatechecks+=len(rr)
  hashes[f'gate_{family}_{method}.npz']=hashlib.sha256((A/f'gate_{family}_{method}.npz').read_bytes()).hexdigest()
region=read(A/'REGION_AUDIT.json');overlap_checks=0
for row in region['rows']:
 for role in ['positive','negative']:
  x=row[role];key=(row['query'],x.get('reference_sha256'))
  if x.get('available') and key in lookup:assert abs(x['query_overlap_inside_fraction']-lookup[key]['support'])<1e-12;overlap_checks+=1
result=dict(status='passed',queries=1117,gallery_rows=len(refs),gallery_unique=len(set(sha)),candidate_ranking_checks=checks,subject_gate_ranking_checks=gatechecks,overlap_fraction_cross_checks=overlap_checks,all_query_hashes_checked_by_remote_region_audit=region['all_query_SHA_matches'],results=summaries,file_hashes=hashes,scope='Independent Python sorting, target-rank counting and gate replay; not a new neural forward. Native cache numeric producer previously audited separately.')
(A/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print({k:v for k,v in result.items() if k not in ['file_hashes','results']})
