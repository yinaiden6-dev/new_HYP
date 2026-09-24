import hashlib,json,math,os
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC')
OUT=ROOT/'results/rc_h593_qwen_quality_joint_v1'
AUTH=ROOT/'registry/rc_h593_qwen_quality_joint_authority_v1_20260924.json'
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(p),'sha256':digest(p)}
def checked(b):
 p=Path(b['path']);assert digest(p)==b['sha256'],('SHA',str(p));return p
assert all((OUT/k).exists() for k in ['validation.json','result.json','predictions_prelabel_seal.json']),'FORMAL_JOIN_NOT_SEALED_YET'
v=read(OUT/'validation.json');assert v['status']=='QWEN_QUALITY_593_AXES_BASELINE_PARITY_SEALS_COUNTS_PASS'
res=read(checked(v['result']));seal=read(checked(v['prelabel_seal']));a=read(AUTH)
assert v['authority']==res['authority']==seal['authority']==binding(AUTH)
for b in a['sources']:checked(b)
cache=read(checked(a['cache']));assert cache['labels_included'] is False
assert seal['predictions']==593 and seal['cache']==a['cache']
# Open identity labels only after validating all submitted folds and the formal seal.
folds={};parameters={};old={};preds={};parity={}
for f,b in enumerate(seal['fold_validations']):
 fv=read(checked(b));p=read(checked(fv['payload']));assert fv['fold']==p['fold']==f
 assert p['authority']==binding(AUTH) and p['heldout_label_reads']==0
 assert set(p['parameters'])=={loss+'_'+arm for loss in ['CE','COST1'] for arm in a['arms']}
 for r in p['predictions']:
  assert r['query_id'] not in preds;preds[r['query_id']]=(f,r)
 parameters[f]=p['parameters'];folds[f]=p
 previous=read(checked(a['folds'][str(f)]['qwen_payload']))
 parity[f]={}
 for loss in ['CE','COST1']:
  newtheta=p['parameters'][loss+'_QWEN3'];oldtheta=previous['parameters'][loss]
  err=max(abs(float.fromhex(x)-float.fromhex(y)) for x,y in zip(newtheta,oldtheta));assert err<2e-10
  parity[f][loss]={'parameters_bit_exact':newtheta==oldtheta,'max_parameter_error':err}
 for r in previous['predictions']:old[r['query_id']]=r
assert len(preds)==593
roles={r['query_id']:r for r in read(checked(a['curator']))['records']}
gallery={r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
split=read(checked(a['split']))
result_rows={r['query_id']:r for r in res['rows']}
assert len(result_rows)==593 and set(preds)==set(result_rows)
trainroles={f:read(checked(a['folds'][str(f)]['train_roles']))['records'] for f in range(5)}
verified=[];dotmax=0.;xmax=0.;logits_checked=0;oldmatches=0;raw_winner_checks=0
for row in cache['rows']:
 qid=row['query_id'];axis=row['candidate_physical_rows'];w=row['winner'];c=row['challenger_positions'];role=roles[qid];f,p=preds[qid]
 assert len(axis)==len(set(axis))==128 and c==[i for i in range(128) if i!=w]
 assert role['outer_fold']==f and qid in split['folds'][f]['heldout_query_ids']
 assert role['identity'] not in {t['identity'] for t in trainroles[f]}
 assert role['component'] not in {t['component'] for t in trainroles[f]}
 assert folds[f]['train_query_ids']==a['folds'][str(f)]['train_query_ids']
 raw=np.asarray(row['raw_scores'],dtype=np.float64);qwen=np.asarray(row['qwen_logit'],dtype=np.float64)
 mass=np.asarray(row['mass'],dtype=np.float64);content=np.asarray(row['free_content'],dtype=np.float64)
 sym=lambda ar:(ar[c]-ar[w])/(np.abs(ar[c])+abs(ar[w])+1e-12)
 independently_x=np.column_stack(((raw[c]-raw[w])/max(float(raw.std()),1e-12),(qwen[c]-qwen[w])/max(float(qwen.std()),1e-12),sym(mass),sym(content)))
 e=float(np.abs(independently_x-np.asarray(row['X'])).max());assert e<2e-10;xmax=max(xmax,e)
 raworder=sorted(range(128),key=lambda i:(-raw[i],axis[i]));assert raworder[0]==w;raw_winner_checks+=1
 rank={i:j for j,i in enumerate(raworder)}
 order_q=sorted(range(128),key=lambda i:(-qwen[i],rank[i],axis[i]))
 orders={'RAW':raworder,'QWEN3_DIRECT':order_q}
 for name,m in p['models'].items():
  arm=name.split('_',1)[1];idx=a['arms'][arm];t=[float.fromhex(v) for v in parameters[f][name]]
  z=[math.fsum(float(xx[k])*tt for k,tt in zip(idx,t[:-1]))+t[-1] for xx in independently_x]
  saved=[float.fromhex(v) for v in m['logits_hex']];assert len(saved)==len(z)==127
  e=max(abs(x-y) for x,y in zip(z,saved));assert e<2e-10;(dotmax:=max(dotmax,e));logits_checked+=127
  full=[0.]*128
  for pos,val in zip(c,z):full[pos]=val
  order=sorted(range(128),key=lambda i:(-full[i],i!=w,i))
  assert axis[order[0]]==m['selected'];orders[name]=order
  if arm=='QWEN3':
   loss=name.split('_',1)[0];o=old[qid]['models'][loss]
   assert m['selected']==o['selected'];assert max(abs(x-float.fromhex(y)) for x,y in zip(saved,o['logits_hex']))<2e-9;oldmatches+=1
 target={i for i,pid in enumerate(axis) if gallery[pid]==role['identity']};assert len(target)<=1
 selected={m:axis[o[0]] for m,o in orders.items()};correct={m:o[0] in target for m,o in orders.items()}
 rr={m:next((1./(i+1) for i,pos in enumerate(o) if pos in target),0.) for m,o in orders.items()}
 prev=result_rows[qid];assert prev['selected']==selected and prev['correct']==correct
 assert prev['target_in_C128']==bool(target) and prev['fold']==f and prev['component']==role['component']
 for m in rr:assert abs(rr[m]-prev['reciprocal_rank_C128'][m])<1e-15
 verified.append({'query_id':qid,'fold':f,'component':role['component'],'target_in_C128':bool(target),'selected':selected,'correct':correct,'rr':rr})
counts={m:sum(r['correct'][m] for r in verified) for m in verified[0]['correct']};assert counts==res['counts']
assert sum(r['target_in_C128'] for r in verified)==570
assert {k:counts[k] for k in ['RAW','QWEN3_DIRECT','CE_QWEN3','COST1_QWEN3']}=={'RAW':426,'QWEN3_DIRECT':522,'CE_QWEN3':523,'COST1_QWEN3':477}
fc={str(f):{m:sum(r['correct'][m] for r in verified if r['fold']==f) for m in counts} for f in range(5)};assert fc==res['folds']
for m in counts:assert abs(sum(r['rr'][m] for r in verified)/593-res['MRR_C128'][m])<1e-14
comparisons={}
for key, original in res['comparisons'].items():
 base,model=key.split('__to__');groups=defaultdict(list)
 for r in verified:groups[r['component']].append(int(r['correct'][model])-int(r['correct'][base]))
 values=[groups[k] for k in sorted(groups)];assert len(values)==64
 total=np.array([sum(v) for v in values],dtype=float);size=np.array([len(v) for v in values],dtype=float)
 sample=np.random.default_rng(20260924).integers(0,64,size=(10000,64))
 macro=(total/size)[sample].mean(1);weighted=total[sample].sum(1)/size[sample].sum(1)
 rescue=sum(r['correct'][model] and not r['correct'][base] for r in verified);br=sum(r['correct'][base] and not r['correct'][model] for r in verified)
 now={'rescue':rescue,'breaks':br,'net':rescue-br,'changed':sum(r['selected'][model]!=r['selected'][base] for r in verified),'component_balanced_delta':float((total/size).mean()),'component_balanced_bootstrap95':list(map(float,np.quantile(macro,[.025,.975]))),'query_weighted_delta':float(total.sum()/size.sum()),'query_weighted_cluster_bootstrap95':list(map(float,np.quantile(weighted,[.025,.975])))}
 for k in now:assert np.allclose(now[k],original[k],atol=1e-14,rtol=0),(key,k,now[k],original[k])
 comparisons[key]=now
summary={'status':'INDEPENDENT_H593_QWEN_QUALITY_AXES_DOTS_IDENTITIES_STATS_PASS','result':binding(OUT/'result.json'),'validation':binding(OUT/'validation.json'),'authority':binding(AUTH),'prediction_seal':binding(OUT/'predictions_prelabel_seal.json'),'queries':593,'candidate_recall':570,'components':64,'held_label_opening':'Only after formal result, validator and all five prediction seals verified','max_independent_feature_error':xmax,'max_independent_dot_error':dotmax,'logits_checked':logits_checked,'raw_winner_checks':raw_winner_checks,'original_qwen_exact_choices':oldmatches,'original_parameter_parity':parity,'counts':counts,'fold_counts':fc,'primary_vs_qwen':comparisons['CE_QWEN3__to__CE_QWEN_ML5'],'primary_vs_free_content':comparisons['CE_QWEN_L4__to__CE_QWEN_ML5'],'comparisons_verified':len(comparisons),'script':binding(Path(__file__))}
p=OUT/'independent_result_audit.json';value=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
if p.exists():assert p.read_text()==value
else:
 tmp=p.with_suffix('.audit.tmp');tmp.write_text(value);os.replace(tmp,p)
print(json.dumps(summary,indent=2,ensure_ascii=False))
