"""New frozen-score combinations and color-only competing explanation, full gallery."""
from pathlib import Path
from collections import defaultdict
import sys,json,math,csv,hashlib
import numpy as np,torch
from PIL import Image,ImageOps
D=Path(__file__).resolve().parent;V=D.parent;O=V.parent;P=O.parent;R=Path('/mnt/lsdf_zyin');sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,zscore,Ranking,metrics
torch.set_num_threads(2);protocol=read(D/'protocol.json');c=read(P/'cohort.json');qids=[r['query'] for r in c]
b=torch.load(R/'cache/scores/colnomic7b/image.pt',map_location='cpu',mmap=True,weights_only=True);ids=b['reference_ids'];rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True);rk=Ranking(ids,rb['sha256'],c);ri={r:j for j,r in enumerate(ids)};qi={q:i for i,q in enumerate(qids)}
old=b['scores'].numpy().astype(float);sub=np.load(V/'prior_subject_scores.npy');new=np.load(V/'selected_image_scores.npy');inc=np.load(V/'incremental_image_scores.npy');text=np.load(V/'production_numeric_z.npy');base=np.load(V/'incremental_joint_scores.npy')
bonus=np.zeros_like(base)
for r in read(V/'identifier_layout_v1/automatic_matches.json'):
 if r['mode']!='one_edit_min8':continue
 groups=defaultdict(list)
 for p in r['matches']:groups[p['query_code']].append(p)
 for pp in groups.values():
  a=np.zeros(len(ids))
  for p in pp:
   js=[ri[x] for x in p['references']];n=len(js);idf=math.log(1+(len(ids)-n+.5)/(n+.5))/math.log(len(ids)+1);a[js]=np.maximum(a[js],idf)
  bonus[qi[r['query']]]+=a
fixed=base+bonus;baseline,_,_=rk.all(fixed);assert (baseline==1).sum()==1051
np.save(D/'fixed1051_scores.npy',fixed);np.save(D/'code_bonus.npy',bonus)
excluded=set(read(V/'identifier_layout_v1/protocol.json')['excluded_queries']);target=np.array([i for i,r in enumerate(c) if r['query'] not in excluded and baseline[i]>1]);assert len(target)==54
scopes={'all':np.arange(len(c)),'validation':np.array([i for i,r in enumerate(c) if r['split']=='val']),'test':np.array([i for i,r in enumerate(c) if r['split']=='test']),'retained':np.array([i for i,r in enumerate(c) if r['query'] not in excluded]),'target54':target,'retained_test':np.array([i for i,r in enumerate(c) if r['split']=='test' and r['query'] not in excluded])}
image={'incremental':inc,'selected_new':new,'new_plus_subject':.8*new+.2*sub,'prior_subject':.8*old+.2*sub,'half_incremental_new':.5*inc+.5*new}
choices=[];curve=[];ranks=[]
def evaluate(name,s,parameters):
 rr,_,_=rk.all(s);vm=metrics(rr,c,scopes['validation'],baseline);choices.append(dict(name=name,parameters=parameters,validation=vm,ranks=rr));
 for scope,ix in scopes.items():curve.append(dict(method=name,scope=scope,**metrics(rr,c,ix,baseline)))
 return rr
for name,a in image.items():
 z=zscore(a)
 for tw in protocol['text_weights']:
  for cw in protocol['code_weights']:
   label=f'{name}/text{tw:g}/code{cw:g}';s=(1-tw)*z+tw*text+cw*bonus;rr=evaluate(label,s,dict(image=name,text_weight=tw,code_weight=cw));ranks.append(rr)
 print('FUSION',name,flush=True)
best=max(choices,key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-abs(x['parameters']['text_weight']-.5),-x['parameters']['code_weight']))
par=best['parameters'];selected=(1-par['text_weight'])*zscore(image[par['image']])+par['text_weight']*text+par['code_weight']*bonus
np.save(D/'selected_fusion_scores.npy',selected);write(D/'FUSION_SELECTION.json',{k:v for k,v in best.items() if k!='ranks'})
csvwrite(D/'fusion_curve.csv',curve);np.save(D/'fusion_ranks.npy',np.stack(ranks));write(D/'fusion_methods.json',[{k:v for k,v in a.items() if k!='ranks'} for a in choices]);print('FUSION SELECT',best['name'],int((best['ranks']==1).sum()),flush=True)
# Chroma is separate raw image information, never OCR or an encoder feature.
qo=read(R/'cache/ocr/photos.json');ro=read(R/'cache/ocr/references.json');policy=read(P/'SUBJECT_REGION_POLICY.json');assert not policy['sam_allowed'];boxes=read(Path(policy['remote_source']))['records']
def hist(im):
 im.thumbnail((256,256));a=np.asarray(im.convert('HSV'),dtype=float);h=a[:,:,0]/255;s=a[:,:,1]/255;v=a[:,:,2]/255;weight=s*np.maximum(v-.08,0)*(s>.15);bins=np.floor(h*36).astype(int)%36
 out=np.bincount(bins.ravel(),weights=weight.ravel(),minlength=36)
 for _ in range(2):out=.5*out+.25*np.roll(out,1)+.25*np.roll(out,-1)
 n=out.sum();out=out/n if n else out
 return np.sqrt(out),float(n/weight.size)
def readim(path):
 with Image.open(path) as im:
  orientation=int(im.getexif().get(274,1));out=im.convert('RGB')
  transforms={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}
  if orientation in transforms:out=out.transpose(transforms[orientation])
  out.info.clear();return out
if (D/'chroma_features.npz').exists():
 f=np.load(D/'chroma_features.npz');qh,rh,qs,rs=[f[k] for k in ['query','reference','query_support','reference_support']]
else:
 qh=[];rh=[];qs=[];rs=[]
 for i,q in enumerate(qids):
  b=boxes[q];assert b['source']=='human' and b['status']=='confirmed';h,s=hist(readim(R/qo[q]['image_path']).crop(tuple(map(int,b['bbox']))));qh.append(h);qs.append(s)
 for j,r in enumerate(ids):
  h,s=hist(readim(R/ro[r]['image_path']));rh.append(h);rs.append(s)
  if j%1000==0:print('CHROMA',j,flush=True)
 qh,rh,qs,rs=map(np.asarray,[qh,rh,qs,rs]);np.savez_compressed(D/'chroma_features.npz',query=qh,reference=rh,query_support=qs,reference_support=rs)
color=qh@rh.T;color[qs<.01]=0;np.save(D/'chroma_similarity.npy',color)
colorrows=[];colorchoices=[]
for family,scores in [('fixed1051',fixed),('selected_fusion',selected),('pure_selected_image',zscore(new))]:
 for weight in protocol['color_weights']:
  s=scores+weight*color;rr,_,_=rk.all(s);vm=metrics(rr,c,scopes['validation'],baseline);colorchoices.append(dict(family=family,weight=weight,validation=vm,ranks=rr))
  for scope,ix in scopes.items():colorrows.append(dict(family=family,weight=weight,scope=scope,**metrics(rr,c,ix,baseline)))
 rr,_,_=rk.all(color)
for scope,ix in scopes.items():colorrows.append(dict(family='color_only',weight=1,scope=scope,**metrics(rr,c,ix,baseline)))
colorbest=max([a for a in colorchoices if a['family']!='pure_selected_image'],key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['weight'],x['family']=='fixed1051'))
selectedcolor=({'fixed1051':fixed,'selected_fusion':selected}[colorbest['family']])+colorbest['weight']*color;np.save(D/'selected_color_scores.npy',selectedcolor)
csvwrite(D/'color_curve.csv',colorrows);write(D/'COLOR_SELECTION.json',{k:v for k,v in colorbest.items() if k!='ranks'})
# Truth-free confidence routing: only compare branch winners and distinct-SHA margins.
def confidence(s):
 ordered=s[:,rk.order];u=np.maximum.reduceat(ordered,rk.starts,axis=1);best2=np.partition(u,-2,axis=1)[:,-2:];return best2.max(1)-best2.min(1)
sources={'fixed1051':fixed,'new_image_text':.5*(zscore(new)+text)+bonus,'image_only':zscore(new),'ocr_only':text+bonus,'selected_color':selectedcolor}
conf={k:confidence(s) for k,s in sources.items()};route=[];routechoices=[]
for alternative in ['new_image_text','image_only','ocr_only','selected_color']:
 for threshold in [.1,.25,.5,1]:
  for ratio in [1,2,4]:
   use=(conf['fixed1051']<threshold)&(conf[alternative]>ratio*conf['fixed1051']);s=np.where(use[:,None],sources[alternative],fixed);rr,_,_=rk.all(s);vm=metrics(rr,c,scopes['validation'],baseline)
   label=f'{alternative}/basegap{threshold:g}/ratio{ratio:g}';routechoices.append(dict(name=label,alternative=alternative,threshold=threshold,ratio=ratio,switched=int(use.sum()),validation=vm,ranks=rr));
   for scope,ix in scopes.items():route.append(dict(method=label,scope=scope,**metrics(rr,c,ix,baseline)))
routechoices.append(dict(name='no_routing',alternative='fixed1051',threshold=0,ratio=0,switched=0,validation=metrics(baseline,c,scopes['validation'],baseline),ranks=baseline))
routebest=max(routechoices,key=lambda x:(x['validation']['macro_accuracy'],x['validation']['macro_mrr'],-x['switched']))
if routebest['alternative']=='fixed1051':routed=fixed
else:
 alt=routebest['alternative'];use=(conf['fixed1051']<routebest['threshold'])&(conf[alt]>routebest['ratio']*conf['fixed1051']);routed=np.where(use[:,None],sources[alt],fixed)
np.save(D/'selected_routing_scores.npy',routed);csvwrite(D/'routing_curve.csv',route);write(D/'ROUTING_SELECTION.json',{k:v for k,v in routebest.items() if k!='ranks'})
# Branch removals at fixed scale separate contribution from normalization changes.
sd=inc.std(1,keepdims=True);mu=inc.mean(1,keepdims=True);components={'original_image':.5*.4*old/sd,'subject_image':.5*.2*sub/sd,'new_image':.5*.4*new/sd,'ocr':.5*text,'code':bonus}
ablations={};abrows=[]
for arm,a in components.items():
 scores=fixed-a;rr,_,_=rk.all(scores);ablations['drop_'+arm]=rr
 for scope,ix in scopes.items():abrows.append(dict(method='drop_'+arm,scope=scope,**metrics(rr,c,ix,baseline)))
for arm,raw in [('original_image',inc-.4*old),('subject_image',inc-.2*sub),('new_image',inc-.4*new)]:
 scores=.5*(zscore(raw)+text)+bonus;rr,_,_=rk.all(scores);ablations['drop_'+arm+'_fresh_norm']=rr
 # Change scale only, keep candidate-dependent numerator unchanged.
 nsd=raw.std(1,keepdims=True);scores=.5*((inc-mu)/nsd+text)+bonus;nr,_,_=rk.all(scores);ablations['only_'+arm+'_norm']=nr
csvwrite(D/'ablation_curve.csv',abrows)
finals={'fixed1051':fixed,'selected_fusion':selected,'selected_color':selectedcolor,'selected_routing':routed};summaries={};per=[]
for name,scores in finals.items():
 rr,_,_=rk.all(scores);summaries[name]={scope:metrics(rr,c,ix,baseline) for scope,ix in scopes.items()}
 for i,r in enumerate(c):
  z=rk.one(scores[i],i);per.append(dict(query=r['query'],split=r['split'],method=name,baseline_rank=int(baseline[i]),**z))
ledger=[]
for i in target:
 r=c[i];z=rk.one(fixed[i],i);p,n=ri[z['positive']],ri[z['negative']];parts={k:float(a[i,p]-a[i,n]) for k,a in components.items()};assert abs(sum(parts.values())-z['margin'])<1e-8
 ledger.append(dict(query=r['query'],positive=z['positive'],negative=z['negative'],rank=z['rank'],parts=parts,margin=z['margin'],ablation_ranks={k:int(v[i]) for k,v in ablations.items()},selected_method_ranks={k:rk.one(s[i],i)['rank'] for k,s in finals.items()},color_positive=float(color[i,p]),color_negative=float(color[i,n]),color_support=float(qs[i])))
csvwrite(D/'repair_cases1117.csv',per);write(D/'REPAIR_SUMMARY.json',dict(protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),methods=summaries,fusion_selection={k:v for k,v in best.items() if k!='ranks'},color_selection={k:v for k,v in colorbest.items() if k!='ranks'},routing_selection={k:v for k,v in routebest.items() if k!='ranks'},limits=['Exploratory historical data; validation-selected, not test-selected.','Color histograms add image-level chromatic evidence, not proof that ColNomic encoded color.','Human subject boxes are existing reviewed annotations.']))
write(D/'MECHANISM54.json',ledger);print('POPULATION DONE',json.dumps({k:v['all']['correct'] for k,v in summaries.items()}),flush=True)
