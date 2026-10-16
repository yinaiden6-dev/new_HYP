"""Feature-only coordinate/row and unseen-identity pair probes, with shuffled labels."""
import sys,re
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,data,preregister,read,write,csvout
from scripts.colnomic_mechanism.geometry import grid


def regression_fit(x,y,alpha,weights=None):
 weights=np.ones(len(x)) if weights is None else weights
 mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)).clip(1e-5)
 design=np.column_stack([(x-mean)/scale,np.ones(len(x))]).astype(np.float64)
 regularizer=np.eye(design.shape[1])*alpha;regularizer[-1,-1]=0
 coefficient=np.linalg.solve(design.T@(design*weights[:,None])+regularizer,design.T@(y*weights[:,None]))
 return dict(mean=mean,scale=scale,coefficient=coefficient)

def regression_predict(model,x):return np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))])@model['coefficient']


def logistic_fit(x,y,C,weights=None):
 weights=np.ones(len(x)) if weights is None else weights
 mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)).clip(1e-5)
 design=torch.from_numpy(np.column_stack([(x-mean)/scale,np.ones(len(x))]).astype(np.float64))
 target=torch.from_numpy(np.asarray(y,np.float64));sample=torch.from_numpy(np.asarray(weights,np.float64));sample/=sample.sum()
 coefficient=torch.zeros(design.shape[1],dtype=torch.float64,requires_grad=True)
 optimizer=torch.optim.LBFGS([coefficient],max_iter=100,lr=1.,line_search_fn='strong_wolfe',tolerance_grad=1e-8)
 def closure():
  optimizer.zero_grad();logits=design@coefficient
  loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*sample).sum()+coefficient[:-1].square().sum()/(2*C*len(x))
  loss.backward();return loss
 optimizer.step(closure)
 return dict(mean=mean,scale=scale,coefficient=coefficient.detach().numpy())

def logistic_predict(model,x):return np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))])@model['coefficient']


def package_metric(values,packages):return float(np.mean([np.mean(values[np.asarray(packages)==p]) for p in sorted(set(packages))]))


def regression_metrics(y,pred,packages):
 error=np.abs(y-pred).mean(1);result=dict(mae=float(error.mean()),package_macro_mae=package_metric(error,packages),
  r2=float(1-np.square(y-pred).sum()/max(1e-12,np.square(y-y.mean(0)).sum())))
 for dimension in range(y.shape[1]):
  true=np.minimum((y[:,dimension]*4).astype(int),3);guess=np.clip((pred[:,dimension]*4).astype(int),0,3)
  result[f'dim{dimension}_four_bin_accuracy']=float(np.mean(true==guess))
  result[f'dim{dimension}_package_macro_four_bin_accuracy']=package_metric((true==guess).astype(float),packages)
 return result


def classification_metrics(y,logits,packages):
 correct=(logits>=0)==y;positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
 auc=float(np.mean((logits[positive,None]>logits[negative][None]).astype(float)+.5*(logits[positive,None]==logits[negative][None]))) if len(positive) and len(negative) else None
 return dict(accuracy=float(correct.mean()),package_macro_accuracy=package_metric(correct.astype(float),packages),auc=auc,
  samples=len(y),packages=len(set(packages)))


def shuffled_labels(labels,packages,seed):
 result=np.array(labels,copy=True);rng=np.random.default_rng(seed)
 for package in sorted(set(packages)):
  positions=np.flatnonzero(np.asarray(packages)==package);result[positions]=result[rng.permutation(positions)]
 return result


def save_head(path,model,extra):
 torch.save(dict(**{k:torch.from_numpy(np.asarray(v)) for k,v in model.items()},**extra),path)


def position_data(d,kind):
 xs=[];ys=[];ids=[];packages=[];splits=[];indices=[];weights=[]
 package_photo_count=Counter(r['package_id'] for r in d['photos'].values())
 for qi,qid in enumerate(d['qb']['ids']):
  spec=d['metadata']['photos'][qi];assert spec['id']==qid
  layout=spec['layout'];tokens=np.arange(layout['start'],layout['start']+layout['count']);centers=grid(spec)[tokens]
  box=d['boxes'][qid];bbox=np.array(box['bbox'])/np.tile([box['width'],box['height']],2)
  keep=np.ones(len(tokens),bool);target=np.column_stack([((tokens-layout['start'])%layout['width']+.5)/layout['width'],((tokens-layout['start'])//layout['width']+.5)/layout['height']])
  if kind!='absolute':
   keep=((centers>=bbox[:2])&(centers<bbox[2:])).all(1);target=(centers-bbox[:2])/(bbox[2:]-bbox[:2])
  if kind=='ocr_row':
   record=d['qo'][qid];eligible=[line for line in record['lines'] if ((np.mean(line['polygon'],axis=0)/[record['width'],record['height']]>=bbox[:2]) & (np.mean(line['polygon'],axis=0)/[record['width'],record['height']]<bbox[2:])).all()]
   if len(eligible)<4:continue
   eligible.sort(key=lambda line:np.mean(line['polygon'],axis=0)[1])
   bounds=np.array([line['bbox'] for line in eligible])/np.tile([record['width'],record['height']],2)
   half=np.array([.5/layout['width'],.5/layout['height']])
   if spec['exif_orientation']>=5:half=half[::-1]
   left=np.maximum(centers[:,None]-half,bounds[None,:,:2]);right=np.minimum(centers[:,None]+half,bounds[None,:,2:])
   intersection=np.maximum(right-left,0).prod(2)/(4*half.prod());candidate=intersection>=.05
   keep&=candidate.sum(1)==1;target=((candidate.argmax(1)+.5)/len(eligible))[:,None]
  tokens=tokens[keep];target=target[keep]
  if not len(tokens):continue
  rng=np.random.default_rng(20261013+qi);selection=np.sort(rng.choice(len(tokens),min(128,len(tokens)),replace=False))
  tokens=tokens[selection];target=target[selection]
  xs.append(d['qb']['embeddings'][qi][tokens].float().numpy());ys.append(target)
  ids.extend([qid]*len(tokens));packages.extend([d['photos'][qid]['package_id']]*len(tokens));splits.extend([d['splits'][qid]]*len(tokens));indices.extend(tokens.tolist())
  weights.extend([1/(len(tokens)*package_photo_count[d['photos'][qid]['package_id']])]*len(tokens))
 return np.concatenate(xs),np.concatenate(ys),np.array(packages),np.array(splits),np.array(weights),ids,indices


def run_positions(d,config,folder):
 tables=[]
 for kind in ['absolute','subject_relative','ocr_row']:
  x,y,packages,splits,weights,ids,indices=position_data(d,kind);train=splits=='train';val=splits=='val';test=splits=='test'
  choices=[]
  for alpha in config['probes']['ridge_grid']:
   model=regression_fit(x[train],y[train],alpha,weights[train]);m=regression_metrics(y[val],regression_predict(model,x[val]),packages[val]);choices.append((m['package_macro_mae'],alpha,model))
  _,alpha,model=min(choices,key=lambda row:(row[0],-row[1]))
  write(folder/(kind+'_selection.json'),dict(alpha=alpha,validation_only=True,train_samples=int(train.sum()),val_samples=int(val.sum()),test_samples=int(test.sum()),
   train_packages=len(set(packages[train])),val_packages=len(set(packages[val])),test_packages=len(set(packages[test])),head_inputs='128 frozen features only',
   ocr_row_labels='largest axis-aligned OCR box intersection; only single-line candidate cells; approximate row labels' if kind=='ocr_row' else None))
  save_head(folder/(kind+'_head.pt'),model,dict(kind=kind,alpha=alpha))
  mean=np.average(y[train],axis=0,weights=weights[train]);majority=[int(np.bincount(np.minimum((y[train,j]*4).astype(int),3),minlength=4).argmax()) for j in range(y.shape[1])]
  for split,mask in [('val',val),('test',test)]:
   tables.append(dict(probe=kind,control='feature_only',split=split,**regression_metrics(y[mask],regression_predict(model,x[mask]),packages[mask])))
   null=regression_metrics(y[mask],np.broadcast_to(mean,y[mask].shape),packages[mask]);null['majority_four_bin_accuracy']=[float(np.mean(np.minimum((y[mask,j]*4).astype(int),3)==majority[j])) for j in range(y.shape[1])]
   tables.append(dict(probe=kind,control='train_mean',split=split,**null))
  for seed in config['probes']['shuffle_seeds']:
   shuffled=shuffled_labels(y[train],packages[train],seed);null=regression_fit(x[train],shuffled,alpha,weights[train])
   tables.append(dict(probe=kind,control='label_shuffle',seed=seed,split='test',**regression_metrics(y[test],regression_predict(null,x[test]),packages[test])))
  csvout(folder/(kind+'_sample_index.csv'),[dict(photo_id=id,token_index=token,package_id=package,split=split) for id,token,package,split in zip(ids,indices,packages,splits)])
  print('Position probe complete',kind,'samples',len(x),'alpha',alpha,flush=True)
 csvout(folder/'position_metrics.csv',tables)


def pooled(bank,spec):
 layout=spec['layout'];z=bank[layout['start']:layout['start']+layout['count']].float().numpy()
 return z.mean(0),z.std(0)


def feature_pair(q,r,score):
 qm,qs=q;rm,rs=r
 return np.r_[np.abs(qm-rm),qm*rm,np.abs(qs-rs),qs*rs,score].astype(np.float32)


def words(text):return set(re.findall(r'[^\W_]+',text.lower(),flags=re.UNICODE))

def text_pair(q,r,bm):
 a,b=words(q['text']),words(r['text']);na={x for x in a if any(c.isdigit() for c in x)};nb={x for x in b if any(c.isdigit() for c in x)}
 return np.array([len(a&b)/max(1,len(a|b)),len(a&b)/max(1,len(a)),len(a&b)/max(1,len(b)),len(na&nb)/max(1,len(na|nb)),len(a),len(b),bm],np.float64)


def coordinate_pair(qspec,rspec,box):
 qxy=grid(qspec);rxy=grid(rspec);qxy=qxy[np.isfinite(qxy).all(1)];rxy=rxy[np.isfinite(rxy).all(1)]
 return np.r_[qxy.mean(0),qxy.std(0),rxy.mean(0),rxy.std(0),np.asarray(box['bbox'])/np.tile([box['width'],box['height']],2)]


def run_identity(d,config,folder):
 ledger={r['photo_id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/candidate_ledger_all1117.json')}
 ri={id:i for i,id in enumerate(d['rb']['ids'])};reference_pools=[pooled(z,spec) for z,spec in zip(d['rb']['embeddings'],d['metadata']['references'])]
 positive_collections={s:sorted({ri[rid] for qid,row in ledger.items() if d['splits'][qid]==s for rid in row['positive_reference_ids']}) for s in ['train','val','test']}
 xs=[];texts=[];coordinates=[];labels=[];packages=[];splits=[];sample_index=[]
 for qi,qid in enumerate(d['qb']['ids']):
  row=ledger[qid];positive=ri[row['positive_reference_ids'][0]];scope=d['splits'][qid]
  eligible=[index for index in positive_collections[scope] if d['rb']['sha256'][index] not in row['positive_reference_sha256']]
  negative=max(eligible,key=lambda index:(float(d['image']['scores'][qi,index]),-index))
  qpool=pooled(d['qb']['embeddings'][qi],d['metadata']['photos'][qi])
  for label,rindex in [(1,positive),(0,negative)]:
   rid=d['rb']['ids'][rindex];score=float(d['image']['scores'][qi,rindex])
   xs.append(feature_pair(qpool,reference_pools[rindex],score));texts.append(text_pair(d['qo'][qid],d['ro'][rid],float(d['bm25']['scores'][qi,rindex])))
   coordinates.append(coordinate_pair(d['metadata']['photos'][qi],d['metadata']['references'][rindex],d['boxes'][qid]))
   labels.append(label);packages.append(row['package_id']);splits.append(scope)
   sample_index.append(dict(photo_id=qid,reference_id=rid,package_id=row['package_id'],split=scope,label=label,text_jaccard=texts[-1][0]))
 x=np.array(xs);text=np.array(texts);coord=np.array(coordinates);y=np.array(labels);packages=np.array(packages);splits=np.array(splits)
 train=splits=='train';val=splits=='val';test=splits=='test';weights=np.array([1/Counter(packages)[p] for p in packages]);results=[];models={}
 for kind,features in [('feature_only',x),('text_only',text),('coordinate_only',coord)]:
  choices=[]
  for C in config['probes']['logistic_C_grid']:
   model=logistic_fit(features[train],y[train],C,weights[train]);m=classification_metrics(y[val],logistic_predict(model,features[val]),packages[val]);choices.append((m['package_macro_accuracy'],C,model))
  _,C,model=max(choices,key=lambda row:(row[0],-row[1]));models[kind]=(model,C)
  write(folder/(kind+'_identity_selection.json'),dict(C=C,validation_only=True,training_reference_scope='training positive designs only',train_packages=len(set(packages[train])),test_packages=len(set(packages[test]))))
  save_head(folder/(kind+'_identity_head.pt'),model,dict(kind=kind,C=C))
  for split,mask in [('val',val),('test',test)]:results.append(dict(probe='unseen_identity_pairs',control=kind,split=split,**classification_metrics(y[mask],logistic_predict(model,features[mask]),packages[mask])))
  if kind=='feature_only':
   for seed in config['probes']['shuffle_seeds']:
    null=logistic_fit(features[train],shuffled_labels(y[train],packages[train],seed),C,weights[train])
    results.append(dict(probe='unseen_identity_pairs',control='label_shuffle',seed=seed,split='test',**classification_metrics(y[test],logistic_predict(null,features[test]),packages[test])))
  print('Identity probe complete',kind,'C',C,flush=True)
 # Natural error/control pairs are a harder evaluation of the same frozen selected head.
 target_qids={r['photo_id'] for r in read(OUT/'stage1_candidate_ledger.json')};challenge=[]
 for qi,qid in enumerate(d['qb']['ids']):
  if qid not in target_qids:continue
  row=ledger[qid];qpool=pooled(d['qb']['embeddings'][qi],d['metadata']['photos'][qi])
  positive=ri[row['positive_reference_ids'][0]]
  for rid in list(dict.fromkeys([row['image_strongest_wrong'],row['fusion_strongest_wrong']])):
   negative=ri[rid];pf=feature_pair(qpool,reference_pools[positive],float(d['image']['scores'][qi,positive]));nf=feature_pair(qpool,reference_pools[negative],float(d['image']['scores'][qi,negative]))
   model,_=models['feature_only'];gap=float(logistic_predict(model,np.stack([pf,nf]))[0]-logistic_predict(model,np.stack([pf,nf]))[1])
   common=len(words(d['ro'][row['positive_reference_ids'][0]]['text'])&words(d['ro'][rid]['text']))/max(1,len(words(d['ro'][row['positive_reference_ids'][0]]['text'])|words(d['ro'][rid]['text'])))
   challenge.append(dict(photo_id=qid,package_id=row['package_id'],split=d['splits'][qid],positive=row['positive_reference_ids'][0],negative=rid,
    raw_gap=float(d['image']['scores'][qi,positive]-d['image']['scores'][qi,negative]),feature_probe_gap=gap,reference_text_jaccard=common,
    shared_text_challenge=common>=.75,same_text_layout_ground_truth='not asserted without face/content audit'))
 csvout(folder/'identity_metrics.csv',results);csvout(folder/'identity_sample_index.csv',sample_index);csvout(folder/'natural_hard_negative_probe.csv',challenge)


def main():
 config=preregister();d=data();torch.set_num_threads(2);folder=OUT/'stage3_probes';folder.mkdir(parents=True,exist_ok=True)
 run_positions(d,config,folder);run_identity(d,config,folder)
 write(folder/'verification.json',dict(status='passed',encoder_frozen=True,head_inputs_exclude_coordinates_indices_and_IDs=True,
  package_identity_group_disjoint=True,parameters_selected_on_validation_only=True,shuffle_seeds=config['probes']['shuffle_seeds'],
  absolute_position_not_treated_as_layout_identity_proof=True,controlled_layout_probe='computed after controlled encodings; separate artifact',
  probe_failure_not_proof_of_absence=True,relative_display_position_labels_use_external_EXIF_metadata=True))
 print('Natural frozen-feature probes complete',flush=True)

if __name__=='__main__':main()
