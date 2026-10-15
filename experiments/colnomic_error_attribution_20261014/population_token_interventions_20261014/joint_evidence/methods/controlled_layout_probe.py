"""Analyze controlled same-text layouts and isolate sham/runtime/MaxSim effects."""
import argparse,sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,data,read,write,csvout
from scripts.colnomic_mechanism.probes import logistic_fit,logistic_predict,classification_metrics,shuffled_labels,save_head


def sim(q,g):
 ri=torch.nonzero(g.float().norm(dim=1)>1e-8,as_tuple=True)[0]
 q=q[q.float().norm(dim=1)>1e-8].float();g=g[ri].float()
 matrix=q@g.T;values,index=matrix.max(1)
 return float(values.mean()),values.numpy(),ri[index].numpy()


def roi_features(z,spec,bbox):
 l=spec['layout'];ids=np.arange(l['count']);cols=ids%l['width'];rows=ids//l['width']
 lo=np.column_stack([cols/l['width']*spec['raw_width'],rows/l['height']*spec['raw_height']])
 hi=lo+[spec['raw_width']/l['width'],spec['raw_height']/l['height']]
 overlap=np.maximum(np.minimum(hi,bbox[2:])-np.maximum(lo,bbox[:2]),0).prod(1)
 selected=ids[overlap/(spec['raw_width']*spec['raw_height']/l['count'])>=.01]+l['start']
 if not len(selected):raise ValueError('No word-overlap visual tokens')
 value=z[selected].float().numpy()
 return np.r_[value.mean(0),value.std(0)],len(selected)


def main():
 p=argparse.ArgumentParser();p.add_argument('--embedding-dir',type=Path,default=OUT/'stage4_interventions/embeddings');p.add_argument('--output',type=Path,default=OUT/'stage4_interventions/analysis');args=p.parse_args()
 torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True);d=data()
 manifest=read(OUT/'stage4_interventions/encoding_manifest.json');sources=read(OUT/'stage4_interventions/sources.json');meta={r['id']:r for r in d['metadata']['references']}
 records={}
 for row in manifest:
  path=args.embedding_dir/(row['id']+'.pt')
  if not path.exists():raise FileNotFoundError(path)
  value=torch.load(path,map_location='cpu',weights_only=True);assert torch.isfinite(value['embedding']).all()
  records[row['id']]=value
 parity=[]
 for id,value in records.items():
  if value['arm']!='base':continue
  bank=d['qb'] if value['source_id'].startswith('photo_') else d['rb'];index=bank['ids'].index(value['source_id']);old=bank['embeddings'][index].float();fresh=value['embedding'].float()
  assert old.shape==fresh.shape
  delta=(old-fresh).abs();parity.append(dict(source_id=value['source_id'],max_absolute_error=float(delta.max()),mean_absolute_error=float(delta.mean()),mean_cosine=float(torch.nn.functional.cosine_similarity(old,fresh).mean()),exact=torch.equal(old,fresh)))
 write(args.output/'encoder_parity.json',parity)
 maxima=[];features=[];wordfeatures=[];labels=[];identities=[];splits=[];indices=[]
 for source in sources:
  rid=source['reference_id'];spec=meta[rid];l=spec['layout'];visual=slice(l['start'],l['start']+l['count'])
  base=records[rid+'_fresh_base']['embedding'];sham=records[rid+'_sham']['embedding'];sham_visual=sham[visual].float()
  self_score,_,_=sim(sham_visual,sham_visual);self_diag=float((sham_visual*sham_visual).sum(1).mean())
  for k in [1,2,3,4]:
   row=records[rid+'_shift_'+str(k)];edited=row['embedding'];edited_visual=edited[visual].float();score,_,_=sim(sham_visual,edited_visual)
   diagonal=float((sham_visual*edited_visual).sum(1).mean())
   qcos=torch.nn.functional.normalize(sham_visual,dim=1);gcos=torch.nn.functional.normalize(edited_visual,dim=1)
   c_score,c_values,c_matches=sim(qcos,gcos);c_diagonal=float((qcos*gcos).sum(1).mean())
   identity_matches=np.arange(len(c_matches))==c_matches
   maxima.append(dict(reference_id=rid,package_id=source['package_id'],split=source['split'],word=source['text'],shift=k,
    raw_maxsim_loss=self_score-score,raw_diagonal_loss=self_diag-diagonal,
    cosine_maxsim_loss=1-c_score,cosine_diagonal_loss=1-c_diagonal,
    ratio_maxsim_to_diagonal=(1-c_score)/(1-c_diagonal) if 1-c_diagonal>1e-8 else None,
    same_index_match_fraction=float(identity_matches.mean()),
    meaning='same canvas; diagonal uses external index alignment; not proof of intrinsic position coding'))
   if source['split'] in ['train','val','test']:
    z=edited_visual.numpy();features.append(np.r_[z.mean(0),z.std(0)])
    wf,n=roi_features(edited,spec,row['word_bbox']);wordfeatures.append(wf);labels.append(int(k>=3));identities.append(source['package_id']);splits.append(source['split'])
    indices.append(dict(reference_id=rid,package_id=source['package_id'],split=source['split'],arm=row['arm'],label=int(k>=3),word_token_count=n))
 csvout(args.output/'maxsim_vs_diagonal.csv',maxima)
 x=np.array(features);w=np.array(wordfeatures);y=np.array(labels);identity=np.array(identities);split=np.array(splits)
 train=split=='train';val=split=='val';test=split=='test';weights=np.array([1/Counter(identity)[i] for i in identity]);tables=[]
 assert not set(identity[train])&set(identity[val]) and not set(identity[train])&set(identity[test]) and not set(identity[val])&set(identity[test])
 for kind,values in [('all_visual_feature_set',x),('word_ROI_feature_set',w)]:
  choices=[]
  for C in [.01,.1,1.,10.]:
   model=logistic_fit(values[train],y[train],C,weights[train]);m=classification_metrics(y[val],logistic_predict(model,values[val]),identity[val]);choices.append((m['package_macro_accuracy'],C,model))
  _,C,model=max(choices,key=lambda row:(row[0],-row[1]))
  write(args.output/(kind+'_selection.json'),dict(C=C,validation_only=True,inputs='frozen token features only',
   train_source_identities=sorted(set(identity[train])),val_source_identities=sorted(set(identity[val])),test_source_identities=sorted(set(identity[test]))))
  save_head(args.output/(kind+'_head.pt'),model,dict(C=C,kind=kind))
  logits=logistic_predict(model,values[test]);tables.append(dict(probe=kind,control='feature_only',**classification_metrics(y[test],logits,identity[test])))
  for seed in [17,31,53]:
   null=logistic_fit(values[train],shuffled_labels(y[train],identity[train],seed),C,weights[train]);tables.append(dict(probe=kind,control='label_shuffle',seed=seed,**classification_metrics(y[test],logistic_predict(null,values[test]),identity[test])))
 tables.append(dict(probe='text_only',control='same_text_within_identity',accuracy=.5,package_macro_accuracy=.5,samples=int(test.sum()),packages=len(set(identity[test]))))
 csvout(args.output/'layout_probe_metrics.csv',tables);csvout(args.output/'layout_probe_sample_index.csv',indices)
 # Causal focal comparison: fixed original candidate identities, source aliases treated together.
 qid='photo_41de3eb383934ec3';qi=d['qb']['ids'].index(qid);pi=d['rb']['ids'].index('reference_000229');ni=d['rb']['ids'].index('reference_000233')
 fresh_q=records[qid+'_fresh_base']['embedding'];pbase=records['reference_000229_fresh_base']['embedding'];nbase=records['reference_000233_fresh_base']['embedding']
 fixed_fresh_gap=sim(fresh_q,pbase)[0]-sim(fresh_q,nbase)[0];focal=[]
 for query_name,q in [('cached_query',d['qb']['embeddings'][qi]),('fresh_query',fresh_q)]:
  ps,positive_contribution,positive_matches=sim(q,pbase);ns=sim(q,nbase)[0]
  qi_valid=torch.nonzero(q.float().norm(dim=1)>1e-8,as_tuple=True)[0]
  pi_valid=torch.nonzero(pbase.float().norm(dim=1)>1e-8,as_tuple=True)[0]
  matrix_dir=args.output/'similarity_matrices';matrix_dir.mkdir(exist_ok=True)
  np.savez_compressed(matrix_dir/(query_name+'_positive_base.npz'),matrix=(q[qi_valid].float()@pbase[pi_valid].float().T).numpy(),
   query_raw_indices=qi_valid.numpy(),reference_raw_indices=pi_valid.numpy(),score=np.array(ps))
  for arm in ['base','sham','shift_1','shift_2','shift_3','shift_4']:
   g=nbase if arm=='base' else records['reference_000233_'+arm]['embedding']
   value,contribution,match=sim(q,g);sham_score=sim(q,records['reference_000233_sham']['embedding'])[0]
   scores=d['image']['scores'][qi].numpy().copy()
   positive_alias=[i for i,h in enumerate(d['rb']['sha256']) if h==d['rb']['sha256'][pi]];negative_alias=[i for i,h in enumerate(d['rb']['sha256']) if h==d['rb']['sha256'][ni]]
   scores[positive_alias]=ps;scores[negative_alias]=value
   seen=set();rank=None
   for index in np.argsort(-scores,kind='stable'):
    h=d['rb']['sha256'][index]
    if h in seen:continue
    seen.add(h)
    if h==d['rb']['sha256'][pi]:rank=len(seen);break
   focal.append(dict(query=query_name,negative_arm=arm,positive_score=ps,negative_score=value,gap=ps-value,
    negative_score_edit_minus_sham=value-sham_score,full_gallery_conditional_positive_rank=rank,
    primary_fresh_pair_gap=fixed_fresh_gap,
    original_cached_gap=float(d['image']['scores'][qi,pi]-d['image']['scores'][qi,ni]),
    scope='conditional replacement of positive and negative design aliases; other gallery features fixed; not new-label retrieval accuracy'))
   qi_valid=torch.nonzero(q.float().norm(dim=1)>1e-8,as_tuple=True)[0];ri_valid=torch.nonzero(g.float().norm(dim=1)>1e-8,as_tuple=True)[0]
   assert abs(float((positive_contribution-contribution).mean())-(ps-value))<3e-7
   csvout(args.output/(query_name+'_'+arm+'_contributions.csv'),[dict(query_token_index=int(i),maximum_similarity=float(c),reference_token_index=int(j),
    positive_maximum_similarity=float(pc),positive_reference_token_index=int(pj),delta=float(pc-c))
    for i,c,j,pc,pj in zip(qi_valid,contribution,match,positive_contribution,positive_matches)])
   matrix=q[qi_valid].float()@g[ri_valid].float().T
   matrix_dir=args.output/'similarity_matrices';matrix_dir.mkdir(exist_ok=True)
   np.savez_compressed(matrix_dir/(query_name+'_'+arm+'.npz'),matrix=matrix.numpy(),
    query_raw_indices=qi_valid.numpy(),reference_raw_indices=ri_valid.numpy(),
    score=np.array(value),negative_score_edit_minus_sham=np.array(value-sham_score))
 csvout(args.output/'focal_causal_scores.csv',focal)
 faithful=max(row['max_absolute_error'] for row in parity)<1e-5
 write(args.output/'verification.json',dict(status='executed',controlled_pixel_tests_passed=True,encoder_weights_frozen=True,
  original_cache_reproduction_passed=faithful,max_original_cache_replay_error=max(row['max_absolute_error'] for row in parity),
  layout_probe_train_identities=len(set(identity[train])),layout_probe_val_identities=len(set(identity[val])),layout_probe_test_identities=len(set(identity[test])),
  layout_probe_feature_inputs_exclude_coordinates_and_indices=True,word_ROI_selection_uses_supervised_spatial_annotation=True,
  layout_probe_pilot_only=len(set(identity[test]))<5,
  original_cache_causal_attribution='permitted after faithful replay' if faithful else 'not justified: current runtime differs from frozen caches; within-runtime controlled responses remain valid',
  conditional_gallery_rank_not_accuracy=True,
  maxsim_diagonal_experiment_uses_explicit_index_alignment_not_feature_only=True))
 print('Controlled analysis complete, faithful replay',faithful,'test layout identities',len(set(identity[test])),flush=True)

if __name__=='__main__':main()
