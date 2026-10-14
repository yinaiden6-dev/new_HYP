"""Controlled layout readouts, paired metrics and no-layout edit-artifact checks."""
import sys
from pathlib import Path
from collections import Counter
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,data,read,write,rows,csvout,sha
from scripts.colnomic_mechanism.probes import logistic_fit,logistic_predict,classification_metrics,paired_metrics,validation_threshold,shuffled_labels,save_head
from scripts.colnomic_mechanism.analyze_controls import sim,roi_features


def vector_features(z,spec):
 l=spec['layout'];v=z[l['start']:l['start']+l['count']].float().numpy()
 return np.r_[v.mean(0),v.std(0)]


def main():
 torch.set_num_threads(2);d=data();folder=OUT/'stage4_extended_photos_v3';output=folder/'analysis';output.mkdir(exist_ok=True)
 completion=read(folder/'encoding_complete.json');sources=read(folder/'sources.json');manifest=read(folder/'encoding_manifest.json')
 specifications={r['id']:r for r in d['metadata']['photos']};qindex={qid:i for i,qid in enumerate(d['qb']['ids'])}
 features=[];roi=[];scalars=[];labels=[];packages=[];splits=[];query_ids=[];sample_index=[];editor_data={};sham_drift=[]
 for source in sources:
  qid=source['photo_id'];spec=specifications[qid];base=d['qb']['embeddings'][qindex[qid]]
  records={arm:torch.load(folder/'embeddings'/(qid+'_'+arm+'.pt'),weights_only=True,map_location='cpu') for arm in ['fresh_base','sham','shift_1','shift_2','editor_1','editor_2']}
  assert torch.equal(records['fresh_base']['embedding'],base),'Recent original cache must replay exactly before causal tests: '+qid
  sham=records['sham']['embedding'];base_sham=sim(base,sham)[0];base_self=sim(base,base)[0]
  sham_drift.append(dict(photo_id=qid,package_id=source['package_id'],split=source['split'],base_self_score=base_self,base_sham_score=base_sham,sham_loss=base_self-base_sham))
  editor_data[qid]=[]
  for k in [1,2]:
   value=records['shift_'+str(k)];z=value['embedding'];wf,n=roi_features(z,spec,value['word_bbox'])
   features.append(vector_features(z,spec));roi.append(wf);scalars.append([sim(base,z)[0]]);labels.append(k-1);packages.append(source['package_id']);splits.append(source['split']);query_ids.append(qid)
   sample_index.append(dict(photo_id=qid,package_id=source['package_id'],group_id=source['group_id'],split=source['split'],arm=value['arm'],label=k-1,word=source['word'],ROI_token_count=n))
   control=records['editor_'+str(k)];control_roi,control_n=roi_features(control['embedding'],spec,control['word_bbox'])
   editor_data[qid].append(dict(moment=vector_features(control['embedding'],spec),roi=control_roi,scalar=[sim(base,control['embedding'])[0]],control_n=control_n,source=source,arm=control['arm']))
 x=np.array(features);w=np.array(roi);s=np.array(scalars);y=np.array(labels);packages=np.array(packages);split=np.array(splits);qids=np.array(query_ids)
 train=split=='train';val=split=='val';test=split=='test';weights=np.array([1/Counter(packages)[p] for p in packages])
 for field in ['package_id','group_id','image_sha256','source_sha256']:
  scopes={part:{r[field] for r in sources if r['split']==part} for part in ['train','val','test']}
  assert all(not scopes[a]&scopes[b] for a,b in [('train','val'),('train','test'),('val','test')]),field
 counts={part:len(set(packages[split==part])) for part in ['train','val','test']};assert counts['test']>=10
 candidates={'maxsim_only':s,'pooled_only':x,'pooled_plus_maxsim':np.column_stack([x,s]),'word_ROI_assisted':w}
 metrics=[];predictions=[];gaps=[];package_tables=[];artifact=[]
 for kind,values in candidates.items():
  choices=[]
  for C in [.01,.1,1.,10.]:
   model=logistic_fit(values[train],y[train],C,weights[train]);logits=logistic_predict(model,values[val]);pm,_=paired_metrics(y[val],logits,qids[val],packages[val]);auc=classification_metrics(y[val],logits,packages[val])['auc'];choices.append((pm['package_macro_pair_accuracy'],auc,C,model))
  _,_,C,model=max(choices,key=lambda r:(r[0],r[1],-r[2]));threshold=validation_threshold(y[val],logistic_predict(model,values[val]),packages[val]);logits=logistic_predict(model,values)
  save_head(output/(kind+'_head.pt'),model,dict(C=C,threshold=threshold,kind=kind))
  write(output/(kind+'_selection.json'),dict(C=C,threshold=threshold,selection='validation package-macro within-source pair ordering, then AUC, then smaller C; validation-only classification threshold',
   inputs='1 raw cached-source/edited-source MaxSim scalar' if kind=='maxsim_only' else '256 visual mean/std features' if kind=='pooled_only' else '256 moments plus one MaxSim scalar' if kind=='pooled_plus_maxsim' else '256 word-ROI mean/std; ROI selection explicitly uses spatial supervision',
   no_token_indices_coordinates_or_IDs_in_linear_head=True,ROI_selection_has_external_coordinates=kind=='word_ROI_assisted',token_summary_probe_not_exhaustive=True))
  predictions.extend(dict(probe=kind,**row,logit=float(logit),zero_threshold_prediction=int(logit>=0),validation_threshold_prediction=int(logit>=threshold),validation_threshold=threshold) for row,logit in zip(sample_index,logits))
  for part,mask in [('val',val),('test',test)]:
   pm,pr=paired_metrics(y[mask],logits[mask],qids[mask],packages[mask]);gaps.extend(dict(probe=kind,split=part,**r) for r in pr)
   for mode,t in [('fixed_zero',0.),('validation_calibrated',threshold)]:metrics.append({'probe':kind,'split':part,'threshold_strategy':mode,**classification_metrics(y[mask],logits[mask],packages[mask],t),**pm})
   for package in sorted(set(packages[mask])):
    keep=mask&(packages==package);package_pm,_=paired_metrics(y[keep],logits[keep],qids[keep],packages[keep]);package_tables.append({'probe':kind,'split':part,'package_id':package,**classification_metrics(y[keep],logits[keep],packages[keep],threshold),**package_pm})
  for seed in [17,31,53]:
   null=logistic_fit(values[train],shuffled_labels(y[train],packages[train],seed),C,weights[train]);null_threshold=validation_threshold(y[val],logistic_predict(null,values[val]),packages[val]);null_logits=logistic_predict(null,values[test]);pm,_=paired_metrics(y[test],null_logits,qids[test],packages[test])
   metrics.append({'probe':kind,'split':'test','threshold_strategy':'label_shuffle_validation_calibrated','seed':seed,**classification_metrics(y[test],null_logits,packages[test],null_threshold),**pm})
  # The words do not move in these controls. This is an artifact response, not layout accuracy.
  for qid,controls in editor_data.items():
   if controls[0]['source']['split'] not in ['val','test']:continue
   vectors=np.array([r['scalar'] if kind=='maxsim_only' else r['moment'] if kind=='pooled_only' else np.r_[r['moment'],r['scalar']] if kind=='pooled_plus_maxsim' else r['roi'] for r in controls])
   scores=logistic_predict(model,vectors);layout_rows=np.flatnonzero(qids==qid);layout_gap=float(logits[layout_rows[1]]-logits[layout_rows[0]])
   artifact.append(dict(probe=kind,photo_id=qid,package_id=controls[0]['source']['package_id'],split=controls[0]['source']['split'],
    editor1_score=float(scores[0]),editor2_score=float(scores[1]),editor_gap=float(scores[1]-scores[0]),layout_gap=layout_gap,
    word_position_unchanged=True,interpretation='response to editing-only position; not class labels or retrieval accuracy'))
 csvout(output/'layout_metrics.csv',metrics);csvout(output/'per_sample_predictions.csv',predictions);csvout(output/'within_source_pair_gaps.csv',gaps)
 csvout(output/'by_package.csv',package_tables);csvout(output/'editing_artifact_responses.csv',artifact);csvout(output/'sham_drift.csv',sham_drift)
 np.savez_compressed(output/'probe_inputs.npz',moments=x,maxsim=s,word_ROI=w,labels=y,photo_ids=qids,package_ids=packages,splits=split)
 write(output/'verification.json',dict(status='passed',source_split_counts=counts,source_identity_group_and_hash_intersections_asserted_empty=True,
  original_source_cache_replay_exact=completion['all_original_query_cache_replays_exact'],original_embeddings_used_for_source_features=True,
  frozen_encoder=True,classification_threshold_and_regularization_selected_validation_only=True,
  metric_scope='within-source synthetic word-position classes, not natural instance retrieval',
  pure_feature_readouts=['maxsim_only','pooled_only','pooled_plus_maxsim'],spatially_assisted_readouts=['word_ROI_assisted'],
  reading_rows_not_a_target=True,metadata_based_source_selection_not_test_prediction_selection=True,
  no_layout_edit_controls_scored=True,pooling_not_a_complete_information_absence_test=True))
 print('Controlled original-photo readouts complete',counts,flush=True)

if __name__=='__main__':main()
