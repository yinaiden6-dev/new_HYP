"""Audit the delivered pilot using its saved heads, not a new fitted model."""
import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,rows,csvout
from scripts.colnomic_mechanism.probes import logistic_predict,classification_metrics,paired_metrics,validation_threshold
from scripts.colnomic_mechanism.analyze_controls import roi_features


def main():
 v1=ROOT/'results/token_mechanism_experiments/20261013_v1';out=OUT/'v1_controlled_probe_audit';out.mkdir(parents=True,exist_ok=True)
 samples=rows(v1/'stage4_interventions/analysis_validated_runtime/layout_probe_sample_index.csv')
 metadata={r['id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['references']};metrics=[]
 for kind in ['all_visual_feature_set','word_ROI_feature_set']:
  stored=torch.load(v1/'stage4_interventions/analysis_validated_runtime'/(kind+'_head.pt'),weights_only=True,map_location='cpu')
  model={k:stored[k].numpy() for k in ['mean','scale','coefficient']};features=[]
  for r in samples:
   rid=r['reference_id'];spec=metadata[rid];value=torch.load(v1/'stage4_interventions/validated_embeddings'/(rid+'_'+r['arm']+'.pt'),weights_only=True,map_location='cpu');z=value['embedding'];l=spec['layout']
   if kind=='all_visual_feature_set':
    v=z[l['start']:l['start']+l['count']].float().numpy();f=np.r_[v.mean(0),v.std(0)]
   else:f=roi_features(z,spec,value['word_bbox'])[0]
   features.append(f)
  logits=logistic_predict(model,np.array(features));y=np.array([int(r['label']) for r in samples]);packages=np.array([r['package_id'] for r in samples]);qids=np.array([r['reference_id'] for r in samples]);split=np.array([r['split'] for r in samples]);val=split=='val';test=split=='test'
  threshold=validation_threshold(y[val],logits[val],packages[val]);m,pairs=paired_metrics(y[test],logits[test],qids[test],packages[test])
  for strategy,t in [('original_zero',0.),('validation_calibrated',threshold)]:
   metrics.append({'head':kind,'threshold_strategy':strategy,**classification_metrics(y[test],logits[test],packages[test],t),**m})
  csvout(out/(kind+'_per_sample.csv'),[dict(**r,logit=float(logit),zero_threshold_prediction=int(logit>=0),validation_calibrated_prediction=int(logit>=threshold),validation_threshold=threshold) for r,logit in zip(samples,logits)])
  csvout(out/(kind+'_within_source_pairs.csv'),pairs)
 write(out/'metrics.json',metrics);write(out/'scope.json',dict(heads_refitted=False,threshold_calibrated_validation_only=True,
  paired_metric='higher-shift class vs lower-shift class within the same source image, all 2x2 comparisons; not instance retrieval positives',
  source_script='scripts/colnomic_mechanism/analyze_controls.py',v1_encoding_backend='PIL; not faithful to recent photo cache'))
 print(metrics,flush=True)

if __name__=='__main__':main()
