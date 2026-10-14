"""Fixed-parameter alignment gate ablation and geometry-conditioned candidate audits."""
import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,data,read,write,csvout
from scripts.colnomic_mechanism.geometry import rank_rows,metrics
from evaluation.scoring import fuse


def main():
 d=data();torch.set_num_threads(2);folder=OUT/'stage2_geometry';cache=np.load(folder/'full_gallery_scores.npz');fits=read(folder/'alignments.json');selection=read(folder/'selection.json')['selections']
 qindex={id:i for i,id in enumerate(d['qb']['ids'])};rindex={id:i for i,id in enumerate(d['rb']['ids'])};available=np.zeros_like(cache['original'])
 for fit in fits:
  qi=qindex[fit['photo_id']];ri=rindex[fit['reference_id']];valid=int((d['qb']['embeddings'][qi].float().norm(dim=1)>1e-8).sum());available[qi,ri]=fit['subject_tokens']/valid
 tables=[];predictions={};has_text=[bool(d['qo'][id]['text'].strip()) for id in d['qb']['ids']]
 for config in ['B03','F03']:
  baseline=d['image']['scores'].numpy() if config=='B03' else d['fusion']['scores'].numpy();old={r['photo_id']:r for r in rank_rows(baseline,d)}
  raw=cache['original']+selection[config]['lambda_value']*available
  scores=raw if config=='B03' else fuse(torch.from_numpy(raw),d['bm25']['scores'],has_text,.5).numpy();pred=rank_rows(scores,d);predictions[config]=pred
  for split in ['val','test','all']:
   subset=[r for r in pred if split=='all' or r['split']==split];m=metrics(subset)
   m.update(rescued=sum(r['correct'] and not old[r['photo_id']]['correct'] for r in subset),harmed=sum(not r['correct'] and old[r['photo_id']]['correct'] for r in subset))
   tables.append(dict(configuration=config,method='alignment_availability_bonus_without_token_geometry',split=split,**m))
 csvout(folder/'alignment_gate_control.csv',tables);write(folder/'alignment_gate_predictions.json',predictions)
 write(folder/'alignment_gate_control_protocol.json',dict(status='posthoc_diagnostic',purpose='Separate independent OCR alignment availability from spatially constrained token correspondence',
  declared_after_primary_test_results_inspected=True,parameters_reused_from_original_validation_selection=True,new_test_parameter_selection=False,
  matching_features_changed=False,alignment_rules_changed=False,not_a_new_validation_selected_primary_method=True))
 # Paired source-group bootstrap for the already-selected primary method.
 primary=read(folder/'predictions.json');bootstrap=[]
 for config in ['B03','F03']:
  old={r['photo_id']:r for r in primary[config+'_original']};new=primary[config+'_geo_matching']
  test=[r for r in new if r['split']=='test'];groups=sorted({d['groups'][r['package_id']] for r in test})
  differences=np.array([np.mean([int(r['correct'])-int(old[r['photo_id']]['correct']) for r in test if d['groups'][r['package_id']]==group]) for group in groups])
  rng=np.random.default_rng(20261013);resamples=differences[rng.integers(0,len(groups),(5000,len(groups)))].mean(1)
  bootstrap.append(dict(configuration=config,unit='held-out source group',groups=len(groups),macro_r1_difference=float(differences.mean()),ci95=np.quantile(resamples,[.025,.975]).tolist(),seed=20261013))
 write(folder/'paired_group_bootstrap.json',bootstrap)
 # Audit foreground ambiguity exclusions separately; identities are never rewritten.
 ambiguous=read(ROOT/'results/token_mechanism_analysis/F03/summary.json')['targets'];ambiguous_ids={r['photo_id'] for r in ambiguous if r['identity_ambiguity']}
 audit=[]
 for config in ['B03','F03']:
  for method in ['original','geo_matching']:
   remaining=[r for r in primary[config+'_'+method] if r['photo_id'] not in ambiguous_ids]
   audit.append(dict(configuration=config,method=method,scope='fixed removal of 19 previously flagged F03 ambiguous query cases (sensitivity only)',excluded_queries=len(ambiguous_ids),**metrics(remaining)))
 csvout(folder/'identity_ambiguity_sensitivity.csv',audit)
 print('Gate diagnostic, bootstrap and ambiguity sensitivity complete',flush=True)

if __name__=='__main__':main()
