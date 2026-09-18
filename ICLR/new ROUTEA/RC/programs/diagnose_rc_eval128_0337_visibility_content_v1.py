#!/usr/bin/env python3
"""Post-hoc decomposition of two references in the sole opened EVAL128 regression; no fitting."""
from pathlib import Path
import json,hashlib
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_eval128_0337_visibility_content_v1'
QID='E128-2c09646865d5f430ec81f977'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def need(x,m):
 if not bool(x):raise RuntimeError(m)
def hten(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def token_sha(t):
 t=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode('ascii'));h.update(json.dumps(list(t.shape),separators=(',',':')).encode('ascii'));h.update(t.view(torch.uint8).numpy().tobytes());return h.hexdigest()
def score(q,r,wq,wr):
 sim=F.normalize(q.double(),dim=1)@F.normalize(r.double(),dim=1).T
 local=(sim*wr[None]).max(1).values;m=torch.sqrt(wq.mean()*wr.mean())
 return m*(wq*local).sum()/wq.sum().clamp_min(1e-12),m,sim,local

def main():
 need(not OUT.exists(),'APPEND_ONLY_OUTPUT_EXISTS');torch.set_num_threads(8);torch.set_num_interop_threads(1)
 rp=ROOT/'results/rc_original7_eval128_full_evidence_v1/result.json';vp=rp.with_name('validation.json');ip=rp.with_name('independent_review.json')
 need(sha(rp)=='096c9575ffa05a3f9b813adbe5e65c879fbdee77507aa757de09420bef0be866','EXPECTED_OPENED_RESULT')
 result=read(rp);val=read(vp);review=read(ip)
 need(val['result']==review['result']==bind(rp) and review['source_validation']==bind(vp),'COMPLETED_RESULT_BINDINGS')
 need(review['status']=='ORIGINAL7_EVAL128_POST_RESULT_INDEPENDENT_REVIEW_PASS','REVIEW_PASS')
 a=next(a for a in result['actions']['REAL'] if a['query_id']==QID)
 need(a['base_correct'] and not a['final_correct'] and result['primary_break_query_ids']==[QID],'SOLE_REGRESSION')
 shard=a['execution_ordinal']//8;tp=ROOT/f'results/rc_original7_eval128_token_raw_v1/shard{shard:02d}/payload.pt';ep=ROOT/f'results/rc_original7_eval128_roma_v1/shard{shard:02d}/payload.pt'
 qualification=read(rp.with_name('input_qualification.json'))
 for key,p in [('raw_shards',tp),('roma_shards',ep)]:need(qualification[key][shard]['payload']==bind(p),'QUALIFIED_PAYLOAD:'+key)
 t=torch.load(tp,map_location='cpu',mmap=True,weights_only=True);maps=torch.load(ep,map_location='cpu',mmap=True,weights_only=True)
 q=next(r for r in t['records'] if r['query_id']==QID);e=next(r for r in maps['records'] if r['query_id']==QID)
 need(token_sha(q['query_tokens'])==q['query_tokens_sha256']==e['query_tokens_sha256'],'QUERY_TOKENS')
 need(sha(q['query_source_path'])==q['query_source_sha256']==e['query_source_sha256'],'QUERY_IMAGE');rows=[];arrays={}
 for role,physical in [('TARGET',a['target_physical_row_in_full_rank']),('SELECTED_WRONG',a['final_physical_row'])]:
  ref=t['references'][physical];c=next(c for c in e['candidates'] if c['physical_row']==physical)
  need(token_sha(ref['tokens'])==ref['tokens_sha256']==c['reference_tokens_sha256'],'REFERENCE_TOKENS');need(sha(ref['source_path'])==ref['source_image_sha256']==c['reference_image_sha256'],'REFERENCE_IMAGE')
  wq=c['query_visibility'];wr=c['reference_visibility'];need(hten(wq)==c['query_map_sha256'] and hten(wr)==c['reference_map_sha256'],'VISIBILITY_BYTES')
  s,m,sim,local=score(q['query_tokens'],ref['tokens'],wq,wr)
  qc=score(q['query_tokens'],ref['tokens'],wq.roll(max(1,len(wq)//2)),wr)[0];rc=score(q['query_tokens'],ref['tokens'],wq,wr.roll(max(1,len(wr)//2)))[0]
  for key,value in [('real_score',s),('visibility_mass',m),('query_control_score',qc),('reference_control_score',rc)]:need(float(value).hex()==float(c['old_scores'][key]).hex(),'ORIGINAL_C4_BITS:'+key)
  prob=wq/wq.sum();free=sim.max(1).values;chosen=(sim*wr[None]).argmax(1);cos=sim[torch.arange(len(wq)),chosen];vis=wr[chosen]
  ell=(wq*local).sum()/wq.sum();cm=(prob*cos).sum();vm=(prob*vis).sum();cov=(prob*(cos-cm)*(vis-vm)).sum();need(abs(float(ell-vm*cm-cov))<1e-14,'COVARIANCE_IDENTITY')
  image_dot=q['query_tokens'].float()@ref['tokens'].float().T
  rows.append({'role':role,'physical_row':physical,'image':bind(ref['source_path']),'grid_shape':ref['grid_shape'],'raw_full_gallery_rank':q['raw_ranked_physical_rows'].index(physical)+1,
   'original_RAW':c['old_scores']['raw_score'],'S':float(s),'M':float(m),'L_readout':float(s)/max(float(m),1e-12),'L_direct':float(ell),
   'query_visibility_mean':float(wq.mean()),'reference_visibility_mean':float(wr.mean()),'reference_visibility_max':float(wr.max()),
   'unweighted_image_cosine_MaxSim_mean':float(free.mean()),'query_weighted_free_reference_cosine':float((prob*free).sum()),
   'uniform_query_real_reference_weighted_MaxSim_mean':float(local.mean()),'cosine_at_original_weighted_argmax_query_weighted_mean':float(cm),
   'visibility_at_original_weighted_argmax_query_weighted_mean':float(vm),'selected_cosine_visibility_covariance':float(cov),
   'L_over_mean_reference_visibility':float(ell/wr.mean()),'L_over_max_reference_visibility':float(ell/wr.max()),'image_only_dot_MaxSim_sum_FP32':float(image_dot.max(1).values.sum()),
   'query_geometry':e['query_geometry'],'reference_geometry':c['reference_geometry'],'query_map_sha256':c['query_map_sha256'],'reference_map_sha256':c['reference_map_sha256']})
  arrays[role]={'wq':wq.tolist(),'wr':wr.tolist(),'query_mass_probability':prob.tolist(),'local_weighted_maxsim':local.tolist(),'free_maxsim':free.tolist(),'argmax_reference_cell':chosen.tolist(),'cosine_at_argmax':cos.tolist(),'visibility_at_argmax':vis.tolist(),'L_atom_contribution':(prob*local).tolist()}
 keys=['original_RAW','S','M','L_readout','query_visibility_mean','reference_visibility_mean','reference_visibility_max','unweighted_image_cosine_MaxSim_mean','query_weighted_free_reference_cosine','cosine_at_original_weighted_argmax_query_weighted_mean','visibility_at_original_weighted_argmax_query_weighted_mean','L_over_mean_reference_visibility','L_over_max_reference_visibility']
 target,wrong=rows;comparison={k:{'target':target[k],'wrong':wrong[k],'wrong_over_target':wrong[k]/target[k] if target[k] else None,'target_better':target[k]>wrong[k]} for k in keys}
 value={'status':'OPENED_EVAL128_0337_TWO_REFERENCE_FACTOR_DECOMPOSITION_COMPLETE','evidence_level':'post-hoc two-reference explanation of one observed regression; not a new model or accuracy',
  'program':bind(__file__),'sources':{'result':bind(rp),'validation':bind(vp),'independent_review':bind(ip),'token_payload':bind(tp),'roma_payload':bind(ep)},
  'query_id':QID,'original_query_id':a['original_query_id'],'query_image':bind(q['query_source_path']),'query_grid_shape':q['query_grid_shape'],'original_action':a,'candidates':rows,'comparisons':comparison,'arrays':arrays,
  'fitted_parameters':0,'new_RoMa_or_encoder_calls':0,'new_C128_predictions':0,
  'factorization':'At original wr-weighted argmax j*(i), L=E_p[wr(j*) cos(i,j*)]=E_p[wr(j*)]E_p[cos(i,j*)]+Cov_p; p=wq/sum(wq).',
  'limitations':['Case selected from already opened sole EVAL regression, not prospectively from TRAIN.','Uniform and rescaled statistics cover only the two chosen references, not a full127 HOLD/SWITCH intervention.','Reference mean/max rescalings are descriptive, with no score replacement or deployment.','L still includes visibility amplitude and selected content; no pixel-level identity causality claimed.']}
 OUT.mkdir();p=OUT/'result.json';p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');p.chmod(0o444)
 print(json.dumps({'result':bind(p),'comparisons':comparison},indent=2))
if __name__=='__main__':main()
