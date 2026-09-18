#!/usr/bin/env python3
"""Post-hoc additive accounting of the two already frozen heads. No fitting."""
import csv,hashlib,json,math
from fractions import Fraction
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_free8_component_accounting_v1'
RUN=ROOT/'results/rc_same_support_specificity_v1'
INPUT=ROOT/'results/rc_same_support_specificity_inputs_v1/result.json'
PINS={INPUT:'13ec8fa9f79282f2cd94e9b451660c99b432c0989ead279cf96c4bfdf3f94bc2',RUN/'result.json':'174195a1d5c5bb99602ea822d6702b2317740a0a7f265b2d19227059803207d1',RUN/'parameters.json':'9a13f78f33470754fbb74263fd6de9313c2a7bd36aabb31ab7cec6698dcf7344',RUN/'independent_validation.json':'6cabaebb90e5453d607d33faf7c6c11c1f02278a63b57a64f891dd51096211c2',RUN/'post_result_independent_review.json':'050ff440708c4edf2979230a9adab4047d8edb99aae992a825a913a6532d564c'}
def need(v,m):
 if not bool(v):raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def matrix(x):return torch.tensor([[float.fromhex(v) for v in row] for row in x],dtype=torch.float64)
def hx(v):return float(v).hex()
def scalar(v):return {'value':float(v),'binary64':hx(v)}
def frac(v):return Fraction.from_float(float(v))
def main():
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 need(not OUT.exists(),'APPEND_ONLY_OUTPUT')
 for p,h in PINS.items():need(sha(p)==h,'SOURCE_PIN:'+str(p))
 inputs=read(INPUT);params=read(RUN/'parameters.json');result=read(RUN/'result.json');validation=read(RUN/'independent_validation.json');review=read(RUN/'post_result_independent_review.json')
 need(validation['result_sha256']==sha(RUN/'result.json') and validation['status']=='SAME_SUPPORT_SPECIFICITY_INDEPENDENT_REEXECUTION_PASS','INDEPENDENT_RESULT_QUALIFICATION')
 review_result=Path(review['result']['path']);review_result=review_result if review_result.is_absolute() else ROOT/review_result
 need(review_result.resolve()==(RUN/'result.json').resolve() and review['result']['sha256']==sha(RUN/'result.json') and review['status']=='SAME_SUPPORT_SPECIFICITY_POST_RESULT_INDEPENDENT_REVIEW_PASS','POST_RESULT_REVIEW')
 seal=read(RUN/'eval_prejoin_seal.json');need(seal['parameters_sha256']==sha(RUN/'parameters.json') and seal['eval_prejoin_sha256']==sha(RUN/'eval_prejoin.json'),'ORIGINAL_PARAMETER_PREDICTION_SEAL')
 pre={x['query_id']:x for x in read(RUN/'eval_prejoin.json')};source={x['query_id']:x for x in inputs['records'] if x['kind']=='FULL'}
 a0=result['actions']['EVAL']['ORIGINAL7']['REAL'];a8={x['query_id']:x for x in result['actions']['EVAL']['FREE8']['REAL']}
 def par(name):
  p=params[name];return torch.tensor([float.fromhex(x) for x in p['weight_binary64']],dtype=torch.float64),float.fromhex(p['bias_binary64'])
 w0,b0=par('ORIGINAL7');w8,b8=par('FREE8');primary=[];supplemental=[];checks=0;max_residual=0.
 for act in a0:
  q=act['query_id'];r=source[q];x0=matrix(r['feature_binary64']['REAL']['ORIGINAL_C']);x8=matrix(r['feature_binary64']['REAL']['FREE'])
  need([[hx(v) for v in row] for row in x0]==[[hx(v) for v in row[:6]] for row in x8],'ORIGINAL_SIX_FEATURE_BITS')
  z0=x0@w0+b0;z8=x8@w8+b8;zero=x8.clone();zero[:,6]=0.;six=zero@w8+b8;extra=x8[:,6]*w8[6]
  for name,z in [('ORIGINAL7',z0),('FREE8',z8)]:
   need([hx(v) for v in z]==pre[q]['predictions'][name]['REAL']['all127_logits_binary64'],'ALL127_FROZEN_LOGITS:'+q+':'+name);checks+=127
  if not act['final_correct']:continue
  other=a8[q];cohort='ORIGINAL7_RAW_RESCUE' if not act['base_correct'] else 'ORIGINAL7_RAW_CORRECT_HOLD'
  primary_pos=act['target_position'] if not act['base_correct'] else other['proposed_challenger']
  positions=[('PRIMARY',primary_pos)]
  if act['proposed_challenger']!=primary_pos:positions.append(('ORIGINAL7_SELECTED_CHALLENGER',act['proposed_challenger']))
  for anchor,pos in positions:
   k=r['challenger_positions'].index(pos);oldterms=x0[k]*w0;newterms=x0[k]*w8[:6]
   exact0=sum((frac(v)*frac(w) for v,w in zip(x0[k],w0)),frac(b0));exact6=sum((frac(v)*frac(w) for v,w in zip(x0[k],w8[:6])),frac(b8));exactextra=frac(x8[k,6])*frac(w8[6]);exactfull=exact6+exactextra
   residual=float(z8[k]-(six[k]+extra[k]));max_residual=max(max_residual,abs(residual))
   need(max(abs(float(exact0)-float(z0[k])),abs(float(exact6)-float(six[k])),abs(float(exactfull)-float(z8[k])))<1e-12,'INDEPENDENT_FRACTION_COMPONENT_ACCOUNTING')
   row={'query_id':q,'execution_ordinal':act['execution_ordinal'],'cohort':cohort,'anchor':anchor,'challenger_position':pos,'challenger_physical_row':r['candidate_physical_rows'][pos],
    'base_winner_physical_row':act['base_winner_physical_row'],'target_physical_row':act['target_physical_row'],'challenger_is_target':pos==act['target_position'],
    'ORIGINAL7_selected_challenger':act['proposed_challenger'],'FREE8_selected_challenger':other['proposed_challenger'],
    'ORIGINAL7_decision':act['decision'],'FREE8_decision':other['decision'],'FREE8_final_correct':other['final_correct'],
    'dF':scalar(x8[k,6]),'ORIGINAL7_logit':scalar(z0[k]),'FREE8_refitted_six_plus_bias':scalar(six[k]),'FREE8_added_F_term':scalar(extra[k]),'FREE8_full_logit':scalar(z8[k]),
    'refitted_six_bias_change_from_original':scalar(six[k]-z0[k]),'bias_change':scalar(b8-b0),
    'six_original_terms':dict(zip(('RAW','S','M','L','Q','R'),map(float,oldterms))),
    'six_refitted_terms':dict(zip(('RAW','S','M','L','Q','R'),map(float,newterms))),
    'six_term_changes':dict(zip(('RAW','S','M','L','Q','R'),map(float,newterms-oldterms))),
    'recomposition_binary64_residual':residual,'exact_rational_signs':{'original':(exact0>0)-(exact0<0),'refitted_six_bias':(exact6>0)-(exact6<0),'extra':(exactextra>0)-(exactextra<0),'full':(exactfull>0)-(exactfull<0)}}
   (primary if anchor=='PRIMARY' else supplemental).append(row)
 need(len(primary)==28 and sum(x['cohort']=='ORIGINAL7_RAW_RESCUE' for x in primary)==3 and checks==8128,'FIXED28_AND_ALL_FROZEN_LOGIT_CHECKS')
 holds=[x for x in primary if x['cohort']=='ORIGINAL7_RAW_CORRECT_HOLD'];r220=next(x for x in primary if x['query_id']=='OUTCOME-0220')
 value={'status':'FREE8_FROZEN_COMPONENT_ACCOUNTING_COMPLETE','sources':{str(p.relative_to(ROOT)):bind(p) for p in PINS},'program':bind(Path(__file__)),
  'original_prejoin':bind(RUN/'eval_prejoin.json'),'original_prejoin_seal':bind(RUN/'eval_prejoin_seal.json'),'parameter_sha256':{n:params[n]['parameter_sha256'] for n in ('ORIGINAL7','FREE8')},
  'FREE8_coefficient_F':scalar(w8[6]),'ORIGINAL7_bias':scalar(b0),'FREE8_bias':scalar(b8),'primary_rows':primary,'supplemental_fixed_original_anchors':supplemental,
  'counts':{'frozen_logit_bit_checks':checks,'original_correct_queries':28,'original_RAW_rescues':3,'original_RAW_correct_holds':25,'supplemental_anchors':len(supplemental)},
  'holds_at_FREE8_selected_challenger':{'F_term_negative':sum(x['FREE8_added_F_term']['value']<0 for x in holds),'F_term_positive':sum(x['FREE8_added_F_term']['value']>0 for x in holds),'F_term_zero':sum(x['FREE8_added_F_term']['value']==0 for x in holds),'refitted_six_bias_positive':sum(x['FREE8_refitted_six_plus_bias']['value']>0 for x in holds),'FREE8_full_positive':sum(x['FREE8_full_logit']['value']>0 for x in holds)},
  'maximum_component_recomposition_rounding_residual':max_residual,'OUTCOME0220':r220,
  'interpretation':'0220 refitted six+bias is already negative; the added F term is also negative. Both contribute to the lost SWITCH. This is fixed-input model arithmetic, not pixel causality or a newly fitted model.',
  'partial_definition':'Keep frozen FREE8 coefficients/bias and matrix width7, set only column7 to zero; component addition may differ from canonical full matmul by reported FP64 rounding.',
  'anchor_definition':'Three original rescues use the target challenger;25 original RAW-correct HOLDs use the frozen FREE8-selected challenger. The original selected challenger is additionally reported whenever different.',
  'new_training_updates':0,'new_encoder_or_RoMa_calls':0,'coefficient_or_threshold_selection':False,'deployment_changed':False}
 OUT.mkdir();p=OUT/'result.json';p.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n');p.chmod(0o444)
 columns=['query_id','cohort','challenger_physical_row','challenger_is_target','dF','ORIGINAL7_logit','FREE8_refitted_six_plus_bias','FREE8_added_F_term','FREE8_full_logit','bias_change']
 with (OUT/'primary_rows.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
  for r in primary:writer.writerow({k:r[k]['value'] if isinstance(r[k],dict) else r[k] for k in columns})
 print(json.dumps({'result':bind(p),'counts':value['counts'],'holds':value['holds_at_FREE8_selected_challenger'],'OUTCOME0220':{k:r220[k] for k in columns if k in r220}},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
