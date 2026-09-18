#!/usr/bin/env python3
"""Fresh exact endpoint/certificate validation; no solver or witness predictions."""
from pathlib import Path
from fractions import Fraction
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/rc_opened_eval_strict_no_break_capacity_v1';PRODUCER=ROOT/'programs/analyze_rc_opened_eval_strict_no_break_capacity_v1.py'
def need(x,m):
 if not bool(x):raise RuntimeError(m)
def equations(rows):
 values=[];meta=[]
 for r in rows:
  cs=list(map(int,r['challenger_positions']));t=int(r['target_position']);w=int(r['base_winner_position']);x={c:[Fraction.from_float(float(v)) for v in r['real_native_features']['C_PAIRED'][i]]+[Fraction(1)] for i,c in enumerate(cs)}
  if t==w:
   for c in cs:values.append([-v for v in x[c]]);meta.append({'query_id':r['query_id'],'kind':'BASE_CORRECT_WRONG_BELOW_ZERO','candidate_position':c})
  else:
   values.append(x[t]);meta.append({'query_id':r['query_id'],'kind':'TRUE_CHALLENGER_ABOVE_HOLD','candidate_position':t})
   for c in cs:
    if c!=t:values.append([a-b for a,b in zip(x[t],x[c])]);meta.append({'query_id':r['query_id'],'kind':'TRUE_CHALLENGER_ABOVE_WRONG','candidate_position':c})
 return values,meta

def main():
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);saved=json.loads((OUT/'result.json').read_text());manifest=json.loads((OUT/'execution_manifest.json').read_text())
 need(hashlib.sha256(PRODUCER.read_bytes()).hexdigest()==manifest['sources']['program']['sha256'],'PRODUCER_PIN')
 sp=importlib.util.spec_from_file_location('capacity_diagnostic_source_rebuild',PRODUCER);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 for item in saved['sources'].values():need(m.sha(ROOT/item['path'])==item['sha256'],'SOURCE_HASH')
 need(saved['execution_manifest']==m.binding(OUT/'execution_manifest.json') and saved['source_feature_endpoints']==m.binding(OUT/'source_feature_endpoints.json'),'ARTIFACT_HASH')
 sources,rows,endpoints,correct,oldtheta=m.inputs(independent=True);need(sources==saved['sources'] and m.read(OUT/'source_feature_endpoints.json')=={'diagnostic_label':m.LABEL,'rows':endpoints},'RAW_SCALAR_ENDPOINTS_INDEPENDENT')
 oldrows,_=equations([r for r in rows if r['query_id'] in correct]);minimum=min(sum((a*b for a,b in zip(row,oldtheta)),Fraction(0)) for row in oldrows)
 need(minimum>0 and str(minimum)==saved['baseline_28_exact_min_margin']==manifest['baseline_28_exact_min_margin'],'OLD28_EXACT_WITNESS')
 checks=[]
 for err in m.ERRORS:
  c=saved['systems'][err];required=[r for r in rows if r['query_id'] in correct or r['query_id']==err];a,meta=equations(required)
  need(len(a)==c['constraint_count']==3683 and c['parameter_count']==7 and c['required_query_ids']==[r['query_id'] for r in required] and c['unconstrained_original_error_query_ids']==[q for q in m.ERRORS if q!=err],'EXACT_FOUR_SYSTEM_SCOPE')
  need(hashlib.sha256(m.encode(meta)).hexdigest()==c['constraint_order_sha256'],'EXACT_CONSTRAINT_ORDER')
  need(c['diagnostic_label']==m.LABEL and c['witness_or_dual_is_not_model'] is True,'NON_DEPLOYABLE_CERTIFICATE_LABEL')
  if c['status']=='EXACT_RATIONAL_STRICT_ACTION_FEASIBLE':
   theta=[Fraction(v) for v in c['rational_unit_margin_theta']];need(len(theta)==7,'PRIMAL_DIMENSION');margin=min(sum((x*y for x,y in zip(row,theta)),Fraction(0)) for row in a);need(margin>=1,'EXACT_PRIMAL_UNIT_MARGIN')
   check={'query_id':err,'certificate_status':c['status'],'exact_certificate_valid':True,'exact_minimum_margin':str(margin)}
  elif c['status']=='EXACT_RATIONAL_STRICT_ACTION_INFEASIBLE':
   cert=c['certificate'];need(cert and len({x['constraint_index'] for x in cert})==len(cert),'DUAL_SUPPORT');weights=[Fraction(x['weight']) for x in cert];need(all(w>=0 for w in weights) and sum(weights)==1,'DUAL_NORMALIZED_NONNEGATIVE')
   for x in cert:need(0<=x['constraint_index']<len(a) and x['constraint']==meta[x['constraint_index']],'EXACT_DUAL_SOURCE_REFERENCE')
   need(all(sum((w*a[x['constraint_index']][j] for w,x in zip(weights,cert)),Fraction(0))==0 for j in range(7)),'FARKAS_EXACT_ZERO')
   check={'query_id':err,'certificate_status':c['status'],'exact_certificate_valid':True,'dual_support_size':len(cert)}
  else:
   need(c['status']=='NUMERIC_INFEASIBLE_WITHOUT_EXACT_CERTIFICATE' and c['exact_constraint_validation'] is False,'UNRECOGNIZED_STATUS');check={'query_id':err,'certificate_status':c['status'],'exact_certificate_valid':False,'interpretation':'unresolved; floating solver status is not proof'}
  checks.append(check)
 need(len(checks)==4 and len(saved['systems'])==4,'ALL_FOUR_SYSTEMS');complete=all(x['exact_certificate_valid'] for x in checks)
 value={'status':'OPENED_EVAL_STRICT_NO_BREAK_CAPACITY_INDEPENDENT_EXACT_VALIDATION_PASS' if complete else 'OPENED_EVAL_STRICT_NO_BREAK_CAPACITY_INDEPENDENT_VALIDATION_UNRESOLVED',
 'diagnostic_label':m.LABEL,'result_sha256':m.sha(OUT/'result.json'),'validator':m.binding(Path(__file__).resolve()),'systems':checks,'source_scalar_features_independently_rebuilt':True,'exact_original_FP64_endpoints':True,
 'original28_exact_positive_margin_verified':True,'all32_original_model_predictions_replayed':True,'certificate_solver_called':False,'certificate_witness_deployed_or_scored':False,'new_accuracy_claimed':False,'HYP_GO_claimed':False,
 'scope':'Four post-hoc strict positive-margin capacity systems on opened EVAL labels, not new learned parameters or population confirmation.'}
 m.atomic(OUT/'independent_validation.json',value);print(json.dumps({'status':value['status'],'systems':checks,'result_sha256':value['result_sha256']},sort_keys=True),flush=True)
if __name__=='__main__':main()
