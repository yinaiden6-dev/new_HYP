#!/usr/bin/env python3
"""Verify the user's literal projection-foot distance on sealed TRAIN4 only."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/rc_reference_support_projection_distance_train_description_v2'
P=ROOT/'results/rc_reference_support_maxmin_train_pilot_v1'
def binding(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def exact(x):return Fraction(int(x['numerator']),int(x['denominator']))
def main():
 dp=D/'description.json';assert binding(dp)['sha256']=='44da9f6aebb868c524299091a8f3466c0f1be2b1dad2668e1ac16172e398a4e3'
 desc=json.loads(dp.read_text())
 for name,v in desc['sources'].items():assert binding(P/name)==v
 assert binding(Path(desc['addendum']['path']))==desc['addendum']
 seal=json.loads((P/'prejoin_seal.json').read_text());result=json.loads((P/'result.json').read_text());val=json.loads((P/'validation.json').read_text())
 assert val['status']=='TRAIN_MAXMIN_PILOT_INDEPENDENT_EXACT_CERTIFICATE_VALIDATION_PASS' and val['result']==binding(P/'result.json')
 assert val['prejoin_seal']==result['prejoin_seal']==binding(P/'prejoin_seal.json')
 indexed={(r['query_id'],r['candidate_position']):r for r in desc['candidate_rows']};assert len(indexed)==512
 scores={}
 for r in seal['records']:
  row=indexed[(r['query_id'],r['candidate_position'])];assert row['physical_row']==r['physical_row']
  lower=exact(r['certificate']['lower']);baseline=exact(r['original_exact_support_margin'])
  j=float.fromhex(r['old_FP64']['J_binary64']);t=float(lower);total=j+t;distance=abs(total)/math.sqrt(2.0)
  assert (row['J_hex'],row['T_hex'],row['sum_hex'],row['distance_hex'])==(j.hex(),t.hex(),total.hex(),distance.hex())
  assert row['distance']==distance
  exact_sum=abs(baseline+lower)
  assert Fraction(int(row['absolute_sum_exact_numerator']),int(row['absolute_sum_exact_denominator']))==exact_sum
  scores[(r['query_id'],r['candidate_position'])]=(exact_sum,distance,r['physical_row'])
 qrows=[]
 for target in result['TRAIN_postjoin']['targets']:
  q=target['query_id'];p=target['target_position'];items=[(g,scores[(q,g)]) for g in range(128)]
  exact_order=sorted(items,key=lambda x:(-x[1][0],x[1][2]));production_order=sorted(items,key=lambda x:(-x[1][1],x[1][2]))
  competitor=max((x for x in items if x[0]!=p),key=lambda x:(x[1][1],-x[1][2]))
  qrows.append({'query_id':q,'target_exact_distance_rank':[x[0] for x in exact_order].index(p)+1,
   'target_production_distance_rank':[x[0] for x in production_order].index(p)+1,
   'target_distance':scores[(q,p)][1],'strongest_wrong_distance':competitor[1][1],'strongest_wrong_position':competitor[0]})
 assert qrows==desc['query_rows'] and [r['target_production_distance_rank'] for r in qrows]==[6,1,1,1]
 assert desc['target_first_count']==3 and desc['query_count']==4 and desc['group_count']==3 and desc['EVAL_reads']==0 and desc['new_model_accuracy'] is None
 out={'status':'LITERAL_PROJECTION_DISTANCE_TRAIN4_INDEPENDENT_REVIEW_PASS','description':binding(dp),'reviewer_program':binding(Path(__file__).resolve()),
  'sources':desc['sources'],'addendum':desc['addendum'],'candidate_count_checked':512,'query_rows':qrows,
  'formula':'binary64 abs(J+float(exact_L))/math.sqrt(2.0); exact plotting ranks use abs(exact_fixed_margin+exact_L)',
  'target_first_count':3,'TRAIN_queries':4,'TRAIN_groups':3,'new_LP_calls':0,'new_training_updates':0,'EVAL_reads':0,'new_model_accuracy':None}
 op=D/'independent_review.json'
 with op.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'status':out['status'],'output':binding(op)},sort_keys=True))
if __name__=='__main__':main()
