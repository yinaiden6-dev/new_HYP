#!/usr/bin/env python3
"""All32 frozen ORIGINAL7 explanation; reuse qualified S/QR, no training or subset search."""
import argparse,hashlib,importlib.util,json,math,os,sys
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
HEAD=ROOT/'results/rc_reference_support_maxmin_readout_v1'
SPEC=ROOT/'results/rc_same_support_specificity_inputs_v1/result.json'
PRIOR=ROOT/'results/rc_frozen_group_effect_native64_v1'
OUT=ROOT/'results/rc_original7_six_feature_decisions_v1'
REPORT=ROOT/'reports/REPORT_NEW_HYP_ORIGINAL7_SIX_FEATURE_DECISIONS_V1_20260910.md'
HELPER=ROOT/'programs/review_rc_reference_support_maxmin_readout_v1.py'
NAMES=('RAW','S','M','L','Q','R')
CONDITIONS={'ORIGINAL':[],'DROP_RAW':[0],'DROP_S':[1],'DROP_M':[2],'DROP_L':[3],'DROP_Q':[4],'DROP_R':[5],'DROP_QR':[4,5],'DROP_S_QR':[1,4,5]}
REUSED={'DROP_S','DROP_QR','DROP_S_QR'}
PINS={HEAD/'result.json':'a7be9cd34e7509c85a5cb06bab79ce92ae63e75614933287279cbd9853760424',HEAD/'parameters.json':'32969083441ec1071cc88777e253e4d0a333894962d5cb30d32d35d2d6bb3ec1',HEAD/'eval_prejoin.json':'4dbd5c5e4fe359e35075493846df1cfd59d7354ea1e0f222fff9aaedc31e5d1c',HEAD/'independent_validation.json':'56d747ac3888687be9455d34086dbf5e12032a246b07151285dd6246b55dcdd2',HEAD/'independent_review.json':'d4c05cf450ebc83f9eaa54a4195b61e578023eb45cf5cd6707f05e83d9624eef',SPEC:'13ec8fa9f79282f2cd94e9b451660c99b432c0989ead279cf96c4bfdf3f94bc2',PRIOR/'result.json':'eebbd6e9b15be6181ed375f04f6c8cbc7a985f5f871073c41b29c1f03a5975fe',PRIOR/'independent_validation.json':'c0d4fd6e729d3e23e36fc3b2533cb92da823c33931569a740c88539f1e2445cd',PRIOR/'all64_prejoin.json':'76549c79c05fc142a2f17827e4cc098dfcc52e21cc754ae29bdfb8fe9a770bc4',PRIOR/'frozen_parameters.json':'019cfeac14575b97371f66f81e5ad6ec55ec321af502a0b9110e4cc207c79b43',PRIOR/'prejoin_seal.json':'6a12ff9a521ce7e40f10160ef5732b903a9ef60254debc6caabc0e15f246dd31',HELPER:'fd0860d8bbf58b032fc1f7f9ffaadce18679ccaf5c96ae4621653fa32f1023df'}
B=None

def need(v,m):
 if not bool(v):raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def hx(x):return float(x).hex()
def F(x):return Fraction.from_float(float(x))
def frac(x):return {'numerator':str(x.numerator),'denominator':str(x.denominator),'value':float(x)}
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def predict(z,row,independent):
 values=list(map(float,z));cs=row['challenger_positions'];axis=row['candidate_physical_rows']
 need(len(values)==127 and all(math.isfinite(v) for v in values),'ALL127_FINITE')
 index=sorted(range(127),key=lambda i:(-values[i],axis[cs[i]]))[0] if independent else max(range(127),key=lambda i:(values[i],-axis[cs[i]]))
 return {'final_position':cs[index] if values[index]>0 else row['base_winner_position'],'decision':'SWITCH' if values[index]>0 else 'HOLD','all127_logits_binary64':[hx(v) for v in z]}
def terms(x,z,w,b,index,row):
 exact=[F(x[index,i])*F(w[i]) for i in range(6)];affine=sum(exact,F(b));observed=F(z[index]);pos=row['challenger_positions'][index]
 return {'candidate_position':pos,'physical_row':row['candidate_physical_rows'][pos],'features_binary64':[hx(v) for v in x[index]],'six_signed_terms':{n:float(exact[i]) for i,n in enumerate(NAMES)},'six_signed_terms_exact':{n:frac(exact[i]) for i,n in enumerate(NAMES)},'bias':b,'exact_affine_sum':frac(affine),'actual_FP64_logit':float(z[index]),'actual_FP64_logit_binary64':hx(z[index]),'matmul_bias_roundoff_exact':frac(observed-affine)}

def build(independent=False):
 global B
 import torch
 for p,h in PINS.items():need(sha(p)==h,'SOURCE_PIN:'+str(p))
 spec=importlib.util.spec_from_file_location('qualified_frozen_action_math',HELPER);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
 current=read(HEAD/'result.json');val=read(HEAD/'independent_validation.json');rev=read(HEAD/'independent_review.json')
 need(val['status']=='MAXMIN_READOUT_INDEPENDENT_REEXECUTION_PASS' and val['result_sha256']==PINS[HEAD/'result.json'] and rev['status']=='MAXMIN_READOUT_POST_RESULT_INDEPENDENT_ACTION_REVIEW_PASS','QUALIFIED_CURRENT_HEAD')
 old=read(PRIOR/'result.json');oldv=read(PRIOR/'independent_validation.json');olds=read(PRIOR/'prejoin_seal.json')
 need(oldv['status']=='FROZEN_NATIVE64_GROUP_EFFECT_FRESH_REEXECUTION_PASS' and oldv['result_sha256']==PINS[PRIOR/'result.json'] and all(v is True for v in oldv['checks'].values()),'QUALIFIED_PRIOR_S_QR')
 need(olds['parameters_sha256']==PINS[PRIOR/'frozen_parameters.json'] and olds['predictions_sha256']==PINS[PRIOR/'all64_prejoin.json'],'PRIOR_S_QR_SEAL')
 p=read(HEAD/'parameters.json')['ORIGINAL7'];op=read(PRIOR/'frozen_parameters.json')['ORIGINAL7']
 for k in ('weight_binary64','bias_binary64','parameter_sha256'):need(p[k]==op[k],'SAME_ORIGINAL_HEAD')
 w=torch.tensor([float.fromhex(v) for v in p['weight_binary64']],dtype=torch.float64);b=float.fromhex(p['bias_binary64'])
 anchors=current['actions']['EVAL']['ORIGINAL7']['REAL'];qids=[a['query_id'] for a in anchors]
 need(len(qids)==len(set(qids))==32 and sum(a['base_correct'] for a in anchors)==25 and sum(a['final_correct'] for a in anchors)==28,'EVAL32_RAW25_ORIGINAL28')
 rows={r['query_id']:r for r in read(SPEC)['records'] if r['kind']=='FULL' and r['query_id'] in qids}
 sealed={r['query_id']:r for r in read(HEAD/'eval_prejoin.json') if r['query_id'] in qids}
 previous={r['query_id']:r for r in read(PRIOR/'all64_prejoin.json') if r['query_id'] in qids}
 groups={a['query_id']:a['supergroup'] for a in old['actions']['EVAL']['REAL']['ORIGINAL7']}
 need(set(rows)==set(sealed)==set(previous)==set(groups)==set(qids) and len(set(groups.values()))==11,'EXACT_CURRENT32_ONLY')
 all_actions={name:[] for name in CONDITIONS};ledger=[];maxres=0.
 for anchor in anchors:
  q=anchor['query_id'];row=rows[q];prior=previous[q];target=anchor['target_position'];cs=row['challenger_positions'];winner=row['base_winner_position'];axis=row['candidate_physical_rows']
  for k in ('candidate_physical_rows','challenger_positions','base_winner_position','base_scores_binary64'):need(row[k]==prior[k],'PRIOR_AXIS')
  features=row['feature_binary64']['REAL']['ORIGINAL_C'];need(features==prior['modes']['REAL']['features_binary64'],'SAME_NATIVE6_BITS')
  x=torch.tensor([[float.fromhex(v) for v in r] for r in features],dtype=torch.float64);need(x.shape==(127,6),'NATIVE6_AXIS');z=x@w+b
  need([hx(v) for v in z]==sealed[q]['predictions']['ORIGINAL7']['REAL']['all127_logits_binary64']==prior['modes']['REAL']['predictions']['ORIGINAL7']['all127_logits_binary64'],'ORIGINAL_ALL127_LOGITS_BIT_EXACT')
  conditions={}
  for name,cols in CONDITIONS.items():
   if name=='ORIGINAL':zz=z
   elif name in REUSED:zz=torch.tensor([float.fromhex(v) for v in prior['modes']['REAL']['predictions'][name]['all127_logits_binary64']],dtype=torch.float64)
   else:xx=x.clone();xx[:,cols]=0.;zz=xx@w+b
   pr=predict(zz,row,independent);action=B.replay(row,pr,target,anchor['track'],anchor['heldout_fold']);all_actions[name].append(action)
   if name=='ORIGINAL':need(B.encode(action)==B.encode(anchor),'ALL_ORIGINAL_ACTION_FIELDS_EXACT')
   if name in REUSED:
    a0=next(a for a in old['actions']['EVAL']['REAL'][name] if a['query_id']==q)
    for k in ('decision','final_position','proposed_challenger','base_correct','final_correct','switch_logit'):need(action[k]==a0[k],'EXACT_PRIOR_INTERVENTION_ACTION')
   conditions[name]={'zeroed_columns':cols,'source':'reused prior independently validated full127 intervention' if name in REUSED else 'current full width6 FP64 matrix multiply','prediction':pr,'action':action}
  action=conditions['ORIGINAL']['action'];ti=None if target==winner else cs.index(target)
  wi=max((i for i in range(127) if cs[i]!=target),key=lambda i:(float(z[i]),-axis[cs[i]]))
  wt=terms(x,z,w,b,wi,row);tt=None if ti is None else terms(x,z,w,b,ti,row)
  threshold=-float(z[wi]) if ti is None else float(z[ti]);comp=None if ti is None else float(z[ti])-float(z[wi])
  if ti is None:
   contrib=[-F(x[wi,k])*F(w[k]) for k in range(6)];ideal=sum(contrib,-F(b));observed=-F(z[wi])
   margin_parts={'kind':'HOLD0_minus_strongest_wrong','RAW_policy_score':0.,'bias_contribution':-b,'six_contributions':{n:float(contrib[k]) for k,n in enumerate(NAMES)},'exact_affine_margin':frac(ideal),'observed_logit_endpoints_margin':frac(observed),'roundoff_exact':frac(observed-ideal)}
  else:
   contrib=[(F(x[ti,k])-F(x[wi,k]))*F(w[k]) for k in range(6)];ideal=sum(contrib,Fraction());observed=F(z[ti])-F(z[wi])
   margin_parts={'kind':'target_minus_strongest_wrong','bias_cancels':True,'six_contributions':{n:float(contrib[k]) for k,n in enumerate(NAMES)},'six_contributions_exact':{n:frac(contrib[k]) for k,n in enumerate(NAMES)},'exact_affine_margin':frac(ideal),'observed_logit_endpoints_margin':frac(observed),'roundoff_exact':frac(observed-ideal)}
  residual=z-((x*w).sum(dim=1)+b);maxres=max(maxres,float(residual.abs().max()))
  category='RAW_CORRECT_HELD' if anchor['base_correct'] else 'ORIGINAL_RESCUE' if anchor['final_correct'] else 'ORIGINAL_REMAINING_WRONG'
  failure=None if anchor['final_correct'] else 'TARGET_CANNOT_TRIGGER_SWITCH' if threshold<=0 else 'TARGET_LOSES_CHALLENGER_COMPETITION'
  ledger.append({'query_id':q,'execution_ordinal':row['execution_ordinal'],'supergroup':groups[q],'category':category,'failure_description':failure,'target_physical_row':axis[target],'RAW_winner_physical_row':axis[winner],'candidate_physical_rows':axis,'challenger_positions':cs,'original_features_binary64':features,'all127_signed_FP64_terms_binary64':[[hx(v) for v in r] for r in x*w],'all127_sum_terms_vs_matmul_residual_binary64':[hx(v) for v in residual],'original_action':action,'switch_threshold_or_HOLD_protection_margin':threshold,'target_vs_strongest_wrong_margin':comp,'correct_action_margin':threshold if comp is None else min(threshold,comp),'target_terms':tt,'strongest_wrong_terms':wt,'margin_decomposition':margin_parts,'conditions':conditions})
 need(sum(r['category']=='RAW_CORRECT_HELD' for r in ledger)==25 and sum(r['category']=='ORIGINAL_RESCUE' for r in ledger)==3,'FULL28_BREAKDOWN')
 need(all(r['original_action']['decision']=='HOLD' for r in ledger if r['category']=='RAW_CORRECT_HELD'),'ALL25_CORRECT_HOLD')
 stats={n:B.metrics(a) for n,a in all_actions.items()};comparisons={n:B.paired(a,all_actions['ORIGINAL'],groups) for n,a in all_actions.items()}
 rescues=[r for r in ledger if r['category']=='ORIGINAL_RESCUE'];held=[r for r in ledger if r['category']=='RAW_CORRECT_HELD']
 return {'status':'ORIGINAL7_FULL_SIX_FEATURE_DECISION_LEDGER_COMPLETE','theory_name':'new HYP','program':bind(Path(__file__).resolve()),'sources':{str(p.relative_to(ROOT)):bind(p) for p in PINS},'model':'ORIGINAL7/NATIVE7 six original features plus shared bias','parameter_sha256':p['parameter_sha256'],'weights':{n:float(w[i]) for i,n in enumerate(NAMES)},'bias':b,'parameter_binary64':{'weights':p['weight_binary64'],'bias':p['bias_binary64']},'feature_definitions':{'RAW':'standardized original C128 challenger-minus-RAW-winner score','S':'symmetric(S_c,S_w)','M':'symmetric(M_c,M_w)','L':'symmetric(S_c/max(M_c,1e-12),S_w/max(M_w,1e-12))','Q':'symmetric(S_c-SQc,S_w-SQw)','R':'symmetric(S_c-SRc,S_w-SRw)'},'coherent_readout_blocks':{'RAW':[0],'S':[1],'M':[2],'L':[3],'QR':[4,5]},'new_zero_conditions':['DROP_RAW','DROP_M','DROP_L','DROP_Q','DROP_R'],'reused_qualified_interventions':sorted(REUSED),'query_count':32,'supergroup_count':11,'candidate_count':128,'original_bit_exact_logit_checks':4064,'new_zeroing_logit_count':20320,'reused_zeroing_logit_count':12192,'condition_summaries':stats,'comparisons_vs_original':comparisons,'query_ledger':ledger,'rescue_query_ids':[r['query_id'] for r in rescues],'original_RAW_M_L_coefficients_preserve_three_rescues_without_S_Q_R':{r['query_id']:r['conditions']['DROP_S_QR']['action']['final_correct'] for r in rescues},'closest_RAW_correct_HOLD_margins':[{'query_id':r['query_id'],'margin':r['correct_action_margin'],'strongest_wrong_physical_row':r['strongest_wrong_terms']['physical_row']} for r in sorted(held,key=lambda r:r['correct_action_margin'])],'maximum_sum_vs_matmul_recomposition_residual':maxres,'all64_feature_subsets_enumerated':False,'subset_enumeration_not_needed':'Prior DROP_S_QR already provides a sufficient retained RAW/M/L combination for three rescues but fails to preserve all28; no minimality claim is required.','new_training_updates':0,'new_LP_calls':0,'new_encoder_forwards':0,'counterfactual_model_adopted':False,'expanded_universe_reads':0,'evidence_level':'Post-hoc fixed-head computation on all previously opened matched EVAL32, original RAW C128; not external confirmation.','limits':['The six coordinates are coupled; zeroing one does not remove the physical geometry/content or all information represented by that quantity.','Feature zeroing may leave the natural feature manifold; it is a computational counterfactual, not image-level causality.','Conditional loss after removing one coordinate is not universal necessity; a sufficient retained subset is not proved minimal, unique or population sufficient.','Bias remains in SWITCH versus HOLD but cancels between challengers. RAW winner policy score is0, not the learned bias.','All127 competitors are reselected in every condition; no target-only action, new model fitting or deployment selection occurs.','Four originally failed queries remain in the full32 ledger; larger-population stability and pixel-level identity specificity remain open.']}

def report(r):
 rescues=[x for x in r['query_ledger'] if x['category']=='ORIGINAL_RESCUE'];stats=r['condition_summaries']
 lines=['# new HYP：原七参数头为何救回3条并保住RAW25','', '对象为原RAW C128、RoMa soft visibility × full-reference image-token MaxSim的ORIGINAL7/NATIVE7。已打开matched EVAL32（11组），RAW25→28；本轮没有训练、换评分或读取扩展样本。','', '原头实际使用六维：','', '    z = 1.007810 RAW − 3.780124 S + 5.716638 M + 5.912594 L', '        − 0.148723 Q − 0.238728 R − 1.477875。','', '各列均相对RAW winner；L=S/max(M,epsilon)，Q/R为控制响应对比。取127个challenger最大logit，严格>0才SWITCH，否则HOLD。RAW winner策略分数为0，不是bias。','', '## 三次救回过了阈值门和竞争门','', '| Query | target row | target logit | 最强wrong logit | target−wrong |','| --- | ---: | ---: | ---: | ---: |']
 for x in rescues:lines.append(f"| {x['query_id']} | {x['target_physical_row']} | {x['target_terms']['actual_FP64_logit']:.9f} | {x['strongest_wrong_terms']['actual_FP64_logit']:.9f} | {x['target_vs_strongest_wrong_margin']:.9f} |")
 lines+=['','target的六项有符号贡献（最后仍加bias −1.477874513）：','', '| Query | RAW | S | M | L | Q | R |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for x in rescues:lines.append('| '+x['query_id']+' | '+' | '.join(f"{x['target_terms']['six_signed_terms'][n]:+.6f}" for n in NAMES)+' |')
 lines+=['','三条target均靠M与L的正贡献，越过RAW、S、Q/R及bias的负贡献；同时必须压过其它完整reference。0212中，target相对最强wrong的RAW项贡献约+0.867069，最终竞争margin只有+0.659635，RAW先验直接参与了身份竞争。','0220的竞争margin约1.518921，但SWITCH余量仅0.012491441；它的脆弱点是是否换掉RAW winner。','', '## 原正确的保持机制','', '25个RAW正确例全部通过HOLD保留。每条完整127个wrong中的最强者、六项和保护margin均已记录。bias在challenger间比较时抵消，在HOLD门中仍保留。最紧的保护余量：','']
 for x in r['closest_RAW_correct_HOLD_margins'][:5]:lines.append(f"- {x['query_id']}：{x['margin']:.9f}；最强wrong row={x['strongest_wrong_physical_row']}。")
 lines+=['','## 固定参数置零','', '| 条件 | 正确/32 | 原3个rescue保留 | 损失的原正确 |','| --- | ---: | ---: | --- |']
 for name in CONDITIONS:
  kept=sum(x['conditions'][name]['action']['final_correct'] for x in rescues);lost=r['comparisons_vs_original'][name]['break_query_ids']
  lines.append(f"| {name} | {stats[name]['final_top1']} | {kept}/3 | {', '.join(lost) if lost else '无'} |")
 lines+=['','去M丢掉3次rescue而保持RAW25；去L还丢掉RAW正确0211。去RAW丢0212救回及0419/0618的HOLD。S和Q/R不直接决定本批3次rescue是否成立，却保护某些原正确决策：去S损失0419/0618，单去Q、单去R或去QR均损失0618。','S/QR及S+QR联合置零复用已独立验证的完整logits。RAW、M、L、S块与各自单列相同；QR是两列响应块，不能把这些输入块当成独立的图像信息。','既有DROP_S_QR已证明原RAW/M/L系数能保留3次rescue，但只得26/32，无法替代完整头。因此没有枚举64个子集，也没有宣称最小或唯一组合。','', '## 原4个失败仍保留','', '| Query | target logit | 最强wrong logit | 原动作 | 计算层面失败位置 |','| --- | ---: | ---: | --- | --- |']
 for x in r['query_ledger']:
  if x['category']=='ORIGINAL_REMAINING_WRONG':lines.append(f"| {x['query_id']} | {x['target_terms']['actual_FP64_logit']:.9f} | {x['strongest_wrong_terms']['actual_FP64_logit']:.9f} | {x['original_action']['decision']} | {x['failure_description']} |")
 lines+=['','这些是固定模型的计算解释：六列来源和代数耦合，置零可能离开自然数据关系，不能解释为删除真实空间/内容信息或证明像素因果。局部条件必要性不是跨数据必要性。','仍未知：这些分量的像素级身份特异性，以及冻结原头在更大群体上的保持/救回稳定性。本账本可直接用于后续经授权的更大样本，此轮没有读取扩展universe。',f"全部4,064原logits逐bit回放；分项求和与矩阵运算最大舍入残差{r['maximum_sum_vs_matmul_recomposition_residual']:.3g}，动作使用完整原运算。全部32条、所有127 logits、六项和反事实动作均在result.json。",f"结果SHA：{sha(OUT/'result.json')}。",'']
 return '\n'.join(lines)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('produce','validate'),required=True);phase=parser.parse_args().phase
 import torch
 torch.set_num_threads(8);torch.set_num_interop_threads(1);r=build(phase=='validate')
 if phase=='produce':
  need(not OUT.exists(),'APPEND_ONLY_NEW_OUTPUT');B.append(OUT/'result.json',B.encode(r)+b'\n');B.append(REPORT,report(r).encode())
 else:
  need(read(OUT/'result.json')==r,'FRESH_SOURCE_AND_SORTING_ACTION_REPLAY');B.append(OUT/'validation.json',B.encode({'status':'ORIGINAL7_SIX_FEATURE_LEDGER_FRESH_REPLAY_PASS','result':bind(OUT/'result.json'),'report':bind(REPORT),'program':bind(Path(__file__).resolve()),'checks':{'all32_original127_logits_bit_exact':True,'all127_reselected_for_every_condition':True,'prior_S_QR_reused_same_head_and_native6':True,'exact_endpoint_threshold_competition_decomposition':True,'all32_retained_no_subset_deployment_selection':True},'new_training_updates':0,'scope':'Fresh source replay; alternate sort-based prediction selector, with previously qualified independent action/metric arithmetic.'})+b'\n')
 print(json.dumps({'status':r['status'],'phase':phase,'result':bind(OUT/'result.json'),'counts':{k:v['final_top1'] for k,v in r['condition_summaries'].items()},'closest_HOLD':r['closest_RAW_correct_HOLD_margins'][:5]},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
