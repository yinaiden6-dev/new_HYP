#!/usr/bin/env python3
"""Independent arithmetic review of a fixed producer snapshot; no training/model calls."""
from __future__ import annotations
import hashlib,json,math,os,statistics,sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/rc_train_group_jackknife_stability_v1'
OUT=ROOT/'results/rc_train_group_jackknife_producer_arithmetic_review_v1/result.json'
EXPECTED='ce7eaeb9a3407e4626bb2bf00d8f8c2c0c328ce20ae28215f17069db1ec21536'
HEADS=('JOINT4','PRODUCT5','RESPONSE6','ORIGINAL7');MODES=('REAL','CBIND');ROLES=('TRAIN','EVAL')
EDGES=(('PRODUCT5','JOINT4'),('ORIGINAL7','RESPONSE6'),('RESPONSE6','JOINT4'),('ORIGINAL7','PRODUCT5'))
PARENT=ROOT/'results/rc_product_response_factorial_v1'
META=ROOT/'results/rc_frozen_group_effect_native64_v1'
CHECKS=0

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def need(v,message):
 global CHECKS
 if not bool(v):raise RuntimeError(message)
 CHECKS+=1

def equal(a,b,label):
 if isinstance(a,dict):
  need(isinstance(b,dict) and set(a)==set(b),label+':keys')
  for k in a:equal(a[k],b[k],label+'/'+str(k))
 elif isinstance(a,list):
  need(isinstance(b,list) and len(a)==len(b),label+':length')
  for i,(x,y) in enumerate(zip(a,b,strict=True)):equal(x,y,label+'/'+str(i))
 elif isinstance(a,float):need(math.isclose(a,b,rel_tol=0,abs_tol=1e-15),label+':float')
 else:need(a==b,label+':value')

def corrected_labels():
 p=ROOT/'registry/gallery_identity_repair_v1.json';need(sha(p)=='9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f','IDENTITY_MANIFEST_PIN')
 m=read(p);catalog=ROOT.parents[2]/'dailymed/data/box_flat_20000_images/data/raw_images';labels=[]
 for directory,subdirs,names in os.walk(catalog):
  subdirs.sort()
  labels.extend((Path(directory)/name).stem.strip() for name in sorted(names) if (Path(directory)/name).is_file() and Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'})
 need(len(labels)==5413,'CATALOGUE5413')
 digest=hashlib.sha256(json.dumps(labels,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 need(digest==m['source_legacy_setids_sha256'],'LEGACY_LABEL_AXIS')
 for override in m['filename_collision_overrides']:
  i=override['physical_row'];need(labels[i]==override['legacy_label'],'ORIGINAL_OVERRIDE_LABEL');labels[i]=override['corrected_identity']
 need(len(set(labels))==5412,'CORRECTED_IDENTITY_COUNT')
 return labels

def metric(actions):
 n=len(actions);base=sum(a['base_correct'] for a in actions);final=sum(a['final_correct'] for a in actions)
 return {'query_count':n,'base_top1':base,'final_top1':final,'base_R@1':base/n,'final_R@1':final/n,
  'base_MRR':math.fsum(1/a['base_target_rank'] for a in actions)/n,'final_MRR':math.fsum(1/a['final_target_rank'] for a in actions)/n,
  'rescue':sum(not a['base_correct'] and a['final_correct'] for a in actions),'break':sum(a['base_correct'] and not a['final_correct'] for a in actions),
  'retained_correct':sum(a['base_correct'] and a['final_correct'] for a in actions),'retained_wrong':sum(not a['base_correct'] and not a['final_correct'] for a in actions),
  'wrong_to_wrong':sum(a['wrong_to_wrong'] for a in actions),'switch_count':sum(a['decision']=='SWITCH' for a in actions),'hold_count':sum(a['decision']=='HOLD' for a in actions)}

def paired(new,base,groups):
 ni={a['query_id']:a for a in new};bi={a['query_id']:a for a in base};need(set(ni)==set(bi),'PAIRED_QUERY_AXIS')
 new_correct={q for q,a in ni.items() if a['final_correct']};base_correct={q for q,a in bi.items() if a['final_correct']}
 rescue=sorted(new_correct-base_correct);broken=sorted(base_correct-new_correct);per={}
 for g in sorted({groups[q] for q in ni}):
  ids={q for q in ni if groups[q]==g};nc=len(ids&new_correct);bc=len(ids&base_correct)
  per[g]={'queries':len(ids),'new_correct':nc,'base_correct':bc,'net':nc-bc}
 return {'rescue_query_ids':rescue,'break_query_ids':broken,'rescue':len(rescue),'break':len(broken),'net':len(rescue)-len(broken),
  'supergroups':per,'group_balanced_accuracy_difference':math.fsum(v['net']/v['queries'] for v in per.values())/len(per),
  'groups_improved':sum(v['net']>0 for v in per.values()),'groups_harmed':sum(v['net']<0 for v in per.values())}

def dist(v):
 need(len(v)==12,'DELETION_DENOMINATOR12');s=sorted(v)
 return {'count':12,'min':s[0],'median':(s[5]+s[6])/2,'max':s[-1],
  'positive':sum(x>0 for x in v),'zero':sum(x==0 for x in v),'negative':sum(x<0 for x in v)}

def main():
 need(sha(SOURCE/'result.json')==EXPECTED,'PINNED_PRODUCER_SNAPSHOT');result=read(SOURCE/'result.json')
 authority_path=ROOT/'registry/rc_train_group_jackknife_stability_authority_v1_20260909.json'
 need(sha(authority_path)==result['authority_sha256'],'AUTHORITY_BOUND');authority=read(authority_path)
 for key,binding in authority['sources'].items():
  if isinstance(binding,dict) and 'path' in binding and 'sha256' in binding:need(sha(ROOT/binding['path'])==binding['sha256'],'AUTHORITY_SOURCE:'+key)
 for relative,digest in authority['sources']['parent_source_pins'].items():need(sha(ROOT/relative)==digest,'PARENT_SOURCE:'+relative)
 for file,key in (('parameters.json','parameters_sha256'),('input_closure.json','input_closure_sha256'),('eval_prejoin_seal.json','eval_prejoin_seal_sha256')):
  need(sha(SOURCE/file)==result[key],'PRODUCER_BINDING:'+file)
 seal=read(SOURCE/'eval_prejoin_seal.json');need(sha(SOURCE/'parameters.json')==seal['parameters_sha256'] and sha(SOURCE/'all_conditions_prejoin.json')==seal['predictions_sha256'],'SEALED_INPUTS')
 closure=read(SOURCE/'input_closure.json');params=read(SOURCE/'parameters.json');predictions=read(SOURCE/'all_conditions_prejoin.json')
 equal(result['conditions'],closure['conditions'],'RESULT_CONDITIONS');equal(result['conditions'],authority['condition_manifest'],'AUTHORIZED_CONDITIONS')
 conditions=[x['condition'] for x in result['conditions']];deleted=conditions[1:]
 need(len(conditions)==13 and conditions[0]=='FULL_TRAIN' and len(deleted)==12,'COMPLETE13_CONDITIONS')
 need(set(params)==set(predictions)==set(conditions),'ALL52_PARAMETER_AND_PREDICTION_SETS')
 need(params['FULL_TRAIN']==read(PARENT/'parameters.json'),'EXACT_BASELINE_ALL4_PARAMETERS')
 need(predictions['FULL_TRAIN']==read(PARENT/'eval_prejoin.json'),'EXACT_BASELINE_FULL64_SEALED_PREDICTIONS')
 parent=read(PARENT/'result.json');parentvalidation=read(PARENT/'independent_validation.json')
 need(parentvalidation['result_sha256']==sha(PARENT/'result.json') and all(v is True for v in parentvalidation['checks'].values()),'PARENT_INDEPENDENT_VALIDATION')
 for c in conditions:
  need(set(params[c])==set(HEADS),'FOUR_HEADS_PER_CONDITION')
  for name,p in params[c].items():
   need(p['finite_training']['all_finite'] and p['finite_training']['finite_loss_step_count']==2000,'RECORDED_FINITE2000')
   need(all(math.isfinite(float.fromhex(x)) for x in p['weight_binary64']+[p['bias_binary64']]),'FINITE_RECORDED_PARAMETERS')
 need(sha(META/'result.json')=='eebbd6e9b15be6181ed375f04f6c8cbc7a985f5f871073c41b29c1f03a5975fe','GROUP_META_RESULT_PIN')
 need(sha(META/'independent_validation.json')=='c0d4fd6e729d3e23e36fc3b2533cb92da823c33931569a740c88539f1e2445cd','GROUP_META_VALIDATION_PIN')
 mv=read(META/'independent_validation.json');need(mv['result_sha256']==sha(META/'result.json') and all(v is True for v in mv['checks'].values()),'GROUP_META_VALIDATED')
 facts={a['query_id']:a for a in read(META/'result.json')['all64_query_facts']};groups={q:a['supergroup'] for q,a in facts.items()}
 full=[]
 for binding in closure['parent_input_closure']['parent_input_closure']['source_shards']:
  p=ROOT/binding['path'];need(sha(p)==binding['sha256'],'FULL_PAYLOAD_SOURCE')
  full.extend(torch.load(p,map_location='cpu',mmap=True,weights_only=True)['records'])
 full=sorted(full,key=lambda r:r['execution_ordinal']);need(len(full)==64 and len({r['query_id'] for r in full})==64,'FULL64_CANDIDATE_AXES')
 byrole={role:[r for r in full if r['data_split_role']==role] for role in ROLES};need(all(len(v)==32 for v in byrole.values()),'FULL32_ROLE_AXES')
 train=byrole['TRAIN'];sorted_groups=sorted({groups[r['query_id']] for r in train});need(len(sorted_groups)==12,'ALL12_TRAIN_GROUPS')
 for spec,g in zip(result['conditions'],[None,*sorted_groups],strict=True):
  removed=[r for r in train if g is not None and groups[r['query_id']]==g];kept=[r for r in train if g is None or groups[r['query_id']]!=g]
  expected={'condition':'FULL_TRAIN' if g is None else 'DROP_GROUP_'+str(sorted_groups.index(g)+1).zfill(2),'removed_supergroup':g,
   'removed_query_ids':[r['query_id'] for r in removed],'removed_execution_ordinals':[r['execution_ordinal'] for r in removed],
   'retained_training_query_ids':[r['query_id'] for r in kept],'retained_training_execution_ordinals':[r['execution_ordinal'] for r in kept],
   'remaining_TRAIN_count':len(kept),'PAIR_count':64}
  equal(spec,expected,'INDEPENDENT_EXHAUSTIVE_CONDITION')
 labels=corrected_labels();derived={};metrics={};within={};edges={};retained={};action_count=logit_count=0
 for c in conditions:
  pred={r['query_id']:r for r in predictions[c]};need(set(pred)==set(facts),'SEALED64_QUERY_AXIS');derived[c]={};metrics[c]={};within[c]={};edges[c]={}
  for role,rows in byrole.items():
   derived[c][role]={};metrics[c][role]={};within[c][role]={};edges[c][role]={}
   for mode in MODES:
    derived[c][role][mode]={};metrics[c][role][mode]={};within[c][role][mode]={}
    for name in HEADS:
     rebuilt=[]
     for row in rows:
      q=row['query_id'];a=pred[q]['predictions'][name][mode];cs=list(map(int,row['challenger_positions']));axis=list(map(int,row['candidate_physical_rows']));raw=list(map(float,row['base_scores']))
      need(len(cs)==127 and len(axis)==128 and len(a['all127_logits_binary64'])==127,'C128_ALL127')
      z=[float.fromhex(v) for v in a['all127_logits_binary64']];need(all(math.isfinite(v) for v in z),'FINITE_SEALED_LOGITS')
      i=sorted(range(127),key=lambda k:(-z[k],axis[cs[k]]))[0];best=cs[i];w=int(row['base_winner_position']);target=int(facts[q]['target_position'])
      switch=z[i]>0;final=best if switch else w;order=sorted(range(128),key=lambda j:(-raw[j],axis[j]));need(order[0]==w,'RAW_WINNER')
      finalorder=([best]+[v for v in order if v!=best]) if switch else order
      need(final==a['final_position'] and a['decision']==('SWITCH' if switch else 'HOLD'),'SEALED_ACTION_SELECTION')
      need(sum(labels[v]==labels[axis[target]] for v in axis)==1,'UNIQUE_TARGET_IDENTITY_IN_C128')
      bc=labels[axis[w]]==labels[axis[target]];fc=labels[axis[final]]==labels[axis[target]]
      out={'execution_ordinal':row['execution_ordinal'],'query_id':q,'track':row['track'],'heldout_fold':row['heldout_fold'],
       'base_winner':w,'base_winner_physical_row':axis[w],'target_position':target,'target_physical_row':axis[target],
       'base_target_rank':order.index(target)+1,'final_target_rank':finalorder.index(target)+1,'proposed_challenger':best,
       'proposed_challenger_physical_row':axis[best],'proposed_challenger_base_rank':order.index(best)+1,
       'switch_logit':z[i],'decision':'SWITCH' if switch else 'HOLD','final_position':final,'final_physical_row':axis[final],
       'base_correct':bc,'final_correct':fc,'wrong_to_wrong':not bc and not fc and final!=w}
      rebuilt.append(out);action_count+=1;logit_count+=len(z)
     need(rebuilt==result['actions'][c][role][mode][name],'ALL_ACTION_FIELDS_EXACT')
     if c=='FULL_TRAIN':need(rebuilt==parent['actions'][role][name][mode],'BASELINE_PARENT_ALL_ACTIONS_EXACT')
     derived[c][role][mode][name]=rebuilt;metrics[c][role][mode][name]=metric(rebuilt)
     within[c][role][mode][name]=paired(rebuilt,derived['FULL_TRAIN'][role][mode][name],groups)
    edges[c][role][mode]={a+'_vs_'+b:paired(derived[c][role][mode][a],derived[c][role][mode][b],groups) for a,b in EDGES}
  kept=set(next(s for s in result['conditions'] if s['condition']==c)['retained_training_query_ids'])
  retained[c]={mode:{name:metric([a for a in derived[c]['TRAIN'][mode][name] if a['query_id'] in kept]) for name in HEADS} for mode in MODES}
 equal(metrics,result['metrics'],'ALL_CONDITION_METRICS');equal(within,result['same_head_vs_FULL_TRAIN'],'ALL_WITHIN_HEAD_PAIRED_GROUP');equal(edges,result['factorial_comparisons'],'ALL_FACTORIAL_EDGES');equal(retained,result['remaining_TRAIN_subset_metrics'],'ALL_RETAINED_TRAIN_METRICS')
 sensitivity={};frequencies={}
 for role in ROLES:
  sensitivity[role]={};frequencies[role]={}
  for mode in MODES:
   ss={'head_accuracy_counts':{},'within_head_vs_FULL_TRAIN':{},'factorial_edges':{}};frequencies[role][mode]={}
   for name in HEADS:
    ss['head_accuracy_counts'][name]=dist([metrics[c][role][mode][name]['final_top1'] for c in deleted])
    ss['within_head_vs_FULL_TRAIN'][name]={'query_net':dist([within[c][role][mode][name]['net'] for c in deleted]),'group_balanced_accuracy_difference':dist([within[c][role][mode][name]['group_balanced_accuracy_difference'] for c in deleted])}
    values=[]
    for i,base in enumerate(derived['FULL_TRAIN'][role][mode][name]):
     variants=[derived[c][role][mode][name][i] for c in deleted];correct=sum(v['final_correct'] for v in variants);dec=sum(v['decision']==base['decision'] for v in variants)
     phys=sum(v['final_physical_row']==base['final_physical_row'] for v in variants);identity=sum(labels[v['final_physical_row']]==labels[base['final_physical_row']] for v in variants)
     values.append({'query_id':base['query_id'],'baseline_final_correct':base['final_correct'],'baseline_decision':base['decision'],
      'baseline_final_physical_row':base['final_physical_row'],'baseline_final_reference_identity':labels[base['final_physical_row']],
      'deletion_condition_count':12,'correct_count':correct,'correct_frequency':correct/12,'decision_agreement_count':dec,'decision_agreement_rate':dec/12,
      'final_physical_candidate_agreement_count':phys,'final_physical_candidate_agreement_rate':phys/12,'final_reference_identity_agreement_count':identity,'final_reference_identity_agreement_rate':identity/12,
      'incorrect_when_baseline_correct_count':sum(base['final_correct'] and not v['final_correct'] for v in variants),
      'correct_when_baseline_incorrect_count':sum(not base['final_correct'] and v['final_correct'] for v in variants)})
    frequencies[role][mode][name]=values
   for a,b in EDGES:
    k=a+'_vs_'+b;ss['factorial_edges'][k]={'query_net':dist([edges[c][role][mode][k]['net'] for c in deleted]),'group_balanced_accuracy_difference':dist([edges[c][role][mode][k]['group_balanced_accuracy_difference'] for c in deleted])}
   sensitivity[role][mode]=ss
 equal(sensitivity,result['deletion_condition_sensitivity'],'ALL_SENSITIVITY_DISTRIBUTIONS');equal(frequencies,result['query_correctness_and_agreement'],'ALL_QUERY_FREQUENCIES_AND_AGREEMENTS')
 need(action_count==6656 and logit_count==845312,'COMPLETE_REVIEW_POPULATION')
 table=[{'condition':c,'remaining_TRAIN_count':spec['remaining_TRAIN_count'],**{h:metrics[c]['EVAL']['REAL'][h]['final_top1'] for h in HEADS}} for c,spec in zip(conditions,result['conditions'],strict=True)]
 original_frequencies=frequencies['EVAL']['REAL']['ORIGINAL7']
 review={'status':'PRODUCER_ARITHMETIC_REVIEW_COMPLETE_PENDING_INDEPENDENT_RETRAIN','producer_result_sha256':EXPECTED,
  'producer_parameters_sha256':sha(SOURCE/'parameters.json'),'producer_predictions_sha256':sha(SOURCE/'all_conditions_prejoin.json'),
  'source_program_sha256':sha(Path(__file__)),'independent_arithmetic_checks':CHECKS,'sealed_action_rows_recomputed':action_count,'sealed_logits_examined':logit_count,
  'all13_condition_metrics_paired_group_distributions_frequencies_recomputed':True,'baseline_all4_parameters_predictions_actions_equal_validated_parent':True,
  'all12_EVAL_REAL_condition_counts':table,'all12_EVAL_REAL_factorial_sensitivity':sensitivity['EVAL']['REAL']['factorial_edges'],
  'ORIGINAL7_all32_EVAL_query_stability':original_frequencies,
  'original_wrong_queries_correct_under_any_deletion':sorted(q['query_id'] for q in original_frequencies if not q['baseline_final_correct'] and q['correct_count']>0),
  'formal_independent_retrain_validation_checked':False,'new_training_runs':0,'new_model_predictions':0,'scheduler_queries':0,
  'scope':'Fixed producer snapshot arithmetic review. It does not verify that deletion parameters reproduce under fresh training, and does not replace the pending full independent retrain validator.',
  'scientific_GO_claimed':False,'model_adopted':False,'limits':['All12 perturbations share one previously opened EVAL32; no independent population confirmation.','No favorable deletion was selected.','Floating aggregate agreement tolerance1e-15 allows only summation-order differences; actions and baseline parameter/prediction objects are exact.']}
 OUT.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(review,sort_keys=True,indent=2)+'\n'
 if OUT.exists():need(OUT.read_text()==payload,'APPEND_ONLY_ARITHMETIC_REVIEW')
 else:OUT.write_text(payload)
 print(json.dumps({'status':review['status'],'review_result_sha256':sha(OUT),'checks':CHECKS,'action_rows':action_count,'logits':logit_count,'condition_counts':table},indent=2))
if __name__=='__main__':main()
