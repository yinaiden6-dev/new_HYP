"""Target-free prediction and post-seal reducer for the exploratory GAP follow-up.

Executed into the main runner's namespace; there are no top-level data reads.
"""
READOUT = ROOT / 'results/rc_original7_train128_readout_v1'
READOUT_AUTH = ROOT / 'registry/rc_original7_train128_readout_authority_v1_20260910.json'
READOUT_AUTH_SHA = '78289a13690a4c662adfe5d22ea03dd721d99469be144b455053d8fa12b04b95'
EVAL_CORE = ROOT / 'programs/run_rc_original7_eval128_full_evidence_v2_compat.py'
EVAL_CORE_SHA = '806a8506693e2d9e88aba4331702c2d2c8df58e4d3f693b9b30e0601de49304c'
EVAL_RELEASED = False
POST_RELEASED = False
EVAL_PATHS = set()
POST_PATHS = set()
READOUT_PINS = {
 'predictions.json':'844d2321846ac83caedd013861a0b5f0bc04a38eaa4d5b55961182c063a8c59e',
 'prediction_seal.json':'f260e07eec3d0329c23b2c420dc63b34474dd944be59252bd80cbcfb05e644c9',
 'prediction_validation.json':'44daffaf14671b34ca590447efa9324109da7ee54824a7a49d73e7ee6b5aa8af',
 'parameters.json':'138d08e89d22f10ae008e1ae38df1cfe6b15b36d0e08f2b655400f6aa05de45a',
 'parameter_seal.json':'538df8340b114f97b0ab1666fe91120d4100ffbd329494a79b30913a4b15d91f',
 'fit_validation.json':'83d723e506e6d6a5df67323aa0e23eeeeda980a62005430c435b37d435dd0889'}
READOUT_POST_PINS = {
 'result.json':'45c1ea0552e2614225e08a2e52c0dd6d3397fe3eff6775bb52b087bf46118020',
 'validation.json':'c10c7a47de24284beed02d1a523bf76c362df1a32838e7252ab2ff1a121250ce',
 'review.json':'5808c1e23a27877a4ca4b27a14050788fa5bb03b52aeed282fed4b8ee3a820d6'}
CONTRACT.update({'performance_gate':['EVAL128 strictly above99 and all old99 retained','EVAL32 all old28 retained'],
 'MRR_scope':{'EVAL32':'natural C128','EVAL128':'full5412 identity rank'}})


def evaluation_sources():
 need(sha(READOUT_AUTH)==READOUT_AUTH_SHA,'PRIOR_READOUT_AUTHORITY_PIN')
 parent=read(READOUT_AUTH);sources=parent['source_bindings']
 public={name:{'path':str(READOUT/name),'sha256':digest} for name,digest in READOUT_PINS.items()}
 public.update({name:value for name,value in sources.items() if name.startswith('shared_EVAL_') and name.endswith(('_metadata','_arrays'))})
 need(sum(name.startswith('shared_EVAL_') for name in public)==64,'ALL32_TARGET_FREE_METADATA_RAW_ARRAYS')
 for name in ('EVAL128_prejoin_records','EVAL128_prejoin_seal','EVAL128_prejoin_validation'):public[name]=sources[name]
 post={name:{'path':str(READOUT/name),'sha256':digest} for name,digest in READOUT_POST_PINS.items()}
 for name in ('EVAL32_role_manifest','EVAL32_role_validation','EVAL128_curator','EVAL128_metadata_validation'):
  post[name]=parent['postjoin_source_bindings'][name]
 return public,post


def evaluation_operators():
 need(sha(EVAL_CORE)==EVAL_CORE_SHA,'BOUND_ACTION_MRR_STATS_CORE')
 names={'predict','target_action','metrics','paired_actions','exact_group_signflip','primary_statistics'}
 nodes=[n for n in ast.parse(EVAL_CORE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 need({n.name for n in nodes}==names,'COMPLETE_BOUND_EVALUATION_OPERATORS')
 namespace=dict(globals());exec(compile(ast.Module(body=nodes,type_ignores=[]),str(EVAL_CORE),'exec'),namespace)
 return {name:namespace[name] for name in names}


def release_fit():
 global EVAL_RELEASED
 val=read(OUT/'fit_validation.json');seal=read(checked(val['seal']))
 need(val['status']=='ORIGINAL7_GAP_FULL_TRAIN_FRESH_EXACT_SOLVER_PASS' and val['authority']==seal['authority']==bind(AUTH)
  and val['program']==seal['program']==bind(PROGRAM),'FRESH_FIT_BEFORE_EVAL')
 need(val['EVAL_reads']==val['base_refits']==val['optimizer_steps']==val['forbidden_read_attempts']==0
  and val['base_parameter_sha256']==BASE_PARAMETER_SHA,'TRAIN_FIT_BOUNDARY')
 params=read(checked(val['parameters']))
 need(val['parameters']==seal['parameters']==bind(OUT/'parameters.json') and params['base']['parameter_sha256']==BASE_PARAMETER_SHA
  and val['alpha_binary64']==params['alpha_binary64'],'SINGLE_SEALED_EC7_ALPHA')
 EVAL_RELEASED=True
 return params


def existing_evaluation_inputs():
 release_fit();public,post=evaluation_sources();authority=read(AUTH)
 need(authority['evaluation_sources']==public and authority['postjoin_sources']==post,'FROZEN_EVAL_POSTJOIN_SOURCE_MAPS')
 EVAL_PATHS.update(Path(b['path']).resolve() for b in public.values());POST_PATHS.update(Path(b['path']).resolve() for b in post.values())
 for b in public.values():checked(b)
 seal=read(READOUT/'prediction_seal.json');val=read(READOUT/'prediction_validation.json')
 params=read(READOUT/'parameters.json')['ORIGINAL7'];fitval=read(READOUT/'fit_validation.json');pseal=read(READOUT/'parameter_seal.json')
 need(seal['status']=='TRAIN128_ALL160_ALLTHREE_REAL_CBIND_PREDICTIONS_SEALED' and seal['authority']==val['authority']==bind(READOUT_AUTH),'OLD_PREDICTION_AUTHORITY')
 need(val['status']=='TRAIN128_FRESH_ALL160_PREDICTION_REPLAY_PASS' and val['prediction_seal']==bind(READOUT/'prediction_seal.json')
  and val['predictions']==seal['predictions']==bind(READOUT/'predictions.json') and val['explicit_fresh_subprocess'] and val['all127_logit_count']==121920,'QUALIFIED_ORIGINAL127_PREDICTIONS')
 need(fitval['status']=='TRAIN128_FRESH_INDEPENDENT_REFIT_PASS' and fitval['all_three_heads_bit_exact'] and fitval['original7_parameter_sha256']==BASE_PARAMETER_SHA
  and fitval['parameter_seal']==bind(READOUT/'parameter_seal.json') and fitval['parameters']==pseal['parameters']==bind(READOUT/'parameters.json'),'OLD_EC7_FIT_CHAIN')
 _,_,head=fixed_head()
 need(params['parameter_sha256']==BASE_PARAMETER_SHA and params['weight_binary64']==head['weight_binary64'] and params['bias_binary64']==head['bias_binary64'],'PREDICTIONS_ARE_ORIGINAL_EC7')
 previous=read(READOUT/'predictions.json');ops=evaluation_operators();small=[]
 for name,binding in public.items():
  if not (name.startswith('shared_EVAL_') and name.endswith('_metadata')):continue
  meta=read(checked(binding));need(meta['original_C_scalars_bit_exact'] is True,'OLD32_METADATA_PARITY')
  array_path=checked(meta['arrays']);need(array_path in EVAL_PATHS and any(Path(v['path']).resolve()==array_path and v['sha256']==meta['arrays']['sha256'] for v in public.values()),'BOUND_RAW_ARRAY')
  with np.load(array_path,allow_pickle=False) as arrays:raw=arrays['raw_scores'].tolist()
  axis=list(meta['axis']);winner=int(meta['winner']);ranked=sorted(range(128),key=lambda p:(-raw[p],axis[p]))
  need(len(raw)==len(set(axis))==128 and ranked[0]==winner,'OLD32_RAW_AXIS')
  small.append({'query_id':meta['query_id'],'execution_ordinal':meta['execution_ordinal'],'source_image_sha256':meta['query_source_image_sha256'],
   'track':meta.get('track','OLD_PANEL'),'candidate_physical_rows':axis,'base_winner_position':winner,'challenger_positions':[p for p in range(128) if p!=winner],
   'raw_ranked_physical_rows':[axis[p] for p in ranked],'rank_scope':'natural C128'})
 small.sort(key=lambda r:r['execution_ordinal'])
 large=read(checked(public['EVAL128_prejoin_records']));pv=read(checked(public['EVAL128_prejoin_validation']));ps=read(checked(public['EVAL128_prejoin_seal']))
 need(pv['status']=='ORIGINAL7_EVAL128_PREJOIN_INDEPENDENT_CPU_REPLAY_PASS' and pv['records']==ps['records']==public['EVAL128_prejoin_records']
  and pv['seal']==public['EVAL128_prejoin_seal'] and ps['head_parameter_sha256']==BASE_PARAMETER_SHA and all(pv['checks'].values()),'OLD128_TARGET_FREE_SEAL')
 fields=('query_id','execution_ordinal','source_image_sha256','track','candidate_physical_rows','base_winner_position','challenger_positions','raw_ranked_physical_rows')
 banks={'EVAL32':small,'EVAL128':[{**{k:r[k] for k in fields},'rank_scope':'full5412 identity rank'} for r in large]}
 for panel,count in (('EVAL32',32),('EVAL128',128)):
  need(len(banks[panel])==len(previous[panel])==count,'COMPLETE_PANEL')
  for row,old in zip(banks[panel],previous[panel],strict=True):
   need(row['query_id']==old['query_id'] and row['execution_ordinal']==old['execution_ordinal'],'PRIOR_QUERY_ORDER')
   axis=row['candidate_physical_rows'];winner=row['base_winner_position']
   need(len(set(axis))==128 and row['challenger_positions']==[i for i in range(128) if i!=winner],'FULL127_AXIS')
   expected=old['predictions']['ORIGINAL7']['REAL'];values=torch.tensor(list(map(float.fromhex,expected['all127_logits_binary64'])),dtype=torch.float64)
   need(ops['predict'](row,values,True)==expected,'OLD_BASE_ALL127_ACTION_SCHEMA_REPLAY');row['original_prediction']=expected
   if panel=='EVAL128':need(expected==large[row['execution_ordinal']]['predictions']['REAL'],'TWO_PRIOR_EC7_CARRIERS_EXACT')
 return banks


def evaluation_predictions(independent=False):
 params=release_fit();alpha=float.fromhex(params['alpha_binary64']);banks=existing_evaluation_inputs();ops=evaluation_operators();outputs={}
 for panel,rows in banks.items():
  outputs[panel]=[]
  for row in rows:
   baseline=row['original_prediction'];values=torch.tensor(list(map(float.fromhex,baseline['all127_logits_binary64'])),dtype=torch.float64);index=top_index(values,row)
   if independent:
    data=list(map(float,values))
    if data[index]<=0 and alpha>0:data[index]=float(np.add(np.float64(data[index]),np.float64(alpha)))
    new=torch.tensor(data,dtype=torch.float64)
   else:new=lifted(values,row,alpha,1.)
   proposed=ops['predict'](row,new,independent)
   for j in range(127):
    if j!=index or baseline['decision']=='SWITCH' or alpha==0:need(hx(new[j])==baseline['all127_logits_binary64'][j],'LOCKED_SWITCH_AND_OTHER126')
   need(proposed['proposed_challenger']==baseline['proposed_challenger'],'ORIGINAL_TOP_IDENTITY_RETAINED')
   outputs[panel].append({**{k:v for k,v in row.items() if k!='original_prediction'},'predictions':{'ORIGINAL7':baseline,'GAP_HOLD1':proposed}})
 return outputs


def predict_all(replay=False,nonce=None):
 authority=require_authority()
 if replay:need(nonce and os.environ.get('RC_ORIGINAL7_GAP_FRESH_NONCE')==nonce,'FRESH_PREDICTION_NONCE')
 predictions=evaluation_predictions(replay)
 if replay:
  seal=read(OUT/'prediction_seal.json')
  need(seal['authority']==authority and seal['program']==bind(PROGRAM) and seal['parameters']==bind(OUT/'parameters.json')
   and seal['fit_validation']==bind(OUT/'fit_validation.json'),'NEW_PREDICTION_SEAL_LINEAGE')
  need(read(checked(seal['predictions']))==predictions,'FRESH_NUMPY_ALL160_PREDICTIONS_EXACT')
  write(OUT/'prediction_validation.json',{'status':'ORIGINAL7_GAP_ALL160_FRESH_PREDICTIONS_PASS','authority':authority,'program':bind(PROGRAM),
   'prediction_seal':bind(OUT/'prediction_seal.json'),'predictions':seal['predictions'],'fresh_process_nonce':nonce,'all127_logit_count':160*127*2,
   'EVAL_role_outcome_reads':0,'forbidden_read_attempts':len(BLOCKED)})
 else:
  write(OUT/'predictions.json',predictions)
  write(OUT/'prediction_seal.json',{'status':'ORIGINAL7_GAP_ALL160_PREDICTIONS_SEALED_BEFORE_ROLES','authority':authority,'program':bind(PROGRAM),
   'parameters':bind(OUT/'parameters.json'),'fit_validation':bind(OUT/'fit_validation.json'),'predictions':bind(OUT/'predictions.json'),
   'all127_logit_count':160*127*2,'panels':{'EVAL32':32,'EVAL128':128},'EVAL_role_outcome_reads':0,'forbidden_read_attempts':len(BLOCKED)})
  fresh('validate-predictions')
 print(json.dumps({'phase':PHASE,'status':'PASS','new_predictions_sealed':160}),flush=True)


def released_predictions():
 global POST_RELEASED
 val=read(OUT/'prediction_validation.json');seal=read(checked(val['prediction_seal']))
 need(val['status']=='ORIGINAL7_GAP_ALL160_FRESH_PREDICTIONS_PASS' and val['authority']==seal['authority']==bind(AUTH)
  and val['program']==seal['program']==bind(PROGRAM) and val['all127_logit_count']==160*127*2 and val['EVAL_role_outcome_reads']==0,'ALL_NEW_PREDICTIONS_VALIDATED')
 sealed=read(checked(val['predictions']))
 need(val['predictions']==seal['predictions']==bind(OUT/'predictions.json') and sealed==evaluation_predictions(True),'BOTH_PANELS_REPLAY_BEFORE_ROLE_RELEASE')
 POST_RELEASED=True;_,post=evaluation_sources()
 for binding in post.values():checked(binding)
 return sealed,post


def postjoin_roles(banks,post):
 need(POST_RELEASED and PHASE in ('join','validate-result'),'CURATOR_POSTSEAL_RELEASE')
 manifest=read(checked(post['EVAL32_role_manifest']));val=read(checked(post['EVAL32_role_validation']))
 need(manifest['status']=='RGH_P0_A0_ROLE_MANIFEST_READY' and val['status']=='RGH_P0_A0_MANIFEST_V2_INDEPENDENT_VALIDATION_PASS','OLD32_ROLES_QUALIFIED')
 entries={int(e['execution_ordinal']):e for e in manifest['shards']};small={}
 for row in banks['EVAL32']:
  binding=entries[row['execution_ordinal']];p=Path(binding['path']);p=p.resolve() if p.is_absolute() else (ROOT/p).resolve();POST_PATHS.add(p)
  role=read(checked(binding));need(role['query_id']==row['query_id'],'OLD32_ROLE_QUERY')
  small[row['query_id']]={'identity':role['identity'],'group':role['supergroup'],'original_query_id':role['query_id'],'track':role['track']}
 meta=read(checked(post['EVAL128_metadata_validation']));roles=read(checked(post['EVAL128_curator']))
 need(meta['status']=='EXPANDED_EVAL128_INDEPENDENT_METADATA_SELECTION_PASS' and roles['status']=='CURATOR_ONLY_EXPANDED_EVAL128_ROLES_FROZEN'
  and meta['curator_ledger']==post['EVAL128_curator'],'OLD128_CURATOR_CHAIN')
 large={r['query_id']:r for r in roles['records']};need(len(large)==128 and set(large)=={r['query_id'] for r in banks['EVAL128']},'ALL128_CURATOR_SET')
 for row in banks['EVAL128']:
  role=large[row['query_id']];need(role['execution_ordinal']==row['execution_ordinal'] and role['source_image_sha256']==row['source_image_sha256'],'CURATOR_IMAGE_ORDINAL')
 return {'EVAL32':small,'EVAL128':large}


def result_payload():
 authority=require_authority();banks,post=released_predictions();roles=postjoin_roles(banks,post)
 previous=read(checked(post['result.json']));val=read(checked(post['validation.json']))
 need(val['status']=='ORIGINAL7_TRAIN128_READOUT_FRESH_RESULT_REPLAY_PASS' and val['result']==post['result.json']
  and val['authority']==bind(READOUT_AUTH),'QUALIFIED_OLD28_99_RESULT')
 review=read(checked(post['review.json']));need('PASS' in review['status'],'PRIOR_RESULT_INDEPENDENT_REVIEW')
 labels,mapping=input_operators()['gallery_labels']();ops=evaluation_operators();panels={}
 for panel,rows in banks.items():
  actions={m:[] for m in ('ORIGINAL7','GAP_HOLD1')}
  for row in rows:
   role=roles[panel][row['query_id']];report_row=dict(row,track=role.get('track',row['track']))
   for model in actions:actions[model].append(ops['target_action'](report_row,row['predictions'][model],role,labels))
  old={a['query_id']:a for a in previous['actions'][panel]['ORIGINAL7']['REAL']}
  need(set(old)=={a['query_id'] for a in actions['ORIGINAL7']},'SAME_ORIGINAL_QUERY_UNIVERSE')
  for action in actions['ORIGINAL7']:
   before=old[action['query_id']]
   for key in ('final_correct','base_correct','final_physical_row','decision','raw_target_rank_full_gallery','final_target_rank_full_gallery'):
    need(action[key]==before[key],'ORIGINAL_BASELINE_REPLAY:'+panel+':'+key)
   need(hx(action['switch_logit'])==hx(before['switch_logit']),'ORIGINAL_SWITCH_LOGIT_BITS')
  metrics={m:ops['metrics'](a) for m,a in actions.items()};good={a['query_id'] for a in actions['ORIGINAL7'] if a['final_correct']};new={a['query_id'] for a in actions['GAP_HOLD1'] if a['final_correct']}
  need(len(good)==(28 if panel=='EVAL32' else 99) and metrics['ORIGINAL7']['RAW_top1']==(25 if panel=='EVAL32' else 88),'ORIGINAL28_99_RAW25_88_REGRESSION')
  paired=[dict(a,base_correct=b['final_correct']) for a,b in zip(actions['GAP_HOLD1'],actions['ORIGINAL7'],strict=True)];group=ops['primary_statistics'](paired)
  need(group['group_count']==(11 if panel=='EVAL32' else 21),'ALL_ORIGINAL_GROUPS')
  for g in group['groups']:g['baseline_correct']=g.pop('RAW_correct');g['model_correct']=g.pop('REAL_correct')
  group['baseline']='ORIGINAL7';group['model']='GAP_HOLD1'
  panels[panel]={'rank_scope':CONTRACT['MRR_scope'][panel],'metrics':metrics,'RAW':{'correct':metrics['ORIGINAL7']['RAW_top1'],'MRR':metrics['ORIGINAL7']['RAW_full_gallery_MRR']},
   'paired_vs_ORIGINAL7':ops['paired_actions'](actions['GAP_HOLD1'],actions['ORIGINAL7']),'original_correct_count':len(good),'retained_original_correct':len(good&new),
   'lost_original_correct_query_ids':sorted(good-new),'added_query_ids':sorted(new-good),'groups_vs_RAW':{m:ops['primary_statistics'](a) for m,a in actions.items()},
   'groups_GAP_vs_ORIGINAL7':group,'actions':actions}
 gates={'EVAL128_strictly_above99':panels['EVAL128']['metrics']['GAP_HOLD1']['final_top1']>99,
  'EVAL128_all99_retained':panels['EVAL128']['retained_original_correct']==99,'EVAL32_all28_retained':panels['EVAL32']['retained_original_correct']==28}
 return {'status':'ORIGINAL7_GAP_EXPLORATORY_FOLLOWUP_COMPLETE','authority':authority,'program':bind(PROGRAM),'parameters':bind(OUT/'parameters.json'),
  'fit_validation':bind(OUT/'fit_validation.json'),'prediction_validation':bind(OUT/'prediction_validation.json'),'contract':CONTRACT,
  'alpha_binary64':read(OUT/'parameters.json')['alpha_binary64'],'base_parameter_sha256':BASE_PARAMETER_SHA,'panels':panels,'performance_gates':gates,
  'exploratory_performance_goal_pass':all(gates.values()),'new_content_HYP_proved':False,'parent_CONTENT_primary_NO_GO_unchanged':True,
  'source_gallery_mapping_sha256':mapping,'new_encoder_RoMa_calls':0,'new_features':0,'base_refits':0,'optimizer_steps':0,'EVAL_coefficient_adjustments':0,
  'evidence_level':'exploratory internal reuse of already opened TRAIN and both EVAL panels','deployment_changed':False,'forbidden_read_attempts':len(BLOCKED)}


def join(replay=False,nonce=None):
 if replay:need(nonce and os.environ.get('RC_ORIGINAL7_GAP_FRESH_NONCE')==nonce,'FRESH_REDUCER_NONCE')
 result=result_payload()
 if replay:
  need(read(OUT/'result.json')==result,'FRESH_BASELINE_CORRECT_SETS_MRR_GROUPS_REPLAY')
  write(OUT/'validation.json',{'status':'ORIGINAL7_GAP_FRESH_REDUCER_BASELINE_AND_RESULT_PASS','authority':bind(AUTH),'program':bind(PROGRAM),
   'result':bind(OUT/'result.json'),'fresh_process_nonce':nonce,'all160_actions_and_correct_sets_replayed':True,
   'exploratory_performance_goal_pass':result['exploratory_performance_goal_pass'],'EVAL_coefficient_adjustments':0,'forbidden_read_attempts':len(BLOCKED)})
 else:
  write(OUT/'result.json',result);fresh('validate-result')
 print(json.dumps({'phase':PHASE,'exploratory_goal_pass':result['exploratory_performance_goal_pass'],
  'correct':{p:result['panels'][p]['metrics']['GAP_HOLD1']['final_top1'] for p in result['panels']}}),flush=True)
