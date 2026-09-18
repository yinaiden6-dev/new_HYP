#!/usr/bin/env python3
"""Frozen V7 TRAIN32 development; raw-input closure, exact resume, full replay."""
from __future__ import annotations
import argparse, dataclasses, gc, hashlib, importlib.util, json, math, os
from pathlib import Path
import signal, sys, tempfile, time
from typing import Any
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
AUTH_REL = 'registry/rc_grouped_residual_development_authority_v7_20260908.json'
SCHEMA = 'rc_grouped_residual_development_checkpoint_v7'
OLD_CONTRACT_SHA = '46c8cacb95673f3eb3a5887561fa0722ef0de995ccdfaaaabd54be4d0615ea21'
OLD_ORDER_SHA = '66935b39e3ce26296d7fc2f5fe0b8dcd5c3f0862969426347662278dcfaf4977'
TRAINING = {'updates':512,'seed':17,'dtype':'torch.float64','lr':0.03,
            'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0,
            'head_parameters_per_arm':256,'common_eligible_queries':29,
            'masked_updates':'graph_connected_zero_loss_with_AdamW_step',
            'order_contract_sha256':OLD_CONTRACT_SHA,'order_sha256':OLD_ORDER_SHA}
STOP = False

def need(value, message):
    if not bool(value): raise RuntimeError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def logical(value):return hashlib.sha256(encode(value)).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)

def source(root,binding):
    path=root/binding['path']
    need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),'UNSAFE_SOURCE')
    need(not any(t in str(path).lower() for t in ('d1_mi','d1-mi','grozi')),'PROTECTED_SOURCE')
    need(sha(path)==binding['sha256'],'SOURCE_HASH_DRIFT:'+binding['path'])
    return path

def module(root,binding,name):
    path=source(root,binding);spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,'IMPORT_FAILED')
    item=importlib.util.module_from_spec(spec);sys.modules[name]=item;spec.loader.exec_module(item);return item

def read_binding(root,binding):return json.loads(source(root,binding).read_text())

def authority(root,path,*,engineering_draft=False):
    value=json.loads(path.read_text())
    need(value['status']=='RC_GROUPED_RESIDUAL_V7_DEVELOPMENT_AUTHORITY_READY' or (engineering_draft and value['status']=='DRAFT_REQUIRES_ROOT_REVIEW_AND_FREEZE'),'V7_AUTHORITY_NOT_READY')
    need(logical({k:v for k,v in value.items() if k!='logical_sha256'})==value['logical_sha256'],'AUTHORITY_LOGICAL_HASH')
    need(value['training']==TRAINING,'TRAINING_CONTRACT_DRIFT')
    need(value['automatic_stage_advance'] is False and value['formal_outcomes_before_development_go_authorized'] is False,'STAGE_BOUNDARY_DRIFT')
    need(value['new_method_variant_count']==1,'VARIANT_COUNT_DRIFT')
    delayed={value['postjoin']['authority']['path'],value['postjoin']['validation']['path']}
    delayed.update(binding['path'] for binding in value['old_v6'].values())
    for binding in value['sources']:
        if binding['path'] not in delayed:source(root,binding)
    e0=read_binding(root,value['e0'])
    need(e0['status']==value['e0']['expected_status'] and all(e0['checks'].values()),'V7_E0_NOT_CLOSED')
    return value,sha(path)

def load_roles(root,a,contexts,gate):
    joined=read_binding(root,a['postjoin']['authority']);validation=read_binding(root,a['postjoin']['validation'])
    for item in (joined,validation):
        need(logical({k:v for k,v in item.items() if k!='logical_sha256'})==item['logical_sha256'],'POSTJOIN_LOGICAL_HASH')
    need(joined['status']=='RC_LTH_P_ONLY_NATURAL_OPTIMIZATION_POSTJOIN_AUTHORITY_READY'
         and joined['record_count']==32 and joined['fold_id']==1 and joined['target_insertion_count']==0,'POSTJOIN_AUTHORITY_INVALID')
    need(validation['status']=='RC_LTH_P_ONLY_NATURAL_OPTIMIZATION_POSTJOIN_AUTHORITY_INDEPENDENT_VALIDATION_PASS'
         and validation['authority_sha256']==a['postjoin']['authority']['sha256']
         and validation['record_count']==32 and validation['target_absent_count']==0
         and validation['target_insertion_count']==0 and validation['protected_filesystem_access_count']==0
         and validation['checks'] and all(x is True for x in validation['checks'].values()),'POSTJOIN_VALIDATION_INVALID')
    roles={};groups=set()
    for row in joined['records']:
        need(set(row)=={'query_resource_key','target_candidate_resource_keys','supergroup_hash','fold_id'},'POSTJOIN_FIELDS')
        key=row['query_resource_key'];targets=tuple(row['target_candidate_resource_keys']);group=row['supergroup_hash']
        need(key in contexts and key not in roles and row['fold_id']==1 and len(group)==64 and bool(targets)
             and targets==tuple(sorted(set(targets))),'POSTJOIN_ROLE_AXIS')
        axis=tuple(contexts[key].candidate_keys)
        need(all(t in axis for t in targets),'POSTJOIN_TARGET_ABSENT')
        roles[key]=gate.PostjoinProposalRole(target_destination_indices=tuple(axis.index(t) for t in targets),supergroup=group).validate(128)
        groups.add(group)
    need(set(roles)==set(contexts) and len(groups)==12,'POSTJOIN_POPULATION')
    return roles

def prepare(root,path,torch,*,engineering_draft=False):
    a,a_sha=authority(root,path,engineering_draft=engineering_draft)
    adapter=module(root,a['adapter'],'rc_grouped_residual_inputs_v7')
    core=module(root,a['core'],'rc_grouped_residual_core_v7')
    gate=module(root,a['gate'],'rc_grouped_residual_gate_v7')
    bundle=adapter.InputBundle(root,expected_manifest_sha256=a['input_manifest_sha256'])
    contexts={};meta={};summaries=[]
    for episode in bundle.iter_queries():
        context=core.build_context(episode);key=episode['query_resource_key']
        need(key not in contexts and context.query_resource_key==key and tuple(context.candidate_keys)==tuple(episode['candidate_keys']),'CONTEXT_AXIS_DRIFT')
        contexts[key]=context
        meta[key]={'fold':1,'raw_atom_present':tuple(bool(x) for x in episode['valid']['REAL'].any(dim=1))}
        summaries.append(core.context_summary(context))
        print(json.dumps({'status':'V7_CONTEXT_SEALED','query':key,'contexts':len(contexts)}),flush=True)
        del episode
    need(len(contexts)==32,'TRAIN32_CONTEXT_COUNT')
    closure=bundle.prejoin_receipt()
    context_sha=logical(sorted((key,ctx.context_sha256) for key,ctx in contexts.items()))
    # This is the first label-bearing open in the runtime. All 32 contexts already exist.
    roles=load_roles(root,a,contexts,gate)
    sources={'input_manifest_sha256':bundle.manifest_sha256,'input_manifest':bundle.manifest,
             'target_free_prejoin_receipt':closure,'context_axis_sha256':context_sha,
             'context_summaries':sorted(summaries,key=lambda x:x['query_resource_key']),
             'postjoin_after_complete_context_count':len(contexts),'preclosure_postjoin_open_count':bundle.barrier.blocked_read_attempts}
    need(sources['preclosure_postjoin_open_count']==0,'PREJOIN_LABEL_OPEN_ATTEMPT')
    old=read_binding(root,a['old_v6']['result']);valid=read_binding(root,a['old_v6']['validation'])
    old_stats=read_binding(root,a['old_v6']['statistics'])
    need(valid['status']=='RC_COMPETITIVE_WITNESS_V6_DEVELOPMENT_INDEPENDENT_VALIDATION_PASS' and valid['gate_go'] is False and valid['result_sha256']==a['old_v6']['result']['sha256'] and valid['checks'] and all(x is True for x in valid['checks'].values()),'OLD_V6_VALIDATION_BINDING')
    need(logical(old_stats['records'])==old['statistics_sha256'],'OLD_V6_STATISTICS_BINDING')
    old_rows={r['query_resource_key']:r for r in old_stats['records']}
    need(set(old_rows)==set(contexts) and sum(r['query_only_margin']>0 for r in old_rows.values())==22,'OLD_V6_QUERY_CONTROL_DRIFT')
    state=read_binding(root,a['old_v6']['fresh_state'])
    order=gate.deterministic_training_order(tuple(contexts),OLD_CONTRACT_SHA)
    need(logical(list(order))==OLD_ORDER_SHA==state['query_order_sha256'] and state['update_index']==512,'SAVED_V6_QUERY_ORDER_DRIFT')
    # Eligibility is based only on REAL structural availability, before any parameter update.
    eligible={key:any(bool(contexts[key].controls['REAL'].families[i])
                      for i in roles[key].target_destination_indices) for key in sorted(contexts)}
    need(sum(eligible.values())==29,'FROZEN_COMMON_29_QUERY_ELIGIBILITY_DRIFT')
    mask=[bool(eligible[key]) for key in order]
    sources['training_eligibility']={'rule':'REAL_target_has_structural_H1_before_fit_shared_all_arms',
        'query_mask':eligible,'query_mask_sha256':logical(eligible),'update_mask':mask,
        'update_mask_sha256':logical(mask),'eligible_queries':29,'eligible_updates':sum(mask),
        'masked_updates':512-sum(mask),'all_32_evaluated':True}
    del bundle;gc.collect()
    return a,a_sha,core,gate,contexts,roles,meta,sources,order,eligible,old_rows

def initialize(core,torch):
    torch.manual_seed(17);heads=core.GroupedResidualHeads();arms=heads.optimizable_arms()
    need(tuple(arms)==tuple(core.ARM_NAMES),'HEAD_ARM_AXIS')
    for name,head in arms.items():
        params=tuple(head.parameters())
        need(sum(p.numel() for p in params)==256 and all(p.dtype==torch.float64 and not bool(torch.count_nonzero(p.detach())) for p in params),'HEAD_INITIALIZATION_DRIFT:'+name)
    opts={name:torch.optim.AdamW(head.parameters(),lr=.03,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,maximize=False,foreach=False,fused=False) for name,head in arms.items()}
    return heads,opts

def pack(value,torch):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous()
        return {'__tensor__':True,'dtype':str(t.dtype),'shape':list(t.shape),'hex':t.numpy().tobytes().hex(),'byte_order':sys.byteorder}
    if isinstance(value,dict):return {'__dict__':[[pack(k,torch),pack(v,torch)] for k,v in value.items()]}
    if isinstance(value,tuple):return {'__tuple__':[pack(v,torch) for v in value]}
    if isinstance(value,list):return [pack(v,torch) for v in value]
    need(value is None or isinstance(value,(str,int,float,bool)),'CHECKPOINT_UNSUPPORTED_TYPE');return value

def unpack(value,torch):
    if isinstance(value,list):return [unpack(v,torch) for v in value]
    if not isinstance(value,dict):return value
    if '__tensor__' in value:
        need(value['byte_order']==sys.byteorder,'CHECKPOINT_BYTE_ORDER')
        dtype=getattr(torch,value['dtype'].split('.')[-1]);data=bytearray.fromhex(value['hex'])
        return torch.frombuffer(data,dtype=dtype).clone().reshape(value['shape']) if data else torch.empty(value['shape'],dtype=dtype)
    if '__dict__' in value:return {unpack(k,torch):unpack(v,torch) for k,v in value['__dict__']}
    if '__tuple__' in value:return tuple(unpack(v,torch) for v in value['__tuple__'])
    raise RuntimeError('CHECKPOINT_ENCODING_INVALID')

def checkpoint(heads,opts,index,contract_sha,order_sha,eligibility_sha,histories,gradients,torch):
    return encode({'schema':SCHEMA,'update_index':index,'contract_sha256':contract_sha,'order_sha256':order_sha,
        'eligibility_sha256':eligibility_sha,'model':pack(heads.state_dict(),torch),
        'optimizers':{name:pack(opt.state_dict(),torch) for name,opt in opts.items()},
        'histories':histories,'finite_gradients':gradients})

def restore(data,heads,opts,contract_sha,order_sha,eligibility_sha,torch):
    value=json.loads(data)
    need(value['schema']==SCHEMA and value['contract_sha256']==contract_sha and value['order_sha256']==order_sha
         and value['eligibility_sha256']==eligibility_sha,'CHECKPOINT_CONTRACT_DRIFT')
    heads.load_state_dict(unpack(value['model'],torch),strict=True)
    need(set(value['optimizers'])==set(opts),'CHECKPOINT_OPTIMIZER_AXIS')
    for name,opt in opts.items():opt.load_state_dict(unpack(value['optimizers'][name],torch))
    need(checkpoint(heads,opts,value['update_index'],contract_sha,order_sha,eligibility_sha,value['histories'],value['finite_gradients'],torch)==data,'CHECKPOINT_ROUNDTRIP_BYTE_DRIFT')
    return value['update_index'],value['histories'],value['finite_gradients']

def arm_for_loss(raw,role,core,gate):
    output={};targets=tuple(role.target_destination_indices)
    for name in core.ARM_NAMES:
        arm=raw['arms'][name];available=tuple(bool(x) for x in arm['available_mask'])
        output[name]=gate.arm_forward_from_scores(name=name,scores=arm['scores'],c_bind_scores=arm['c_bind_scores'],
            p_coord_scores=arm['p_coord_scores'],target_indices=targets,opaque_keys=raw['candidate_keys'],
            target_witness_score=arm['scores'][list(targets)].max(),target_witness_missing=not any(available[i] for i in targets),
            h1_decisions=available if name==core.ARM_REAL else None)
    return output

def loss(raw,role,eligible,core,gate):
    arms=arm_for_loss(raw,role,core,gate)
    return {name:(gate.five_term_loss(arms[name],role.target_destination_indices).total if eligible
                  else raw['arms'][name]['scores'].sum()*0.0) for name in core.ARM_NAMES}

def statistic(raw,role,meta,core,gate):
    arms=arm_for_loss(raw,role,core,gate);targets=tuple(role.target_destination_indices)
    present=tuple(bool(x) for x in raw['arms'][core.ARM_REAL]['available_mask']);real=arms[core.ARM_REAL]
    return gate.EpisodeStatistic(query_resource_key=raw['query_resource_key'],fold=meta['fold'],supergroup=role.supergroup,
        target_present=True,raw_target_atom_present=any(meta['raw_atom_present'][i] for i in targets),
        selected_target_connected_h1_present=any(present[i] for i in targets),selected_target_positive_h1_present=any(present[i] for i in targets),
        h1_candidate_count=sum(present),candidate_count=len(raw['candidate_keys']),real_margin=float(real.margin.detach()),
        allpatch_margin=float(arms[core.ARM_ALLPATCH].margin.detach()),query_only_margin=float(arms[core.ARM_QUERY_ONLY].margin.detach()),
        c_bind_margin=float(real.c_bind_margin.detach()),p_coord_margin=float(real.p_coord_margin.detach()))

def train(root,path,lane,max_seconds,torch):
    start=time.monotonic();a,a_sha,core,gate,contexts,roles,meta,sources,order,eligible,old_rows=prepare(root,path,torch)
    work=root/a['work_rel'];state_path=work/(lane+'_state.json');heads,opts=initialize(core,torch)
    contract_sha=a['contract']['sha256'];mask_sha=sources['training_eligibility']['update_mask_sha256']
    history={name:[] for name in core.ARM_NAMES};gradients={name:0 for name in core.ARM_NAMES}
    index=loads=0;forced=False;milestone=None
    if state_path.exists():
        old=json.loads(state_path.read_text());need(old['authority_sha256']==a_sha and old['context_axis_sha256']==sources['context_axis_sha256'],'RESUME_INPUT_DRIFT')
        data=bytes.fromhex(old['checkpoint_hex']);need(hashlib.sha256(data).hexdigest()==old['checkpoint_sha256'],'RESUME_CHECKPOINT_HASH')
        index,history,gradients=restore(data,heads,opts,contract_sha,OLD_ORDER_SHA,mask_sha,torch)
        loads=old['resume_load_count']+1;forced=old['forced_resume_at_256'];milestone=old['checkpoint_256_sha256']
    atomic_json(work/(lane+'_inputs.json'),{'authority_sha256':a_sha,**sources})
    while index<512 and not STOP and time.monotonic()-start<max_seconds:
        key=order[index]
        for opt in opts.values():opt.zero_grad(set_to_none=True)
        raw=core.score_episode(contexts[key],heads);losses=loss(raw,roles[key],eligible[key],core,gate)
        sum(losses.values()).backward()
        for name,head in heads.optimizable_arms().items():
            need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()),'NONFINITE_OR_MISSING_GRADIENT')
            if not eligible[key]:need(all(not bool(torch.count_nonzero(p.grad)) for p in head.parameters()),'MASKED_UPDATE_NONZERO_GRADIENT')
            opts[name].step();need(all(bool(torch.isfinite(p).all()) for p in head.parameters()),'NONFINITE_PARAMETER')
            history[name].append(float(losses[name].detach()));gradients[name]+=1
        index+=1;data=checkpoint(heads,opts,index,contract_sha,OLD_ORDER_SHA,mask_sha,history,gradients,torch);cp_sha=hashlib.sha256(data).hexdigest()
        if index==256:
            milestone=cp_sha
            if lane=='resume' and not forced:
                heads,opts=initialize(core,torch);index,history,gradients=restore(data,heads,opts,contract_sha,OLD_ORDER_SHA,mask_sha,torch)
                loads+=1;forced=True
        state={'schema':'rc_grouped_residual_development_state_v7','lane':lane,'status':'COMPLETE' if index==512 else 'RESUMABLE',
            'update_index':index,'authority_sha256':a_sha,'context_axis_sha256':sources['context_axis_sha256'],
            'query_order_sha256':OLD_ORDER_SHA,'training_eligibility_sha256':mask_sha,'checkpoint_hex':data.hex(),
            'checkpoint_sha256':cp_sha,'checkpoint_256_sha256':milestone,'resume_load_count':loads,'forced_resume_at_256':forced,
            'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
        atomic_json(state_path,state)
        if index%16==0:print(json.dumps({'status':'V7_TRAIN_UPDATE','lane':lane,'update':index,'elapsed_seconds':round(time.monotonic()-start,2),'checkpoint_sha256':cp_sha,'loss':{name:history[name][-1] for name in core.ARM_NAMES}}),flush=True)
        del raw,losses
    need(state_path.exists(),'NO_TRAIN_UPDATE_COMPLETED');final=json.loads(state_path.read_text())
    print(json.dumps({'status':final['status'],'lane':lane,'update_index':index,'state':str(state_path)}),flush=True)
    return final

def vectors(raw,core,torch):
    for name in core.ARM_NAMES:
        arm=raw['arms'][name]
        for field in ('scores','c_bind_scores','p_coord_scores'):
            yield {'arm':name,'field':field},arm[field]
        for field in ('decisions','c_bind_decisions','p_coord_decisions'):
            for decision in arm[field]:
                yield {'arm':name,'field':field,'candidate_resource_key':decision.candidate_resource_key},decision.complete_seed_scores
        selectors=arm.get('query_selector_complete_scores')
        if selectors is not None:
            for control,value in (selectors.items() if isinstance(selectors,dict) else [('REAL',selectors)]):
                yield {'arm':name,'field':'query_selector_complete_scores','control':control},value
    selectors=raw.get('query_selector_complete_scores')
    if selectors is not None:
        for control,value in (selectors.items() if isinstance(selectors,dict) else [('REAL',selectors)]):
            yield {'arm':core.ARM_QUERY_ONLY,'field':'query_selector_complete_scores','control':control},value

def replay(root,path,mode,torch):
    a,a_sha,core,gate,contexts,roles,meta,sources,order,eligible,old_rows=prepare(root,path,torch)
    work=root/a['work_rel'];output=root/a['output_rel'];fresh,resumed=(json.loads((work/(name+'_state.json')).read_text()) for name in ('fresh','resume'))
    mask_sha=sources['training_eligibility']['update_mask_sha256']
    for state in (fresh,resumed):
        need(state['status']=='COMPLETE' and state['update_index']==512 and state['authority_sha256']==a_sha and state['context_axis_sha256']==sources['context_axis_sha256'] and state['training_eligibility_sha256']==mask_sha,'DUAL_LANE_NOT_COMPLETE')
    need(resumed['forced_resume_at_256'] and fresh['checkpoint_hex']==resumed['checkpoint_hex'] and fresh['checkpoint_256_sha256']==resumed['checkpoint_256_sha256'],'FRESH_RESUME_PARITY')
    data=bytes.fromhex(fresh['checkpoint_hex']);need(hashlib.sha256(data).hexdigest()==fresh['checkpoint_sha256']==resumed['checkpoint_sha256'],'FINAL_CHECKPOINT_HASH')
    heads,opts=initialize(core,torch);index,histories,gradients=restore(data,heads,opts,a['contract']['sha256'],OLD_ORDER_SHA,mask_sha,torch)
    need(index==512 and all(v==512 for v in gradients.values()),'UPDATE_COUNT');heads.eval()
    if mode=='reduce':
        need(not output.exists(),'APPEND_ONLY_OUTPUT_EXISTS');output.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent));stream=(staging/'complete_score_vectors.bin').open('wb')
    else:staging=None;stream=(output/'complete_score_vectors.bin').open('rb')
    records=[];score_rows=[];vector_index=[];digest=hashlib.sha256();offset=0;pc_empty=0
    with torch.no_grad():
        for key in sorted(contexts):
            raw=core.score_episode(contexts[key],heads);deployed=core.score_episode(contexts[key],heads)
            serialized=core.serialize_episode_scores(raw);need(encode(serialized)==encode(core.serialize_episode_scores(deployed)),'REPEATED_DEPLOY_SEAL_DRIFT')
            left=list(vectors(raw,core,torch));right=list(vectors(deployed,core,torch))
            selectors=[d for d,t in left if d['field']=='query_selector_complete_scores']
            need(len(selectors)==3 and {d['control'] for d in selectors}=={'REAL','C_BIND','P_COORD'},'COMPLETE_QUERY_SELECTOR_VECTORS_REQUIRED')
            need(len(left)==len(right),'VECTOR_AXIS_DRIFT')
            for (description,tensor),(other,twin) in zip(left,right,strict=True):
                need(description==other and torch.equal(tensor,twin),'DEPLOY_COMPLETE_VECTOR_DRIFT')
                tensor=tensor.detach().cpu().contiguous();need(tensor.dtype==torch.float64 and bool(torch.isfinite(tensor).all()),'VECTOR_DTYPE_OR_FINITE')
                binary=tensor.numpy().tobytes()
                if mode=='reduce':stream.write(binary)
                else:need(stream.read(len(binary))==binary,'INDEPENDENT_VECTOR_BYTES_MISMATCH')
                digest.update(binary);vector_index.append({'query_resource_key':key,**description,'offset_bytes':offset,'shape':list(tensor.shape),'value_count':tensor.numel(),'dtype':'float64','byte_order':sys.byteorder,'binary_sha256':hashlib.sha256(binary).hexdigest()});offset+=len(binary)
            pc_empty+=int(not any(bool(d.h1) for d in raw['arms'][core.ARM_REAL]['p_coord_decisions']))
            records.append(statistic(raw,roles[key],meta[key],core,gate));score_rows.append(serialized)
    if mode=='reduce':stream.flush();os.fsync(stream.fileno())
    else:need(stream.read(1)==b'','VECTOR_TRAILING_BYTES')
    stream.close();index_bytes=encode(vector_index)+b'\n'
    if mode=='reduce':(staging/'complete_score_vectors.index.json').write_bytes(index_bytes)
    else:need((output/'complete_score_vectors.index.json').read_bytes()==index_bytes,'VECTOR_INDEX_DRIFT')
    reduced=gate.reduce_natural_gate(records,stage='optimization');statistics=[dataclasses.asdict(r) for r in records]
    rescues=sum(r.real_margin>0 and old_rows[r.query_resource_key]['query_only_margin']<=0 for r in records)
    breaks=sum(r.real_margin<=0 and old_rows[r.query_resource_key]['query_only_margin']>0 for r in records)
    old_control={'baseline_success':22,'rescue':rescues,'break_count':breaks,'paired_net':rescues-breaks,'required_net':4,'pass':rescues-breaks>=4 and rescues>breaks}
    failures=list(reduced.failures)
    if not old_control['pass']:failures.append('P_FROZEN_V6_QUERY_ONLY_INCREMENT_NO_GO')
    result={'schema':'rc_grouped_residual_development_result_v7','status':'RC_GROUPED_RESIDUAL_V7_DEVELOPMENT_COMPLETE',
        'claim_level':'POSTHOC_TRAIN32_DEVELOPMENT_ONLY','authority_sha256':a_sha,'core_sha256':a['core']['sha256'],
        'final_checkpoint_sha256':hashlib.sha256(data).hexdigest(),'context_axis_sha256':sources['context_axis_sha256'],
        'gate_go':not failures,'all_simultaneous_failures':failures,'matched_gate_reduction':dataclasses.asdict(reduced),
        'frozen_v6_query_only_control':old_control,'statistics_sha256':logical(statistics),'score_records_sha256':logical(score_rows),
        'complete_vector_artifacts':{'binary_sha256':digest.hexdigest(),'index_sha256':hashlib.sha256(index_bytes).hexdigest(),'bytes':offset,'vector_count':len(vector_index),'dtype':'float64','byte_order':sys.byteorder},
        'fresh_resume_checkpoint_byte_equal':True,'checkpoint_256_byte_equal':True,'training_updates':512,'finite_gradient_updates':gradients,
        'input_closure':sources,'p_coord_structural_degeneracy':{'all_H0_query_count':pc_empty,'query_count':32,'spatial_identity_causality_established_by_this_control':False,'interpretation':'PCOORD removal of all proposals tests generator dependence; it does not isolate identity spatial relations'},
        'automatic_stage_advance':False,'formal_panel_consumed':False,'scientific_GO_or_NO_GO':None}
    if mode=='reduce':
        (staging/'final_checkpoint.json').write_bytes(data);atomic_json(staging/'statistics.json',{'record_count':32,'records':statistics})
        (staging/'score_records.jsonl').write_bytes(b''.join(encode(row)+b'\n' for row in score_rows));atomic_json(staging/'result.json',result)
        staging.rename(output);print(json.dumps({'status':result['status'],'gate_go':result['gate_go'],'failures':failures,'output':str(output)}),flush=True);return result
    need(encode(json.loads((output/'result.json').read_text()))==encode(result) and (output/'final_checkpoint.json').read_bytes()==data,'INDEPENDENT_FINAL_REPLAY_MISMATCH')
    need(encode(json.loads((output/'statistics.json').read_text()))==encode({'record_count':32,'records':statistics}),'INDEPENDENT_STATISTICS_REPLAY_MISMATCH')
    need(encode([json.loads(line) for line in (output/'score_records.jsonl').read_text().splitlines()])==encode(score_rows),'INDEPENDENT_SCORE_RECORDS_REPLAY_MISMATCH')
    validation_output=root/a['validation_rel'];need(not validation_output.exists(),'APPEND_ONLY_VALIDATION_EXISTS');validation_output.mkdir(parents=True)
    validation={'schema':'rc_grouped_residual_development_validation_v7','status':'RC_GROUPED_RESIDUAL_V7_DEVELOPMENT_INDEPENDENT_VALIDATION_PASS',
        'claim_level':'INDEPENDENT_RAW_INPUT_CHECKPOINT_FULL_SCORE_REPLAY','gate_go':result['gate_go'],'all_simultaneous_failures':failures,
        'result_sha256':sha(output/'result.json'),'authority_sha256':a_sha,'checkpoint_sha256':hashlib.sha256(data).hexdigest(),
        'checks':{'all_raw_inputs_reloaded_and_contexts_rebuilt':True,'all_32_C128_statistics_replayed':True,'complete_score_and_query_selection_vectors_bytes_replayed':True,'fresh_resume_checkpoints_byte_equal':True,'same_saved_v6_512_query_order':True,'common_29_eligibility_replayed':True,'frozen_old_query_only_control_replayed':True,'canonical_JSON_comparison':True,'postjoin_after_all_contexts':True,'coordinate_control_degeneracy_disclosed':True},
        'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    atomic_json(validation_output/'result.json',validation);print(json.dumps(validation,sort_keys=True),flush=True);return validation

def synthetic_checkpoint_test(torch):
    class Core:
        ARM_NAMES=('REAL','ALL_PATCH_NO_HYP','QUERY_ONLY_REGION')
        class GroupedResidualHeads(torch.nn.Module):
            def __init__(self):
                super().__init__();self.real=torch.nn.Linear(256,1,bias=False,dtype=torch.float64);self.allpatch=torch.nn.Linear(256,1,bias=False,dtype=torch.float64);self.query_only=torch.nn.Linear(256,1,bias=False,dtype=torch.float64)
                for p in self.parameters():torch.nn.init.zeros_(p)
            def optimizable_arms(self):return dict(zip(Core.ARM_NAMES,(self.real,self.allpatch,self.query_only)))
    def lane(resume):
        heads,opts=initialize(Core,torch);history={k:[] for k in Core.ARM_NAMES};grads={k:0 for k in Core.ARM_NAMES};boundary=None
        for i in range(512):
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            x=torch.arange(1,257,dtype=torch.float64)/(i+257)
            masked=i%32 in (2,10,21)
            losses={name:(head(x).square().sum()+head(x).sum()+1 if not masked else head(x).sum()*0) for name,head in heads.optimizable_arms().items()}
            sum(losses.values()).backward()
            for name,head in heads.optimizable_arms().items():
                need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()),'SYNTHETIC_GRADIENT')
                if masked:need(all(not bool(torch.count_nonzero(p.grad)) for p in head.parameters()),'SYNTHETIC_MASK')
                opts[name].step();history[name].append(float(losses[name].detach()));grads[name]+=1
            data=checkpoint(heads,opts,i+1,'c'*64,'o'*64,'m'*64,history,grads,torch)
            if i==255:
                boundary=data
                if resume:
                    heads,opts=initialize(Core,torch);index,history,grads=restore(data,heads,opts,'c'*64,'o'*64,'m'*64,torch);need(index==256,'SYNTHETIC_INDEX')
        return boundary,data
    fresh=lane(False);resumed=lane(True);need(fresh==resumed,'SYNTHETIC_512_FRESH_RESUME_NOT_EXACT')
    return {'status':'V7_SYNTHETIC_512_CHECKPOINT_PARITY_PASS','checks':{'model_and_all_AdamW_tensor_bits':True,'256_boundary':True,'512_final':True,'masked_zero_grad_steps_retained':True},'checkpoint_256_sha256':hashlib.sha256(fresh[0]).hexdigest(),'checkpoint_512_sha256':hashlib.sha256(fresh[1]).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--authority',type=Path);p.add_argument('--mode',choices=('sources','prepare','train','reduce','validate','synthetic-checkpoint','engineering-smoke'),required=True);p.add_argument('--out',type=Path);p.add_argument('--lane',choices=('fresh','resume'),default='fresh');p.add_argument('--max-seconds',type=int,default=3200);args=p.parse_args();root=args.root.resolve();path=args.authority or root/AUTH_REL
    if args.mode=='sources':
        a,digest=authority(root,path);print(json.dumps({'status':'V7_DEVELOPMENT_SOURCE_PREFLIGHT_PASS','authority_sha256':digest,'sources':len(a['sources']),'label_or_prior_outcome_sources_deferred_until_full_context_closure':6}));return
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.mode=='synthetic-checkpoint':print(json.dumps(synthetic_checkpoint_test(torch),sort_keys=True));return
    def stop(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop);need(60<=args.max_seconds<=3300,'WORKER_TIME_BUDGET')
    if args.mode=='engineering-smoke':
        need(args.out is not None and not args.out.exists(),'APPEND_ONLY_SMOKE_OUTPUT_REQUIRED')
        a,a_sha=authority(root,path,engineering_draft=True)
        adapter=module(root,a['adapter'],'rc_grouped_residual_inputs_v7')
        core=module(root,a['core'],'rc_grouped_residual_core_v7')
        gate=module(root,a['gate'],'rc_grouped_residual_gate_v7')
        bundle=adapter.InputBundle(root,expected_manifest_sha256=a['input_manifest_sha256'])
        episode=next(bundle.iter_queries());context=core.build_context(episode)
        heads,opts=initialize(core,torch)
        role=gate.PostjoinProposalRole(target_destination_indices=(0,),supergroup='f'*64).validate(128)
        meta={'fold':1,'raw_atom_present':tuple(bool(x) for x in episode['valid']['REAL'].any(dim=1))}
        with torch.no_grad():
            raw=core.score_episode(context,heads)
            stat=statistic(raw,role,meta,core,gate)
            listed=list(vectors(raw,core,torch))
            selectors=[d for d,t in listed if d['field']=='query_selector_complete_scores']
            need(len(selectors)==3 and {d['control'] for d in selectors}=={'REAL','C_BIND','P_COORD'},'SMOKE_COMPLETE_SELECTOR_AXIS')
            need(all(t.dtype==torch.float64 and bool(torch.isfinite(t).all()) for d,t in listed),'SMOKE_VECTOR_FINITE')
            need(stat.candidate_count==128 and stat.fold==1,'SMOKE_STATISTIC_SCHEMA')
            need(encode(core.serialize_episode_scores(raw))==encode(core.serialize_episode_scores(core.deploy_episode(context,heads))),'SMOKE_DEPLOY_PARITY')
        need(not bundle.barrier.closed and bundle.barrier.blocked_read_attempts==0,'SMOKE_LABEL_BARRIER_NOT_CLOSED')
        receipt={'status':'RC_GROUPED_RESIDUAL_V7_RUNNER_ONE_QUERY_SMOKE_PASS','claim_level':'ENGINEERING_ONLY_SYNTHETIC_ROLE_NO_TARGET_READS',
                 'authority_sha256':a_sha,'core_sha256':a['core']['sha256'],'runner_sha256':sha(Path(__file__)),
                 'context_sha256':context.context_sha256,'input_manifest_sha256':bundle.manifest_sha256,
                 'query_count_closed':1,'candidate_count_each':128,'full_32_closure_performed':False,
                 'scored_query_count':1,'score_vector_count':len(listed),'query_selector_vector_count':len(selectors),
                 'checks':{'source_manifest_bound':True,'real_full_C128_query_loaded':True,'synthetic_role_only':True,
                           'postjoin_label_barrier_unreleased':True,'score_vectors_FP64_finite':True,
                           'statistic_conversion':True,'deploy_serialization_parity':True},
                 'optimizer_step_count':0,'natural_gate_reduction_count':0,'actual_target_reads':0,
                 'scientific_GO_or_NO_GO':None,'automatic_stage_advance':False}
        atomic_json(args.out,receipt);print(json.dumps(receipt,sort_keys=True));return
    if args.mode=='prepare':
        values=prepare(root,path,torch);print(json.dumps({'status':'V7_RAW_INPUTS_AND_COMMON_MASK_PREPARED','source_record':values[7]},sort_keys=True));return
    if args.mode=='train':train(root,path,args.lane,args.max_seconds,torch)
    else:replay(root,path,args.mode,torch)
if __name__=='__main__':main()
