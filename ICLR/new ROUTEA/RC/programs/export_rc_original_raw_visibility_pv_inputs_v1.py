#!/usr/bin/env python3
"""Export old RAW C128 full visibility maps without consuming adapted D1 values.

Ordinary historical RAW/D1 map archives are opened with weights_only+mmap.
Their dictionary envelopes are deserialized; only RAW-selected visibility
storage is dereferenced. No D1 scores/tensors are accessed or exported. Raw
ColNomic tokens remain references to their separately sealed RAW-only archive.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_original_raw_visibility_pv_inputs_v1'
MIXED=ROOT/'results/routea_d1_current_runtime_matched_three_arm_prejoin_v1'
TOKENS=ROOT/'results/romav2_colnomic_current_runtime_bridge_prejoin_v1'
SCORES=ROOT/'results/romav2_colnomic_current_runtime_bridge_roma_prejoin_v1'
PINS={
 'mixed_validation':(MIXED/'validation.json','60bd07480bf4963503993ecb926b696e3fc568c588189c557c0bc9975f32c66f'),
 'raw_token_validation':(TOKENS/'validation.json','f776472a70ee61a66624d9d750666168fb0c72cd484a248de5cabad77a5a248c'),
 'old_score_validation':(SCORES/'validation.json','7693c5b09f014f1189ad9ee5722b98bb19e0dd68947afadab2f7e556b197ff52'),
 'mixed_producer':(ROOT/'programs/materialize_routea_d1_current_runtime_matched_three_arm_shard_v1.py','ca10470f271da2d6fef5cd59d11ad930c5d9045774774d2caac7df8d1b339383'),
 'raw_token_producer':(ROOT/'programs/materialize_romav2_colnomic_current_runtime_bridge_shard_v1.py','3df9c314df8e1da6fa9b4f60c41e3db7117c6cad3784f35a012c76fdb2fde03e')}
MIXED_RECORD_KEYS={'role','execution_ordinal','query_id','track','heldout_fold','raw_candidate_physical_rows','d1_candidate_physical_rows',
 'union_physical_rows','raw_cbind_destination_to_source_physical_rows','d1_cbind_destination_to_source_physical_rows','candidates',
 'target_role_read_count','target_insertion_count','model_update_count'}
MIXED_CANDIDATE_KEYS={'union_ordinal','physical_row','raw_position','d1_position','reference_tokens_sha256','reference_source_image_sha256',
 'reference_source_logical_sha256','query_map','reference_map','query_map_sha256','reference_map_sha256','raw_arm_scores','d1_arm_scores'}
OLD_SCORE_KEYS={'candidate_position','physical_row','query_control_score','query_map_sha256','raw_score','real_score',
 'reference_control_score','reference_map_sha256','visibility_mass'}

def need(value,message):
    if not bool(value):raise RuntimeError(message)

def safe(path):
    path=Path(path);need(path.resolve().is_relative_to(ROOT) and not path.is_symlink(),'UNSAFE_PATH')
    need(not any(x in str(path).lower() for x in ('d1_mi','d1-mi','d1_minimal_intervention','grozi','gisc_prerecall_universe')),'PROTECTED_PATH')
    return path

def sha(path):
    path=safe(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def encode(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def logical(value):return hashlib.sha256(encode(value)).hexdigest()
def read(path):return json.loads(safe(path).read_text())
def binding(path):return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
def check(path,expected):need(sha(path)==expected,'SOURCE_HASH_DRIFT:'+str(path));return path

def map_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def token_meta(tensor):return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device)}

def json_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(encode(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    path.chmod(0o444)

def sources():
    for path,pin in PINS.values():check(path,pin)
    mixed=read(PINS['mixed_validation'][0]);tokens=read(PINS['raw_token_validation'][0]);scores=read(PINS['old_score_validation'][0])
    need(mixed['status']=='ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_VALIDATED'
       and tokens['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_VALIDATION_PASS'
       and scores['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_VALIDATION_PASS','INPUT_VALIDATION_STATUS')
    for value in (mixed,tokens,scores):
        need(value['query_count']==64 and value['checks'] and all(x is True for x in value['checks'].values()),'INPUT_POPULATION_OR_CHECKS')
    need(mixed['roles']=={'TRAIN':32,'EVAL':32} and tokens['train_count']==scores['train_count']==32
       and tokens['eval_count']==scores['eval_count']==32,'INPUT_ROLE_AXIS')
    return ({int(x['shard']):x for x in v['shards']} for v in (mixed,tokens,scores))

def export():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    need(not OUT.exists(),'APPEND_ONLY_OUTPUT_EXISTS')
    ms,ts,ss=sources();staging=Path(tempfile.mkdtemp(prefix='.'+OUT.name+'.',dir=OUT.parent))
    all_records=[];shards=[];source_bindings={k:binding(v[0]) for k,v in PINS.items()};roles=collections.Counter();count=0
    for shard in range(8):
        mixed_path=MIXED/f'shard{shard:02d}/payload.pt';token_path=TOKENS/f'shard{shard:02d}/payload.pt';score_path=SCORES/f'shard{shard:02d}/result.json'
        check(mixed_path,ms[shard]['payload_sha256']);check(token_path,ts[shard]['payload_sha256']);check(score_path,ss[shard]['sha256'])
        # Opaque package hashes above read bytes; mmap below never inspects D1 score/tensor values.
        mixed=torch.load(mixed_path,map_location='cpu',mmap=True,weights_only=True)
        token=torch.load(token_path,map_location='cpu',mmap=True,weights_only=True)
        old=read(score_path)
        need(mixed['status']=='ROUTEA_D1_CURRENT_RUNTIME_MATCHED_THREE_ARM_PREJOIN_SHARD_READY'
           and token['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_PREJOIN_SHARD_READY'
           and old['status']=='ROMAV2_COLNOMIC_CURRENT_RUNTIME_BRIDGE_ROMA_PREJOIN_SHARD_READY'
           and mixed['shard']==token['shard']==old['shard']==shard,'SHARD_ENVELOPE')
        need(mixed['bindings']['raw_c_authority_sha256']==ss[shard]['sha256']
           and old['bindings']['source_payload_sha256']==ts[shard]['payload_sha256'],'OLD_RAW_LINEAGE')
        mixed_rows={int(r['execution_ordinal']):r for r in mixed['records']}
        old_rows={int(r['execution_ordinal']):r for r in old['rows']}
        token_rows={int(r['execution_ordinal']):(i,r) for i,r in enumerate(token['records'])}
        need(len(mixed_rows)==len(old_rows)==len(token_rows)==8 and set(mixed_rows)==set(old_rows)==set(token_rows),'SHARD_QUERY_AXIS')
        exported=[]
        for execution in sorted(token_rows):
            index,tr=token_rows[execution];mr=mixed_rows[execution];sr=old_rows[execution]
            need(set(mr)==MIXED_RECORD_KEYS,'MIXED_RECORD_SCHEMA_CHANGED')
            need(tr['query_id']==mr['query_id']==sr['query_id'] and tr['role']==mr['role']==sr['role']
               and tr['track']==mr['track']==sr['track'],'QUERY_ROLE_CONTENT_BINDING')
            axis=list(tr['candidate_physical_rows']);old_candidates=sr['candidates']
            need(axis==list(mr['raw_candidate_physical_rows'])==[int(c['physical_row']) for c in old_candidates]
               and axis==sorted(set(axis)) and len(axis)==128,'ORIGINAL_RAW_C128_AXIS_DRIFT')
            needed=set(axis);selected={}
            # Only field names and physical_row are consulted before RAW-axis selection.
            for candidate in mr['candidates']:
                physical=int(candidate['physical_row'])
                if physical in needed:
                    need(set(candidate)==MIXED_CANDIDATE_KEYS and physical not in selected,'RAW_CANDIDATE_SCHEMA')
                    selected[physical]=candidate
            need(set(selected)==needed,'RAW_CANDIDATE_MAP_COVERAGE')
            raw_scores=tr['candidate_raw_scores'].tolist()
            need(raw_scores==[float(x['raw_score']) for x in old_candidates],'OLD_RAW_SCORE_VECTOR_BIT_DRIFT')
            qshape=tuple(tr['query_grid_shape']);qtoken_meta=token_meta(tr['query_tokens'])
            need(qtoken_meta['shape'][0]==math.prod(qshape),'QUERY_GRID_TOKEN_AXIS')
            query_key='raw_query_'+logical({'query_id':tr['query_id'],'execution_ordinal':execution,'query_tokens_sha256':tr['query_tokens_sha256']})
            candidates=[]
            for position,physical in enumerate(axis):
                mapped=selected[physical];saved=old_candidates[position];reference=token['references'][physical]
                need(set(saved)==OLD_SCORE_KEYS and saved['candidate_position']==position and mapped['raw_position']==position,'RAW_POSITION_OR_SCORE_SCHEMA')
                need(reference['physical_row']==physical and mapped['reference_tokens_sha256']==reference['tokens_sha256'],'RAW_REFERENCE_TOKEN_BINDING')
                qv=mapped['query_map'];rv=mapped['reference_map'];rshape=tuple(reference['grid_shape'])
                for values,shape in ((qv,qshape),(rv,rshape)):
                    need(values.dtype==torch.float64 and values.device.type=='cpu' and values.shape==(math.prod(shape),)
                       and bool(torch.isfinite(values).all()) and bool(((values>=0)&(values<=1)).all()),'FULL_VISIBILITY_MAP_DOMAIN')
                qsha,rsha=map_sha(qv),map_sha(rv)
                need(qsha==mapped['query_map_sha256']==saved['query_map_sha256']
                   and rsha==mapped['reference_map_sha256']==saved['reference_map_sha256'],'OLD_MAP_HASH_REPLAY')
                qmean=qv.mean();rmean=rv.mean();mass=torch.sqrt(qmean*rmean)
                need(float(mass).hex()==float(saved['visibility_mass']).hex(),'OLD_VISIBILITY_MASS_NOT_BIT_EXACT')
                source_binding={'query_tokens_sha256':tr['query_tokens_sha256'],'reference_tokens_sha256':reference['tokens_sha256'],
                    'source_maps_sha256':logical({'query_map_sha256':qsha,'reference_map_sha256':rsha}),
                    'raw_token_payload_sha256':ts[shard]['payload_sha256'],'mixed_map_payload_sha256':ms[shard]['payload_sha256'],
                    'old_score_shard_sha256':ss[shard]['sha256']}
                key='raw_gallery_row_'+str(physical).zfill(6)
                candidates.append({'candidate_position':position,'physical_row':physical,'candidate_resource_key':key,'reference_resource_key':key,
                    'reference_grid_shape':list(rshape),'reference_tokens_sha256':reference['tokens_sha256'],
                    'reference_token_metadata':token_meta(reference['tokens']),
                    'query_visibility':qv.clone(),'reference_visibility':rv.clone(),
                    'query_visibility_mean_binary64':float(qmean).hex(),'reference_visibility_mean_binary64':float(rmean).hex(),
                    'visibility_mass_binary64':float(mass).hex(),'query_map_sha256':qsha,'reference_map_sha256':rsha,
                    'source_binding':source_binding,'old_scores':dict(saved)})
                count+=1
            record={'role':tr['role'],'query_id':tr['query_id'],'execution_ordinal':execution,'track':tr['track'],
                'query_resource_key':query_key,'query_grid_shape':list(qshape),'query_tokens_sha256':tr['query_tokens_sha256'],
                'query_token_metadata':qtoken_meta,'token_source':{'path':str(token_path.relative_to(ROOT)),
                    'sha256':ts[shard]['payload_sha256'],'record_index':index,'reference_table_key':'references'},
                'candidate_physical_rows':axis,'candidate_ranked_physical_rows':list(tr['candidate_ranked_physical_rows']),
                'candidate_raw_scores':raw_scores,'candidates':candidates}
            exported.append(record);roles[record['role']]+=1
        payload={'schema':'rc_original_raw_visibility_pv_input_shard_v1','status':'RC_ORIGINAL_RAW_VISIBILITY_PV_INPUT_SHARD_READY',
                 'shard':shard,'records':exported,'source_bindings':{'mixed_map_payload':binding(mixed_path),
                    'raw_token_payload':binding(token_path),'old_scores':binding(score_path)},
                 'claim_level':'RAW_ONLY_LOSSLESS_INTERFACE_EXPORT_NO_NEW_SCIENTIFIC_METHOD'}
        dest=staging/f'shard{shard:02d}';dest.mkdir();torch.save(payload,dest/'payload.pt');(dest/'payload.pt').chmod(0o444)
        payload_hash=sha(dest/'payload.pt')
        receipt={'status':'RC_ORIGINAL_RAW_VISIBILITY_PV_INPUT_SHARD_EXACT_EXPORT_PASS','shard':shard,'query_count':8,
          'candidate_count_per_query':128,'candidate_occurrence_count':1024,'payload_sha256':payload_hash,
          'source_bindings':payload['source_bindings'],'all_original_map_hashes_exact':True,'all_old_visibility_mass_binary64_exact':True,
          'old_RAW_candidate_axis_and_score_vectors_exact':True,'d1_score_or_tensor_fields_accessed':False,
          'new_model_forward_count':0,'training_updates':0,'scientific_GO_or_NO_GO':None}
        json_new(dest/'receipt.json',receipt)
        shard_entry={'shard':shard,'path':f'shard{shard:02d}/payload.pt','sha256':payload_hash,
                     'receipt_path':f'shard{shard:02d}/receipt.json','receipt_sha256':sha(dest/'receipt.json')};shards.append(shard_entry)
        for i,record in enumerate(exported):all_records.append({'shard':shard,'record_index':i,'query_id':record['query_id'],
          'execution_ordinal':record['execution_ordinal'],'query_resource_key':record['query_resource_key'],'role':record['role'],
          'payload_path':shard_entry['path'],'payload_sha256':payload_hash})
        print(json.dumps({'event':'RAW_VISIBILITY_EXPORT_SHARD_CLOSED','shard':shard,'queries':len(all_records),'candidates':count}),flush=True)
        del mixed,token,old,payload,exported,selected
    need(roles=={'TRAIN':32,'EVAL':32} and len(all_records)==64 and len({x['query_id'] for x in all_records})==64
       and len({x['execution_ordinal'] for x in all_records})==64 and count==8192,'FULL64_RAW_C128_CLOSURE')
    manifest={'schema':'rc_original_raw_visibility_pv_inputs_v1','status':'RC_ORIGINAL_RAW_VISIBILITY_PV_INPUTS_FULL64_EXACT_EXPORT_PASS',
      'claim_level':'LOSSLESS_ORIGINAL_RAW_P_V_INPUT_INTERFACE_REPAIR_ONLY','query_count':64,'roles':dict(roles),
      'candidate_count_per_query':128,'candidate_occurrence_count':8192,'shards':shards,'records':sorted(all_records,key=lambda x:x['execution_ordinal']),
      'sources':source_bindings,'exporter':binding(Path(__file__)),
      'checks':{'all_original_RAW_C128_axes_and_RAW_scores_exact':True,'all8192_query_and_reference_visibility_map_hashes_exact':True,
        'all8192_full_bilateral_visibility_mass_binary64_exact':True,'RAW_tokens_referenced_in_original_sealed_RAW_archives':True,
        'complete_TRAIN32_EVAL32_without_target_join':True,'no_protected_source_or_new_candidate_axis':True},
      'access':{'ordinary_mixed_package_mmap_deserializations':8,'mixed_dictionary_envelopes_deserialized':True,
        'd1_score_or_tensor_fields_accessed':False,'d1_values_exported':False,'raw_token_tensor_values_recomputed_or_copied':False,
        'raw_token_archive_metadata_read':True,'opaque_source_file_hash_reads':True,'target_label_reads':0,
        'new_model_forward_count':0,'training_updates':0},
      'verification_boundary':'Original RAW axes, scalar records, complete visibility maps and exact mass replay; RAW token/scorer and action parity belong to the separate consumer.',
      'new_scientific_GO_claim':False,'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    json_new(staging/'manifest.json',manifest)
    for path,pin in PINS.values():check(path,pin)
    need(not OUT.exists(),'OUTPUT_APPEARED_DURING_EXPORT');staging.rename(OUT)
    print(json.dumps({'status':manifest['status'],'output':str(OUT),'manifest_sha256':sha(OUT/'manifest.json'),'queries':64,'candidates':8192}),flush=True)

def inspect():
    import torch
    ms,ts,ss=sources();path=MIXED/'shard00/payload.pt';check(path,ms[0]['payload_sha256'])
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=True);record=payload['records'][0]
    need(set(record)==MIXED_RECORD_KEYS,'MIXED_RECORD_SCHEMA_CHANGED')
    physical=record['raw_candidate_physical_rows'][0];candidate=next(x for x in record['candidates'] if x['physical_row']==physical)
    need(set(candidate)==MIXED_CANDIDATE_KEYS,'MIXED_CANDIDATE_SCHEMA_CHANGED')
    print(json.dumps({'status':'RAW_ONLY_MIXED_PACKAGE_SCHEMA_INSPECTION_PASS','query_map_metadata':token_meta(candidate['query_map']),
      'reference_map_metadata':token_meta(candidate['reference_map']),'d1_score_or_tensor_fields_accessed':False,
      'read_policy':'mmap and field whitelist; only selected RAW maps dereferenced, excluded D1 values never accessed'},sort_keys=True))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('inspect','export'),required=True);args=parser.parse_args()
    if args.phase=='inspect':inspect()
    else:export()
if __name__=='__main__':main()
