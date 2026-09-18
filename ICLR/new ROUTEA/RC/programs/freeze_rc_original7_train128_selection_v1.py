#!/usr/bin/env python3
"""Metadata-only TRAIN128: preserve FULL32+PAIR64, add32 by fixed group counts."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import hashlib, heapq, json, os, re, subprocess, sys, uuid
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
OUT = ROOT / 'results/rc_original7_train128_manifest_v1'
SEED = 'RC_ORIGINAL7_TRAIN128_20260910'
PATHS = {
 'exclusions': 'registry/rc_eval128_identity_group_exclusion_v1_20260910.json',
 'formal_negative': 'registry/rc_eval128_formal392_negative_exclusion_v1_20260910.json',
 'identity_index': 'results/l0_natural_hardneg_v_representation_v2_identityfix/target_join/query_target_join_index.json',
 'query_ledger': 'cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json',
 'legacy_token_index': 'cache/l0_natural_hardneg_v2_redacted_query_tokens_v1/index_v1.json',
 'old_training_manifest': 'results/rc_shared_query_target_prior_cache_v1/manifest.json',
 'eval128_worker': 'results/rc_original7_expanded_eval128_manifest_v1/worker_manifest.json',
 'eval128_curator': 'results/rc_original7_expanded_eval128_manifest_v1/curator_roles.json',
 'eval128_validation': 'results/rc_original7_expanded_eval128_manifest_v1/independent_metadata_validation.json',
 'RAW600_membership': 'protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json',
 'old_cache_producer': 'programs/prepare_rc_shared_query_target_prior_cache_v1.py',
 'old_training_loader': 'programs/run_rc_absolute_evidence_scale_calibration_v1.py',
 'old_runtime_token_validation': 'results/romav2_colnomic_current_runtime_bridge_prejoin_v1/validation.json',
}
HASHES = ('e1ee298521ab9dfad5f8611b0011b5aeea9808055e50eea156288eb07f3bd0df',
 '0e09f9bbf33998f1553d0b3aedd4a6d28ced9da3c57da6c45d7013abda81e243',
 'f99b08737791f9f30329a2b6388cb76421a4ffb6642ce88b2fcb42aca767eb9b',
 'df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec',
 '2a78dbde7d375f5f3640ad523b08c813a436ef0b0549dcea92960f73b1f0f26d',
 'f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c',
 'f2c2e6f9181e103a61a181edf43e3309aa1de72743cfa6cf6097a9deb108e4e7',
 '47277ce66f7aba445fb80a93643039c7e8a47c1f5cc8237f60157a98e3e93381',
 'a70936a24fc2f5c6996228aee673a6de505c3022487b903878d268ad9a34cd94',
 '44df51fe97a8db6de1a1101d0b310320da398d1eb88dd21b282c33cbcc6c6bc8',
 'be92313a0597c7f924f52040512eedd55afcb6a04d7183927a97b2f07d516de0',
 '09de5883aafc8625a56d13b99560cd3cc8b02a6e9dc370851bf351480ac29404',
 'f776472a70ee61a66624d9d750666168fb0c72cd484a248de5cabad77a5a248c')
TARGET_FIELDS = {'version','contract_sha256','source_query_ledger_logical_sha256','query_ordinal','query_id','identity','track','group_id','target_join_performed','opened_runtime_read_count','sealed_runtime_read_count','a10_runtime_read_count','home_files_modified','logical_sha256'}
WORKER_FIELDS = {'query_id','execution_ordinal','track','query_image_path','source_image_sha256','grid_hw','redacted_token_artifact','redacted_token_artifact_declared_sha256','image_tokens_sha256','template_tokens_sha256'}
BLOCKED = []

def need(value, message):
 if not bool(value): raise RuntimeError(message)
def audit(event, args):
 if event != 'open' or not args or not isinstance(args[0], (str,bytes,os.PathLike)): return
 p = Path(os.fsdecode(args[0])); s = str(p).lower()
 forbidden = p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.pt','.npz','.npy')
 forbidden |= any(t in s for t in ('d1_mi','d1-mi','grozi','gisc_prerecall_universe','/rc_opened_eval_strict_'))
 forbidden |= '/rc_original7_eval128_full_evidence_v1/' in s or '/reports/' in s
 if forbidden: BLOCKED.append(str(p)); raise PermissionError('METADATA_ONLY_NO_PIXEL_TENSOR_PREDICTION_READ')
def encode(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): return {'path':str(Path(path).absolute()),'sha256':sha(path)}
def read(path): return json.loads(Path(path).read_text())
def path_of(binding):
 p=Path(binding['path']); return p if p.is_absolute() else ROOT/p
def write(path, value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream: stream.write(encode(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
 path.chmod(0o444)
def key(domain, text): return hashlib.sha256((SEED+'|'+domain+'|'+text).encode('utf-8')).hexdigest()
def sources():
 result={}
 for (name,rel),digest in zip(PATHS.items(),HASHES,strict=True):
  p=ROOT/rel; need(sha(p)==digest,'SOURCE_PIN:'+name); result[name]=bind(p)
 result['program']=bind(PROGRAM); return result
def label_record(index, entry):
 p=ROOT/Path(PATHS['identity_index']).parent/entry['path']; value=read(p)
 need(set(value)==TARGET_FIELDS,'LABEL_METADATA_ONLY_SCHEMA')
 need(value['query_id']==entry['query_id'] and value['query_ordinal']==entry['query_ordinal'],'LABEL_QUERY_ID_ADDRESS')
 need(value['contract_sha256']==index['contract_sha256'] and value['source_query_ledger_logical_sha256']==index['source_query_ledger_logical_sha256'],'LABEL_PROVENANCE')
 logical=hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
 need(logical==value['logical_sha256']==entry['record_logical_sha256'],'LABEL_LOGICAL_SHA')
 return value,bind(p)

def inputs(independent=False):
 docs={k:read(ROOT/v) for k,v in PATHS.items() if not k.startswith('old_cache_') and not k.startswith('old_training_loader')}
 ex,formal,idx,ledger=docs['exclusions'],docs['formal_negative'],docs['identity_index'],docs['query_ledger']
 manifest=docs['old_training_manifest']; entries=manifest['records']
 # These two orders are exactly those retained by the original training loader.
 full=sorted((e for e in entries if e['kind']=='FULL' and e['role']=='TRAIN'),key=lambda e:e['execution_ordinal'])
 pair=[e for e in entries if e['kind']=='PAIR']
 need(len(full)==32 and len(pair)==64 and all(e['role']=='TRAIN' for e in pair),'PRESERVE_ORIGINAL32_PLUS64')
 retained=full+pair; old_ids=[e['query_id'] for e in retained]
 need(len(set(old_ids))==96,'ORIGINAL96_DISTINCT_QUERY_IDS')
 need(set(old_ids)==set(ex['split_sets']['FULL_TRAIN32']['query_ids'])|set(ex['split_sets']['PAIR64']['query_ids']),'EXACT_OLD_TRAINING_MEMBERSHIP')
 groups=set(ex['training_union_groups']); identities=set(ex['training_union_identities'])
 e32=ex['split_sets']['FULL_EVAL32']; e128=docs['eval128_curator']['records']
 eval_groups=set(e32['groups'])|{e['group'] for e in e128}; eval_ids=set(e32['identities'])|{e['identity'] for e in e128}
 need(len(groups)==len(identities)==32 and not(groups&eval_groups) and not(identities&eval_ids),'TRAIN32_GROUPS_AND_IDS_EVAL_DISJOINT')
 byq={e['query_id']:e for e in ledger['queries']}; ix={e['query_id']:e for e in idx['records']}; tokens={e['query_id']:e for e in docs['legacy_token_index']['records']}
 need(len(byq)==len(ix)==len(tokens)==987 and set(byq)==set(ix)==set(tokens),'QUERY_ID_CATALOG_JOIN')
 eval_sha={byq[q]['source_image_sha256'] for q in e32['query_ids']}|{q['source_image_sha256'] for q in e128}
 excluded_q=set(formal['exclude_query_ids']); excluded_sha=set(formal['exclude_image_sha256'])
 historical={e['query_id'] for e in docs['RAW600_membership']['records']}
 rows=[]; iterator=reversed(ledger['queries']) if independent else ledger['queries']
 for q in iterator:
  if q['query_id'] in excluded_q or q['source_image_sha256'] in excluded_sha or q['source_image_sha256'] in eval_sha: continue
  label,source=label_record(idx,ix[q['query_id']])
  if label['group_id'] not in groups or label['identity'] not in identities: continue
  need(label['group_id'] not in eval_groups and label['identity'] not in eval_ids,'NO_TEST_ID_OR_GROUP_IN_TRAIN')
  need(q['track']==label['track'] and q['track'] in ('difficult','outcome','new_difficult_train'),'SUPPORTED_PROCESSOR_FRAME')
  t=tokens[q['query_id']]; token=ROOT/Path(PATHS['legacy_token_index']).parent/t['artifact']
  need(Path(q['path']).is_file() and token.is_file(),'AVAILABLE_IMAGE_AND_LEGACY_TOKEN_FILES')
  rows.append({'query_id':q['query_id'],'query_ordinal':q['query_ordinal'],'track':q['track'],'source_image_sha256':q['source_image_sha256'],
   'query_image_path':q['path'],'grid_hw':[q['grid_h'],q['grid_w']],'redacted_token_artifact':str(token),
   'redacted_token_artifact_declared_sha256':t['artifact_file_sha256'],'image_tokens_sha256':t['image_tokens_sha256'],'template_tokens_sha256':t['template_tokens_sha256'],
   'identity':label['identity'],'group':label['group_id'],'identity_metadata_source':source,'in_historical_RAW600_C128_catalog':q['query_id'] in historical})
 byhash=defaultdict(list)
 for row in rows: byhash[row['source_image_sha256']].append(row)
 unique=[]; duplicates=[]
 for image_sha,members in byhash.items():
  need(len({(m['identity'],m['group']) for m in members})==1,'DUPLICATE_IMAGE_LABEL_CONFLICT')
  forced=[m for m in members if m['query_id'] in old_ids]; need(len(forced)<=1,'ORIGINAL96_IMAGE_COLLISION')
  canonical=forced[0] if forced else min(members,key=lambda m:m['query_id'])
  unique.append(canonical)
  if len(members)>1: duplicates.append({'source_image_sha256':image_sha,'query_ids':sorted(m['query_id'] for m in members),'kept_query_id':canonical['query_id'],'kept_original_training_image':bool(forced),'labels_and_groups_consistent':True})
 unique.sort(key=lambda m:m['query_id']); duplicates.sort(key=lambda m:m['source_image_sha256'])
 lookup={row['query_id']:row for row in unique}
 need(len(rows)==270 and len(unique)==269 and len(duplicates)==1,'STRICT_TRAIN_POOL270_RECORDS269_IMAGES')
 need(set(old_ids)<=set(lookup) and len({lookup[q]['source_image_sha256'] for q in old_ids})==96,'ALL_ORIGINAL96_IMAGES_RETAINED')
 base=[lookup[q] for q in old_ids]
 return unique,base,retained,duplicates

def select(unique,base,independent=False):
 selected=list(base); counts=Counter(row['group'] for row in base); used={row['source_image_sha256'] for row in base}
 remaining=[row for row in unique if row['source_image_sha256'] not in used]; trace=[]
 if independent:
  # Full rescan of all remaining images, independent of producer heap/queues.
  while len(selected)<128:
   row=min(remaining,key=lambda x:(counts[x['group']],key('group',x['group']),x['group'],key('image',x['source_image_sha256']),x['source_image_sha256']))
   trace.append({'added_ordinal':len(selected)-96,'group':row['group'],'selected_group_count_before':counts[row['group']],
    'group_tie_sha256':key('group',row['group']),'image_order_sha256':key('image',row['source_image_sha256']),'original_query_id':row['query_id'],'source_image_sha256':row['source_image_sha256']})
   selected.append(row); counts[row['group']]+=1; remaining.remove(row)
 else:
  queues=defaultdict(list)
  for row in remaining: queues[row['group']].append(row)
  queues={g:deque(sorted(v,key=lambda x:(key('image',x['source_image_sha256']),x['source_image_sha256']))) for g,v in queues.items()}
  heap=[(counts[g],key('group',g),g) for g in queues]; heapq.heapify(heap)
  while len(selected)<128:
   need(bool(heap),'INSUFFICIENT_NEW_TRAIN_IMAGES'); before,tie,g=heapq.heappop(heap); row=queues[g].popleft()
   need(before==counts[g],'HEAP_GROUP_COUNT_CONSISTENCY')
   trace.append({'added_ordinal':len(selected)-96,'group':g,'selected_group_count_before':before,
    'group_tie_sha256':tie,'image_order_sha256':key('image',row['source_image_sha256']),'original_query_id':row['query_id'],'source_image_sha256':row['source_image_sha256']})
   selected.append(row); counts[g]+=1
   if queues[g]: heapq.heappush(heap,(counts[g],tie,g))
 need(len(selected)==len({r['source_image_sha256'] for r in selected})==128,'SELECTED128_DISTINCT_IMAGES')
 return selected,trace

def provenance_only_field(path,field):
 # Old episode metadata are indented with exactly two spaces at top level.
 # Decode only the requested provenance value; never decode candidates/scalars.
 text=Path(path).read_text(); matches=list(re.finditer(r'^  "'+re.escape(field)+r'":\s*',text,re.MULTILINE))
 need(len(matches)==1,'UNIQUE_TOP_LEVEL_PROVENANCE_FIELD:'+field)
 value,_=json.JSONDecoder().raw_decode(text,matches[0].end()); return value

def artifacts(selected,retained):
 workers=[]; roles=[]; parity=[]
 for ordinal,row in enumerate(selected):
  opaque='T128-'+key('opaque',row['query_id']+'|'+row['source_image_sha256'])[:24]
  image=OUT/'worker_assets/images'/(opaque+Path(row['query_image_path']).suffix.lower()); token=OUT/'worker_assets/tokens'/(opaque+'.pt')
  workers.append({'query_id':opaque,'execution_ordinal':ordinal,'track':row['track'],'query_image_path':str(image),
   'source_image_sha256':row['source_image_sha256'],'grid_hw':row['grid_hw'],'redacted_token_artifact':str(token),
   'redacted_token_artifact_declared_sha256':row['redacted_token_artifact_declared_sha256'],'image_tokens_sha256':row['image_tokens_sha256'],'template_tokens_sha256':row['template_tokens_sha256']})
  origin='ORIGINAL_FULL32' if ordinal<32 else 'ORIGINAL_PAIR64' if ordinal<96 else 'ADDED32'
  roles.append({'query_id':opaque,'execution_ordinal':ordinal,'role':'TRAIN','selection_origin':origin,
   'original_query_id':row['query_id'],'original_query_ordinal':row['query_ordinal'],'original_training_execution_ordinal':retained[ordinal]['execution_ordinal'] if ordinal<96 else None,
   'identity':row['identity'],'group':row['group'],'track':row['track'],'source_image_sha256':row['source_image_sha256'],
   'original_query_image_path':row['query_image_path'],'original_redacted_token_artifact':row['redacted_token_artifact'],
   'identity_metadata_source':row['identity_metadata_source'],'in_historical_RAW600_C128_catalog':row['in_historical_RAW600_C128_catalog']})
  if ordinal<32:
   entry=retained[ordinal]; metadata=ROOT/entry['path']; need(sha(metadata)==entry['sha256'],'ORIGINAL_FULL_METADATA_SHA')
   source=provenance_only_field(metadata,'source')
   need(set(source)=={'maps','tokens'} and provenance_only_field(metadata,'query_id')==row['query_id'],'ORIGINAL_FULL_PROVENANCE')
   need(provenance_only_field(metadata,'query_source_image_sha256')==row['source_image_sha256'],'ORIGINAL_FULL_QUERY_IMAGE_SHA')
   tokens=dict(source['tokens']); tokens['path']=str(path_of(tokens)); maps=dict(source['maps']); maps['path']=str(path_of(maps))
   need(set(tokens)=={'path','sha256','record_index','reference_table_key'} and tokens['reference_table_key']=='references' and 0<=tokens['record_index']<8,'OLD_FULL_TOKEN_REFERENCE_ADDRESS')
   parity.append({'query_id':opaque,'execution_ordinal':ordinal,'original_execution_ordinal':entry['execution_ordinal'],
    'metadata':bind(metadata),'tokens':tokens,'maps':maps})
 need(all(set(r)==WORKER_FIELDS for r in workers) and len({r['query_id'] for r in workers})==128,'OPAQUE_WORKER_NO_LABELS')
 return workers,roles,parity

def counts(roles):
 return {'training_images':len(roles),'unique_image_sha256':len({r['source_image_sha256'] for r in roles}),
  'identities':len({r['identity'] for r in roles}),'groups':len({r['group'] for r in roles}),
  'by_track':dict(sorted(Counter(r['track'] for r in roles).items())),
  'by_selection_origin':dict(sorted(Counter(r['selection_origin'] for r in roles).items())),
  'images_by_group':dict(sorted(Counter(r['group'] for r in roles).items())),
  'images_by_identity':dict(sorted(Counter(r['identity'] for r in roles).items())),
  'historical_RAW600_members':sum(r['in_historical_RAW600_C128_catalog'] for r in roles)}

def aliases(workers,roles,produce=False):
 for worker,role in zip(workers,roles,strict=True):
  for wk,rk in [('query_image_path','original_query_image_path'),('redacted_token_artifact','original_redacted_token_artifact')]:
   alias,source=Path(worker[wk]),Path(role[rk]); need(source.is_file(),'SOURCE_IMAGE_OR_TOKEN_MISSING')
   if produce: alias.parent.mkdir(parents=True,exist_ok=True); os.symlink(str(source),str(alias))
   need(alias.is_symlink() and os.readlink(alias)==str(source),'ANONYMOUS_SOURCE_ALIAS')

def produce():
 need(not OUT.exists(),'APPEND_ONLY_TRAIN_MANIFEST'); src=sources(); unique,base,retained,duplicates=inputs()
 selected,trace=select(unique,base); workers,roles,parity=artifacts(selected,retained); OUT.mkdir(); aliases(workers,roles,True)
 summary=counts(roles); need(summary['identities']==summary['groups']==32,'PRESERVED32_IDS_AND_GROUPS')
 write(OUT/'curator_roles.json',{'status':'CURATOR_ONLY_ORIGINAL7_TRAIN128_ROLES_FROZEN','role':'TRAIN','records':roles,'counts':summary,'sources':src,
  'training_labels_only':True,'evaluation_roles_changed':False,'model_training_authorized_by_this_metadata':False})
 write(OUT/'legacy_full32_parity_index.json',{'status':'ORIGINAL7_TRAIN128_LEGACY_FULL32_PARITY_INDEX_METADATA_ONLY','query_count':32,'records':parity,
  'source_manifest':src['old_training_manifest'],'source_runtime_validation':src['old_runtime_token_validation'],
  'target_identity_group_fields_in_records':False,'cache_reuse_requires_current_runtime_parity':True})
 write(OUT/'worker_manifest.json',{'status':'TRAIN128_WORKER_MANIFEST_FROZEN_METADATA_ONLY','role':'TRAIN','query_count':128,'records':workers,
  'curator_ledger_sha256':sha(OUT/'curator_roles.json'),'legacy_full32_parity_index':bind(OUT/'legacy_full32_parity_index.json'),
  'program_sha256':sha(PROGRAM),'all128_retained_regardless_later_C128_presence':True,
  'legacy_redacted_token_runtime_reuse_authorized':False,'natural_execution_authorized_by_manifest':False,
  'old_FULL32_PAIR64_preserved':True,'execution_order':'original FULL32 sorted execution, original PAIR64 record order, then additional selection order'})
 write(OUT/'selection_receipt.json',{'status':'ORIGINAL7_TRAIN128_METADATA_SELECTION_PENDING_INDEPENDENT_VALIDATION','sources':src,'seed':SEED,
  'eligible_records':270,'eligible_unique_images':269,'counts':summary,'worker_manifest':bind(OUT/'worker_manifest.json'),
  'curator_ledger':bind(OUT/'curator_roles.json'),'legacy_full32_parity_index':bind(OUT/'legacy_full32_parity_index.json'),
  'selection_rule':'Keep original FULL32 then PAIR64; add32 from available group with lowest selected count, tie SHA256(seed+|group|+group); next image SHA256(seed+|image|+image_sha)',
  'selected_original_query_ids':[r['query_id'] for r in selected],'selected_image_sha256':[r['source_image_sha256'] for r in selected],
  'added_selection_trace':trace,'duplicate_resolution':duplicates,'evaluation_identity_group_image_overlap':0,
  'formal_negative_membership_overlap':0,'image_or_tensor_payload_reads':0,'candidate_presence_score_prediction_reads':0,'new_training_updates':0,'new_submissions':0})
 print(json.dumps({'status':'ORIGINAL7_TRAIN128_METADATA_CREATED','counts':summary,'worker_manifest':bind(OUT/'worker_manifest.json')},ensure_ascii=False),flush=True)

def validate_child(nonce):
 need(nonce==os.environ.get('RC_TRAIN128_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('RC_TRAIN128_VALIDATOR_PARENT'),'FRESH_VALIDATOR_REQUIRED')
 src=sources(); receipt=read(OUT/'selection_receipt.json'); worker=read(OUT/'worker_manifest.json'); curator=read(OUT/'curator_roles.json'); legacy=read(OUT/'legacy_full32_parity_index.json')
 need(receipt['sources']==curator['sources']==src,'SOURCE_BINDINGS_UNCHANGED')
 for key,filename in [('worker_manifest','worker_manifest.json'),('curator_ledger','curator_roles.json'),('legacy_full32_parity_index','legacy_full32_parity_index.json')]:need(receipt[key]==bind(OUT/filename),'ARTIFACT_BINDING:'+key)
 unique,base,retained,duplicates=inputs(True); selected,trace=select(unique,base,True); workers,roles,parity=artifacts(selected,retained)
 need(worker['records']==workers and curator['records']==roles and legacy['records']==parity,'INDEPENDENT_METADATA_REPLAY')
 need(trace==receipt['added_selection_trace'] and duplicates==receipt['duplicate_resolution'],'INDEPENDENT_FULL_RESCAN_MATCHES_HEAP')
 need(receipt['selected_original_query_ids']==[r['query_id'] for r in selected] and receipt['selected_image_sha256']==[r['source_image_sha256'] for r in selected],'SELECTED_ORDER_EXACT')
 need(receipt['counts']==curator['counts']==counts(roles),'ALL_SELECTION_COUNTS')
 need(worker['role']==curator['role']=='TRAIN' and all(r['role']=='TRAIN' for r in roles),'TRAIN_ONLY_NO_EVAL_ROLE_REASSIGNMENT')
 need(worker['curator_ledger_sha256']==sha(OUT/'curator_roles.json') and worker['legacy_full32_parity_index']==bind(OUT/'legacy_full32_parity_index.json'),'WORKER_PROVENANCE')
 aliases(workers,roles); need(not BLOCKED,'NO_FORBIDDEN_READ_ATTEMPTS'); sources()
 write(OUT/'independent_metadata_validation.json',{'status':'TRAIN128_INDEPENDENT_METADATA_SELECTION_PASS','sources':src,
  'worker_manifest':bind(OUT/'worker_manifest.json'),'curator_ledger':bind(OUT/'curator_roles.json'),'selection_receipt':bind(OUT/'selection_receipt.json'),
  'legacy_full32_parity_index':bind(OUT/'legacy_full32_parity_index.json'),'counts':counts(roles),'eligible_unique_images':269,
  'original_FULL32_and_PAIR64_images_and_internal_order_preserved':True,'fixed_seed_least_selected_group_selection_independently_replayed':True,
  'exact_image_dedup_and_labels_consistent':True,'old_and_new_EVAL_identities_groups_images_excluded':True,'formal392_ID_and_image_SHA_negative_exclusions':True,
  'worker_records_have_no_identity_group_target_fields':True,'all_existing_EVAL_metadata_source_hashes_unchanged':True,'fresh_explicit_subprocess':True,
  'image_or_tensor_payload_reads':0,'candidate_presence_score_prediction_reads':0,'new_training_updates':0,'new_submissions':0,'execution_authority_still_required':True})
 print(json.dumps({'status':'TRAIN128_INDEPENDENT_METADATA_SELECTION_PASS','validation':bind(OUT/'independent_metadata_validation.json'),'counts':counts(roles)},ensure_ascii=False),flush=True)

def main():
 parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True); group.add_argument('--produce',action='store_true'); group.add_argument('--validate',action='store_true'); group.add_argument('--validator-child'); args=parser.parse_args(); sys.addaudithook(audit)
 if args.produce: produce()
 elif args.validate:
  nonce=uuid.uuid4().hex; env=dict(os.environ,RC_TRAIN128_VALIDATOR_NONCE=nonce,RC_TRAIN128_VALIDATOR_PARENT=str(os.getpid()))
  subprocess.run([sys.executable,str(PROGRAM),'--validator-child',nonce],env=env,check=True)
 else: validate_child(args.validator_child)

if __name__=='__main__': main()
