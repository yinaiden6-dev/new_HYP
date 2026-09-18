#!/usr/bin/env python3
"""Metadata-only group-balanced128 selection, opaque assets, separate curator ledger."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict,deque
import hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PROGRAM=Path(__file__).resolve()
OUT=ROOT/'results/rc_original7_expanded_eval128_manifest_v1'
PLAN=ROOT/'plan/RC_ORIGINAL7_EXPANDED_EVAL128_SELECTION_V1_20260910.md'
SEED='RC_ORIGINAL7_EVAL128_20260910';N=128
AUTH=ROOT/'registry/rc_retrieval_only_eval128_expansion_authority_addendum_v1_20260910.json'
INVENTORY=ROOT/'results/rc_expanded_original7_source_inventory_v1/result.json'
EXCLUSIONS=ROOT/'registry/rc_eval128_identity_group_exclusion_v1_20260910.json'
FORMAL=ROOT/'registry/rc_eval128_formal392_negative_exclusion_v1_20260910.json'
INDEX=ROOT/'results/l0_natural_hardneg_v_representation_v2_identityfix/target_join/query_target_join_index.json'
LEDGER=ROOT/'cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json'
HEAD=ROOT/'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json'
PINS={INVENTORY:'654e23ff7b83bb301e8ac2afe0fc74d2b18e05ef473f89285a1d15d281eb538f',
 EXCLUSIONS:'e1ee298521ab9dfad5f8611b0011b5aeea9808055e50eea156288eb07f3bd0df',
 FORMAL:'0e09f9bbf33998f1553d0b3aedd4a6d28ced9da3c57da6c45d7013abda81e243',
 INDEX:'f99b08737791f9f30329a2b6388cb76421a4ffb6642ce88b2fcb42aca767eb9b',
 LEDGER:'df7af8a116d25881b9dcf274fdd389d268b1811b8e17c9c412ab064925e290ec',
 HEAD:'42c8e503874cb807ca7085a39c802c016de2b1b8c6124bb17a56ed6f06bd174b'}
TARGET_FIELDS={'version','contract_sha256','source_query_ledger_logical_sha256','query_ordinal','query_id','identity','track','group_id','target_join_performed','opened_runtime_read_count','sealed_runtime_read_count','a10_runtime_read_count','home_files_modified','logical_sha256'}
BLOCKED=[]
def need(x,m):
 if not bool(x):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0]));s=str(p).lower()
 if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.pt','.npz','.npy') or any(t in s for t in ('d1_mi','d1-mi','grozi','/rc_opened_eval_strict_')):
  BLOCKED.append(str(p));raise PermissionError('IMAGE_TENSOR_OR_PROTECTED_READ_FORBIDDEN')
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).absolute()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(encode(v)+b'\n');f.flush();os.fsync(f.fileno())
 p.chmod(0o444)
def key(domain,*parts):return hashlib.sha256(encode([SEED,domain,*parts])).hexdigest()
def sources():
 for p,d in PINS.items():need(sha(p)==d,'PIN:'+str(p))
 a=read(AUTH);need(a['minimum_evaluation_queries']==128 and a['natural_expanded_run_started'] is False,'USER_EXPANSION_SCOPE')
 return {'program':bind(PROGRAM),'plan':bind(PLAN),'user_authority':bind(AUTH),'inventory':bind(INVENTORY),'current43_exclusions':bind(EXCLUSIONS),'formal_negative_exclusions':bind(FORMAL),'identity_group_index':bind(INDEX),'query_ledger':bind(LEDGER),'frozen_ORIGINAL7_parameters':bind(HEAD)}
def label_record(index,item):
 p=INDEX.parent/item['path'];v=read(p);need(set(v)==TARGET_FIELDS,'METADATA_ONLY_TARGET_SCHEMA')
 need(v['query_id']==item['query_id'] and v['query_ordinal']==item['query_ordinal'],'QUERY_ID_AND_METADATA_ADDRESS')
 need(v['contract_sha256']==index['contract_sha256'] and v['source_query_ledger_logical_sha256']==index['source_query_ledger_logical_sha256'],'TARGET_METADATA_PROVENANCE')
 logical=hashlib.sha256(encode({k:x for k,x in v.items() if k!='logical_sha256'})).hexdigest()
 need(logical==v['logical_sha256']==item['record_logical_sha256'],'TARGET_METADATA_LOGICAL_SHA')
 return v,bind(p)
def inputs(independent=False):
 inventory=read(INVENTORY);ex=read(EXCLUSIONS);fm=read(FORMAL);idx=read(INDEX);ledger=read(LEDGER)
 need(ex['status']=='EVAL128_OLD128_IDENTITY_GROUP_EXCLUSION_METADATA_PASS' and fm['status']=='FORMAL392_PUBLIC_MEMBERSHIP_NEGATIVE_EXCLUSION_METADATA_PASS','QUALIFIED_EXCLUSION_METADATA')
 ids=set(ex['new_panel_exclude_identities']);groups=set(ex['new_panel_exclude_groups']);old_queries=set(ex['new_panel_exclude_old_query_ids'])
 fq=set(fm['exclude_query_ids']);fh=set(fm['exclude_image_sha256']);need(len(ids)==len(groups)==43 and len(fq)==392,'EXCLUSION_COUNTS')
 byq={q['query_id']:q for q in ledger['queries']};ix={r['query_id']:r for r in idx['records']}
 records=inventory['available_unselected_metadata_records'];need(len(records)==267 and len({r['query_id'] for r in records})==267,'FROZEN267')
 need(len(byq)==len(ix)==987 and set(byq)==set(ix),'CATALOG987_QUERY_ID_AXIS')
 # Independently rebuild the eligible universe without opening protected members' identity records.
 allowed={};metadata_sources=[]
 query_ids=list(byq) if independent else [r['query_id'] for r in records]
 for qid in query_ids:
  q=byq[qid]
  if qid in fq or q['source_image_sha256'] in fh:continue
  target,source=label_record(idx,ix[qid])
  if target['identity'] in ids or target['group_id'] in groups:continue
  need(qid not in old_queries,'OLD_QUERY_ESCAPED_GROUP_EXCLUSION')
  allowed[qid]=(target,source)
 need(set(allowed)=={r['query_id'] for r in records},'COMPLETE_ELIGIBLE_UNIVERSE_BEFORE_ANY_CANDIDATE_TEST')
 enriched=[]
 for r in records:
  q=byq[r['query_id']];target,source=allowed[r['query_id']]
  need(r['source_image_sha256']==q['source_image_sha256'] and r['query_image_path']==q['path'] and r['query_ordinal']==q['query_ordinal'] and r['track']==q['track']==target['track'],'INVENTORY_QUERY_SOURCE')
  need(r['grid_hw']==[q['grid_h'],q['grid_w']],'GRID_METADATA')
  enriched.append({**r,'identity':target['identity'],'group':target['group_id'],'identity_metadata_source':source})
  metadata_sources.append(source)
 byhash=defaultdict(list)
 for r in enriched:byhash[r['source_image_sha256']].append(r)
 duplicates=[];unique=[]
 for h,rs in byhash.items():
  need(len({r['identity'] for r in rs})==len({r['group'] for r in rs})==1,'DUPLICATE_IMAGE_CONFLICTING_LABELS')
  ordered=sorted(rs,key=lambda r:(key('record',r['query_id']),r['query_id']))
  unique.append(ordered[0])
  if len(rs)>1:duplicates.append({'source_image_sha256':h,'query_ids':sorted(r['query_id'] for r in rs),'kept_query_id':ordered[0]['query_id'],'identity':rs[0]['identity'],'group':rs[0]['group'],'identity_and_group_consistent':True})
 need(len(unique)==266 and len(duplicates)==1,'EXACT_SHA_DEDUP_267_TO266')
 need(len({r['identity'] for r in unique})==24 and len({r['group'] for r in unique})==21,'ELIGIBLE_IDENTITY_GROUP_COUNTS')
 return unique,duplicates,metadata_sources

def organized(rows):
 groups=defaultdict(lambda:defaultdict(list))
 for r in rows:groups[r['group']][r['identity']].append(r)
 for g,ids in groups.items():
  for identity,rs in ids.items():rs.sort(key=lambda r:(key('image',r['source_image_sha256']),r['source_image_sha256']))
 gs=sorted(groups,key=lambda g:(key('group',g),g));id_order={g:sorted(groups[g],key=lambda i:(key('identity',g,i),i)) for g in gs}
 return groups,gs,id_order

def select(rows,independent=False):
 groups,order,identities=organized(rows)
 if independent:
  streams={g:[groups[g][i][level] for level in range(max(map(len,groups[g].values()))) for i in identities[g] if level<len(groups[g][i])] for g in order}
  return [streams[g][level] for level in range(max(map(len,streams.values()))) for g in order if level<len(streams[g])][:N]
 queues={g:deque(identities[g]) for g in order};images={g:{i:deque(groups[g][i]) for i in identities[g]} for g in order}
 active=deque(order);chosen=[]
 while active and len(chosen)<N:
  g=active.popleft();i=queues[g].popleft();chosen.append(images[g][i].popleft())
  if images[g][i]:queues[g].append(i)
  if queues[g]:active.append(g)
 return chosen

def artifacts(selected):
 need(len(selected)==len({r['source_image_sha256'] for r in selected})==N,'SELECTED128_UNIQUE_IMAGES')
 need(len({r['identity'] for r in selected})==24 and len({r['group'] for r in selected})==21,'ALL_AVAILABLE_IDENTITIES_AND_GROUPS_COVERED')
 workers=[];private=[]
 for ordinal,r in enumerate(selected):
  opaque='E128-'+key('opaque-query',r['query_id'],r['source_image_sha256'])[:24]
  image=OUT/'worker_assets/images'/(opaque+Path(r['query_image_path']).suffix.lower())
  token=OUT/'worker_assets/tokens'/(opaque+'.pt')
  workers.append({'query_id':opaque,'execution_ordinal':ordinal,'track':r['track'],'query_image_path':str(image),
   'source_image_sha256':r['source_image_sha256'],'grid_hw':r['grid_hw'],'redacted_token_artifact':str(token),
   'redacted_token_artifact_declared_sha256':r['redacted_token_artifact_declared_sha256'],
   'image_tokens_sha256':r['image_tokens_sha256'],'template_tokens_sha256':r['template_tokens_sha256']})
  private.append({'query_id':opaque,'execution_ordinal':ordinal,'original_query_id':r['query_id'],'original_query_ordinal':r['query_ordinal'],
   'identity':r['identity'],'group':r['group'],'track':r['track'],'source_image_sha256':r['source_image_sha256'],
   'original_query_image_path':r['query_image_path'],'original_redacted_token_artifact':r['redacted_token_artifact'],
   'identity_metadata_source':r['identity_metadata_source'],'in_historical_RAW600_C128_catalog':r['in_historical_RAW600_C128_catalog']})
 need(len({r['query_id'] for r in workers})==N,'OPAQUE_ID_COLLISION')
 return workers,private

def counts(private):
 return {'evaluation_images':N,'unique_image_sha256':len({r['source_image_sha256'] for r in private}),
 'identities':len({r['identity'] for r in private}),'groups':len({r['group'] for r in private}),
 'by_track':dict(Counter(r['track'] for r in private)),
 'images_by_identity':dict(sorted(Counter(r['identity'] for r in private).items())),
 'images_by_group':dict(sorted(Counter(r['group'] for r in private).items())),
 'in_historical_RAW600_C128_catalog':sum(r['in_historical_RAW600_C128_catalog'] for r in private)}

def alias_assets(worker,private,produce=False):
 for w,c in zip(worker,private):
  for alias_key,source_key in [('query_image_path','original_query_image_path'),('redacted_token_artifact','original_redacted_token_artifact')]:
   alias=Path(w[alias_key]);original=Path(c[source_key])
   need(original.is_file(),'SOURCE_FILE_MISSING:'+source_key)
   if produce:
    alias.parent.mkdir(parents=True,exist_ok=True);os.symlink(str(original),str(alias))
   need(alias.is_symlink() and os.readlink(alias)==str(original),'ANONYMOUS_ALIAS_BINDING')

def produce():
 need(not OUT.exists(),'APPEND_ONLY_COHORT');src=sources();rows,duplicates,metadata=inputs(False);selected=select(rows);worker,private=artifacts(selected)
 OUT.mkdir();alias_assets(worker,private,True)
 curator={'status':'CURATOR_ONLY_EXPANDED_EVAL128_ROLES_FROZEN','sources':src,'seed':SEED,'records':private,
  'counts':counts(private),'duplicate_resolution':duplicates,'all_selected_queries_retained_regardless_later_C128_presence':True,
  'scope':'expanded current-training-and-old-EVAL identity/group-heldout development; historically exposed source pools; not untouched external',
  'scorer_read_authorized':False,'metadata_selection_not_model_scoring':True}
 write(OUT/'curator_roles.json',curator)
 wm={'status':'EXPANDED_EVAL128_WORKER_MANIFEST_FROZEN_METADATA_ONLY','query_count':N,'primary_comparison':'FROZEN_ORIGINAL7_VS_RAW',
  'frozen_head_parameters':bind(HEAD),'records':worker,'curator_ledger_sha256':sha(OUT/'curator_roles.json'),
  'program_sha256':sha(PROGRAM),'selection_plan_sha256':sha(PLAN),'all128_retained_regardless_later_C128_presence':True,
  'payload_content_validation':'DEFERRED_TO_EXECUTION_PREFLIGHT; no image or tensor bytes read during selection',
  'untouched_external_claim':False,'natural_scoring_authorized_by_this_manifest':False}
 write(OUT/'worker_manifest.json',wm)
 write(OUT/'selection_receipt.json',{'status':'EXPANDED_EVAL128_SELECTION_FROZEN_PENDING_INDEPENDENT_METADATA_VALIDATION',
  'sources':src,'seed':SEED,'eligible_records':267,'eligible_unique_images':266,'counts':counts(private),
  'worker_manifest':bind(OUT/'worker_manifest.json'),'curator_ledger':bind(OUT/'curator_roles.json'),
  'duplicate_resolution':duplicates,'selected_original_query_ids':[r['query_id'] for r in selected],
  'selected_image_sha256':[r['source_image_sha256'] for r in selected],'selection_algorithm':'hash-ordered group round-robin, identity round-robin, image hash order',
  'image_or_tensor_payload_reads':0,'candidate_presence_or_score_reads':0,'new_training_updates':0,'automatic_submission':False})
 print(json.dumps({'status':'EXPANDED_EVAL128_SELECTION_FROZEN','counts':counts(private),'worker_manifest':bind(OUT/'worker_manifest.json')},ensure_ascii=False),flush=True)

def validate_child(nonce):
 need(nonce==os.environ.get('RC_EVAL128_VALIDATOR_NONCE') and str(os.getppid())==os.environ.get('RC_EVAL128_VALIDATOR_PARENT'),'EXPLICIT_FRESH_PROCESS_REQUIRED')
 src=sources();receipt=read(OUT/'selection_receipt.json');wm=read(OUT/'worker_manifest.json');cur=read(OUT/'curator_roles.json')
 need(receipt['sources']==cur['sources']==src,'SOURCE_BINDING')
 need(receipt['worker_manifest']==bind(OUT/'worker_manifest.json') and receipt['curator_ledger']==bind(OUT/'curator_roles.json'),'SEALED_FILES')
 rows,duplicates,_=inputs(True);selected=select(rows,True);worker,private=artifacts(selected)
 need(wm['records']==worker and cur['records']==private,'INDEPENDENT_ROUND_BASED_SELECTION_REPLAY')
 need(receipt['selected_original_query_ids']==[r['query_id'] for r in selected] and receipt['selected_image_sha256']==[r['source_image_sha256'] for r in selected],'EXACT_SELECTED_ORDER')
 need(receipt['counts']==cur['counts']==counts(private) and receipt['duplicate_resolution']==cur['duplicate_resolution']==duplicates,'COUNTS_AND_DUPLICATE_LABELS')
 need(wm['curator_ledger_sha256']==sha(OUT/'curator_roles.json') and wm['frozen_head_parameters']==bind(HEAD),'CURATOR_HEAD_BINDINGS')
 need(wm['query_count']==128 and wm['primary_comparison']=='FROZEN_ORIGINAL7_VS_RAW' and wm['all128_retained_regardless_later_C128_presence'] is True,'FIXED_PRIMARY_AND_DENOMINATOR')
 allowed={'query_id','execution_ordinal','track','query_image_path','source_image_sha256','grid_hw','redacted_token_artifact','redacted_token_artifact_declared_sha256','image_tokens_sha256','template_tokens_sha256'}
 need(all(set(r)==allowed for r in worker),'WORKER_NO_TARGET_IDENTITY_OR_GROUP_FIELDS')
 alias_assets(worker,private)
 need(not BLOCKED,'FORBIDDEN_PAYLOAD_READ_ATTEMPT')
 write(OUT/'independent_metadata_validation.json',{'status':'EXPANDED_EVAL128_INDEPENDENT_METADATA_SELECTION_PASS','sources':src,
  'worker_manifest':bind(OUT/'worker_manifest.json'),'curator_ledger':bind(OUT/'curator_roles.json'),'selection_receipt':bind(OUT/'selection_receipt.json'),
  'counts':counts(private),'eligible267_universe_independently_rebuilt':True,'duplicate_labels_consistent':True,
  'independent_round_expansion_matches_queue_selection':True,'old43_identity_group_and_formal392_membership_exclusions':True,
  'all21_groups_and24_identities_covered':True,'worker_records_have_no_target_identity_group_fields':True,
  'anonymous_aliases_only_no_payload_content_reads':True,'fresh_explicit_subprocess':True,'validator_pid':os.getpid(),'validator_parent_pid':os.getppid(),
  'image_or_tensor_payload_reads':0,'candidate_presence_or_score_reads':0,'new_training_updates':0,'untouched_external_claim':False,'execution_authority_still_required':True})
 print(json.dumps({'status':'EXPANDED_EVAL128_INDEPENDENT_METADATA_SELECTION_PASS','counts':counts(private),'validation':bind(OUT/'independent_metadata_validation.json')},ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--produce',action='store_true');m.add_argument('--validate',action='store_true');m.add_argument('--validator-child');a=p.parse_args();sys.addaudithook(audit)
 if a.produce:produce()
 elif a.validate:
  n=uuid.uuid4().hex;env=dict(os.environ,RC_EVAL128_VALIDATOR_NONCE=n,RC_EVAL128_VALIDATOR_PARENT=str(os.getpid()))
  subprocess.run([sys.executable,str(PROGRAM),'--validator-child',n],check=True,env=env)
 else:validate_child(a.validator_child)
if __name__=='__main__':main()
