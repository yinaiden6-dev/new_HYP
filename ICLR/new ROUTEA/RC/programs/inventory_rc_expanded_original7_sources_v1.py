#!/usr/bin/env python3
"""Metadata-only availability inventory; no cohort choice or model execution."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_expanded_original7_source_inventory_v1'
BRIDGE=ROOT/'registry/rc_eval128_old128_metadata_exclusion_bridge_v1_20260910.json'
EXCLUDE=ROOT/'registry/rc_eval128_formal392_negative_exclusion_v1_20260910.json'
LEDGER=ROOT/'cache/l0_natural_hardneg_v2_targetfree_inputs_v1/query_ledger.json'
INDEX=ROOT/'cache/l0_natural_hardneg_v2_redacted_query_tokens_v1/index_v1.json'
TARGET=ROOT/'results/l0_natural_hardneg_v_representation_v2_identityfix/target_join'
GALLERY=ROOT.parents[2]/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
def need(v,m):
 if not bool(v):raise RuntimeError(m)
def audit(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
 if p.suffix in('.pt','.npz','.jpg','.jpeg','.png','.webp','.tiff','.bmp'):raise RuntimeError('METADATA_ONLY_NO_PIXEL_OR_TENSOR_READ')
 if any(x in s for x in ('d1_mi','d1-mi','grozi','gisc_prerecall_universe')):raise RuntimeError('PROTECTED_INPUT')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def logical(d):return hashlib.sha256(json.dumps({k:v for k,v in d.items() if k!='logical_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 sys.addaudithook(audit);need(not OUT.exists(),'APPEND_ONLY_INVENTORY')
 bridge,negative,ledger,index=map(read,(BRIDGE,EXCLUDE,LEDGER,INDEX))
 need(sha(BRIDGE)=='9deeef3e9148450605bba5c8adc67cebf34711b1e168e2699ee793f14e8d7f79','QUALIFIED43_BRIDGE')
 need(sha(EXCLUDE)=='0e09f9bbf33998f1553d0b3aedd4a6d28ced9da3c57da6c45d7013abda81e243','PUBLIC_FORMAL_NEGATIVE_MEMBERSHIP')
 ids={r['catalog_identity'] for r in bridge['records']};groups={r['catalog_group_id'] for r in bridge['records']}
 need(len(ids)==len(groups)==43,'ALL43_EXCLUSIONS')
 fq,fh=set(negative['exclude_query_ids']),set(negative['exclude_image_sha256'])
 need(len(fq)==392 and len(fh)==389,'FORMAL_NEGATIVE_LEDGER_POPULATION')
 target_index=read(TARGET/'query_target_join_index.json');metadata={};metadata_sources=[]
 allowed={'version','contract_sha256','source_query_ledger_logical_sha256','query_ordinal','query_id','identity','track','group_id',
  'target_join_performed','opened_runtime_read_count','sealed_runtime_read_count','a10_runtime_read_count','home_files_modified','logical_sha256'}
 for entry in target_index['records']:
  p=TARGET/entry['path'];d=read(p);need(set(d)==allowed,'IDENTITY_GROUP_METADATA_ONLY')
  need(d['query_id']==entry['query_id'] and d['query_ordinal']==entry['query_ordinal'] and logical(d)==d['logical_sha256']==entry['record_logical_sha256'],'METADATA_SOURCE_BINDING')
  metadata[d['query_id']]={k:d[k] for k in ('identity','group_id','track')};metadata_sources.append(bind(p))
 tokens={r['query_id']:r for r in index['records']}
 schedule=read(ROOT/'protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json');six={r['query_id'] for r in schedule['records']}
 current={'FULL_TRAIN32':set(),'PAIR64':set(),'OLD_EVAL32':set()};current_query_ids=set()
 for entry in read(ROOT/'results/rc_shared_query_target_prior_cache_v1/manifest.json')['records']:
  p=Path(entry['path']);p=p if p.is_absolute() else ROOT/p;d=read(p)
  name='PAIR64' if d['kind']=='PAIR' else ('FULL_TRAIN32' if d['role']=='TRAIN' else 'OLD_EVAL32')
  current[name].add(d['query_source_image_sha256']);current_query_ids.add(d['query_id'])
 old_images=set.union(*current.values());need(len(old_images)==128,'OLD128_DISTINCT_IMAGES')
 raw=ledger['queries'];after43=[r for r in raw if metadata[r['query_id']]['identity'] not in ids and metadata[r['query_id']]['group_id'] not in groups]
 afterformal=[r for r in after43 if r['query_id'] not in fq and r['source_image_sha256'] not in fh and r['source_image_sha256'] not in old_images]
 def summary(rows):
  return {'query_records':len(rows),'unique_image_sha256':len({r['source_image_sha256'] for r in rows}),
   'identity_count':len({metadata[r['query_id']]['identity'] for r in rows}),'group_count':len({metadata[r['query_id']]['group_id'] for r in rows}),
   'track_record_counts':dict(Counter(r['track'] for r in rows))}
 inventory=[]
 for r in afterformal:
  t=tokens[r['query_id']];p=Path(t['artifact']);p=p if p.is_absolute() else INDEX.parent/p
  need(p.is_file() and Path(r['path']).is_file(),'AVAILABLE_IMAGE_AND_REDACTED_TOKENS')
  inventory.append({'query_id':r['query_id'],'query_ordinal':r['query_ordinal'],'track':r['track'],
   'source_image_sha256':r['source_image_sha256'],'query_image_path':r['path'],'grid_hw':[r['grid_h'],r['grid_w']],
   'redacted_token_artifact':str(p.resolve()),'redacted_token_artifact_declared_sha256':t['artifact_file_sha256'],
   'image_tokens_sha256':t['image_tokens_sha256'],'template_tokens_sha256':t['template_tokens_sha256'],
   'in_historical_RAW600_C128_catalog':r['query_id'] in six})
 legacy={}
 for name,directory,validation in (
  ('difficult90','romav2_colnomic_difficult90_regression_manifest_v1','romav2_colnomic_difficult90_real_prejoin_v1'),
  ('new_difficult31','romav2_colnomic_new_difficult_sealed_manifest_v1','romav2_colnomic_new_difficult_sealed_roma_prejoin_v1')):
  p=ROOT/'results'/directory/'prejoin_manifest.json';rows=read(p)['rows'];hashes={r['query_sha256'] for r in rows};vp=ROOT/'results'/validation/'validation.json';v=read(vp)
  need(all(x is True for x in v['checks'].values()),'LEGACY_C4_QUALIFICATION')
  legacy[name]={'records':len(rows),'unique_images':len(hashes),'current128_image_overlap':len(hashes&old_images),
   'RAW987_image_overlap':len(hashes&{r['source_image_sha256'] for r in raw}),
   'formal392_image_overlap':len(hashes&fh),'query_images_existing':sum(Path(r['query_path']).is_file() for r in rows),
   'full128_RAW_tokens_and_C4_scalars_exist':True,'persistent_visibility_weights_not_in_original_writer':True,
   'identity_group_eligible_count':'PENDING_SEPARATE_METADATA_BRIDGE','prejoin':bind(p),'C4_validation':bind(vp),
   'C4_shard_root':str(vp.parent)}
 source_paths=[
 'cache/l0_natural_hardneg_v2_redacted_query_tokens_v1/validation/validation_complete.json',
 'protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json',
 'results/dino_rcde_r1_oof_redacted_inputs_v1_0/redacted_oof_schedule.json',
 'results/dino_rcde_p0_v1_2/formal_job5069436/data/model_visible_c128.json',
 'results/dino_rcde_p0_v1_2/formal_job5069436/data/base_prejoin_receipt.json',
 'results/dino_rcde_p0_v1_1/formal_job5068254/base_prejoin_receipt.json',
 'cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_receipt_v2.json',
 'programs/run_dino_rcde_p0_v1_1.py','programs/run_dino_rcde_p0_data_v1_2.py',
 'programs/materialize_romav2_colnomic_current_runtime_bridge_shard_v1.py',
 'programs/materialize_romav2_colnomic_current_runtime_bridge_roma_shard_v1.py',
 'programs/materialize_romav2_colnomic_difficult90_token_fullrank_shard_v1.py',
 'programs/materialize_romav2_colnomic_difficult90_real_shard_v1.py',
 'programs/materialize_romav2_colnomic_new_difficult_sealed_roma_shard_v1.py',
 'programs/run_romav2_colnomic_visibility_xf_six_case_v1.py',
 'programs/run_romav2_colnomic_sealed_source_e0_v2.py',
 'src/rc_aslo_xf/reference_visibility_pv_lossless_v1.py','src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
 'src/rc_aslo_xf/l0_redacted_query_cache.py',
 'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
 'slurm/romav2_colnomic_current_runtime_bridge_prejoin_remaining_v1_30m.sbatch',
 'slurm/romav2_colnomic_current_runtime_bridge_roma_remaining_v1_30m.sbatch',
 'slurm/romav2_colnomic_difficult90_token_fullrank_remaining_v1_30m.sbatch',
 'slurm/romav2_colnomic_difficult90_real_remaining_v1_30m.sbatch']
 result={'status':'EXPANDED_ORIGINAL7_METADATA_INVENTORY_COMPLETE_NO_EVALUATION_STARTED',
  'sources':{'program':bind(__file__),'query_ledger':bind(LEDGER),'redacted_token_index':bind(INDEX),'identity_group_index':bind(TARGET/'query_target_join_index.json'),
   'old43_exclusion_bridge':bind(BRIDGE),'formal392_negative_membership':bind(EXCLUDE),'source_code_and_receipts':[bind(ROOT/p) for p in source_paths]},
  'RAW987_catalog':summary(raw),'after43_identity_group_exclusion_before_any_candidate_test':summary(after43),
  'after43_and_public_formal392_query_and_image_exclusion':summary(afterformal),
  'after_all_exclusions_in_RAW600':summary([r for r in afterformal if r['query_id'] in six]),
  'removed_by_formal392_membership_after43':len(after43)-len(afterformal),
  'available_unselected_metadata_records':inventory,'catalog_identity_metadata_sources':metadata_sources,
  'current_image_overlap':{k:len(v) for k,v in current.items()},'legacy_scalar_ready_panels':legacy,
  'full_gallery_tokens':{'path':str(GALLERY),'registered_sha256':'11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc','exists':GALLERY.is_file(),'binary_content_not_read':True},
  'historical_RAW600_binary':{'path':str(ROOT/'results/dino_rcde_p0_v1_1/formal_job5068254/base_prejoin.pt'),'registered_sha256':'ba1df745e5986abfcad3e82924ec9b745c076126c4666de907495642c5980691','binary_content_not_read':True},
  'head_source':'FROZEN_NATIVE7_CURRENT_ORIGINAL7_NOT_HISTORICAL_FROZEN_C','primary_features':'RAW_gap_plus_original_C4_derived_five_contrasts',
  'J_T_LP_required_for_primary':False,'candidate_presence_or_score_reads':0,'image_or_tensor_payload_reads':0,
  'cohort_selection_performed':False,'new_training_or_evaluation_started':False,'untouched_external_claim':False}
 OUT.mkdir();p=OUT/'result.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');p.chmod(0o444)
 print(json.dumps({'status':result['status'],'after43':result['after43_identity_group_exclusion_before_any_candidate_test'],
  'after43_and_formal':result['after43_and_public_formal392_query_and_image_exclusion'],'RAW600_remaining':result['after_all_exclusions_in_RAW600'],
  'result':bind(p)}),flush=True)
if __name__=='__main__':main()
