#!/usr/bin/env python3
import hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT.parent/'scripts'))
from rc_aslo_xf.conditional_rep_sources import build_gallery_source
import c2_build_new_difficult_token_cache as c2
OUT=ROOT/'results/romav2_colnomic_new_difficult_model_lineage_v1/result.json';ROMA=WORKSPACE/'third_party/RoMaV2';DINO=WORKSPACE/'third_party/model_cache/torch/hub/facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa';WEIGHTS=WORKSPACE/'third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt';GALLERY=WORKSPACE/'colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def gitrev(p):return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
def main():
 if OUT.exists():raise RuntimeError('immutable model lineage exists')
 fp=c2.model_fingerprint();gallery=build_gallery_source(verify_cache_file_sha256=True);programs=['programs/materialize_romav2_colnomic_new_difficult_sealed_token_shard_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_roma_shard_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_fullrank_shard_v1.py','programs/materialize_romav2_colnomic_new_difficult_sealed_cbind_shard_v1.py','src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py'];checks={'roma_checkpoint':sha(WEIGHTS)=='1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7','roma_commit':gitrev(ROMA)=='95c9968145c8906b7b59383258e9f73b02853d89','dino_commit':gitrev(DINO)=='adc254450203739c8149213a7a69d8d905b4fcfa','colnomic_fingerprint':fp['logical_sha256']==c2.EXPECTED_ENCODER_FINGERPRINT_LOGICAL_SHA256 and fp['base_model_manifest_sha256']==c2.EXPECTED_BASE_MODEL_MANIFEST_SHA256,'gallery_hash':sha(GALLERY)==c2.EXPECTED_GALLERY_FILE_SHA256,'gallery_population':len(gallery.raw_paths)==len(gallery.corrected_identities)==5413 and len(set(gallery.corrected_identities))==5412,'programs_present':all((ROOT/p).is_file() for p in programs)};passed=all(checks.values());v={'schema_version':'rc_romav2_colnomic_new_difficult_model_lineage_v1_20260901','status':'ROMAV2_COLNOMIC_NEW_DIFFICULT_MODEL_LINEAGE_FROZEN' if passed else 'ROMAV2_COLNOMIC_NEW_DIFFICULT_MODEL_LINEAGE_FAIL','checks':checks,'romav2':{'source_commit':gitrev(ROMA),'checkpoint_sha256':sha(WEIGHTS)},'dinov3':{'source_commit':gitrev(DINO)},'colnomic':{'encoder_fingerprint':fp,'gallery_cache_sha256':sha(GALLERY)},'gallery':{'physical_row_count':5413,'corrected_exact_label_count':5412,'corrected_mapping_sha256':gallery.corrected_mapping_sha256,'repair_contract_sha256':gallery.repair_contract_sha256,'repair_manifest_sha256':gallery.repair_manifest_sha256},'runtime':{'python':sys.version,'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'colpali_engine':importlib.metadata.version('colpali-engine')},'source_sha256':{p:sha(ROOT/p) for p in programs},'sealed_query_read_count':0,'target_label_read_count':0,'next_authorized_stage':'SEALED_TOP_LEVEL_PREJOIN_SEAL' if passed else None};OUT.parent.mkdir(parents=True,exist_ok=False);OUT.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':v['status'],'checks':checks},sort_keys=True));raise SystemExit(0 if passed else 4)
if __name__=='__main__':main()
