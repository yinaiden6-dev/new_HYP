"""Validate, hash and package final experiment evidence without replacing baseline archives."""
import sys,json,zipfile,hashlib
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.colnomic_mechanism.common import ROOT,OUT,read,write,sha,rows


def eligible(path):
 relative=path.relative_to(OUT);parts=relative.parts
 if '__pycache__' in parts or path.suffix in ['.tmp','.pyc']:return False
 if parts[0]=='stage2_geometry_before_polygon_order_fix':return False
 if len(parts)>1 and parts[0]=='stage4_interventions' and parts[1] in ['embeddings','analysis']:return False
 if path.name=='reference_001410_controlled_layouts.png':return False
 if len(parts)>1 and parts[:2]==('stage4_interventions','images'):
  names={Path(r['path']).name for r in read(OUT/'stage4_interventions/portable_encoding_manifest.json') if r['arm']!='base'}
  return path.name in names
 return True


def verify_evidence():
 torch.set_num_threads(2);analysis=OUT/'stage4_interventions/analysis_validated_runtime';matrix_errors=[];delta_errors=[]
 for path in sorted((analysis/'similarity_matrices').glob('*.npz')):
  z=np.load(path);matrix=z['matrix'];assert matrix.dtype==np.float32 and np.isfinite(matrix).all()
  assert matrix.shape==(len(z['query_raw_indices']),len(z['reference_raw_indices']))
  error=abs(float(matrix.max(1).mean())-float(z['score']));assert error<2e-7;matrix_errors.append(error)
 focal={(r['query'],r['negative_arm']):r for r in rows(analysis/'focal_causal_scores.csv')}
 for (q,arm),r in focal.items():
  table=rows(analysis/(q+'_'+arm+'_contributions.csv'));gap=np.mean([float(t['delta']) for t in table]);error=abs(gap-float(r['gap']));assert error<3e-7;delta_errors.append(error)
 for record in read(OUT/'stage4_interventions/portable_encoding_manifest.json'):
  path=OUT/'stage4_interventions'/record['path'];assert path.exists()
  value=torch.load(OUT/'stage4_interventions/validated_embeddings'/(record['id']+'.pt'),map_location='cpu',weights_only=True)
  assert sha(path)==value['image_sha256']
 runtime=read(OUT/'stage4_interventions/validated_embeddings/complete.json')
 encoder=ROOT/'scripts/colnomic_mechanism/offloaded_encoder.py';assert sha(encoder)==runtime['experimental_encoder_sha256']
 actual_worker=OUT/'source_code/executed/encode_worker.py';assert sha(actual_worker)==runtime['worker_sha256']
 tests=(OUT/'test_results.txt').read_text();assert '19 passed' in tests
 scores=np.load(OUT/'stage2_geometry/full_gallery_scores.npz');assert scores['original'].shape==(1117,5405)
 assert np.array_equal(scores['original'],scores['geo_bonus'][0]) and np.array_equal(scores['original'],scores['geo_penalty'][0])
 axes=read(OUT/'stage2_geometry/score_axes.json');assert len(axes['query_ids'])==1117 and len(axes['reference_ids'])==5405
 verification=read(OUT/'validation.json');verification['export_checks']=dict(status='passed',similarity_matrices=len(matrix_errors),
  max_matrix_score_replay_error=max(matrix_errors),delta_tables=len(delta_errors),max_token_delta_conservation_error=max(delta_errors),
  portable_images_hash_match=True,executed_worker_hash_match=True,numeric_encoder_hash_match=True,
  test_count=19,full_score_axes_verified=True,lambda_zero_exact=True)
 write(OUT/'validation.json',verification)


def main():
 verify_evidence()
 files=sorted(p for p in OUT.rglob('*') if p.is_file() and eligible(p) and p.name!='artifact_manifest.json')
 manifest=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 write(OUT/'artifact_manifest.json',dict(files=manifest,exclusions='superseded polygon-order experiment, duplicate preliminary encodings/analysis and rejected-source previews; retained locally',
  baseline_token_banks='dependency: prior F03/B03 archives or original workspace; not duplicated',model_weights_included=False))
 files.append(OUT/'artifact_manifest.json')
 target=ROOT/'results/token_mechanism_analysis/colnomic7b_four_stage_experiments_20261013_v1.zip';temporary=target.with_suffix('.zip.tmp')
 if target.exists():raise FileExistsError('Use a new version path rather than replacing a delivered experiment bundle')
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for path in files:z.write(path,arcname=str(Path(OUT.name)/path.relative_to(OUT)))
 temporary.replace(target)
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  for record in manifest:
   with z.open(str(Path(OUT.name)/record['path'])) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
 digest=sha(target);target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n')
 write(target.with_suffix('.validation.json'),dict(status='passed',zip_crc=True,all_member_sha256=True,files=len(files),bytes=target.stat().st_size,sha256=digest,
  experimental_limits=read(OUT/'validation.json'),source=OUT.name))
 print(json.dumps(dict(archive=str(target),files=len(files),bytes=target.stat().st_size,sha256=digest),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
