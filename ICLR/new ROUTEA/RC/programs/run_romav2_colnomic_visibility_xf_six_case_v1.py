#!/usr/bin/env python3
import hashlib,json,math,os,sys,tempfile
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[2];sys.path.insert(0,str(ROOT/"src"))
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa:E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import FrozenColNomicBaseV1  # noqa:E402
from romav2 import RoMaV2  # noqa:E402
CASES=(480,20,421,324,485,413);V9B=ROOT/"results/rgh_v9b_balanced_fold2_oof_pilot_v1/result.json";ROMA6=ROOT/"results/romav2_six_case_qualification_v1/result.json";E0=ROOT/"results/romav2_correspondence_e0_v1/result.json";GEOMETRY=ROOT/"cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt";FOLDS=ROOT/"protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json";OUT=ROOT/"results/romav2_colnomic_visibility_xf_six_case_v1/result.json";WEIGHTS=WORKSPACE/"third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def logical(v):return hashlib.sha256(json.dumps({k:x for k,x in v.items() if k!="logical_sha256"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=False);fd,n=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".partial",dir=p.parent);q=Path(n)
 try:
  with os.fdopen(fd,"w") as h:json.dump(v,h,indent=2,sort_keys=True,allow_nan=False);h.write("\n");h.flush();os.fsync(h.fileno())
  q.chmod(0o444);os.link(q,p)
 finally:q.unlink(missing_ok=True)
def oriented(path):
 with Image.open(path) as im:
  try:orientation=int(im.getexif().get(274,1))
  except (TypeError,ValueError):orientation=1
  image=im.copy()
 operation={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}.get(orientation)
 if operation is not None:image=image.transpose(operation)
 return image.convert("RGB")
def cell_means(overlap,geometry):
 t=torch.as_tensor(overlap,dtype=torch.float64);h,w=t.shape;values=[]
 for box,valid in zip(geometry.cell_boxes_xyxy,geometry.valid_patch_mask,strict=True):
  if not bool(valid):values.append(t.new_zeros(()));continue
  x0,y0,x1,y1=[float(v) for v in box];ix0=max(0,min(w-1,int(math.floor(x0*w))));iy0=max(0,min(h-1,int(math.floor(y0*h))));ix1=max(ix0+1,min(w,int(math.ceil(x1*w))));iy1=max(iy0+1,min(h,int(math.ceil(y1*h))));values.append(t[iy0:iy1,ix0:ix1].mean())
 return torch.stack(values)
def score(q,r,wq,wr):
 qn=F.normalize(q.to(torch.float64),dim=1);rn=F.normalize(r.to(torch.float64),dim=1);sim=qn@rn.T;local=(sim*wr[None]).max(1).values;mass=torch.sqrt(wq.mean()*wr.mean());return mass*(wq*local).sum()/wq.sum().clamp_min(1e-12),mass,local
def main():
 if OUT.exists():raise RuntimeError("immutable visibility-XF output exists")
 e0=json.loads(E0.read_text());roma6=json.loads(ROMA6.read_text());assert e0["status"]=="ROMAV2_CORRESPONDENCE_E0_PASS" and roma6["status"]=="ROMAV2_SIX_CASE_CORRESPONDENCE_NO_HEADROOM" and sha(WEIGHTS)==e0["checkpoint_sha256"];v9b=json.loads(V9B.read_text());probes={int(x["execution_ordinal"]):x for x in v9b["probes"]};loader=RGHFull600SourceLoaderV1(rc_root=ROOT,geometry_payload_path=GEOMETRY,prejoin_schedule_path=FOLDS);base=FrozenColNomicBaseV1(ROOT);torch.set_float32_matmul_precision("highest");torch.manual_seed(17);model=RoMaV2();rows=[]
 for ex in CASES:
  source=loader.load_execution(ex);target=int(probes[ex]["target_position"]);raw=base.scores(ex,[x.physical_row for x in source.candidates]);competitor=sorted((i for i in range(128) if i!=target),key=lambda i:(-float(raw[i]),int(source.candidates[i].physical_row)))[0];query_img=oriented(source.query.source_path);candidates=[]
  for role,pos in (("TARGET",target),("COMPETITOR",competitor)):
   ref=source.candidates[pos].source;pred=model.match(query_img,oriented(ref.source_path));wq=cell_means(pred["overlap_AB"][0,...,0].detach().cpu(),source.query.colnomic_geometry);wr=cell_means(pred["overlap_BA"][0,...,0].detach().cpu(),ref.colnomic_geometry);real,mass,_=score(source.query.tokens,ref.tokens,wq,wr);qctrl,_,_=score(source.query.tokens,ref.tokens,wq.roll(1),wr);rctrl,_,_=score(source.query.tokens,ref.tokens,wq,wr.roll(1));allpatch=(F.normalize(source.query.tokens.to(torch.float64),dim=1)@F.normalize(ref.tokens.to(torch.float64),dim=1).T).max(1).values.mean();candidates.append({"role":role,"candidate_position":pos,"physical_row":int(source.candidates[pos].physical_row),"real_score":float(real),"query_spatial_control_score":float(qctrl),"reference_spatial_control_score":float(rctrl),"visibility_mass":float(mass),"query_visibility_mean":float(wq.mean()),"reference_visibility_mean":float(wr.mean()),"query_visibility_std":float(wq.std(unbiased=False)),"reference_visibility_std":float(wr.std(unbiased=False)),"allpatch_colnomic_score":float(allpatch),"query_visibility_sha256":hashlib.sha256(wq.contiguous().numpy().tobytes()).hexdigest(),"reference_visibility_sha256":hashlib.sha256(wr.contiguous().numpy().tobytes()).hexdigest()})
  t,c=candidates;comparisons={"target_real_better":t["real_score"]>c["real_score"],"target_real_gt_query_control":t["real_score"]>t["query_spatial_control_score"],"target_real_gt_reference_control":t["real_score"]>t["reference_spatial_control_score"],"candidate_maps_distinct":t["query_visibility_sha256"]!=c["query_visibility_sha256"] and t["reference_visibility_sha256"]!=c["reference_visibility_sha256"]};rows.append({"execution_ordinal":ex,"query_id":source.query_id,"candidates":candidates,"comparisons":comparisons})
 counts={name:sum(r["comparisons"][name] for r in rows) for name in rows[0]["comparisons"]};gates={"complete":len(rows)==6,"target_discrimination":counts["target_real_better"]>=4,"query_spatial":counts["target_real_gt_query_control"]>=4,"reference_spatial":counts["target_real_gt_reference_control"]>=4,"candidate_binding":counts["candidate_maps_distinct"]==6};passed=all(gates.values());value={"schema_version":"rc_romav2_colnomic_visibility_xf_six_case_v1_20260831","status":"ROMAV2_COLNOMIC_VISIBILITY_XF_HEADROOM" if passed else "ROMAV2_COLNOMIC_VISIBILITY_XF_NO_HEADROOM","claim_level":"SIX_OPENED_CASE_ZERO_TRAINING_COMPLEMENTARITY_ONLY","contract_sha256":sha(ROOT/"plan/ROMAV2_COLNOMIC_VISIBILITY_XF_SIX_CASE_CONTRACT_V1_20260831.md"),"roma_e0_sha256":sha(E0),"roma_six_case_sha256":sha(ROMA6),"checkpoint_sha256":sha(WEIGHTS),"comparison_counts":counts,"gates":gates,"rows":rows,"model_update_count":0,"training_count":0,"retrieval_score_mutation_count":0,"P0_authorized":False,"scientific_GO_or_NO_GO":None,"next_authorized_stage":"ROMAV2_COLNOMIC_VISIBILITY_XF_IDENTITY_DISJOINT_GATE" if passed else None,"logical_sha256":""};value["logical_sha256"]=logical(value);atomic(OUT,value);print(json.dumps({"status":value["status"],"counts":counts,"gates":gates},sort_keys=True))
if __name__=="__main__":main()
