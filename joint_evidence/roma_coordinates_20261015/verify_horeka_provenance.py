"""Bind current frozen features to recovered HoreKa evidence; no model execution."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
import torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin');OUT=D/'roma_coordinates'
sys.path.insert(0,str(P));from experiment_utils import read,write
A=R/'results/colnomic_full_audit/20261013_resume/delivery_support/provenance_risk_audit_20261014'

def tokenhash(t):
    t=t.contiguous();return hashlib.sha256(str(t.dtype).encode()+json.dumps(list(t.shape),separators=(',',':')).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()

torch.set_num_threads(2)
binding=read(A/'original_horeka_gallery_binding.json');inputs=read(A/'horeka_input_binding.json');cohort=read(A/'source_cohort_audit.json')
q=torch.load(R/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',mmap=True,weights_only=True)
r=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)
byid={x['reference_id']:x for x in binding['records']};bad=[];label_differences=[]
for i,rid in enumerate(r['ids']):
    x=byid[rid]
    if x['current_row']!=i or x['tensor_SHA256']!=tokenhash(r['embeddings'][i]):bad.append(rid)
    if not x['same_identity_match']:label_differences.append(x)
assert not bad and len(r['ids'])==5405
qrows=[dict(query=qid,kind=p['kind'],image_sha=q['sha256'][i],provenance=p) for i,(qid,p) in enumerate(zip(q['ids'],q['provenance']))]
counts=dict(Counter(x['kind'] for x in qrows));assert counts==cohort['query_provenance_counts']
jobs=read(A/'historical_sources/horeka/producer_job_4076613_evidence.json');jobfacts=[]
for x in jobs:
    if not isinstance(x,dict):continue
    path=x.get('path','')
    if path.endswith('4076613.out'):
        lines=x['text'].splitlines();selected=[s for s in lines if any(w in s for w in ['job=','Python ','NVIDIA A100','model device=','saved gallery cache','Job ID:','Starttime:','Endtime:'])]
        jobfacts.append(dict(path=path,sha256=x['SHA256'],excerpts=selected))
    elif path.endswith('.pt'):jobfacts.append({k:x[k] for k in ['path','SHA256','bytes'] if k in x})
evidence_names=['original_horeka_gallery_binding.json','horeka_input_binding.json','source_cohort_audit.json','actual_query_cache_vs_intervention_base.json','historical_sources/horeka/producer_job_4076613_evidence.json']
evidence=[dict(path=str(A/name),sha256=hashlib.sha256((A/name).read_bytes()).hexdigest()) for name in evidence_names]
summary=dict(status='passed',current_reference_tensors=5405,verified_tensor_hashes=5405,horeka_original_reference_rows=binding['original_references'],same_original_identity=5390,labels_require_revision_explanation=len(label_differences),upstream_cache_sha256=binding['current_declared_upstream_SHA256'],all_reference_images_sha_match=inputs['image_SHA_matches'],query_provenance_counts=counts,query_contract=q['contract'],original_job=jobfacts,evidence=evidence,
    conclusion='All current gallery tensors bind byte-for-byte to the recovered HoreKa original. Current query bank is mixed:439 reused and678 generated under h593_encoder_v1. i14s45 current coordinate run freezes BOTH banks; RoMa inference is new, ColNomic inference is not.',
    limitations=['Matching current feature bytes proves cache identity, not reconstruction of every historical library or checkpoint byte at production time.','Query source group comparisons are confounded by sample identity and acquisition differences.','Historical15label discrepancies are not newly inferred wrong examples; existing current cohort labels remain unchanged.','Neither a cache difference nor its GPU provenance alone establishes a causal encoder failure.'])
write(OUT/'HOREKA_PROVENANCE.json',summary);write(OUT/'QUERY_PROVENANCE.json',qrows);write(OUT/'HOREKA_LABEL_REVISIONS.json',label_differences)
print(json.dumps({k:v for k,v in summary.items() if k not in ['original_job','evidence','query_contract','limitations']},ensure_ascii=False))
