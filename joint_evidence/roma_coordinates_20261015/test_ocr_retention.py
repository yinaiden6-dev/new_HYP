"""Post-hoc global evidence-retention control, with the prior validation-selected parameters frozen."""
from pathlib import Path
import sys,math,json
import numpy as np
import torch
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin')
sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite,metrics
torch.set_num_threads(2)
c=read(P/'cohort.json');qids=[q['query'] for q in c]
rb=torch.load(R/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',mmap=True,weights_only=True)
ids=rb['ids'];sha=rb['sha256'];ri={r:j for j,r in enumerate(ids)}
selected=read(D/'SELECTIVE_OCR_SUMMARY.json')['selected'];w=selected['weight']
oldcode=np.load(D/'code_bonus.npy');newcode=oldcode.copy()
fixed=np.load(D/'fixed1051_scores.npy');ocr=np.load(D/'selected_ocr_scores.npy')
audit={a['query']:a for a in read(D/'SELECTIVE_OCR_MATCHES.json') if a['threshold']==selected['threshold']}
for i,q in enumerate(qids):
 if q not in audit:continue
 parts={}
 for m in audit[q]['matches']:
  a=parts.setdefault(m['query_code'],np.zeros(len(ids)));ix=[ri[r] for r in m['references']]
  idf=math.log(1+(len(ids)-len(ix)+.5)/(len(ix)+.5))/math.log(len(ids)+1)
  a[ix]=np.maximum(a[ix],idf)
 newcode[i]=sum(parts.values(),start=np.zeros(len(ids)))
score=ocr+w*np.maximum(oldcode-newcode,0) if selected['code_update'] else ocr.copy()
def rank(s,i):
 seen=set();n=0;positive=set(c[i]['positive_sha256'])
 for j in sorted(range(len(ids)),key=lambda j:(-float(s[j]),j)):
  if sha[j] in seen:continue
  seen.add(sha[j]);n+=1
  if sha[j] in positive:return n
 raise AssertionError('No allowed reference')
br=np.array([rank(s,i) for i,s in enumerate(fixed)])
rr=np.array([rank(s,i) for i,s in enumerate(score)])
excluded=set(read(V/'identifier_layout_v1/protocol.json')['excluded_queries'])
scopes={'all':np.arange(len(c)),'validation':np.array([i for i,x in enumerate(c) if x['split']=='val']),'test':np.array([i for i,x in enumerate(c) if x['split']=='test']),'retained':np.array([i for i,q in enumerate(qids) if q not in excluded]),'target54':np.array([i for i,q in enumerate(qids) if q not in excluded and br[i]>1])}
out=dict(status='complete',protocol='One global post-hoc control: keep the larger old/new aggregate code bonus for every gallery row. All OCR threshold/weight parameters frozen from the prior validation selection. Same rule for all1117 queries; no truth or case-specific routing. Original replacement result and primary2x2 remain unchanged.',selection=selected,metrics={s:metrics(rr,c,ix,br) for s,ix in scopes.items()},repairs=[qids[i] for i in range(len(c)) if br[i]>1 and rr[i]==1],harms=[qids[i] for i in range(len(c)) if br[i]==1 and rr[i]>1],limits=['Designed after observing a replacement failure; exploratory control, not independent confirmation.','Can retain a previous false code match; future evaluation must measure that risk.'])
np.save(D/'retained_ocr_scores.npy',score)
csvwrite(D/'ocr_retention_cases1117.csv',[dict(query=q,baseline_rank=int(br[i]),rank=int(rr[i]),winner=ids[int(np.argmax(score[i]))]) for i,q in enumerate(qids)])
write(D/'OCR_EVIDENCE_RETENTION.json',out)
print(json.dumps(out),flush=True)
