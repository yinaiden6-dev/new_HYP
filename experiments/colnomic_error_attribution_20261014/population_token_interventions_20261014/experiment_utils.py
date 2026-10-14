import json,csv
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
def csvwrite(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def zscore(a):
 a=np.asarray(a,dtype=np.float64);mu=a.mean(-1,keepdims=True);sd=a.std(-1,keepdims=True)
 return np.divide(a-mu,sd,out=np.zeros_like(a),where=sd>0)
class Ranking:
 def __init__(self,rids,sha,cohort):
  self.rids=np.array(rids);self.sha=np.array(sha);self.cohort=cohort
  self.order=np.argsort(self.sha,kind='stable');ss=self.sha[self.order]
  self.starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1];self.usha=ss[self.starts]
  self.group=np.searchsorted(self.starts,np.arange(len(ss)),side='right')-1
  self.pos=np.array([[s in set(r['positive_sha256']) for s in self.usha] for r in cohort])
  self.rawpos=np.array([[s in set(r['positive_sha256']) for s in self.sha] for r in cohort])
 def one(self,s,i):
  s=np.asarray(s);ordered=s[self.order];best=np.maximum.reduceat(ordered,self.starts)
  first=np.minimum.reduceat(np.where(ordered==best[self.group],self.order,len(s)),self.starts)
  ps=np.where(self.pos[i],best,-np.inf);p=ps.max();pi=first[ps==p].min()
  rank=int(((best>p)|((best==p)&(first<pi))).sum()+1)
  top=int(np.argmax(s));neg=int(np.argmax(np.where(self.rawpos[i],-np.inf,s)))
  return dict(rank=rank,top1=str(self.rids[top]),positive=str(self.rids[pi]),negative=str(self.rids[neg]),margin=float(s[pi]-s[neg]))
 def all(self,a):
  rank=[];top=[];margins=[]
  for i,s in enumerate(a):
   r=self.one(s,i);rank.append(r['rank']);top.append(r['top1']);margins.append(r['margin'])
  return np.array(rank),np.array(top),np.array(margins)
def metrics(ranks,cohort,selection=None,baseline=None):
 ids=np.arange(len(cohort)) if selection is None else np.asarray(selection);rr=ranks[ids]
 groups={}
 for i in ids:groups.setdefault(cohort[i]['package_id'],[]).append(int(i))
 out=dict(n=len(ids),identities=len(groups),correct=int((rr==1).sum()),accuracy=float((rr==1).mean()),mrr=float((1/rr).mean()),macro_accuracy=float(np.mean([np.mean(ranks[v]==1) for v in groups.values()])),macro_mrr=float(np.mean([np.mean(1/ranks[v]) for v in groups.values()])))
 if baseline is not None:out.update(repairs=int(((baseline[ids]>1)&(rr==1)).sum()),harms=int(((baseline[ids]==1)&(rr>1)).sum()))
 return out
