#!/usr/bin/env python3
"""All-reference, outcome-blind feature materialization and independent index replay."""
import ast,json,math,os,sys,argparse,hashlib,subprocess,uuid
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as FC
OUT=M.OUT;AUTH=ROOT/'registry/rc_new_hyp593_feature_authority_v1_20260911.json'
SOURCE=ROOT/'programs/cache_rc_train128_disagreement_features_v1.py'
CACHE={}
def operator():
 ns={'torch':torch,'np':np,'math':math,'need':M.need};tree=ast.parse(SOURCE.read_text());names={'primary','independent','c4_from_similarity'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];M.need(len(nodes)==3,'SOURCE_FUNCTIONS');exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns);return ns

def guard():
 M.need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED');M.need(M.datetime.now(M.timezone.utc)<M.DEADLINE,'USER_DEADLINE')
 a=M.read(AUTH)
 for b in a['sources'].values():M.checked(b)
 def audit(e,args):
  if e!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  s=os.path.abspath(os.fsdecode(args[0])).lower();M.need(not any(x in s for x in ('curator_roles','role_shards','target_join','rc_opened_','d1-mi','d1_mi','grozi','gisc_prerecall_universe','/reports/')),'NO_LABELS_OR_PROTECTED_RESULTS')
 sys.addaudithook(audit)

def loadsource(b):
 key=b['payload']['sha256']
 if key not in CACHE:
  val=M.read(M.checked(b['validation']));rec=M.read(M.checked(b['receipt']));M.need(val['payload']==rec['payload']==b['payload'] and val['receipt']==b['receipt'],'SOURCE_ENVELOPE')
  M.need(val['status'].endswith('CPU_REPLAY_PASS'),'QUALIFIED_SOURCE')
  CACHE[key]=torch.load(M.checked(b['payload']),map_location='cpu',weights_only=True,mmap=True)
 return CACHE[key]
def selected(kind,shard):
 rs=M.read(M.META/'worker_manifest.json')['records'];rs=[r for r in rs if ('reuse' in r)==(kind=='reuse')]
 return rs[shard*8:(shard+1)*8]
def pair(w):
 if 'reuse' in w:
  z=w['reuse'];raw=loadsource(z['token_raw']);roma=loadsource(z['roma']);j=z['record_index'];expected=z['source_query_id']
 else:
  j=w['missing_ordinal']%8;s=w['missing_ordinal']//8;b={}
  for k in ('raw','roma'):
   folder=OUT/k/f'shard{s:02d}';rec=M.read(folder/'receipt.json');b[k]=dict(payload=rec['payload'],receipt=M.bind(folder/'receipt.json'),validation=M.bind(folder/'validation.json'))
  raw=loadsource(b['raw']);roma=loadsource(b['roma']);expected=w['query_id']
 q,r=raw['records'][j],roma['records'][j]
 M.need(q['query_id']==r['query_id']==expected and q['query_source_sha256']==r['query_source_sha256']==w['source_image_sha256'],'QUERY_SOURCE_ALIGNMENT')
 M.need(q['candidate_physical_rows']==r['candidate_physical_rows'] and torch.equal(q['candidate_raw_scores'],r['candidate_raw_scores']),'C128_ALIGNMENT')
 return q,r,raw['references']
def compute(kind,shard,replay):
 op=operator();out=[]
 for w in selected(kind,shard):
  q,r,refs=pair(w);axis=q['candidate_physical_rows'];M.need(len(axis)==len(r['candidates'])==128 and axis==sorted(set(axis)),'FULL_C128')
  qv=q['query_tokens'].double();qv=qv/qv.norm(p=2,dim=1,keepdim=True).clamp_min(1e-12)
  stats=[];scores=[]
  for j,c in enumerate(r['candidates']):
   ref=refs[axis[j]];M.need(c['candidate_position']==j and c['physical_row']==axis[j] and c['reference_tokens_sha256']==ref['tokens_sha256'],'REFERENCE_MAP_ALIGNMENT')
   rv=ref['tokens'].double();rv=rv/rv.norm(p=2,dim=1,keepdim=True).clamp_min(1e-12);sim=qv@rv.T;wq,wr=c['query_visibility'],c['reference_visibility']
   M.need(wq.shape==(len(qv),) and wr.shape==(len(rv),),'MAP_AXIS')
   c4=op['c4_from_similarity'](sim,wq,wr)
   M.need(all(float(c['old_scores'][k]).hex()==float(v).hex() for k,v in zip(('real_score','visibility_mass','query_control_score','reference_control_score'),c4)),'ORIGINAL_C4_BITS')
   v=op['independent'](sim,wq,wr)[0] if replay else op['primary'](sim,wq,wr)
   stats.append(v);scores.append({k:float(x) for k,x in zip(('real_score','visibility_mass','query_control_score','reference_control_score'),c4)})
  raw=q['candidate_raw_scores'].tolist();winner=axis.index(q['candidate_ranked_physical_rows'][0]);chall=[p for p in range(128) if p!=winner];modes={}
  for mode in ('REAL','CBIND'):
   ids=list(range(128)) if mode=='REAL' else [(i+64)%128 for i in range(128)];ev={i:scores[ids[i]] for i in range(128)}
   X=torch.stack([FC.candidate_feature(raw,ev,c,winner) for c in chall]);context=[];df=[]
   for i,c in enumerate(chall):
    fc,_,dc,rc,_=stats[ids[c]];fw,_,dw,rw,_=stats[ids[winner]]
    context.append([1.,-float(X[i,0]),dw-dc,FC.symmetric(rc,rw)]);df.append(FC.symmetric(fc,fw))
   modes[mode]=dict(X=X,context=torch.tensor(context,dtype=torch.float64),dF=torch.tensor(df,dtype=torch.float64))
  out.append(dict(query_id=w['query_id'],execution_ordinal=w['execution_ordinal'],source_image_sha256=w['source_image_sha256'],candidate_physical_rows=axis,raw_ranked_physical_rows=q['raw_ranked_physical_rows'],winner=winner,challenger_positions=chall,modes=modes))
  print(json.dumps(dict(event='FEATURE_QUERY_REPLAY' if replay else 'FEATURE_QUERY_READY',query_id=w['query_id'])),flush=True)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['reuse','missing']);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--replay',action='store_true');ap.add_argument('--nonce');a=ap.parse_args();guard();torch.set_num_threads(8);torch.set_num_interop_threads(1)
 rs=selected(a.kind,a.shard);M.need(rs,'EMPTY_FEATURE_SHARD');folder=OUT/'features'/a.kind/f'shard{a.shard:02d}';rows=compute(a.kind,a.shard,a.replay)
 if a.replay:
  M.need(a.nonce==os.environ.get('H593_FEATURE_NONCE'),'FRESH_REPLAY');receipt=M.read(folder/'receipt.json');old=torch.load(M.checked(receipt['payload']),map_location='cpu',weights_only=True)
  M.need(len(old['records'])==len(rows),'REPLAY_LENGTH')
  for x,y in zip(old['records'],rows):
   M.need({k:v for k,v in x.items() if k!='modes'}=={k:v for k,v in y.items() if k!='modes'},'REPLAY_METADATA')
   for mode in ('REAL','CBIND'):
    for key in ('X','context','dF'):M.need(torch.equal(x['modes'][mode][key],y['modes'][mode][key]),'INDEPENDENT_FEATURE_BITS')
  M.write(folder/'validation.json',dict(status='H593_FEATURE_INDEPENDENT_INDEX_REPLAY_PASS',payload=receipt['payload'],receipt=M.bind(folder/'receipt.json'),authority=M.bind(AUTH),count=len(rows),fresh_nonce=a.nonce))
 else:
  folder.mkdir(parents=True,exist_ok=True);p=folder/'payload.pt';M.need(not p.exists(),'IMMUTABLE_FEATURES');torch.save(dict(records=rows,authority=M.bind(AUTH)),p)
  M.write(folder/'receipt.json',dict(payload=M.bind(p),authority=M.bind(AUTH),count=len(rows)))
  nonce=uuid.uuid4().hex;subprocess.run([sys.executable,__file__,a.kind,'--shard',str(a.shard),'--replay','--nonce',nonce],env=dict(os.environ,H593_FEATURE_NONCE=nonce,CUDA_VISIBLE_DEVICES=''),check=True)
if __name__=='__main__':main()
