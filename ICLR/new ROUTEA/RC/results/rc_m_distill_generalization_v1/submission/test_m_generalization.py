import sys,importlib.util,tempfile,os,json
from pathlib import Path
r=Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC');sys.path.insert(0,str(r/'programs'))
spec=importlib.util.spec_from_file_location('gen',r/'programs/run_rc_m_distill_generalization_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Source:
 def __init__(self):self.records=[{'query_id':x} for x in ['a','b','c']];self.visits=[]
 def __iter__(self):
  for x in self.records:self.visits.append(x['query_id']);yield x
s=Source();assert list(m.Subset(s,['b']))==[{'query_id':'b'}];assert s.visits==['b'] and len(s.records)==3
os.environ['SLURM_JOB_ID']='7001'
with tempfile.TemporaryDirectory() as tmp:
 m.OUT=Path(tmp);calls=[];m.submit=lambda stage,n=0,dep=None,array=None:calls.append([stage,n,dep,array]) or str(8000+len(calls))
 m.advance({},'cache-join',0);assert calls==[['run',457,'7001',None],['inner',0,'7001',None],['inner',1,'7001',None]]
with tempfile.TemporaryDirectory() as tmp:
 m.OUT=Path(tmp);m.AUTH=Path(tmp)/'authority.json';m.write(m.AUTH,{})
 for arm in range(2):
  m.write(m.OUT/f'inner{arm}/result.json',dict(status='INNER_COMPLETE'))
  for it,err in [(0,1.),(10,.8),(20,.9)]:m.write(m.OUT/f'inner{arm}/snapshots'/f'iter{it:03d}.json',dict(authority=m.bind(m.AUTH),outer_held_reads=0,ridge=[0.,.001][arm],iteration=it,group_validation_error=err))
 m.select({});v=m.read(m.OUT/'inner_selection.json');assert v['iteration']==10 and v['ridge']==.001 and v['outer_held_reads']==0
print(json.dumps(dict(status='GROUP_FILTER_SELECTION_AND_SHARED_CACHE_DISPATCH_PASS')))
