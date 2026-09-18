#!/usr/bin/env python3
"""Freeze/E0 and time-bounded generation-only full-C128 natural coverage gate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'src/rc_aslo_xf/rc_cycle_closed_region_generation_v1.py'
TEST=ROOT/'tests/test_rc_cycle_closed_region_generation_v1.py'
PLAN=ROOT/'plan/RC_CYCLE_CLOSED_REGION_GENERATION_QUALIFICATION_V1_20260908.md'
ATOM_CLASS=ROOT/'src/rc_aslo_xf/reference_conditioned_latent_proposal_p_only_v1.py'
AUTH=ROOT/'registry/rc_cycle_closed_region_generation_authority_v1_20260908.json'
OUT=ROOT/'results/rc_cycle_closed_region_generation_v1'
E0=ROOT/'results/rc_cycle_closed_region_generation_e0_v1/result.json'
POST=ROOT/'registry/rc_lth_p_only_natural_optimization_postjoin_authority_v1_20260907.json'
POSTV=ROOT/'registry/rc_lth_p_only_natural_optimization_postjoin_authority_v1_20260907.independent_validation.json'


def need(x,message):
    if not x:raise RuntimeError(message)


def sha(p):
    p=Path(p).resolve();need(not any(s in str(p).lower() for s in ['d1_mi','grozi','ima++','gisc_prerecall_universe']),'PROTECTED_PATH')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(p):return json.loads(p.read_text())


def binding(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}


def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def atomic(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        need(read(p)==x,'IMMUTABLE_ARTIFACT_DRIFT');return
    fd,n=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(x,f,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.chmod(n,0o444);os.link(n,p)
    finally:os.unlink(n)


def freeze():
    need(not AUTH.exists(),'AUTHORITY_EXISTS')
    tests=load(TEST,'cycle_generation_e0_tests');receipt=tests.run_synthetic_e0()
    need(receipt['status']=='SYNTHETIC_ENGINEERING_E0_PASS','E0_FAILED')
    receipt.pop('runner_output',None)  # Wall time is diagnostic, never a frozen identity.
    receipt['bindings']={'core':binding(CORE),'tests':binding(TEST),'plan':binding(PLAN)}
    atomic(E0,receipt)
    shards=[]
    for s in range(8):
        base=ROOT/f'results/rc_lth_p_only_natural_atom_bank_train_v1/shard{s:02d}'
        p=base/'atom_banks.pt';v=ROOT/f'results/rc_lth_p_only_natural_atom_bank_train_v1/validation/shard{s:02d}.json'
        vd=read(v);need(vd['payload_sha256']==sha(p) and all(vd['checks'].values()),'SOURCE_VALIDATION_FAILED')
        shards.append({'shard':s,'payload':binding(p),'validation':binding(v)})
    a={'status':'RC_CYCLE_CLOSED_REGION_GENERATION_AUTHORIZED','scope':'ONE_SHOT_GENERATION_COVERAGE_ONLY',
       'sources':{'core':binding(CORE),'tests':binding(TEST),'plan':binding(PLAN),'atom_class':binding(ATOM_CLASS),'runner':binding(Path(__file__)),
                  'e0':binding(E0),'postjoin':binding(POST),'postjoin_validation':binding(POSTV)},
       'shards':shards,'output_rel':str(OUT.relative_to(ROOT)),'candidate_count':128,'query_count':32,
       'target_coverage_required':26,'selector_training_authorized':False,'formal_P0_authorized':False,
       'automatic_stage_advance':False,'scientific_GO_or_NO_GO':None}
    atomic(AUTH,a)
    print(json.dumps({'status':a['status'],'e0_tests':receipt['tests_run'],'authority':str(AUTH)}),flush=True)


def independent_component_check(bank,result):
    """Independent union-find check of complete maximal region membership."""
    # Source equality uses a separate integer-ratio calculation from raw coordinates.
    gh,gw=map(int,bank.query_grid_shape);valid=bank.valid_mask.tolist();source=bank.source_query_indices.tolist()
    eligible=[]
    for i,(x,y) in enumerate(bank.source_reverse_query_xy.tolist()):
        if valid[i] and -1<=x<1 and -1<=y<1:
            xn,xd=float(x).as_integer_ratio();yn,yd=float(y).as_integer_ratio()
            col=((xn+xd)*gw)//(2*xd);row=((yn+yd)*gh)//(2*yd)
            if row*gw+col==source[i]:eligible.append(i)
    need(eligible==result['cycle_closed_atom_indices'],'INDEPENDENT_CYCLE_AXIS_MISMATCH')
    q=bank.query_rc.tolist();r=bank.reference_rc.tolist();qi=bank.query_indices.tolist();ri=bank.reference_indices.tolist()
    parent={i:i for i in eligible};loc={tuple(q[i]):i for i in eligible}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in eligible:
        row,col=q[i]
        for other in ((row+1,col),(row,col+1)):
            j=loc.get(other)
            if j is not None and max(abs(r[i][d]-r[j][d]) for d in (0,1))<=2:parent[find(j)]=find(i)
    groups={}
    for i in eligible:groups.setdefault(find(i),[]).append(i)
    expected=[sorted(g,key=lambda i:qi[i]) for g in groups.values() if len(g)>=4 and len({ri[i] for i in g})>=2]
    expected.sort(key=lambda g:tuple(qi[i] for i in g))
    need([c['atom_indices'] for c in result['components']]==expected,'INDEPENDENT_COMPONENT_MISMATCH')
    need(result['structural_h0']==(not expected),'INDEPENDENT_H0_MISMATCH')
    for component in result['components']:
        ids=component['atom_indices']
        need(component['query_indices']==[qi[i] for i in ids] and component['reference_indices']==[ri[i] for i in ids], 'INDEPENDENT_PROVENANCE_MISMATCH')


def run():
    began=time.monotonic();a=read(AUTH);authsha=sha(AUTH)
    need(a['status']=='RC_CYCLE_CLOSED_REGION_GENERATION_AUTHORIZED','AUTHORITY_INVALID')
    for name,b in a['sources'].items():
        if name not in ('postjoin','postjoin_validation'):need(sha(ROOT/b['path'])==b['sha256'],'SOURCE_DRIFT:'+name)
    need(a['sources']['runner']['sha256']==sha(Path(__file__)),'RUNNER_DRIFT')
    # Actual label-bearing reads are blocked until all prejoin output records are closed.
    closed=[False];early=[0]
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.path.abspath(os.fsdecode(args[0]))
            if any(s in p.lower() for s in ('d1_mi','grozi','ima++','gisc_prerecall_universe')):raise PermissionError('PROTECTED_PATH')
            if p in (str(POST),str(POSTV)) and not closed[0]:early[0]+=1;raise PermissionError('EARLY_POSTJOIN')
    sys.addaudithook(audit)
    import torch
    torch.set_num_threads(4)
    core=load(CORE,'cycle_closed_generation_core')
    legacy=load(ATOM_CLASS,'rc_lth_p_only_core_standalone_v1')
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    OUT.mkdir(parents=True,exist_ok=True);qdir=OUT/'prejoin';qdir.mkdir(exist_ok=True)
    source_records=[]
    for shard in a['shards']:
        p=ROOT/shard['payload']['path'];need(sha(p)==shard['payload']['sha256'],'ATOM_PAYLOAD_DRIFT')
        payload=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        need(payload['access_vector']==[0]*12 and len(payload['records'])==4,'ATOM_SCHEMA_DRIFT')
        for record in payload['records']:
            key=record['resource_key'];dest=qdir/(key+'.json')
            keys=record['candidate_keys'];need(len(keys)==len(set(keys))==128,'C128_DRIFT')
            if dest.exists():
                saved=read(dest);need(saved['authority_sha256']==authsha and saved['source_payload_sha256']==shard['payload']['sha256'] and saved['candidate_keys']==keys,'RESUME_LINEAGE_DRIFT')
            else:
                if stop[0] or time.monotonic()-began>2700:
                    print(json.dumps({'status':'GENERATION_RESUMABLE','completed_queries':len(list(qdir.glob('q_*.json')))}),flush=True);return 75
                banks={b.candidate_resource_key:b for b in record['real_banks']}
                controls={'REAL':record['real_banks'],'C_BIND':record['c_bind_banks']}
                need([b.reference_resource_key for b in controls['C_BIND']]==record['c_bind_source_keys'],'C_BIND_DONOR_AXIS_DRIFT')
                need(set(record['c_bind_source_keys'])==set(keys) and all(k!=s for k,s in zip(keys,record['c_bind_source_keys'])),'C_BIND_NOT_DERANGED')
                controls['P_COORD']=[legacy.coordinate_destroy_atom_bank(b,record['p_coord_derivation']['namespace'])[0] for b in record['real_banks']]
                by_control={}
                for name,items in controls.items():
                    generated=[]
                    for bank in items:
                        result=core.enumerate_cycle_closed_regions(bank)
                        independent_component_check(bank,result)
                        need(core.replay_cycle_closed_regions(bank,core.serialize_cycle_closed_regions(result))==result,'SERIALIZATION_REPLAY_FAILED')
                        generated.append(result)
                    need([x['candidate_resource_key'] for x in generated]==keys,'GENERATION_AXIS_DRIFT')
                    by_control[name]=generated
                for real,coord in zip(by_control['REAL'],by_control['P_COORD']):
                    need(real['cycle_closed_atom_indices']==coord['cycle_closed_atom_indices'],'P_COORD_CHANGED_SOURCE_CYCLE_QUALIFICATION')
                real_by_key={r['candidate_resource_key']:r for r in by_control['REAL']}
                for k in reversed(keys):need(core.enumerate_cycle_closed_regions(banks[k])==real_by_key[k],'CANDIDATE_REORDER_CHANGED_GENERATION')
                saved={'query_resource_key':key,'candidate_keys':keys,'authority_sha256':authsha,'source_payload_sha256':shard['payload']['sha256'],
                       'families':by_control,'independent_union_find_all_controls_pass':True,'candidate_reorder_pass':True,'serialization_replay_pass':True}
                atomic(dest,saved)
                print(json.dumps({'event':'PREJOIN_QUERY_SEALED','query':key,'real_nonempty':sum(not x['structural_h0'] for x in by_control['REAL'])}),flush=True)
            source_records.append({'query_resource_key':key,'file':str(dest.relative_to(OUT)),'sha256':sha(dest)})
        del payload
    need(len(source_records)==len({r['query_resource_key'] for r in source_records})==32,'PREJOIN_32_NOT_CLOSED')
    source_records.sort(key=lambda r:r['query_resource_key'])
    atomic(OUT/'prejoin_seal.json',{'status':'GENERATION_PREJOIN_FULL32_C128_CLOSED','authority_sha256':authsha,'records':source_records,'prejoin_label_reads':early[0]})
    closed[0]=True
    need(sha(POST)==a['sources']['postjoin']['sha256'] and sha(POSTV)==a['sources']['postjoin_validation']['sha256'],'POSTJOIN_SOURCE_DRIFT')
    roles={r['query_resource_key']:r for r in read(POST)['records']}
    stats=[];nonempty={n:0 for n in ('REAL','C_BIND','P_COORD')};sizes={n:[] for n in nonempty}
    for src in source_records:
        q=read(OUT/src['file']);role=roles[q['query_resource_key']];targets=set(role['target_candidate_resource_keys']);present=targets.intersection(q['candidate_keys'])
        flags={}
        for name,fs in q['families'].items():
            nonempty[name]+=sum(not f['structural_h0'] for f in fs)
            sizes[name].extend(len(c['atom_indices']) for f in fs for c in f['components'])
            flags[name]=any(f['candidate_resource_key'] in present and not f['structural_h0'] for f in fs)
        stats.append({'query_resource_key':q['query_resource_key'],'target_present':bool(present),'target_legal_region':flags})
    coverage=sum(r['target_legal_region']['REAL'] for r in stats if r['target_present']);absent=sum(not r['target_present'] for r in stats)
    passed=absent==0 and coverage>=26
    status='ABORT_FROZEN_TRAIN32_TARGET_ABSENT_FROM_C128' if absent else ('RC_CYCLE_CLOSED_REGION_GENERATION_COVERAGE_PASS' if passed else 'P_GENERATION_COVERAGE_NO_GO')
    result={'status':status,
            'claim_level':'GENERATION_ONLY_OPENED_TRAIN32_NO_P_EVIDENCE_OR_RETRIEVAL_CLAIM','generation_gate_pass':passed,
            'query_count':32,'candidate_count':128,'candidate_control_records':32*128*3,'target_coverage_count':coverage,
            'target_absent_from_C128_count':absent,'target_absence_class':'CANDIDATE_RECALL_NOT_P_FAILURE' if absent else None,'target_coverage_required':26,'nonempty_candidate_counts':nonempty,
            'structural_h0_fractions':{n:1-v/4096 for n,v in nonempty.items()},
            'component_size_summaries':{n:{'count':len(v),'min':min(v) if v else None,'max':max(v) if v else None,'mean':sum(v)/len(v) if v else None} for n,v in sizes.items()},
            'records':stats,'prejoin_seal_sha256':sha(OUT/'prejoin_seal.json'),'authority_sha256':authsha,
            'checks':{'all_records_independent_union_find_checked':True,'serialization_replay':True,'candidate_reorder':True,'P_COORD_source_cycle_invariant':True,'prejoin_label_reads_zero':early[0]==0},
            'selector_training_count':0,'new_backbone_forward_count':0,'target_insertion_count':0,'formal_panel_read_count':0,
            'margin_comparator_binding_coordinate_P_evidence_gates_tested':False,'scientific_GO_or_NO_GO':None,
            'automatic_stage_advance':False,'next_stage':'SHARED_SELECTOR_AND_P_EVIDENCE_CONTRACT_REQUIRED' if passed else None}
    atomic(OUT/'result.json',result)
    print(json.dumps({k:result[k] for k in ['status','target_coverage_count','nonempty_candidate_counts','next_stage']}),flush=True);return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['freeze','run'],required=True);args=parser.parse_args()
    if args.phase=='freeze':freeze()
    else:raise SystemExit(run())
