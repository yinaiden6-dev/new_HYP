#!/usr/bin/env python3
"""Existing open population only: pixels, token collisions and exposure inventory."""
import hashlib,json,os,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'programs'))
import run_rc_paired_local_evidence_oof4_v1 as N
OUT=ROOT/'results/rc_six_cause_isolation_v1/inventory'
AUTH=ROOT/'registry/rc_six_cause_inventory_authority_v1_20260913.json'
read,bind,checked,write,need=N.read,N.bind,N.checked,N.write,N.need

def main():
    need(os.environ.get('SLURM_JOB_ID'),'SLURM_REQUIRED')
    a=read(AUTH);need(a['program']==bind(__file__),'CODE_PIN')
    w=read(checked(a['worker']));roles=read(checked(a['curator']))['records']
    val=read(checked(a['validation']));need(val['worker']==a['worker'] and val['curator']==a['curator'],'METADATA_CHAIN')
    ledger=read(checked(a['ledger']))['queries'];formal=read(checked(a['exclusion']))
    ids=set(formal['exclude_query_ids']);hs=set(formal['exclude_image_sha256'])
    eligible=[r for r in ledger if r['query_id'] not in ids and r['source_image_sha256'] not in hs]
    rolemap={r['query_id']:r for r in roles};byimage={};token_rows={};sources={};cache={};result=[]
    # Only the existing H593 metadata and its already qualified public RAW inputs.
    for i,r in enumerate(w['records']):
        if 'reuse' in r:
            b=r['reuse']['token_raw'];j=r['reuse']['record_index']
        else:
            s,j=divmod(r['missing_ordinal'],8);folder=ROOT/f'results/rc_new_hyp593_oof5_v1/raw/shard{s:02d}'
            v=read(folder/'validation.json');rec=read(checked(v['receipt']))
            need(v['status']=='H593_RAW_CPU_REPLAY_PASS' and v['payload']==rec['payload'],'MISSING_RAW_CHAIN')
            b=dict(payload=rec['payload'],receipt=bind(folder/'receipt.json'),validation=bind(folder/'validation.json'))
        p=b['payload']['path']
        if p not in sources:
            rv=read(checked(b['validation']));rr=read(checked(b['receipt']))
            need(rv['payload']==rr['payload']==b['payload'] and rv['receipt']==b['receipt'],'RAW_CHAIN')
            # Retain only query arrays' fingerprints, never accumulate large reference tensors.
            data=torch.load(checked(b['payload']),weights_only=True,mmap=True,map_location='cpu')
            cache={p:[dict(query_id=q['query_id'],image=q['query_source_sha256'],token=N.tsha(q['query_tokens']),stored=q['query_tokens_sha256'],shape=list(q['query_tokens'].shape)) for q in data['records']]}
            sources[p]=b
            token_rows[p]=cache[p]
            del data
        q=token_rows[p][j];need(q['image']==r['source_image_sha256'] and q['token']==q['stored'],'QUERY_BINDING')
        path=Path(r['query_image_path']);need(N.sha(path)==r['source_image_sha256'],'IMAGE_BYTES')
        with Image.open(path) as im:
            exif=ImageOps.exif_transpose(im).convert('RGB');shape=exif.size
            pixel=hashlib.sha256(json.dumps(shape).encode()+exif.tobytes()).hexdigest()
        role=rolemap[r['query_id']]
        result.append(dict(query_id=r['query_id'],original_query_id=role['original_query_id'],image_sha256=r['source_image_sha256'],pixel_sha256=pixel,token_sha256=q['token'],token_shape=q['shape'],identity=role['identity'],group=role['group'],component=role['component']))
        if (i+1)%64==0:print(dict(completed=i+1),flush=True)
    def collisions(key):
        groups=defaultdict(list)
        for r in result:groups[r[key]].append(r)
        duplicate=[rs for rs in groups.values() if len(rs)>1]
        conflict=[rs for rs in duplicate if len({r['identity'] for r in rs})>1]
        upper=sum(max(sum(r['identity']==y for r in rs) for y in {r['identity'] for r in rs}) for rs in groups.values())
        return dict(duplicate_groups=duplicate,conflicting_identity_groups=conflict,deterministic_identity_accuracy_ceiling_from_exact_collisions=upper,total=len(result),no_conflict_does_not_imply_identifiability=True)
    exposed={r['image_sha256'] for r in result};eligible_h={r['source_image_sha256'] for r in eligible}
    need(eligible_h==exposed and len(result)==593,'COMPLETE_OPEN_POPULATION')
    report=dict(status='SIX_CAUSE_OPEN_POPULATION_INVENTORY_COMPLETE',authority=bind(AUTH),sources=list(sources.values()),records=result,pixel=collisions('pixel_sha256'),token=collisions('token_sha256'),byte=collisions('image_sha256'),population=dict(ledger_rows=len(ledger),excluded_rows=len(ledger)-len(eligible),eligible_rows=len(eligible),eligible_unique_images=len(eligible_h),already_opened_unique_images=len(exposed),unopened_eligible_images=len(eligible_h-exposed),unopened_confirmation_available=False),limits=['No private formal labels read','No appearance sufficiency conclusion from absence of collisions','Tokens checked for equality only; readable identity requires separate probes','All593 are already development-exposed; resplitting is not external confirmation'])
    write(OUT/'result.json',report)
    saved=read(OUT/'result.json');need(saved['population']['eligible_unique_images']==len({r['image_sha256'] for r in saved['records']}),'INDEPENDENT_SET_COUNTS')
    write(OUT/'validation.json',dict(status='SIX_CAUSE_INVENTORY_HASH_AND_SET_PASS',result=bind(OUT/'result.json'),authority=bind(AUTH),rows=593,formal_private_label_reads=0))
    print(dict(population=report['population'],pixel_conflicts=len(report['pixel']['conflicting_identity_groups']),token_conflicts=len(report['token']['conflicting_identity_groups'])),flush=True)
if __name__=='__main__':main()
