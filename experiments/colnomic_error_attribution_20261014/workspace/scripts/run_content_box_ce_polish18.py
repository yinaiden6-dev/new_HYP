"""Frozen fold0: cached ColNomic -> natural C128 -> original RoMa -> X11 -> infer.py.

No encoder or training entry point is instantiated. All new files live in a
separate deployment cache. The package's infer.py and all existing caches stay
unchanged. Stages: prepare, run, report, verify, status.
"""
import argparse
import ast
from collections import Counter, OrderedDict
from contextlib import contextmanager
import csv
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'cache/content_box_ce_polish18/upstream'
RC = UPSTREAM/'ICLR/new ROUTEA/RC'
PACKAGE = ROOT/'weights/content_box_ce_polish18'
SOURCE_COMMIT = 'd0d04731668d9558d0dcb21df540f9f0a46ba09b'
DEPLOY = ROOT/'cache/content_box_ce_polish18/deployment'
MODEL = PACKAGE/'models/fold0.json'
FEATURE_SCHEMA = 'content_box_ce_polish18_x11_v1'
COLUMNS = ['RAW','s','m','l','q','r','f','b','gq','gr','u']
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RC/'src'))
from drugpack.method.token_layout import token_layout
from evaluation.data import file_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csvread(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def csvwrite(path, rows, fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_npz(path, values):
    """Lossless native overlaps, FP64 pooled evidence; fast DEFLATE level 1."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name, array in values.items():
                with archive.open(name+'.npy','w',force_zip64=True) as stream:
                    np.lib.format.write_array(stream,np.asarray(array),allow_pickle=False)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def claim(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False;return
        try:yield True
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def original_functions(path, names, globals_):
    """Load only audited pure definitions, excluding any training/audit main."""
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    require({n.name for n in nodes}==set(names),'Missing original mathematical function')
    scope=dict(globals_)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope


def math_functions():
    core=original_functions(RC/'programs/run_romav2_colnomic_visibility_xf_six_case_v1.py',
        ['oriented','cell_means','score'],dict(torch=torch,F=F,Image=Image,math=math))
    content=original_functions(RC/'programs/run_rc_h593_learned_colnomic_only_v1.py',
        ['active','statistics','numpy_statistics','features'],dict(torch=torch,np=np,need=require))
    spec=importlib.util.spec_from_file_location('polish18_original_gate',RC/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    return core,content,gate


def token_hash(tokens):
    x=tokens.detach().cpu().contiguous()
    return hashlib.sha256(str(x.dtype).encode()+json.dumps(list(x.shape),separators=(',',':')).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()


def natural_candidates(scores, references):
    require(len(scores)==len(references) and np.isfinite(scores).all(),'Invalid full-gallery score row')
    seen=set();ranked=[]
    for row in np.argsort(-np.asarray(scores),kind='stable'):
        row=int(row);identity=references[row]['sha256']
        if identity in seen:continue
        seen.add(identity);ranked.append(row)
        if len(ranked)==128:break
    require(len(ranked)==128,'Natural gallery must contain at least 128 unique designs')
    # Original worker requires sorted physical rows, not score-ordered X11.
    axis=sorted(ranked)
    return axis,ranked,axis.index(ranked[0])


class Banks:
    def __init__(self, processor=True):
        self.q=torch.load(ROOT/'cache/image_embeddings/colnomic7b/photos.pt',map_location='cpu',weights_only=True,mmap=True)
        self.r=torch.load(ROOT/'cache/image_embeddings/colnomic7b/references.pt',map_location='cpu',weights_only=True,mmap=True)
        self.s=torch.load(ROOT/'cache/scores/colnomic7b/image.pt',map_location='cpu',weights_only=True,mmap=True)
        if processor:
            from colpali_engine.models import ColQwen2_5_Processor
            self.processor=ColQwen2_5_Processor.from_pretrained(ROOT/'weights/colnomic7b/adapter',local_files_only=True)

    def inspect(self, record, index, reference=False):
        bank=self.r if reference else self.q
        require(bank['ids'][index]==record['id'],'Token ID mapping changed')
        require(bank['sha256'][index]==record['sha256'],'Token image SHA differs')
        path=ROOT/bank['paths'][index]
        require(path.resolve()==Path(record['path']).resolve(),'Token image path differs')
        require(file_hash(path)==record['sha256'],'Original image bytes changed: '+record['id'])
        with Image.open(path) as image:
            height,width=image.height,image.width
            orientation=int(image.getexif().get(274,1))
        layout=token_layout(self.processor,'colnomic7b',width,height)
        tokens=bank['embeddings'][index]
        require(tuple(tokens.shape)==(layout['total'],128),'Cached token layout differs: '+record['id'])
        image_tokens=tokens[layout['start']:layout['start']+layout['count']]
        require(torch.isfinite(tokens).all().item(),'Nonfinite cached tokens')
        return dict(raw_height=height,raw_width=width,exif_orientation=orientation,layout=layout,
            processor_input_frame='DECODED_RAW_BEFORE_EXIF',image_tokens_sha256=token_hash(image_tokens),
            all_tokens_sha256=token_hash(tokens),valid_content_tokens=int((image_tokens.double().abs().sum(1)>1e-6).sum()))

    def image_tokens(self, record, reference=False):
        bank=self.r if reference else self.q;layout=record['layout'];tokens=bank['embeddings'][record['bank_index']]
        out=tokens[layout['start']:layout['start']+layout['count']]
        require(token_hash(out)==record['image_tokens_sha256'],'Image tokens changed')
        return out


def model_checks():
    model=read(MODEL)
    require(model['fold']==0 and model['feature_schema']==FEATURE_SCHEMA,'Only fixed fold0 is deployed')
    payload=RC/'results/rc_h593_box_ce_polish_v1/fold0/payload.json'
    require(file_hash(payload)==model['provenance']['payload_sha256'],'Original fold0 payload provenance differs')
    require(model['parameters_hex']==read(payload)['parameters']['CONTENT_BOX_CE_POLISH18'],'Head parameters differ from original')
    example=PACKAGE/'prediction.json'
    subprocess.run([sys.executable,str(PACKAGE/'infer.py'),'--model',str(MODEL),'--input',str(PACKAGE/'example_input.json'),'--output',str(example)],check=True)
    require(read(example)==read(PACKAGE/'example_expected.json'),'Standalone example differs')
    return dict(fold=0,model_sha256=file_hash(MODEL),example_exact=True,example_logits=127,threshold=0.0)


def prepare(args):
    torch.set_num_threads(args.cpu_threads)
    checks=model_checks();banks=Banks();photos=csvread(ROOT/'dataset/lists/photos.csv');refs=csvread(ROOT/'dataset/lists/references.csv')
    queries=csvread(args.query_list)
    require(len({r['photo_id'] for r in queries})==len(queries),'Duplicate query IDs')
    require(banks.q['ids']==[p['photo_id'] for p in photos] and banks.r['ids']==[r['reference_id'] for r in refs],'Inventory order differs from token caches')
    require(banks.s['query_ids']==banks.q['ids'] and banks.s['reference_ids']==banks.r['ids'],'Score axes differ from tokens')
    require(banks.q['model']==banks.r['model'] and banks.q['model']['revision']=='09dbc9502b66605d5be56d2226019b49c9fd3293','ColNomic checkpoint revision differs')
    require(banks.q['family']==banks.r['family']=='colnomic_7b','Not ColNomic 7B tokens')
    for bank, inventory in [(banks.q,photos),(banks.r,refs)]:
        require(bank['sha256']==[p['sha256'] for p in inventory],'Image mapping differs from cache')
        require(bank['paths']==['dataset/'+p['path'] for p in inventory],'Image paths differ from cache')
    qi={p['photo_id']:i for i,p in enumerate(photos)}
    definitions=[RC/'programs/run_romav2_colnomic_visibility_xf_six_case_v1.py',
        RC/'programs/run_rc_h593_learned_colnomic_only_v1.py',RC/'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py',
        RC/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v1.py',RC/'src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py']
    checkpoint=ROOT/'weights/romav2/torchhub/hub/checkpoints/romav2.0.1.pt'
    rm=read(ROOT/'weights/romav2/manifest.json');require(file_hash(checkpoint)==rm['checkpoint']['sha256'],'RoMa checkpoint differs')
    files=[args.query_list,ROOT/'dataset/lists/photos.csv',ROOT/'dataset/lists/references.csv',
        ROOT/'cache/image_embeddings/colnomic7b/photos.pt',ROOT/'cache/image_embeddings/colnomic7b/references.pt',
        ROOT/'cache/scores/colnomic7b/image.pt',MODEL,PACKAGE/'infer.py',checkpoint,
        ROOT/'weights/colnomic7b/adapter/preprocessor_config.json',*definitions]
    inputs={str(p.resolve()):file_hash(p) for p in files}
    recipe=dict(model=checks,source_commit=SOURCE_COMMIT,inputs=inputs,columns=COLUMNS,candidate_budget=128,
        candidates='Natural cached full-gallery ranking; SHA-unique; score tie by physical row; sorted physical axis',
        roma=dict(version='v2.0.1',revision=rm['revision'],checkpoint_sha256=rm['checkpoint']['sha256'],setting='precise',
            H_lr=800,W_lr=800,H_hr=1280,W_hr=1280,bidirectional=True,threshold=None,amp='stock_bfloat16',seed=17),
        content='image tokens only; fp64; abs-sum > 1e-6; no new normalization',
        six_features='Original FP64 normalize inside score; exact EXIF geometry; half-length rolls; original candidate_feature',
        memory_execution='Native local correlation tiled by 16 spatial rows; original sampling/multiply and original 32-bit-index channel reduction splits; unchanged model settings',
        cache_storage='Lossless native overlap_AB/BA and FP64 u/v/C4/content statistics; no warp needed by this head',
        encoder_forwards=0,roma_training_updates=0,head_training_updates=0,target_insertion_count=0)
    signature=digest(recipe);output=DEPLOY/signature[:20]
    if (output/'manifest.json').exists():
        old=read(output/'manifest.json');require(old['recipe']==recipe,'Existing deployment recipe changed')
        atomic_json(DEPLOY/'active.json',dict(path=str(output),signature=signature));print(json.dumps(dict(stage='prepare',reused=True,path=str(output),queries=len(old['queries']))));return
    rows=[];needed=set();score_checks=[]
    for ordinal, requested in enumerate(queries):
        index=qi[requested['photo_id']];p=photos[index]
        require(requested['sha256']==p['sha256'],'Query collection differs from full inventory')
        record=dict(id=p['photo_id'],path=str(ROOT/'dataset'/p['path']),sha256=p['sha256'],source_sha256=p['source_sha256'],
            package_id=p['package_id'],bank_index=index,ordinal=ordinal)
        axis,ranked,winner=natural_candidates(banks.s['scores'][index].numpy(),refs)
        record.update(candidate_physical_rows=axis,ranked_physical_rows=ranked,winner=winner,
            challenger_positions=[i for i in range(128) if i!=winner],raw_scores=[float(banks.s['scores'][index,i]) for i in axis])
        record.update(banks.inspect(record,index));rows.append(record);needed.update(axis)
    references={}
    for i in sorted(needed):
        r=refs[i];record=dict(id=r['reference_id'],physical_row=i,bank_index=i,path=str(ROOT/'dataset'/r['path']),sha256=r['sha256'],design_id=r['design_id'])
        record.update(banks.inspect(record,i,True));references[str(i)]=record
    # Outcome-blind numeric checks against cached RAW; no ground-truth read.
    for q in rows[:3]:
        qt=banks.q['embeddings'][q['bank_index']].float();qt=qt[qt.abs().sum(1)>1e-6]
        for row in [q['ranked_physical_rows'][0],q['candidate_physical_rows'][0],q['candidate_physical_rows'][-1]]:
            rt=banks.r['embeddings'][row].float();rt=rt[rt.abs().sum(1)>1e-6]
            actual=float((qt@rt.T).max(1).values.mean());expected=float(banks.s['scores'][q['bank_index'],row])
            require(abs(actual-expected)<2e-6,'Cached RAW does not match cached token MaxSim')
            score_checks.append(dict(query=q['id'],row=row,maxsim_error=abs(actual-expected)))
    manifest=dict(schema='content_box_ce_polish18_deployment.v1',signature=signature,recipe=recipe,query_list=str(args.query_list.resolve()),
        queries=rows,references=references,full_gallery_rows=len(refs),unique_designs=len({r['sha256'] for r in refs}),score_checks=score_checks,
        old_roma_cache_reuse=0,old_roma_rejection='fast512 versus precise800/1280; FP16/area pooling and roll1 versus original canonical FP64/half-grid',
        old_11d_cache_found=False)
    atomic_json(output/'manifest.json',manifest)
    atomic_json(DEPLOY/'active.json',dict(path=str(output),signature=signature))
    print(json.dumps(dict(stage='prepare',path=str(output),queries=len(rows),candidate_pairs=len(rows)*128,references_needed=len(needed),encoder_forwards=0)),flush=True)


def active(args):
    output=args.deployment or Path(read(DEPLOY/'active.json')['path'])
    if not output.is_absolute():output=DEPLOY/output
    manifest=read(output/'manifest.json')
    require(manifest['recipe']['model']['model_sha256']==file_hash(MODEL),'Fixed head changed')
    return output,relocate_manifest(manifest,ROOT)


def relocate_manifest(manifest, root):
    """Relocate a transfer's image paths without changing signed evidence axes.

    The sealed recipe and its input hashes retain their original absolute keys.
    Pair bindings depend on that original recipe signature and image-token
    hashes, not on the mount point where the same files are unpacked.
    """
    transfer=manifest.get('transfer')
    if transfer is None:return manifest
    origin=Path(transfer['source_root'])
    def moved(path):
        return str(Path(root)/Path(path).relative_to(origin))
    manifest=dict(manifest)
    manifest['queries']=[dict(q,path=moved(q['path'])) for q in manifest['queries']]
    manifest['references']={key:dict(r,path=moved(r['path'])) for key,r in manifest['references'].items()}
    manifest['query_list']=moved(manifest['query_list'])
    return manifest


def recipe_input_hash(manifest, relative):
    origin=Path(manifest.get('transfer',{}).get('source_root',str(ROOT)))
    return manifest['recipe']['inputs'][str(origin/relative)]


def geometry(record, processor_sha):
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2
    return build_colnomic_canonical_geometry_v2(source_image_sha256=record['sha256'],source_key=record['id'],
        processor_config_sha256=processor_sha,raw_size_hw=(record['raw_height'],record['raw_width']),
        exif_orientation=record['exif_orientation'],merged_grid_shape=(record['layout']['height'],record['layout']['width']),
        processor_input_frame=record['processor_input_frame'])


class DescriptorCache(torch.nn.Module):
    """Original frozen module outputs reused, with containers copied for consumers."""
    def __init__(self, module, limit, cpu_limit=0):
        super().__init__();self.module=module;self.limit=limit;self.cache=OrderedDict();self.used=0;self.calls=Counter()
        self.cpu_limit=cpu_limit;self.cpu_cache=OrderedDict();self.cpu_used=0;self.hits=Counter()
    @staticmethod
    def transfer(value, device):
        if isinstance(value,dict):return {k:t.detach().to(device) for k,t in value.items()}
        return [t.detach().to(device) for t in value]
    def start(self, a,b):
        self.ids=(a,b);self.calls.clear()
    def forward(self, image):
        shape=tuple(image.shape[-2:]);side=self.calls[shape]%2;self.calls[shape]+=1;key=(self.ids[side],shape)
        if key not in self.cache:
            if key in self.cpu_cache:
                cpu,size=self.cpu_cache.pop(key);self.cpu_cache[key]=(cpu,size)
                value=self.transfer(cpu,image.device);self.hits['cpu']+=1
            else:
                value=self.module(image);self.hits['computed']+=1
            items=value.values() if isinstance(value,dict) else value
            size=sum(t.numel()*t.element_size() for t in items)
            if key not in self.cpu_cache and size<=self.cpu_limit:
                while self.cpu_cache and self.cpu_used+size>self.cpu_limit:
                    _,(_,old)=self.cpu_cache.popitem(last=False);self.cpu_used-=old
                self.cpu_cache[key]=(self.transfer(value,'cpu'),size);self.cpu_used+=size
            while self.cache and self.used+size>self.limit:
                _,(_,old)=self.cache.popitem(last=False);self.used-=old
            if size<=self.limit:self.cache[key]=(value,size);self.used+=size
        else:value,size=self.cache.pop(key);self.cache[key]=(value,size);self.hits['gpu']+=1
        if isinstance(value,list):return list(value)
        if isinstance(value,dict):return dict(value)
        return value


class OriginalRoMa:
    def __init__(self, core, descriptor_cache=True):
        sys.path.insert(0,str(ROOT/'weights/romav2/source/src'));sys.path.insert(0,str(ROOT/'cache/method/intermediate/runtime'))
        os.environ['TORCH_HOME']=str(ROOT/'weights/romav2/torchhub')
        from romav2 import RoMaV2
        import romav2.local_correlation as local_correlation_module
        # Spatial tiling keeps the original window, sampling and C reduction;
        # it avoids the original >3.5 GiB product allocation on an 11 GiB GPU.
        # Keep the audited native backend even if the destination environment
        # happens to have the optional fused extension installed.
        local_correlation_module.local_corr = None
        local_correlation_module.native_torch_local_corr = tiled_local_corr
        torch.manual_seed(17);torch.set_float32_matmul_precision('highest')
        self.model=RoMaV2();self.model.requires_grad_(False);self.model.eval();self.core=core;self.images=OrderedDict()
        require((self.model.H_lr,self.model.W_lr,self.model.H_hr,self.model.W_hr,self.model.bidirectional,self.model.threshold)==(800,800,1280,1280,True,None),'Original RoMa config differs')
        self.f=DescriptorCache(self.model.f,256<<20,2<<30);self.fine=DescriptorCache(self.model.refiner_features,512<<20,8<<30)
        self.model.f=self.f;self.model.refiner_features=self.fine
    def image(self, record):
        key=record['sha256']
        if key not in self.images:
            require(file_hash(record['path'])==key,'RoMa image SHA changed')
            self.images[key]=self.core['oriented'](record['path'])
        value=self.images.pop(key);self.images[key]=value
        while len(self.images)>16:self.images.popitem(last=False)
        return value
    def match(self, a,b):
        self.f.start(a['sha256'],b['sha256']);self.fine.start(a['sha256'],b['sha256'])
        result=self.model.match(self.image(a),self.image(b))
        return {k:result[k][0,...,0].float().cpu().numpy() for k in ['overlap_AB','overlap_BA']}


def tiled_local_corr(feature0,feature1,warp,local_window,B,K,c,r,h,w,device,
                     padding_mode='zeros',sample_mode='bilinear',dtype=torch.float32):
    corr=torch.empty((B,K,h,w),device=device,dtype=dtype)
    # Native Torch's original >2 GiB product is reduced through TensorIterator
    # channel splits. Keep those FP32 partial sums even though each spatial
    # tile fits in memory; a single sum over all C otherwise changes rounding.
    plane=h*w*K;itemsize=feature0.element_size()
    def splits(lo,hi):
        if (hi-lo)*plane*itemsize-itemsize <= 2**31-1:
            return [(lo,hi)]
        mid=lo+(hi-lo)//2
        return splits(lo,mid)+splits(mid,hi)
    ranges=splits(0,c)
    for batch in range(B):
        for start in range(0,h,16):
            stop=min(start+16,h)
            with torch.no_grad():
                coords=(warp[batch,start:stop,:,None]+local_window[:,None,None]).reshape(1,stop-start,w*K,2)
                window=F.grid_sample(feature1[batch:batch+1],coords,padding_mode=padding_mode,
                    align_corners=False,mode=sample_mode).reshape(c,stop-start,w,K)
            product=feature0[batch,:,start:stop,...,None]/(c**.5)*window
            sums=[product[lo:hi].sum(dim=0) for lo,hi in ranges]
            reduced=sums[0]
            for value in sums[1:]:reduced=reduced+value
            corr[batch,:,start:stop]=reduced.permute(2,0,1)
    return corr


def pair_path(output,q,row):
    return output/'pairs'/q['id']/f'{row:06d}.npz'


def accepted_pair_bindings(manifest,q,r):
    signatures=[manifest['signature']]
    for reuse in manifest.get('evidence_reuse',[]):
        if q['id'] in reuse['query_ids']:
            require(reuse['validated_recipe_equivalence'] is True,'Unvalidated evidence reuse')
            signatures.append(reuse['source_signature'])
    return {digest([signature,q['id'],q['image_tokens_sha256'],r['image_tokens_sha256']]) for signature in signatures}


def validate_reuse_contract(source,destination):
    """A changed query population may reuse identical frozen per-pair evidence."""
    for manifest in [source,destination]:
        require(manifest['signature']==digest(manifest['recipe']),'Recipe signature differs')
    def relative_inputs(manifest):
        origin=Path(manifest.get('transfer',{}).get('source_root',str(ROOT)))
        return {str(Path(key).relative_to(origin)):value for key,value in manifest['recipe']['inputs'].items()},str(Path(manifest['query_list']).relative_to(origin))
    left,lq=relative_inputs(source);right,rq=relative_inputs(destination)
    require(all(left[k]==right[k] for k in left.keys()&right.keys()),'Frozen input bytes differ')
    ignored={lq,rq}
    require({k:v for k,v in left.items() if k not in ignored}=={k:v for k,v in right.items() if k not in ignored},'Non-query-list input differs')
    a={k:v for k,v in source['recipe'].items() if k!='inputs'}
    b={k:v for k,v in destination['recipe'].items() if k!='inputs'}
    require(a==b,'Frozen feature/model recipe differs')
    query_fields=['sha256','source_sha256','package_id','bank_index','raw_height','raw_width','exif_orientation','layout',
        'processor_input_frame','image_tokens_sha256','all_tokens_sha256','candidate_physical_rows','ranked_physical_rows',
        'winner','challenger_positions','raw_scores']
    ref_fields=['id','physical_row','bank_index','sha256','design_id','raw_height','raw_width','exif_orientation','layout',
        'processor_input_frame','image_tokens_sha256','all_tokens_sha256']
    indexed={q['id']:q for q in destination['queries']};matched=[]
    for q in source['queries']:
        if q['id'] not in indexed:continue
        new=indexed[q['id']]
        require(all(q[k]==new[k] for k in query_fields),'Query evidence/RAW/C128 differs: '+q['id'])
        for row in q['candidate_physical_rows']:
            old=source['references'][str(row)];current=destination['references'][str(row)]
            require(all(old[k]==current[k] for k in ref_fields),'Reference geometry/tokens differ: '+str(row))
        matched.append(q['id'])
    return matched


def reuse(args):
    require(args.reuse_from is not None,'reuse requires --reuse-from')
    output,manifest=active(args);source_path=args.reuse_from.resolve();source=read(source_path/'manifest.json')
    require(output.resolve()!=source_path,'Source and destination must be separate deployments')
    matched=validate_reuse_contract(source,manifest)
    source_queries={q['id']:q for q in source['queries']};new_queries={q['id']:q for q in manifest['queries']}
    imported=[key for key in matched if (source_path/'queries'/key/'prediction.json').exists() or any((source_path/'pairs'/key).glob('*.npz'))]
    record=dict(source_deployment=str(source_path),source_signature=source['signature'],source_manifest_sha256=file_hash(source_path/'manifest.json'),
        query_ids=imported,validated_recipe_equivalence=True,validation='Exact frozen inputs, checkpoint, feature definitions, token geometry, RAW and natural C128; only query-list population differs')
    # The original NPZ bindings remain unchanged. Validated source signatures
    # are accepted only for the explicitly imported query identities.
    sealed=read(output/'manifest.json')
    aliases=[r for r in sealed.get('evidence_reuse',[]) if r['source_signature']!=source['signature']]
    aliases.append(record)
    for previous in source.get('evidence_reuse',[]):
        retained=[key for key in previous['query_ids'] if key in imported]
        if retained:
            require(previous['validated_recipe_equivalence'] is True,'Unvalidated source reuse chain')
            aliases=[r for r in aliases if r['source_signature']!=previous['source_signature']]
            aliases.append(dict(previous,query_ids=retained))
    sealed['evidence_reuse']=aliases;atomic_json(output/'manifest.json',sealed)
    pairs=linked=copied=completed=0
    for key in imported:
        old=source_queries[key];new=new_queries[key]
        for origin in sorted((source_path/'pairs'/key).glob('*.npz')):
            row=int(origin.stem);require(row in new['candidate_physical_rows'],'Imported pair outside natural C128')
            with np.load(origin,allow_pickle=False) as z:
                require(str(z['binding']) in accepted_pair_bindings(source,old,source['references'][str(row)]),'Source pair binding differs')
                require(str(z['query_tokens_sha256'])==new['image_tokens_sha256'] and str(z['reference_tokens_sha256'])==manifest['references'][str(row)]['image_tokens_sha256'],'Source pair token identity differs')
                require(z['c4'].shape==(4,) and z['content_statistics'].shape==(5,),'Source pair incomplete')
            target=pair_path(output,new,row);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try:os.link(origin,target);linked+=1
                except OSError:shutil.copy2(origin,target);copied+=1
            with np.load(target,allow_pickle=False) as z:
                require(str(z['binding']) in accepted_pair_bindings(sealed,new,manifest['references'][str(row)]),'Imported pair binding differs')
            pairs+=1
        folder=source_path/'queries'/key
        if (folder/'prediction.json').exists():
            receipt=read(folder/'receipt.json');require(receipt['manifest_signature']==source['signature'],'Source query signature differs')
            require(receipt['model_sha256']==manifest['recipe']['model']['model_sha256'],'Source head differs')
            require(file_hash(folder/'input.json')==receipt['input_sha256'] and file_hash(folder/'prediction.json')==receipt['prediction_sha256'],'Source query bytes changed')
            target=output/'queries'/key;target.mkdir(parents=True,exist_ok=True)
            if not (target/'prediction.json').exists():
                for name in ['input.json','features.npz']:shutil.copy2(folder/name,target/name)
                imported_receipt=dict(receipt,manifest_signature=manifest['signature'],imported_from_deployment=str(source_path),
                    source_manifest_signature=source['signature'],source_receipt_sha256=file_hash(folder/'receipt.json'),reused_pairs=128)
                atomic_json(target/'receipt.json',imported_receipt)
                temp=target/'.import_prediction.tmp.json';shutil.copy2(folder/'prediction.json',temp);temp.replace(target/'prediction.json')
            require(file_hash(target/'prediction.json')==receipt['prediction_sha256'],'Destination completion differs')
            completed+=1
        print(json.dumps(dict(event='evidence_reused',query=key,pairs=pairs,completed_queries=completed)),flush=True)
    result=dict(source_deployment=str(source_path),destination_deployment=str(output),validated_query_overlap=len(matched),
        queries_completed_reused=completed,pairs_reused=pairs,pairs_hardlinked=linked,pairs_copied=copied,
        encoder_forwards=0,roma_pair_recomputations=0,head_recomputations=0)
    atomic_json(output/'reuse_receipt.json',result);print(json.dumps(result,indent=2),flush=True)


def compute_pair(output,manifest,q,r,banks,core,content,matcher_holder):
    path=pair_path(output,q,r['physical_row']);binding=digest([manifest['signature'],q['id'],q['image_tokens_sha256'],r['image_tokens_sha256']])
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            require(str(data['binding']) in accepted_pair_bindings(manifest,q,r),'Completed pair binding differs')
        return True
    qt=banks.image_tokens(q);rt=banks.image_tokens(r,True)
    pre=ROOT/'cache/content_box_ce_polish18/preflight_native'
    premeta=read(pre/'metadata.json') if (pre/'metadata.json').exists() else {}
    if (premeta.get('query_sha256')==q['sha256'] and premeta.get('reference_sha256')==r['sha256'] and
            premeta.get('setting')=='precise' and premeta.get('checkpoint_sha256')==manifest['recipe']['roma']['checkpoint_sha256']):
        with np.load(pre/'overlap.npz',allow_pickle=False) as z:maps={k:z[k].copy() for k in ['overlap_AB','overlap_BA']}
        reused_native=True
    else:
        if not matcher_holder:matcher_holder.append(OriginalRoMa(core))
        maps=matcher_holder[0].match(q,r);reused_native=False
    processor_sha=recipe_input_hash(manifest,'weights/colnomic7b/adapter/preprocessor_config.json')
    a=geometry(q,processor_sha);b=geometry(r,processor_sha)
    u=core['cell_means'](maps['overlap_AB'],a);v=core['cell_means'](maps['overlap_BA'],b)
    real,mass,_=core['score'](qt,rt,u,v)
    qc=core['score'](qt,rt,u.roll(max(1,u.numel()//2)),v)[0]
    rc=core['score'](qt,rt,u,v.roll(max(1,v.numel()//2)))[0]
    stats=content['statistics'](content['active'](qt),content['active'](rt)).numpy()
    values=dict(**maps,u=u.numpy(),v=v.numpy(),c4=np.array([float(real),float(mass),float(qc),float(rc)],dtype=np.float64),
        content_statistics=stats,binding=np.array(binding),reused_native=np.array(reused_native),query_tokens_sha256=np.array(q['image_tokens_sha256']),
        reference_tokens_sha256=np.array(r['image_tokens_sha256']))
    require(all(np.isfinite(values[k]).all() for k in ['overlap_AB','overlap_BA','u','v','c4','content_statistics']),'Nonfinite paired evidence')
    save_npz(path,values)
    return reused_native


def construct_x11(q, evidence, statistics, content, gate):
    require(np.asarray(evidence).shape==(128,4),'X11 requires S, M and both original spatial controls for all 128 candidates')
    require(np.asarray(statistics).shape==(128,5),'X11 requires all five original content statistics')
    keys=['real_score','visibility_mass','query_control_score','reference_control_score']
    records={i:dict(zip(keys,map(float,evidence[i]))) for i in range(128)}
    x6=torch.stack([gate.candidate_feature(q['raw_scores'],records,i,q['winner']) for i in q['challenger_positions']]).numpy()
    more=np.array(content['features'](q['raw_scores'],statistics.tolist(),q['winner']),dtype=np.float64)
    require(np.array_equal(x6[:,0],more[:,0]),'RAW standardization differs between original feature branches')
    x11=np.concatenate((x6,more[:,1:]),axis=1)
    require(x11.shape==(127,11) and np.isfinite(x11).all(),'Invalid complete X11')
    return x11


def run(args):
    output,manifest=active(args);torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda'):
        require(torch.cuda.is_available(),'RoMa requires an available CUDA device');torch.cuda.set_device(args.device)
    else:raise ValueError('Use CUDA for the original RoMa configuration')
    core,content,gate=math_functions();banks=Banks(False);matcher=[]
    queries=manifest['queries'][:args.limit] if args.limit is not None else manifest['queries']
    completed=0
    for q in queries:
        if q['ordinal']%args.workers!=args.worker_index:continue
        folder=output/'queries'/q['id'];prediction=folder/'prediction.json'
        if prediction.exists():
            require(read(folder/'receipt.json')['manifest_signature']==manifest['signature'],'Completed query signature differs')
            completed+=1;continue
        with claim(folder/'.lock') as acquired:
            if not acquired:continue
            evidence=[];statistics=[];reused=0;start=time.perf_counter()
            for position,row in enumerate(q['candidate_physical_rows']):
                reused+=compute_pair(output,manifest,q,manifest['references'][str(row)],banks,core,content,matcher)
                with np.load(pair_path(output,q,row),allow_pickle=False) as z:
                    evidence.append(z['c4'].copy());statistics.append(z['content_statistics'].copy())
                if (position+1)%16==0:
                    print(json.dumps(dict(event='pair_progress',query=q['id'],pairs=position+1,
                        ordinal=q['ordinal'],worker=args.worker_index,seconds=time.perf_counter()-start)),flush=True)
            x11=construct_x11(q,np.array(evidence),np.array(statistics),content,gate)
            input_=dict(feature_schema=FEATURE_SCHEMA,query_id=q['id'],X11=x11.tolist(),
                candidate_physical_rows=q['candidate_physical_rows'],winner=q['winner'],challenger_positions=q['challenger_positions'])
            atomic_json(folder/'input.json',input_)
            save_npz(folder/'features.npz',dict(X11=x11,c4=np.array(evidence),content_statistics=np.array(statistics),raw=np.array(q['raw_scores']),columns=np.array(COLUMNS)))
            temp=folder/'.prediction.tmp.json'
            subprocess.run([sys.executable,str(PACKAGE/'infer.py'),'--model',str(MODEL),'--input',str(folder/'input.json'),'--output',str(temp)],check=True)
            p=read(temp)
            require(p['fold']==0 and len(p['logits'])==127 and p['action']==('SWITCH' if p['top_logit']>0 else 'HOLD'),'Frozen inference contract differs')
            receipt=dict(manifest_signature=manifest['signature'],input_sha256=file_hash(folder/'input.json'),model_sha256=file_hash(MODEL),
                prediction_sha256=file_hash(temp),candidate_pairs=128,reused_pairs=reused,seconds=time.perf_counter()-start,
                encoder_forwards=0,roma_training_updates=0,head_training_updates=0)
            atomic_json(folder/'receipt.json',receipt);temp.replace(prediction);completed+=1
            print(json.dumps(dict(query=q['id'],ordinal=q['ordinal'],completed=completed,action=p['action'],RAW=q['ranked_physical_rows'][0],
                selected=p['selected'],reused_pairs=reused,seconds=receipt['seconds']),ensure_ascii=False),flush=True)
    atomic_json(output/f'worker_{args.worker_index}.json',dict(completed=completed,workers=args.workers,limit=args.limit,manifest_signature=manifest['signature']))


def overlap_audit(manifest):
    source=ROOT/'cache/content_box_ce_polish18/source_reference'
    workers=read(source/'worker_manifest.json')['records'];curator=read(source/'curator_roles.json')['records']
    payload=read(RC/'results/rc_h593_box_ce_polish_v1/fold0/payload.json')
    train=set(payload['train_query_ids']);effective=set(payload['effective_train_query_ids'])
    by_sha={r['source_image_sha256']:r['query_id'] for r in workers}
    by_id={r['query_id']:r for r in curator};matched=[]
    for q in manifest['queries']:
        old=by_sha.get(q['sha256']) or by_sha.get(q['source_sha256'])
        matched.append(dict(query_id=q['id'],original_query_id=old,fold0_train_role=bool(old and old in train),
            fold0_effective_training=bool(old and old in effective),original_component=by_id.get(old,{}).get('component')))
    return dict(method='Current image SHA and preserved source SHA matched against all 593 original worker records',
        original_train_role_queries=len(train),original_effective_fit_queries=len(effective),
        exact_image_matches=sum(bool(r['original_query_id']) for r in matched),
        fold0_train_role_overlap=sum(r['fold0_train_role'] for r in matched),fold0_effective_training_overlap=sum(r['fold0_effective_training'] for r in matched),
        scope='Byte-exact image/source matches; distinct captures and alternate image encodings are not excluded by SHA alone',records=matched)


def report(args):
    output,manifest=active(args);positives={}
    for r in csvread(ROOT/'dataset/lists/positives.csv'):positives.setdefault(r['package_id'],set()).add(r['reference_id'])
    refs=csvread(ROOT/'dataset/lists/references.csv');rows=[];counts=Counter()
    for q in manifest['queries']:
        path=output/'queries'/q['id']/'prediction.json'
        if not path.exists():continue
        p=read(path);target={r['sha256'] for r in refs if r['reference_id'] in positives[q['package_id']]}
        raw=q['ranked_physical_rows'][0];selected=p['selected'];raw_ok=refs[raw]['sha256'] in target;final_ok=refs[selected]['sha256'] in target
        present=any(refs[i]['sha256'] in target for i in q['candidate_physical_rows'])
        t='CORRECT_HOLD' if raw_ok and final_ok else 'RESCUE' if final_ok else 'BREAK' if raw_ok else ('WRONG_HOLD' if refs[raw]['sha256']==refs[selected]['sha256'] else 'WRONG_SWITCH')
        counts[t]+=1;counts['RAW_correct']+=raw_ok;counts['optimized_correct']+=final_ok;counts['target_absent_C128']+=not present;counts[p['action']]+=1
        rows.append(dict(query_id=q['id'],query_path=q['path'],package_id=q['package_id'],RAW_row=raw,RAW_reference=refs[raw]['reference_id'],
            final_row=selected,final_reference=refs[selected]['reference_id'],RAW_correct=raw_ok,final_correct=final_ok,
            target_in_C128=present,action=p['action'],transition=t,top_logit=p['top_logit'],fold=0))
    result=dict(model='CONTENT_BOX_CE_POLISH18',fold=0,threshold=0,queries_expected=len(manifest['queries']),queries_completed=len(rows),
        status='complete' if len(rows)==len(manifest['queries']) else 'partial',**dict(counts),
        RAW_accuracy=counts['RAW_correct']/len(rows) if rows else None,optimized_accuracy=counts['optimized_correct']/len(rows) if rows else None,
        encoder_forwards=0,training_updates=0,historical_496_593_not_used=True,overlap=overlap_audit(manifest))
    if rows:csvwrite(output/'predictions.csv',rows,list(rows[0]))
    atomic_json(output/'summary.json',result)
    lines=['# CONTENT_BOX_CE_POLISH18：本机固定 fold0 测试','',
        f"测试清单：`{manifest['query_list']}`；已完成 {len(rows)}/{len(manifest['queries'])}。",'',
        f"RAW 正确 {counts['RAW_correct']}；优化头正确 {counts['optimized_correct']}；救回 {counts['RESCUE']}；误伤 {counts['BREAK']}；target 未进自然 C128 {counts['target_absent_C128']}。",'',
        '模型固定 models/fold0.json，HOLD 阈值固定 0；全部 127 个挑战者均通过原 infer.py 评分。没有训练或 ColNomic 编码。',
        '原 496/593 是历史五折结果，本报告是固定单头的当前测试结果。','',
        f"原 fold0 的字节级训练图片重叠：角色训练 {result['overlap']['fold0_train_role_overlap']}、实际拟合 {result['overlap']['fold0_effective_training_overlap']}。SHA 检查不排除不同编码或不同拍摄的同源图片。",'',
        '缓存保存在此目录：manifest.json 记录原图、token、图库和自然 C128 映射；pairs/ 保存原生双向 RoMa 覆盖图及 FP64 u/v、C4、内容五维；queries/ 保存完整 X11、infer.py 输入、全部挑战者分数和 HOLD/SWITCH。',
        '旧 fast512 RoMa 输出及旧八维特征不兼容本头，未用于本次计算；新 RoMa 保持原默认 precise800/1280、双向、原 BF16 配置。',
        '六维与内容五维直接执行原源码数学函数；模板 token 只参与原检索，X11 使用图像 tokens，内容五维不额外归一化。','',
        '恢复：使用同一个 deployment 运行 run；已完成配对和 query 不会重复计算。','']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k!='overlap'},ensure_ascii=False,indent=2),flush=True)


def verify(args):
    output,manifest=active(args);torch.set_num_threads(args.cpu_threads)
    core,content,gate=math_functions();banks=Banks(False);checked=0;pair_checks=0;max_error=0.;max_c4_error=0.
    processor_sha=recipe_input_hash(manifest,'weights/colnomic7b/adapter/preprocessor_config.json')
    for q in manifest['queries']:
        folder=output/'queries'/q['id']
        if not (folder/'prediction.json').exists():continue
        record=read(folder/'input.json');p=read(folder/'prediction.json');receipt=read(folder/'receipt.json')
        require(file_hash(folder/'input.json')==receipt['input_sha256'] and file_hash(folder/'prediction.json')==receipt['prediction_sha256'],'Saved query bytes changed')
        require(record['candidate_physical_rows']==sorted(set(record['candidate_physical_rows'])) and len(record['candidate_physical_rows'])==128,'Non-natural axis')
        axis,ranked,winner=natural_candidates(banks.s['scores'][q['bank_index']].numpy(),csvread(ROOT/'dataset/lists/references.csv'))
        require(axis==record['candidate_physical_rows'] and ranked==q['ranked_physical_rows'] and winner==record['winner'],'Candidate generation differs')
        x=np.asarray(record['X11'],dtype=np.float64)
        theta=np.array([float.fromhex(s) for s in read(MODEL)['parameters_hex']])
        basis=np.concatenate((x[:,:6],x[:,:6]**2,x[:,6:],np.ones((127,1))),axis=1);scores=basis@theta
        require(np.array_equal(scores,np.array(p['logits'])),'Independent saved-head scores differ')
        require((p['top_logit']>0)==(p['action']=='SWITCH'),'Threshold changed')
        with np.load(folder/'features.npz',allow_pickle=False) as feature:
            reconstructed=construct_x11(q,feature['c4'],feature['content_statistics'],content,gate)
            require(np.array_equal(reconstructed,x),'Stored X11 differs from original six-plus-five construction')
        # Independent NumPy active-token statistics for fixed candidate positions.
        for row in [axis[0],ranked[0],axis[-1]]:
            r=manifest['references'][str(row)]
            expected=content['numpy_statistics'](banks.image_tokens(q).numpy(),banks.image_tokens(r,True).numpy())
            with np.load(pair_path(output,q,row),allow_pickle=False) as z:
                error=float(np.max(np.abs(expected-z['content_statistics'])))
                require(error<2e-10,'Independent five-dimensional content statistics differ')
                require(z['c4'].shape==(4,) and z['u'].dtype==z['v'].dtype==np.float64,'Incomplete RoMa evidence')
                # Independent direct NumPy pooling and the literal original C4.
                vectors=[]
                for name,record in [('overlap_AB',q),('overlap_BA',r)]:
                    image=z[name].astype(np.float64);height,width=image.shape;g=geometry(record,processor_sha);values=[]
                    for box,valid in zip(g.cell_boxes_xyxy.numpy(),g.valid_patch_mask.numpy()):
                        if not valid:values.append(0.);continue
                        x0,y0,x1,y1=box
                        left=max(0,min(width-1,math.floor(x0*width)));top=max(0,min(height-1,math.floor(y0*height)))
                        right=max(left+1,min(width,math.ceil(x1*width)));bottom=max(top+1,min(height,math.ceil(y1*height)))
                        values.append(float(image[top:bottom,left:right].mean()))
                    vectors.append(np.array(values))
                u,v=vectors
                np.testing.assert_allclose(u,z['u'],atol=2e-14,rtol=2e-14);np.testing.assert_allclose(v,z['v'],atol=2e-14,rtol=2e-14)
                qt=banks.image_tokens(q).double().numpy();rt=banks.image_tokens(r,True).double().numpy()
                qt=qt/np.maximum(np.linalg.norm(qt,axis=1,keepdims=True),1e-12);rt=rt/np.maximum(np.linalg.norm(rt,axis=1,keepdims=True),1e-12)
                sim=qt@rt.T
                def scalar(a,b):
                    mass=np.sqrt(a.mean()*b.mean())
                    return mass*np.sum(a*np.max(sim*b[None],axis=1))/max(float(a.sum()),1e-12),mass
                real,mass=scalar(u,v);qc=scalar(np.roll(u,max(1,len(u)//2)),v)[0];rc=scalar(u,np.roll(v,max(1,len(v)//2)))[0]
                c4_error=float(np.max(np.abs(np.array([real,mass,qc,rc])-z['c4'])))
                require(c4_error<2e-10,'Independent original six-feature C4 evidence differs');max_c4_error=max(max_c4_error,c4_error)
            max_error=max(max_error,error);pair_checks+=1
        checked+=1
    result=dict(status='passed',queries_verified=checked,logits_verified=checked*127,independent_content_pairs=pair_checks,
        max_content_error=max_error,max_c4_error=max_c4_error,complete_x11_reconstructed=True,fold=0,threshold=0,encoder_forwards=0,target_insertions=0)
    atomic_json(output/'verification.json',result);print(json.dumps(result,indent=2))


def status(args):
    output,manifest=active(args)
    done=sum((output/'queries'/q['id']/'prediction.json').exists() for q in manifest['queries'])
    native_maps=sum(len(list((output/'pairs'/q['id']).glob('*.npz'))) for q in manifest['queries'])
    pairs=sum(128 if (output/'queries'/q['id']/'prediction.json').exists() else len(list((output/'pairs'/q['id']).glob('*.npz'))) for q in manifest['queries'])
    print(json.dumps(dict(deployment=str(output),queries_completed=done,queries=len(manifest['queries']),pairs_completed=pairs,
        pairs=len(manifest['queries'])*128,stored_roma_maps=native_maps),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','reuse','run','report','verify','status'])
    p.add_argument('--query-list',type=Path,default=ROOT/'dataset/lists/test.csv')
    p.add_argument('--deployment',type=Path)
    p.add_argument('--reuse-from',type=Path)
    p.add_argument('--limit',type=int)
    p.add_argument('--worker-index',type=int,default=0)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-threads',type=int,default=2)
    args=p.parse_args()
    require(args.workers>0 and 0<=args.worker_index<args.workers,'Invalid worker partition')
    require(args.limit is None or args.limit>0,'Limit must be positive')
    globals()[args.stage](args)


if __name__=='__main__':
    main()
