#!/usr/bin/env python3
"""OCR support pilot: frozen natural C128, offline OCR, identity-only readout.

No RoMa/ColNomic inference, held-label fitting, candidate expansion or manual OCR.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import numpy as np

RC = Path(__file__).resolve().parents[1]
WS = RC.parents[2]
OUT = RC/'results/rc_ocr_support_pilot128_v1'
ARMS = ['CALIBRATION_ONLY', 'TEXT', 'TEXT_M', 'TEXT_UV', 'TEXT_UV_SHUFFLED']
MODELS = WS/'cache/paddlex/official_models'
sys.path[:0] = [str(RC/'programs'), str(RC/'src')]

def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest())
_verified=set()
def checked(b):
    key=(b['path'],b['sha256'])
    if key not in _verified:assert bind(b['path'])['sha256']==b['sha256'],('SOURCE_DRIFT',b['path']);_verified.add(key)
    return Path(b['path'])
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n');os.replace(tmp,p)
def emit(**d):print(json.dumps(d),flush=True)
def tokens(s):
    # Retain single digits, decimals, units and original line association.
    return re.findall(r'\d+(?:\.\d+)?|[a-z]+',unicodedata.normalize('NFKC',s).casefold())
def protocol():
    p=read(OUT/'protocol.json')
    for b in p['code_sources']:checked(b)
    return p

def prepare():
    assert not (OUT/'protocol.json').exists(),'PROTOCOL_ALREADY_FROZEN'
    a=read(RC/'registry/rc_h593_simple_explanations_authority_v1_20260924.json')
    cache=read(RC/'results/rc_h593_simple_explanations_v1/cache.json')
    workers=read(RC/'results/rc_crisp_manual_baseline_v1/H593_workers.json')['records']
    md=read(checked(a['public']['worker']))['records'];qpaths={r['query_id']:r['query_image_path'] for r in md}
    cat=read(RC/'cache/rc_h593_shared_native_v1/catalog.json')['images'];bysha={r['sha256']:r for r in cat};bypath={r['path']:r for r in cat}
    gallery=read(checked(a['public']['gallery']))['records'];gallery={r['physical_row']:r for r in gallery}
    queries=[];images={};refs={}
    # Selection is execution order, independent of labels, OCR availability and outcome.
    for i,row in enumerate(cache['rows'][:128]):
        w=workers[i];assert w['query_id']==row['query_id'] and row['execution_ordinal']==i
        q=bysha[row['source_image_sha256']];images[q['sha256']]=dict(image=dict(path=q['path'],sha256=q['sha256']))
        rshas=[]
        for physical in row['axis']:
            r=bypath.get(gallery[physical]['image_path'])
            if r is None:r=bind(gallery[physical]['image_path'])
            images[r['sha256']]=dict(image=dict(path=r['path'],sha256=r['sha256']))
            refs[physical]=r['sha256'];rshas.append(r['sha256'])
        queries.append(dict(index=i,query_id=row['query_id'],query_sha=q['sha256'],reference_shas=rshas,axis=row['axis'],roma=w['roma']['payload'],source_index=w['source_index'],source_query_id=w['source_query_id']))
    weights=[]
    for model in ['PP-OCRv5_server_det','en_PP-OCRv5_mobile_rec']:
        weights.extend(bind(MODELS/model/n) for n in ['inference.pdiparams','inference.json','inference.yml'])
    p=dict(status='OCR_SUPPORT_PILOT128_FROZEN',queries=queries,images=[images[s] for s in sorted(images)],reference_shas=refs,
        cache=bind(RC/'results/rc_h593_simple_explanations_v1/cache.json'),original_authority=bind(RC/'registry/rc_h593_simple_explanations_authority_v1_20260924.json'),
        original_worker=bind(RC/'results/rc_crisp_manual_baseline_v1/H593_workers.json'),split=a['public']['split'],gallery=a['public']['gallery'],folds=a['folds'],
        original_evaluation=bind(RC/'results/rc_h593_quality_operator_eval_v1/result.json'),models=weights,
        code_sources=[bind(Path(__file__)),bind(RC/'slurm/rc_ocr_support_pilot128_v1.sbatch'),bind(RC/'plan/RC_OCR_SUPPORT_PILOT128_20261009.md')],
        arms=ARMS,ocr=dict(det='PP-OCRv5_server_det',rec='en_PP-OCRv5_mobile_rec',max_side=960,recognition_threshold=0.,doc_rotation=False,unwarp=False,textline_rotation=False),
        loss='COST1_original_unit',updates=2000,lr=.03,weight_decay=.001,seed=17,ocr_shards=32,feature_shards=16,
        supervision='retrieval identity only; pretrained OCR frozen; no human transcriptions',
        evidence='opened first128 of H593, original group folds, pilot not external confirmation',
        no_new_RoMa=True,no_new_ColNomic=True,missing_ocr='zero text evidence; base candidate scores unchanged at zero residual')
    write(OUT/'protocol.json',p);emit(stage='PREPARED',queries=128,unique_images=len(images),references=len(refs))

def ocr(shard,budget,smoke=False):
    p=protocol();deadline=time.monotonic()+budget
    import tempfile
    temp=OUT/'runtime_tmp'/str(os.getpid());temp.mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(temp);tempfile.tempdir=str(temp)
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True';os.environ['PADDLE_PDX_HOME']=str(WS/'cache/paddlex')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from PIL import Image,ImageOps
    from paddleocr import PaddleOCR
    smoke_shas=[p['queries'][0]['query_sha'],p['queries'][0]['reference_shas'][0]]
    todo=[next(r for r in p['images'] if r['image']['sha256']==s) for s in smoke_shas] if smoke else p['images'][shard::p['ocr_shards']]
    model=None
    for r in todo:
        b=r['image'];out=OUT/'ocr'/f"{b['sha256']}.json"
        if out.exists():assert read(out)['source']==b;continue
        if time.monotonic()>deadline-40:return 75
        if model is None:
            for seal in p['models']:checked(seal)
            model=PaddleOCR(text_detection_model_name=p['ocr']['det'],text_detection_model_dir=str(MODELS/p['ocr']['det']),text_recognition_model_name=p['ocr']['rec'],text_recognition_model_dir=str(MODELS/p['ocr']['rec']),use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,device='cpu',cpu_threads=4,enable_mkldnn=False,text_det_limit_side_len=960,text_det_limit_type='max',text_rec_score_thresh=0.0)
        path=checked(b);start=time.monotonic()
        with Image.open(path) as im:rgb=ImageOps.exif_transpose(im).convert('RGB')
        w,h=rgb.size
        result=list(model.predict(np.asarray(rgb)[:,:,::-1]))[0].json
        result=result.get('res',result);lines=[]
        assert len(result['rec_texts'])==len(result['rec_scores'])==len(result['rec_polys'])
        for text,score,poly in zip(result['rec_texts'],result['rec_scores'],result['rec_polys']):
            a=np.asarray(poly,float);box=[float(a[:,0].min()/w),float(a[:,1].min()/h),float(a[:,0].max()/w),float(a[:,1].max()/h)]
            box=np.clip(box,0,1).tolist();assert math.isfinite(score) and 0<=score<=1
            lines.append(dict(text=text,recognition_score=float(score),polygon_pixels=a.tolist(),box=box,tokens=tokens(text)))
        write(out,dict(status='OCR_IMAGE_COMPLETE',source=b,oriented_size_hw=[h,w],ocr=p['ocr'],lines=lines,raw=result,seconds=time.monotonic()-start))
        emit(stage='OCR',sha=b['sha256'],lines=len(lines),seconds=time.monotonic()-start)
    if smoke:write(OUT/'smoke_validation.json',dict(status='PADDLE_CPU_SOURCE_AND_BOX_SCHEMA_PASS',images=[bind(OUT/'ocr'/f'{s}.json') for s in smoke_shas]))
    else:write(OUT/'ocr_shards'/f'{shard:02d}.json',dict(status='OCR_SHARD_COMPLETE',images=len(todo)))
    return 0

def index():
    p=protocol();df=Counter();seals=[]
    for r in p['images']:
        f=OUT/'ocr'/f"{r['image']['sha256']}.json";d=read(f);assert d['status']=='OCR_IMAGE_COMPLETE' and d['source']==r['image'];seals.append(bind(f))
    for sha in p['reference_shas'].values():
        d=read(OUT/'ocr'/f'{sha}.json');df.update({t for l in d['lines'] for t in l['tokens']})
    n=len(p['reference_shas']);idf={t:math.log(1+(n-v+.5)/(v+.5)) for t,v in df.items()}
    write(OUT/'index.json',dict(status='ALL_OCR_VALIDATED',ocr_files=seals,idf=idf,unseen_idf=math.log(1+(n+.5)/.5),references=n))

def pool(boxes,cells,weights):
    boxes=np.asarray(boxes,float).reshape(-1,4);cells=np.asarray(cells,float);weights=np.asarray(weights,float)
    if not len(boxes):return np.zeros(0)
    area=np.maximum(0,np.minimum(boxes[:,None,2:],cells[None,:,2:])-np.maximum(boxes[:,None,:2],cells[None,:,:2])).prod(2)
    denom=area.sum(1);out=area@weights/np.maximum(denom,1e-15)
    assert np.all(denom>0),'EMPTY_BOX_OVERLAP'
    return out

def lexical(q,r,uq,vr,idf,unseen):
    # Free lookup over all reference text lines; no RoMa coordinate hard binding.
    best={};where={}
    for j,(line,w) in enumerate(zip(r,vr)):
        v=float(w)*line['recognition_score']
        for t in line['tokens']:
            if t not in best or v>best[t]:best[t]=v;where[t]=j
    num=den=0.;matched=[]
    for i,(line,w) in enumerate(zip(q,uq)):
        for t in line['tokens']:
            base=idf.get(t,unseen)*line['recognition_score'];den+=base
            value=base*float(w)*best.get(t,0.);num+=value
            if t in where:matched.append([i,where[t],t,value])
    return num/max(den,1e-15),matched

def geometry(g,sha):
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2
    return build_colnomic_canonical_geometry_v2(source_image_sha256=sha,source_key=sha,processor_config_sha256=g['processor_config_sha256'],raw_size_hw=tuple(g['raw_size_hw']),exif_orientation=g['exif_orientation'],merged_grid_shape=tuple(g['grid_shape']),processor_input_frame=g['processor_input_frame']).cell_boxes_xyxy.numpy()

def features(shard,budget):
    import torch
    torch.set_num_threads(4);p=protocol();deadline=time.monotonic()+budget;ix=read(OUT/'index.json');assert ix['status']=='ALL_OCR_VALIDATED'
    last=None;payload=None
    for qr in p['queries'][shard::p['feature_shards']]:
        out=OUT/'features'/f"{qr['index']:03d}.json"
        if out.exists():continue
        if time.monotonic()>deadline-60:return 75
        if qr['roma']['path']!=last:
            payload=torch.load(checked(qr['roma']),weights_only=True,map_location='cpu',mmap=True);last=qr['roma']['path']
        old=payload['records'][qr['source_index']];assert old['query_id']==qr['source_query_id'] and old['candidate_physical_rows']==qr['axis']
        qocr=read(OUT/'ocr'/f"{qr['query_sha']}.json");qc=geometry(old['query_geometry'],qr['query_sha']);q=qocr['lines']
        assert qocr['oriented_size_hw']==old['query_geometry']['oriented_size_hw']
        assert len(old['candidates'])==128
        rows=[]
        for j,c in enumerate(old['candidates']):
            sha=qr['reference_shas'][j];assert sha==c['reference_image_sha256']
            rocr=read(OUT/'ocr'/f'{sha}.json');r=rocr['lines'];rc=geometry(c['reference_geometry'],sha)
            assert rocr['oriented_size_hw']==c['reference_geometry']['oriented_size_hw']
            u=c['query_visibility'].numpy();v=c['reference_visibility'].numpy()
            uw=pool([x['box'] for x in q],qc,u);vw=pool([x['box'] for x in r],rc,v)
            uw=uw/max(float(u.mean()),1e-12);vw=vw/max(float(v.mean()),1e-12)
            seed=int(hashlib.sha256((qr['query_id']+'|'+str(c['physical_row'])+'|OCR_SUPPORT_V1').encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed)
            us=uw[rng.permutation(len(uw))];vs=vw[rng.permutation(len(vw))]
            flat,_=lexical(q,r,np.ones(len(q)),np.ones(len(r)),ix['idf'],ix['unseen_idf'])
            real,matches=lexical(q,r,uw,vw,ix['idf'],ix['unseen_idf'])
            shuffled,_=lexical(q,r,us,vs,ix['idf'],ix['unseen_idf'])
            mass=math.sqrt(float(u.mean()*v.mean()));assert abs(mass-c['old_scores']['visibility_mass'])<2e-12
            rows.append(dict(position=j,physical_row=c['physical_row'],text=flat,uv=real,shuffled=shuffled,mass=mass,
                query_box_weights=uw.tolist(),reference_box_weights=vw.tolist(),query_shuffled=us.tolist(),reference_shuffled=vs.tolist(),matched_terms=matches,
                query_nonempty=any(l['tokens'] for l in q),reference_nonempty=any(l['tokens'] for l in r)))
        write(out,dict(status='OCR_SUPPORT_FEATURES_COMPLETE',index=qr['index'],query_id=qr['query_id'],axis=qr['axis'],rows=rows,source=qr['roma']))
        emit(stage='FEATURE',index=qr['index'])
    return 0

def unit(z,y):
    import torch
    from torch.nn import functional as F
    raw=y==-1;ii=torch.arange(len(y));idx=y.clamp_min(0);wrong=z.clone();wrong[ii[~raw],idx[~raw]]=-torch.inf
    return torch.where(raw,F.softplus(z.amax(1)),F.softplus(-z[ii,idx])+F.softplus(wrong.amax(1))).mean()

def fit(fold,budget):
    import torch
    torch.set_num_threads(4);torch.manual_seed(17);p=protocol();deadline=time.monotonic()+budget
    cache=read(checked(p['cache']));rows=cache['rows'][:128];spl=read(checked(p['split']))['folds'][fold]
    roles={r['query_id']:r for r in read(checked(p['folds'][str(fold)]['train_roles']))['records']}
    gallery={r['physical_row']:r['identity'] for r in read(checked(p['gallery']))['records']}
    tr=[i for i,r in enumerate(rows) if r['query_id'] in set(spl['train_query_ids'])];he=[i for i,r in enumerate(rows) if r['query_id'] in set(spl['heldout_query_ids'])]
    assert set(tr).isdisjoint(he) and len(tr)+len(he)==128
    valid=[];ys=[]
    for i in tr:
        r=rows[i];order=[r['winner']]+r['challengers'];target=roles[r['query_id']]['identity'];k=next((j for j,v in enumerate(order) if gallery[r['axis'][v]]==target),None)
        if k is not None:valid.append(i);ys.append(k-1)
    # The original fold head was trained without this outer held group. Recompute
    # every row with this fold's head (never use other folds' predictions as TRAIN).
    th=np.asarray([float.fromhex(v) for v in cache['native_parameters'][fold]['COST1']]);base=np.asarray([np.asarray(r['native_X'])@th[:-1]+th[-1] for r in rows])
    fs=[read(OUT/'features'/f'{i:03d}.json') for i in range(128)]
    for i,(r,f) in enumerate(zip(rows,fs)):assert f['query_id']==r['query_id'] and f['axis']==r['axis']
    models={};predictions={};y=torch.tensor(ys,dtype=torch.long);fdir=OUT/f'fold{fold}';fdir.mkdir(parents=True,exist_ok=True)
    for arm in ARMS:
        af=fdir/f'{arm}.json'
        if af.exists():d=read(af);models[arm]=d['model'];predictions[arm]=d['predictions'];continue
        if time.monotonic()>deadline-30:return 75
        extras=[]
        for i,r in enumerate(rows):
            rr=fs[i]['rows'];w=r['winner'];cc=r['challengers'];m=np.asarray([z['mass'] for z in rr]);t=np.asarray([z['text' if arm in ['TEXT','TEXT_M','CALIBRATION_ONLY'] else 'uv' if arm=='TEXT_UV' else 'shuffled'] for z in rr])
            t=np.log1p(t);mt=m*t
            col1=t[cc]-t[w];col2=mt[cc]-mt[w]
            if arm=='TEXT':col2*=0
            if arm=='CALIBRATION_ONLY':col1*=0;col2*=0
            extras.append(np.stack([base[i],col1,col2,np.ones(127)],-1))
        x=np.asarray(extras);scale=np.sqrt((x[valid]**2).mean((0,1)));scale[scale<1e-12]=1.;scale[0]=scale[-1]=1.;x=x/scale
        xt=torch.tensor(x[valid],dtype=torch.float64);bt=torch.tensor(base[valid],dtype=torch.float64)
        theta=torch.nn.Parameter(torch.zeros(4,dtype=torch.float64));opt=torch.optim.AdamW([theta],lr=p['lr'],weight_decay=p['weight_decay']);start=0;ck=fdir/f'{arm}.pt'
        if ck.exists():
            z=torch.load(ck,weights_only=True,map_location='cpu');theta.data.copy_(z['theta']);opt.load_state_dict(z['optimizer']);start=z['step'];assert np.array_equal(z['scale'].numpy(),scale)
        initial=float(unit(bt,y));last=initial
        for step in range(start,p['updates']):
            opt.zero_grad();loss=unit(bt+xt@theta,y);assert torch.isfinite(loss);loss.backward();opt.step();last=float(loss.detach())
            if (step+1)%50==0:
                tmp=ck.with_suffix('.tmp');torch.save(dict(theta=theta.detach(),optimizer=opt.state_dict(),step=step+1,scale=torch.tensor(scale)),tmp);os.replace(tmp,ck)
                if time.monotonic()>deadline-15:return 75
        z=base+x@theta.detach().numpy();held=[];maxerr=0.
        for i in he:
            scalar=base[i]+np.sum(x[i]*theta.detach().numpy(),axis=1);maxerr=max(maxerr,float(np.abs(scalar-z[i]).max()));assert maxerr<2e-10
            k=int(z[i].argmax());pos=rows[i]['challengers'][k] if z[i,k]>0 else rows[i]['winner']
            held.append(dict(index=i,query_id=rows[i]['query_id'],selected=rows[i]['axis'][pos],logits127=z[i].tolist(),base_logits127=base[i].tolist()))
        d=dict(model=dict(theta=theta.detach().tolist(),scale=scale.tolist(),initial_train_loss=initial,final_train_loss=last,updates=p['updates'],train_query_ids=[rows[i]['query_id'] for i in valid],scalar_max_error=maxerr),predictions=held)
        write(af,d);models[arm]=d['model'];predictions[arm]=held;emit(stage='FIT_COMPLETE',fold=fold,arm=arm,initial=initial,final=last)
    write(fdir/'result.json',dict(status='OCR_SUPPORT_FOLD_COMPLETE',fold=fold,models=models,predictions=predictions,held_labels_read=False,features=[bind(OUT/'features'/f'{i:03d}.json') for i in range(128)]))
    return 0

def join():
    p=protocol();cache=read(checked(p['cache']));labs={r['query_id']:r for r in read(checked(p['original_evaluation']))['rows']};gallery={r['physical_row']:r['identity'] for r in read(checked(p['gallery']))['records']}
    folds=[read(OUT/f'fold{f}/result.json') for f in range(5)];assert all(d['status']=='OCR_SUPPORT_FOLD_COMPLETE' and not d['held_labels_read'] for d in folds)
    assembled={};maxerr=0.
    for f,d in enumerate(folds):
        roles=read(checked(p['folds'][str(f)]['train_roles']))['records'];traincomp={r['component'] for r in roles};trainid={r['identity'] for r in roles}
        for arm,preds in d['predictions'].items():
            for pred in preds:
                i=pred['index'];r=cache['rows'][i];lab=labs[r['query_id']];assert lab['fold']==f and lab['component'] not in traincomp and lab['identity'] not in trainid
                z=np.asarray(pred['base_logits127']);k=int(z.argmax());basepick=r['axis'][r['challengers'][k] if z[k]>0 else r['winner']];assert basepick==lab['selected']['COST1_FROZEN_NATIVE'],'BASE_REPLAY'
                if i not in assembled:assembled[i]=dict(index=i,query_id=r['query_id'],component=lab['component'],fold=f,target_in_C128=lab['target_in_C128'],selected={'RAW':lab['selected']['RAW'],'COST1_FROZEN':basepick},correct={'RAW':lab['correct']['RAW'],'COST1_FROZEN':lab['correct']['COST1_FROZEN_NATIVE']})
                assembled[i]['selected'][arm]=pred['selected'];assembled[i]['correct'][arm]=gallery[pred['selected']]==lab['identity']
    assert len(assembled)==128
    rows=[assembled[i] for i in range(128)];counts={a:sum(r['correct'][a] for r in rows) for a in ['RAW','COST1_FROZEN',*ARMS]};comparisons={}
    for a,b in [(a,'COST1_FROZEN') for a in ARMS]+[('TEXT_UV','TEXT_M'),('TEXT_UV','TEXT_UV_SHUFFLED')]:
        groups={g:[] for g in sorted({r['component'] for r in rows})}
        for r in rows:groups[r['component']].append(int(r['correct'][a])-int(r['correct'][b]))
        v=np.asarray([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(20261009);boot=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
        comparisons[a+' minus '+b]=dict(rescue=sum(r['correct'][a] and not r['correct'][b] for r in rows),breaks=sum(r['correct'][b] and not r['correct'][a] for r in rows),component_mean=float(v.mean()),exploratory_group95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(status='OCR_SUPPORT_PILOT128_COMPLETE',queries=128,groups=len({r['component'] for r in rows}),candidate_recall=sum(r['target_in_C128'] for r in rows),counts=counts,comparisons=comparisons,rows=rows,scope=p['evidence'],primary='TEXT_UV vs TEXT_M and TEXT_UV_SHUFFLED; no threshold or arm selection on held outcomes')
    write(OUT/'result.json',result);write(OUT/'validation.json',dict(status='FIVE_FOLD_COVERAGE_BASE_REPLAY_AND_SCALAR_PASS',result=bind(OUT/'result.json')))
    table='\n'.join(f'| {a} | {v}/128 |' for a,v in counts.items())
    (OUT/'report.md').write_text('# OCR空间支持首轮结果\n\n'+p['evidence']+'\n\n| 方法 | 正确数 |\n|---|---:|\n'+table+'\n\n具体救回、误伤、组区间及逐候选分数见result.json和fold*/result.json。所有模型同一OCR、候选与原五折；新增头仅用本128子集的TRAIN组。冻结COST1来自原H593对应折，不能把该首轮当成全H593提升或外部确认。\n')
    emit(stage='COMPLETE',counts=counts)

def self_test():
    assert tokens('5 mg 0.5 mL 10mg')==['5','mg','0.5','ml','10','mg']
    cells=np.array([[0,0,.5,1],[.5,0,1,1.]])
    assert np.array_equal(pool([[0,0,1,1],[0,0,.5,1]],cells,[.2,.8]),[.5,.2])
    q=[dict(tokens=['5','mg'],recognition_score=1.)];r=[dict(tokens=['10','mg'],recognition_score=1.)]
    s,match=lexical(q,r,[1],[1],{'5':1.,'10':1.,'mg':1.},1.);assert s==.5 and match[0][2]=='mg'
    assert lexical([],r,[],[1],{},1.)[0]==0
    assert lexical(q,[],[1],[],{},1.)[0]==0
    import torch
    x=torch.tensor([[.1,.2],[-.4,.2]],dtype=torch.float64,requires_grad=True);y=torch.tensor([-1,0]);l=unit(x,y);l.backward();assert torch.isfinite(x.grad).all()
    write(OUT/'self_test.json',dict(status='OCR_TOKEN_POOL_FREE_MATCH_AND_LOSS_PASS',checks=['single_digit_decimal_retention','area_pool','uniform_support_equivalence','free_text_exact_numeric_distinction','empty_modality','cost1_finite_gradient']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','smoke','ocr','index','features','fit','join','self-test']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--budget',type=float,default=430);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='self-test':self_test()
    elif a.stage in ('smoke','ocr'):sys.exit(ocr(a.shard,a.budget,a.stage=='smoke'))
    elif a.stage=='index':index()
    elif a.stage=='features':sys.exit(features(a.shard,a.budget))
    elif a.stage=='fit':sys.exit(fit(a.fold,a.budget))
    elif a.stage=='join':join()
