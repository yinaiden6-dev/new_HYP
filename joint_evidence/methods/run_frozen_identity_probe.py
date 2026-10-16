"""Local, family-blocked, full frozen-token paired readout. No encoder training."""
import os,sys,json,csv,hashlib,time,collections,argparse
from pathlib import Path
B=Path(__file__).resolve().parent; O=B/'identity_probe_local'; O.mkdir(exist_ok=True)
os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(O/'dependencies'));sys.path.insert(0,str(B.parent))
import numpy as np
import torch
from torch_numpy_reader import load_tensor_archive
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OLD=Path('C:/Users/shazer/Desktop/B03_token_mechanism_analysis_20261013/token_mechanism_analysis/B03/reproduction')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def rows(p):return list(csv.DictReader(open(p,encoding='utf-8-sig',newline='')))
def hs(x):return int(hashlib.sha256(x.encode()).hexdigest()[:12],16)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
P=rows(B/'token_deepening/token_pair_partitions.csv');Q=rows(B/'token_deepening/query_outcomes.csv')
meta0=read(OLD/'token_metadata.json');meta={r['id']:r for v in meta0.values() for r in v}
refs=rows(B/'source/dataset/lists/references.csv');parent={r['reference_id']:r['reference_id'] for r in refs}
edges=[]
def root(x):
    while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
    return x
def union(a,b,why):
    if a not in parent or b not in parent:return
    ra,rb=root(a),root(b)
    if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
    edges.append([a,b,why])
def group(items,why):
    d={}
    for rid,key in items:
        if key:
            if key in d:union(d[key],rid,why)
            else:d[key]=rid
group(((r['reference_id'],r['sha256']) for r in refs),'exact_reference_sha')
group(((Path(r['image']).stem,r.get('dailymed',{}).get('url')) for r in read(B/'source/dataset/references/metadata.json')),'same_dailymed_set')
conf=read(B/'source/dataset/annotations/confusion_metadata.json')
def confirmed(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k=='confirmed_pair_ids':
                for pair in v:
                    a,b=pair.split('__');union(a,b,'human_confirmed_confusion_component')
            else:confirmed(v)
    elif isinstance(x,list):
        for v in x:confirmed(v)
confirmed(conf)
V=read(O/'visibility_inventory.json')
for r in V:union(r['positive'],r['negative'],'focal39_conservative_confusion_block_not_claimed_biological_family')
ledger=read(OLD/'candidate_ledger_all1117.json')
if isinstance(ledger,dict):ledger=ledger.get('records',ledger.get('queries',ledger))
assert isinstance(ledger,list)
for r in ledger:
    ids=r['positive_reference_ids']
    for rid in ids[1:]:union(ids[0],rid,'same_query_positive_equivalence')
group(((r['image_positive'],r['group_id']) for r in Q),'same_source_group')
group(((r['image_positive'],meta[r['query']]['image_sha256']) for r in Q),'duplicate_query_sha')
families={rid:root(rid) for rid in parent}
for r in P:r['family']=families[r['positive']];r['negative_family']=families[r['negative']]
splits=[]
for fold in range(3):
    te=[i for i,r in enumerate(P) if hs('20261014|'+r['family'])%3==fold]
    blocked={f for i in te for f in [P[i]['family'],P[i]['negative_family']]}
    eligible=[i for i,r in enumerate(P) if r['family'] not in blocked and r['negative_family'] not in blocked]
    va=[i for i in eligible if hs('val|'+P[i]['family'])%5==0]
    vb={f for i in va for f in [P[i]['family'],P[i]['negative_family']]}
    tr=[i for i in eligible if P[i]['family'] not in vb and P[i]['negative_family'] not in vb]
    def fs(ids):return {f for i in ids for f in [P[i]['family'],P[i]['negative_family']]}
    assert not fs(tr)&fs(va) and not fs(tr)&fs(te) and not fs(va)&fs(te)
    assert tr and va and te
    splits.append(dict(fold=fold,train=tr,validation=va,test=te,counts={k:dict(pairs=len(v),queries=len({P[i]['query'] for i in v}),positive_families=len({P[i]['family'] for i in v})) for k,v in [('train',tr),('validation',va),('test',te)]},candidate_family_overlap=0))
assert sorted(i for s in splits for i in s['test'])==list(range(len(P)))
protocol=dict(created_before_fitting=True,question='Can a small shared readout of full frozen token vectors improve natural positive/negative discrimination for unseen source/confusion components?',scope='Historical three-fold cross-fit on 1860 fixed pairs / 1117 photos, not a fresh blind test and not full-gallery retrieval.',features='Original cached valid 128-dimensional tokens only, including valid template tokens; no OCR, coordinates, positions, IDs, order, ROI or edited pixels enter readout.',architecture='Shared bias-free Linear(128,16), tanh, L2 normalization; directional mean MaxSim; residual baseline + beta * learned score.',parameters=2048,epochs=4,seeds=[17,31,53],optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),loss='pair logistic softplus(-target*(baseline_margin + .2*learned_margin)/.03), equal pair weight',selection='Best validation positive-component macro accuracy, averaging pairs within photo first; epoch0..4 and beta in [0,.05,.1,.2,.5,1]. Ties prefer beta0, then smaller beta, then earlier epoch. Separate head-only selection on same validation metric.',labels='True positive-versus-negative. Null controls flip signs for exactly half of training positive components, deterministically per seed/fold; all rows in a component share a flip. Validation and test labels stay true for both pipelines.',controls=['original MaxSim','random untrained projection','matched shuffled-label training'],splits='Union SHA duplicates, DailyMed set, source group, positive equivalents, human confusion edges and all focal39 confusions. Train/validation/test candidate components are disjoint, including negatives. Grouping is conservative but not an exhaustive independent brand audit.',split_counts=[s['counts'] for s in splits],limitations=['Positive references are historical ledger choices; positive equivalence preserved. Hard negatives were mined in earlier full-gallery analyses.', 'Selection of focal39 and grouping use previously observed historical errors; no claim of untouched test.', 'Full token cache is frozen; no claim that failure of this 2048-parameter readout proves absence of information.','Color, text, layout, background and templates remain mixed in token vectors; improvement would not isolate layout information.'])
write(O/'probe_protocol.json',protocol);write(O/'family_blocks.json',dict(reference_components=families,edges=edges));write(O/'probe_splits.json',splits)
write(O/'probe_pair_inventory.json',[{k:r[k] for k in ['query','positive','negative','family','negative_family','package_id','group_id','split','roles','image_margin']} for r in P])
print('Split counts:',protocol['split_counts'],flush=True)
banks={}
for name in ['photos','references']:
    b=load_tensor_archive(OLD/f'cache/image_embeddings/colnomic7b/{name}.pt')
    banks.update({rid:e for rid,e in zip(b['ids'],b['embeddings'])})
cache=collections.OrderedDict();validated=set()
def feat(rid):
    if rid in cache:cache.move_to_end(rid);return cache[rid]
    raw=np.asarray(banks[rid]);m=meta[rid]
    if rid not in validated:
        token_hash=hashlib.sha256(m['dtype'].encode()+json.dumps(list(raw.shape),separators=(',',':')).encode()+raw.tobytes()).hexdigest()
        assert token_hash==m['all_tokens_sha256'],rid
        validated.add(rid)
    x=torch.from_numpy(np.array(raw[m['valid_token_indices']],dtype=np.float32,copy=True))
    assert x.shape[1]==128 and torch.isfinite(x).all()
    cache[rid]=x
    if len(cache)>24:cache.popitem(last=False)
    return x
def model(seed):
    torch.manual_seed(seed);return torch.nn.Linear(128,16,bias=False)
def enc(m,x):return torch.nn.functional.normalize(torch.tanh(m(x)),dim=-1)
def delta(m,r):
    q,p,n=[enc(m,feat(r[k])) for k in ['query','positive','negative']]
    return (q@p.T).max(1).values.mean()-(q@n.T).max(1).values.mean()
@torch.no_grad()
def evaluate(m,ids):
    return np.array([delta(m,P[i]).item() for i in ids],np.float64)
def macro(ids,v):
    by=collections.defaultdict(list)
    for i,x in zip(ids,v):by[(P[i]['family'],P[i]['query'])].append(float(x>1e-9)+.5*float(abs(x)<=1e-9))
    fam=collections.defaultdict(list)
    for (f,q),x in by.items():fam[f].append(np.mean(x))
    return float(np.mean([np.mean(x) for x in fam.values()]))
def baseline(ids):return np.array([float(P[i]['image_margin']) for i in ids])
args=argparse.ArgumentParser();args.add_argument('--benchmark',action='store_true');a=args.parse_args()
if a.benchmark:
    m=model(17);opt=torch.optim.AdamW(m.parameters(),lr=.001);start=time.time();diff=[]
    for r in P[:30]:
        opt.zero_grad();d=delta(m,r);loss=torch.nn.functional.softplus(-(float(r['image_margin'])+.2*d)/.03);loss.backward();opt.step()
        with torch.no_grad():
            q,p,n=[feat(r[k]) for k in ['query','positive','negative']];z=(q@p.T).max(1).values.mean()-(q@n.T).max(1).values.mean();diff.append(abs(z.item()-float(r['image_margin'])))
    result=dict(seconds=time.time()-start,steps=30,max_baseline_replay_error=max(diff),validated_token_arrays=len(validated));write(O/'probe_benchmark.json',result);print(result,flush=True);sys.exit()
out=O/'probe_runs';out.mkdir(exist_ok=True)
for s in splits:
    fold=s['fold'];tr,va,te=[s[k] for k in ['train','validation','test']];bv=baseline(va);bt=baseline(te)
    for seed in protocol['seeds']:
        for null in [False,True]:
            tag=f'fold{fold}_seed{seed}_{"shuffled" if null else "true"}';dest=out/tag
            if dest.with_suffix('.json').is_file():print('Resume skip',tag,flush=True);continue
            start=time.time();m=model(seed);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed+1000*fold)
            tf=sorted({P[i]['family'] for i in tr},key=lambda f:hs(f'{seed}|{fold}|null|{f}'));flipped=set(tf[:len(tf)//2]) if null else set()
            best=(-1,0,0);best_head=(-1,0);chosen=None;chosen_head=None;history=[];random_test=evaluate(m,te)
            for epoch in range(5):
                if epoch:
                    losses=[]
                    for i in rng.permutation(tr):
                        r=P[i];target=-1 if r['family'] in flipped else 1;opt.zero_grad(set_to_none=True);d=delta(m,r)
                        loss=torch.nn.functional.softplus(-target*(float(r['image_margin'])+.2*d)/.03);loss.backward();opt.step();losses.append(loss.item())
                else:losses=[]
                dv=evaluate(m,va);scores=[]
                for beta in [0,.05,.1,.2,.5,1]:
                    value=macro(va,bv+beta*dv);key=(value,-beta,-epoch);scores.append([beta,value])
                    if key>best:best=key;chosen={k:v.detach().clone() for k,v in m.state_dict().items()}
                hv=macro(va,dv);hk=(hv,-epoch)
                if hk>best_head:best_head=hk;chosen_head={k:v.detach().clone() for k,v in m.state_dict().items()}
                history.append(dict(epoch=epoch,mean_train_loss=float(np.mean(losses)) if losses else None,validation_scores=scores,head_only_validation=hv))
                print(tag,'epoch',epoch,'val',round(best[0],4),'beta',-best[1],'elapsed',round(time.time()-start,1),flush=True)
            m.load_state_dict(chosen);d=evaluate(m,te);weight=chosen['weight'].numpy();m.load_state_dict(chosen_head);hd=evaluate(m,te)
            np.savez_compressed(dest.with_suffix('.npz'),indices=np.array(te),baseline=bt,learned_margin=d,selected_margin=bt-best[1]*d,head_only_margin=hd,random_margin=random_test,weight=weight,head_weight=chosen_head['weight'].numpy())
            write(dest.with_suffix('.json'),dict(fold=fold,seed=seed,shuffled=null,beta=-best[1],epoch=-best[2],validation_macro=best[0],head_epoch=-best_head[1],head_validation_macro=best_head[0],flipped_training_components=sorted(flipped),history=history,seconds=time.time()-start))
write(O/'probe_input_validation.json',dict(validated_token_arrays=len(validated),validated_ids=sorted(validated),token_sha_validation='All used arrays checked against original metadata SHA256 before use',bank_mode='read-only memory-mapped zip storages',source_files={str(p):sha(p) for p in [OLD/'token_metadata.json',B/'token_deepening/token_pair_partitions.csv',Path(__file__),O/'probe_protocol.json',O/'probe_splits.json']},torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads()))
print('All probe fits complete',flush=True)
