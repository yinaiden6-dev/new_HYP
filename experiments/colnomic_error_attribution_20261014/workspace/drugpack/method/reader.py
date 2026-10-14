import argparse
import sys
import time

import torch
from torch.nn import functional as F

from .common import RUNS, ROOT, WORK, RESULTS, key, read, save, signature
from .heads import cost, features, fit, load_rows, summary, wait_rows
from .post import Inputs

sys.path.insert(0, str(ROOT/'drugpack/method/legacy'))
from rc_colnomic_relation_reader_v1 import Reader


def relation_score(model, inputs, item, index, intervention='native'):
    h = inputs.hidden.get(key(item['row']))
    if h is None:
        inputs.get(item['row'])
        h = inputs.hidden[key(item['row'])]
    q = F.normalize(h['native'][h['mask']].to(inputs.device).float(),dim=-1)
    rid = item['row']['candidates'][index]
    r = inputs.reference(rid)
    _,qh,qw = h['grid'][0].tolist()
    rg = inputs.bank.layout[(True,rid)]
    return model(q,r,(qh//2,qw//2),(rg['height'],rg['width']),intervention=intervention)[0]


@torch.no_grad()
def scores(model,inputs,item,intervention='native'):
    if model.arm == 'COMPRESSED' and intervention != 'native':
        raise ValueError('Spatial neighborhood interventions are defined for SET and SPATIAL readers only')
    return torch.stack([relation_score(model,inputs,item,i,intervention)
                        for i in range(len(item['row']['candidates']))]).double()


def logits(item,relation,head):
    base = features(item['features']['raw'].to(head.device),item['features']['values'].to(head.device))
    return base@head + relation[1:]-relation[0]


def update(model,inputs,item,head,optimizer):
    started=time.monotonic()
    relation=scores(model,inputs,item).requires_grad_()
    value=cost(logits(item,relation,head),item['target'])
    dr,dh=torch.autograd.grad(value,(relation,head))
    optimizer.zero_grad(set_to_none=True)
    head.grad=dh.detach()
    active=dr.nonzero().flatten().tolist()
    error=0.
    for i in active:
        s=relation_score(model,inputs,item,i)
        error=max(error,abs(float(s.detach())-float(relation[i].detach())))
        (s*dr[i].detach()).backward()
    if not error < 2e-6:
        raise ValueError(f'Reader gradient replay differs: {error}; query={item["row"]["id"]}; active={active}')
    parameters=list(model.parameters())
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError(f'Non-finite Reader gradient: {item["row"]["id"]}')
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
    torch.nn.utils.clip_grad_norm_([head],1.)
    optimizer.step()
    return dict(loss=float(value.detach()),gradient_norm=norm,replay_error=error,
                active=len(active),seconds=time.monotonic()-started)


@torch.inference_mode()
def validate(model,inputs,rows,head,intervention='native'):
    results={}
    all_scores=[scores(model,inputs,row,intervention).cpu() for row in rows]
    for k in (8,16,32,64,128):
        orders=[]
        for item,ss in zip(rows,all_scores):
            n=min(k,len(ss))
            sliced=dict(item,features=dict(item['features'],raw=item['features']['raw'][:n],
                                          values=item['features']['values'][:n]))
            z=logits(sliced,ss[:n],head.detach().cpu())
            order=torch.argsort(torch.cat([torch.zeros(1,dtype=z.dtype),z]),descending=True,stable=True).tolist()
            orders.append(order+list(range(n,len(ss))))
        report,predictions=summary(rows,orders)
        results[str(k)]=dict(metrics=report,predictions=predictions)
    return results,all_scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['COMPRESSED','SET','SPATIAL'],default='SPATIAL')
    parser.add_argument('--dataset',default='medicine')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.smoke:
        manifest=read(WORK/'medicine/manifest.json')
        labels=read(WORK/'medicine/labels.json')
        train=[]
        for row in manifest['rows']:
            path=WORK/'medicine/features'/(key(row)+'.pt')
            if row['split']=='train' and path.exists():
                p=[i for i,r in enumerate(row['candidates']) if r in labels[row['id']]]
                train.append(dict(row=row,features=torch.load(path,weights_only=True),positions=p,target=p[0]))
            if len(train)==4:
                break
        val=[]
    else:
        manifest,train=load_rows(args.dataset,'train')
        val=[]
    for item in train+val:
        item['features']['values'][:,1:]=0.
    config=dict(arm=args.arm,seed=args.seed,epochs=args.epochs,loss='cost',dataset=args.dataset,manifest=manifest['signature'],
                feature_recipe=train[0]['features']['support_signature'],
                training_rows=[x['row']['id'] for x in train],teacher_inputs=False)
    sig=signature(config)
    run=f'reader_{args.arm}_s{args.seed}'
    if args.dataset!='medicine':
        run=args.dataset+'_'+run
    folder=(RESULTS/'smoke' if args.smoke else RUNS)/run
    if (folder/'complete.json').exists():
        assert read(folder/'complete.json')['signature']==sig
        return
    model=Reader(args.arm,args.seed).to('cuda:0')
    head=torch.nn.Parameter(fit(train,'INTERNAL3').to('cuda:0'))
    optimizer=torch.optim.AdamW([dict(params=model.parameters(),lr=.001),dict(params=[head],lr=.003)],weight_decay=.001)
    inputs=Inputs(manifest,'cuda:0')
    step,best,history=0,(-1.,-1.),[]
    checkpoint=folder/'resume.pt'
    def snapshot():
        return dict(signature=sig,config=config,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    theta=head.detach().cpu(),step=step,best=best,history=history)
    if checkpoint.exists() and not args.smoke:
        old=torch.load(checkpoint,weights_only=True)
        assert old['signature']==sig
        model.load_state_dict(old['model']);head.data.copy_(old['theta'].to(head.device))
        optimizer.load_state_dict(old['optimizer']);step,best,history=old['step'],tuple(old['best']),old['history']
    total=4 if args.smoke else len(train)*args.epochs
    while step<total:
        epoch,offset=divmod(step,len(train))
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        item=train[order[offset]]
        record=update(model,inputs,item,head,optimizer)
        step+=1;record.update(step=step,id=item['row']['id']);history.append(record)
        print(args.arm,record,flush=True)
        if step%25==0 or step==total:
            save(checkpoint,dict(**snapshot(),optimizer=optimizer.state_dict()))
            save(folder/'progress.json',dict(step=step,total=total,last=record,updated=time.time()))
        if not args.smoke and step%len(train)==0 and (epoch+1)%2==0:
            if not val:
                _,val=wait_rows(args.dataset,'val')
                for item in val:
                    item['features']['values'][:,1:]=0.
            reports,contents=validate(model,inputs,val,head)
            metric=max((v['metrics']['macro']['R@1'],v['metrics']['macro']['MRR']) for v in reports.values())
            save(folder/f'val_epoch{epoch+1}.json',reports)
            if metric>best:
                best=metric;save(folder/'best.pt',snapshot());save(folder/'best_validation.json',reports)
                save(folder/'best_validation_contents.pt',contents)
            save(checkpoint,dict(**snapshot(),optimizer=optimizer.state_dict()))
    save(folder/'final.pt',snapshot())
    save(folder/'complete.json',dict(signature=sig,config=config,best=best,step=step,
            parameters=sum(p.numel() for p in model.parameters()),
            mean_update_seconds=sum(r['seconds'] for r in history)/len(history)))


if __name__=='__main__':
    main()
