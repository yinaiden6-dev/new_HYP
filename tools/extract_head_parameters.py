#!/usr/bin/env python3
"""Archive learned tensors from mixed training payloads without prediction caches."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    import torch
    rc=a.workspace/'ICLR/new ROUTEA/RC'
    def encode(v):
        if isinstance(v,torch.Tensor):
            return {'dtype':str(v.dtype),'shape':list(v.shape),'values_hex':[float(x).hex() for x in v.reshape(-1)]}
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [encode(x) for x in v]
        return v
    count=0
    for family in ['rc_new_hyp593_oof5_v1','rc_h593_group_risk_strong_base_v1']:
        for receipt in sorted((rc/'results'/family/'fits').glob('fold*/receipt.json')):
            rec=json.loads(receipt.read_text());bound=rec['payload'];p=Path(bound['path']);assert sha(p)==bound['sha256']
            value=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            param_keys=[k for k in value if k in ('parameters','params','theta','beta','models')]
            # Explicitly reject unidentified envelopes; never call predictions parameters.
            assert param_keys,(family,list(value))
            data={'source':bound,'receipt_sha256':sha(receipt),'family':family,'fold':value.get('fold'),
                  'parameters':{k:encode(value[k]) for k in param_keys},'prediction_caches_included':False}
            dest=a.output/family/(receipt.parent.name+'.json');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(data,indent=2)+'\n');count+=1
    family='rc_six_cause_isolation_v1/loss_binding'
    for p in sorted((rc/'results'/family).glob('fold*/payload.json')):
        value=json.loads(p.read_text());vpath=p.with_name('validation.json');val=json.loads(vpath.read_text());expected=val.get('payload',{})
        if expected.get('sha256'):assert sha(p)==expected['sha256']
        data={'source':{'path':str(p),'sha256':sha(p)},'validation_sha256':sha(vpath),'family':family,'fold':value['fold'],
              'parameters_hex':value['parameters'],'prediction_caches_included':False}
        dest=a.output/'rc_six_cause_loss_binding'/f"fold{value['fold']}.json";dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(data,indent=2)+'\n');count+=1
    print(json.dumps({'status':'HEAD_PARAMETERS_EXTRACTED_NO_TRAINING','files':count}))

if __name__=='__main__':main()
