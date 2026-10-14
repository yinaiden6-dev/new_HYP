from evaluation.paths import project_path
import argparse

import numpy as np
import torch

from .common import RUNS, ROOT, WORK, RESULTS, read, save, sha
from .heads import features, load_rows
from .ranking import baseline_ranks, full_rank, paired_comparison, summarize


def validation_orders(name,record):
    dataset='medicine_catalog' if record['regime']=='catalog' else 'medicine'
    _,rows=load_rows(dataset,'val')
    checkpoint=torch.load(project_path(record['checkpoint']),weights_only=True)
    folder=RUNS/name
    content_path=folder/'best_validation_contents.pt'
    contents=torch.load(content_path,weights_only=True) if content_path.exists() else None
    k=record['candidate_k']
    expected=(read(folder/'best_validation.json')[str(k)]['predictions']
              if (folder/'best_validation.json').exists() else read(folder/f'val_k{k}.json')['predictions'])
    scalar = None
    if 'mlp' in checkpoint:
        from .scalar_head import ScalarHead
        scalar = ScalarHead(checkpoint['config']['width'], checkpoint['config']['seed'], checkpoint['normalization']).eval()
        scalar.load_state_dict(checkpoint['mlp'])
    orders={}
    for i,item in enumerate(rows):
        raw=item['features']['raw'][:k]
        values=item['features']['values'][:k]
        if 'adapter' in checkpoint:
            z=features(raw,values,content=contents[i][:k])@checkpoint['theta']
        elif scalar is not None:
            with torch.inference_mode():
                z=scalar(raw,values,checkpoint['theta'])
        elif 'model' in checkpoint:
            z=features(raw,values)@checkpoint['theta']+contents[i][1:k]-contents[i][0]
        else:
            z=features(raw,values,checkpoint['config']['kind'])@checkpoint['theta']
        order=torch.argsort(torch.cat([torch.zeros(1,dtype=z.dtype),z]),descending=True,stable=True).tolist()
        full_order=order+list(range(k,len(item['row']['candidates'])))
        assert [item['row']['candidates'][p] for p in full_order[:10]]==expected[i]['top_reference_indices']
        orders[item['row']['id']]=order
    return orders


class Fusion:
    def __init__(self,split):
        from evaluation.bm25_image_fusion import load_source
        from evaluation.data import Dataset
        from evaluation.scoring import tokenize
        self.manifest=read(WORK/'medicine/manifest.json')
        self.rows=[r for r in self.manifest['rows'] if r['split']==split]
        self.labels=read(WORK/'medicine/labels.json')
        self.raw=baseline_ranks(self.manifest)
        dataset=Dataset(ROOT)
        image,_,image_source=load_source(dataset,'colnomic7b','image')
        text,_,text_source=load_source(dataset,'bm25','bm25')
        ocr=read(ROOT/'cache/ocr/photos.json')
        self.gallery=self.manifest['gallery_indices']
        self.column={r:i for i,r in enumerate(self.gallery)}
        self.data={}
        for row in self.rows:
            qi=row['query_index']
            ranks=[]
            for bank in (image,text):
                order=np.argsort(-bank['scores'][qi,self.gallery].numpy(),kind='stable')
                inverse=np.empty(len(self.gallery),dtype=np.int64)
                inverse[order]=np.arange(1,len(order)+1)
                ranks.append(inverse)
            assert np.argsort(ranks[0])[:len(row['candidates'])].tolist()==[self.column[r] for r in row['candidates']]
            self.data[row['id']]=(*ranks,bool(tokenize(ocr[row['original_id']]['text'])))
        self.sources=dict(image=image_source,text=text_source)

    def evaluate(self,alpha,orders=None):
        predictions=[]
        for row in self.rows:
            image,text,available=self.data[row['id']]
            image=image.copy()
            if orders:
                order=orders[row['id']]
                columns=[self.column[row['candidates'][i]] for i in order]
                image[columns]=np.arange(1,len(order)+1)
            scores=(1-alpha)/(60+image)+alpha/(60+text) if available else 1/(60+image)
            target={self.column[r] for r in self.labels[row['id']]}
            rank=full_rank(scores,list(range(len(scores))),target)
            predictions.append(dict(id=row['id'],group=row['group'],identity=row['identity'],
                  rank=rank,raw_rank=self.raw[row['id']],target_in_candidates=True))
        return predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','final'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    selection=read(RESULTS/'selection.json')
    split='val' if args.phase=='validation' else 'test'
    engine=Fusion(split)
    path=RESULTS/'fusion_selection.json'
    settings=read(path) if path.exists() else {}
    results={}
    methods=['RAW']+list(dict.fromkeys(selection['selected'].values()))
    for name in methods:
        orders=None
        if name!='RAW':
            record=selection['runs'][name]
            if args.phase=='validation':
                orders=validation_orders(name,record)
            else:
                ds='medicine_catalog' if record['regime']=='catalog' else 'medicine'
                source=RESULTS/'evaluation'/ds/'test'/f"{name}_k{record['candidate_k']}"/'predictions.json'
                orders={r['id']:r['permutation'] for r in read(source)}
        if args.phase=='validation' and name not in settings:
            candidates=[]
            for i in range(21):
                metrics=summarize(engine.evaluate(i/20,orders),False)
                candidates.append(dict(alpha=i/20,macro=metrics['macro'],micro=metrics['micro']))
            best=max(candidates,key=lambda c:(c['macro']['R@1'],c['macro']['MRR'],-abs(c['alpha']-.5)))
            settings[name]=dict(alpha=best['alpha'],validation=best,grid=candidates,
                scoring='weighted reciprocal ranks with offset 60; full SHA-canonical gallery; original representative row scores',
                normalization='No test or transfer fitting. Empty-query OCR falls back to image ranking.',
                selection='Medicine validation macro R@1 then macro MRR; ties nearest 0.5.',
                checkpoint_sha256=None if name=='RAW' else selection['runs'][name]['checkpoint_sha256'],
                model_selection_sha256=sha(RESULTS/'selection.json'))
            save(path,settings)
        assert settings[name]['model_selection_sha256']==sha(RESULTS/'selection.json')
        predictions=engine.evaluate(settings[name]['alpha'],orders)
        results[name]=dict(alpha=settings[name]['alpha'],metrics=summarize(predictions),predictions=predictions)
    for name,result in results.items():
        result['comparison_to_calibrated_RAW_RRF']=paired_comparison(
            result['predictions'], results['RAW']['predictions'], 'validation-calibrated RAW + BM25 RRF')
    fixed=engine.evaluate(.5)
    results['RAW_fixed_0.5']=dict(alpha=.5,metrics=summarize(fixed),predictions=fixed)
    save(RESULTS/'fusion'/f'{split}.json',dict(methods=results,sources=engine.sources,
                                            selection_sha256=sha(path)))
    print('RRF',split,{n:round(r['metrics']['micro']['R@1']*100,2) for n,r in results.items()},flush=True)


if __name__=='__main__':
    main()
