"""OCR-content-only control for the reviewed normalized vertical-order proxy.

The head receives train-vocabulary word presence, never coordinates, Token/line
indices, IDs or OCR line count. OCR/polygon-based sample association is extra
supervision. This tests a content/position correlation, not a causal mechanism.
"""
from collections import Counter
import re,json
import numpy as np
import torch
from scripts.colnomic_mechanism.common import ROOT,data,read,rows,write,csvout,sha
from scripts.colnomic_mechanism.probes import regression_fit,regression_predict,regression_metrics,save_head
from scripts.colnomic_mechanism.completion_audit import regression_fit as independent_fit,regression_predict as independent_predict
BASE=ROOT/'results/colnomic_full_audit/20261013_resume'


def words(text):return set(re.findall(r'[^\W_]+',text.lower(),flags=re.UNICODE))


def main():
    torch.set_num_threads(2);d=data();folder=BASE/'position_content_control';folder.mkdir(exist_ok=True)
    index=rows(BASE/'positions_reviewed/ocr_vertical_order_sample_index.csv');boxes=read(BASE/'manual_box_review/subject_boxes_after_user_review.json')['records']
    lines={};samples=[];line_words=[]
    for row in index:
        qid=row['photo_id']
        if qid not in lines:
            b=boxes[qid];box=np.array(b['bbox'])/np.tile([b['width'],b['height']],2);record=d['qo'][qid];size=np.array([record['width'],record['height']])
            selected=[line for line in record['lines'] if np.all((np.mean(line['polygon'],axis=0)/size>=box[:2])&(np.mean(line['polygon'],axis=0)/size<box[2:]))]
            lines[qid]=sorted(selected,key=lambda line:np.mean(line['polygon'],axis=0)[1]);assert len(lines[qid])>=4
        y=json.loads(row['target'])[0];li=int(round(y*len(lines[qid])-.5));assert abs((li+.5)/len(lines[qid])-y)<1e-12
        text=lines[qid][li]['text'];line_words.append(words(text));samples.append(dict(**row,OCR_text=text,OCR_vertical_index=li,
            OCR_association_scope='same uniquely intersecting OCR line as verified vertical-order task; index/count/coordinates are not head features'))
    y=np.array([json.loads(r['target']) for r in samples]);p=np.array([r['package_id'] for r in samples]);splits=np.array([r['split'] for r in samples]);train=splits=='train';val=splits=='val'
    counts=Counter(w for i,text in enumerate(line_words) if train[i] for w in text)
    vocabulary=sorted(sorted(counts,key=lambda w:(-counts[w],w))[:256]);assert vocabulary and set(vocabulary)<=set(counts)
    x=np.array([[int(word in text) for word in vocabulary] for text in line_words],dtype=np.float64)
    photo_counts=Counter(r['package_id'] for r in d['photos'].values());sample_counts=Counter(r['photo_id'] for r in samples)
    weight=np.array([1/(sample_counts[r['photo_id']]*photo_counts[r['package_id']]) for r in samples])
    candidates=[]
    for alpha in [.001,.01,.1,1.,10.,100.]:
        head=regression_fit(x[train],y[train],alpha,weight[train]);m=regression_metrics(y[val],regression_predict(head,x[val]),p[val]);candidates.append((m['package_macro_mae'],alpha,head))
    _,alpha,head=min(candidates,key=lambda r:(r[0],-r[1]));save_head(folder/'OCR_content_only_head.pt',head,dict(alpha=alpha,vocabulary=vocabulary,coordinates_supplied=False))
    prediction=regression_predict(head,x);table=[]
    for split in ['train','val','test']:
        mask=splits==split;table.append(dict(control='OCR_content_only',split=split,samples=int(mask.sum()),packages=len(set(p[mask])),**regression_metrics(y[mask],prediction[mask],p[mask])))
    csvout(folder/'sample_index_predictions.csv',[dict(**r,content_prediction=float(prediction[i,0])) for i,r in enumerate(samples)])
    np.savez_compressed(folder/'head_inputs.npz',word_presence=x,targets=y,weights=weight,splits=splits,package_ids=p)
    write(folder/'selection.json',dict(alpha=alpha,vocabulary=vocabulary,train_only_vocabulary=True,train_only_standardization=True,
        validation_candidates=[dict(alpha=a,package_macro_mae=m) for m,a,h in candidates],objective='validation identity macro MAE then larger ridge penalty',
        input='256 train-vocabulary word presence features; no Token embeddings, coordinates, index, IDs, OCR confidence or number of OCR lines'))
    refit=independent_fit(x[train],y[train],alpha,weight[train]);assert np.array_equal(head['mean'],refit['mean']) and np.array_equal(head['scale'],refit['scale'])
    assert np.max(np.abs(head['coefficient']-refit['coefficient']))<1e-8 and np.max(np.abs(independent_predict(refit,x)-prediction))<1e-10
    nulls=[]
    for seed in [17,31,53]:
        shuffled=y[train].copy();rng=np.random.default_rng(seed)
        for pkg in sorted(set(p[train])):
            loc=np.flatnonzero(p[train]==pkg);shuffled[loc]=shuffled[rng.permutation(loc)]
        null=regression_fit(x[train],shuffled,alpha,weight[train]);check=independent_fit(x[train],shuffled,alpha,weight[train]);assert np.max(np.abs(null['coefficient']-check['coefficient']))<1e-8
        pred=regression_predict(null,x);mask=splits=='test';table.append(dict(control='label_shuffle',seed=seed,split='test',samples=int(mask.sum()),packages=len(set(p[mask])),**regression_metrics(y[mask],pred[mask],p[mask])))
        save_head(folder/f'OCR_content_shuffle_{seed}_head.pt',null,dict(alpha=alpha,seed=seed,vocabulary=vocabulary))
        np.savez_compressed(folder/f'shuffle_{seed}_predictions.npz',predictions=pred,shuffled_training_targets=shuffled)
        nulls.append(seed)
    csvout(folder/'metrics.csv',table)
    # Lexical recurrence/position spread expose correlated phrase contents.
    association=[]
    for word in vocabulary:
        mask=np.array([word in text for text in line_words]);values=y[mask,0]
        association.append(dict(word=word,samples=int(mask.sum()),identities=len(set(p[mask])),mean_vertical_order=float(values.mean()),std_vertical_order=float(values.std()),
            train_samples=int((mask&train).sum()),val_samples=int((mask&val).sum()),test_samples=int((mask&(splits=='test')).sum())))
    csvout(folder/'content_position_associations.csv',association)
    write(folder/'validation.json',dict(status='passed',samples=len(samples),vocabulary_dimensions=len(vocabulary),selected_alpha=alpha,null_seeds_replayed=nulls,
        source_SHA256={str(path.relative_to(ROOT)):sha(path) for path in [BASE/'positions_reviewed/ocr_vertical_order_sample_index.csv',BASE/'manual_box_review/subject_boxes_after_user_review.json',ROOT/'cache/ocr/photos.json']},
        split_verification=d['split_verification'],head_standardization_and_predictions_independently_refitted=True,
        scope='OCR content-only correlation control for normalized vertical ordinal; not actual reading lines or layout identity separability'))
    (folder/'REPORT.md').write_text('# 文字内容与OCR纵向序号的关联控制\n\n头只输入训练集词表的二值词存在向量，不输入坐标/索引/ID/OCR行数或Token向量。OCR多边形和主体框只用来关联同一已核验标签样本，属于额外监督。参数仅按验证身份MAE选取，全部主头及三个打乱对照保存并独立闭式重拟合。\n\n'+json.dumps(table,ensure_ascii=False,indent=2)+'\n\n这是相关性检查，不能单凭它断言feature-only头使用了文字捷径，也不证明自然同文字包装布局可分。\n')
    print('OCR content/position control passed',len(samples),len(vocabulary),alpha,flush=True)


if __name__=='__main__':main()
