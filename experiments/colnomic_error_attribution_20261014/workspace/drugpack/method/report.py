from evaluation.paths import project_path
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .common import RUNS, ROOT, RESULTS, WORK, read, save
from .ranking import baseline_ranks, paired_comparison, summarize


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def table_row(dataset,split,method,metrics,**extra):
    row = dict(dataset=dataset,split=split,method=method,n=metrics['n'],**extra)
    for key in ('micro','macro'):
        for name,value in metrics[key].items():
            row[key+'_'+name] = value
    row.update(rescued=metrics['rescued'],harmed=metrics['harmed'],candidate_recall=metrics['candidate_recall'])
    interval = metrics.get('paired_intervals',{}).get('micro',{}).get('R@1',{})
    row.update(delta_R1_ci_low=interval.get('low'),delta_R1_ci_high=interval.get('high'))
    return row


def validation(run,record):
    dataset='medicine_catalog' if record['regime']=='catalog' else 'medicine'
    manifest=read(WORK/dataset/'manifest.json')
    labels=read(WORK/dataset/'labels.json')
    rows=[r for r in manifest['rows'] if r['split']=='val']
    folder=RUNS/run
    k=record['candidate_k']
    if (folder/'best_validation.json').exists():
        source=read(folder/'best_validation.json')[str(k)]['predictions']
    else:
        source=read(folder/f'val_k{k}.json')['predictions']
    assert [p['id'] for p in source]==[r['id'] for r in rows]
    raw=baseline_ranks(manifest)
    corrected=[]
    for row,pred in zip(rows,source,strict=True):
        corrected.append(dict(pred,raw_rank=raw[row['id']],
             rank=pred['rank'] if pred['rank'] is not None else raw[row['id']],
             target_in_candidates=bool(set(row['candidates'][:k])&set(labels[row['id']]))))
    return summarize(corrected,False)


def comparisons(selection):
    results, rows = {}, []
    def predictions(dataset, name):
        split = 'test' if dataset == 'medicine' else 'transfer'
        ds = 'medicine_catalog' if dataset == 'medicine' and name.startswith('medicine_catalog_') else dataset
        k = 32 if ds == 'forb32' else selection['runs'][name]['candidate_k']
        return read(RESULTS/'evaluation'/ds/split/f'{name}_k{k}'/'predictions.json')
    for dataset in ('medicine','grozi','milk','forb32'):
        split = 'test' if dataset == 'medicine' else 'transfer'
        baselines = read(RESULTS/'baselines'/dataset/f'{split}.json')
        for regime, name in selection['primary'].items():
            method = predictions(dataset, name)
            references = {selection['selected'][regime+'/'+role]: predictions(dataset, selection['selected'][regime+'/'+role])
                          for role in ('external','reader')}
            control_k = 32 if dataset=='forb32' else 128
            for control in ('L0','M','weighted_L','S'):
                references[control] = read(RESULTS/'evaluation'/dataset/split/f'{control}_k{control_k}'/'predictions.json')
            for baseline, result in baselines.items():
                if baseline in ('colnomic7b','colnomic7b/image','bm25/bm25'):
                    references[baseline] = result['predictions']
            if dataset == 'medicine':
                references['BM25+7B calibrated zscore'] = read(
                    RESULTS/'baselines/medicine/tuned_fusion_test.json')['colnomic7b']['predictions']
            for reference, other in references.items():
                result = paired_comparison(method, other, reference)
                token = '/'.join((dataset, regime, reference))
                results[token] = dict(method=name, **result)
                row = dict(dataset=dataset, regime=regime, method=name, reference=reference,
                           n=result['n'], rescued=result['rescued'], harmed=result['harmed'])
                for metric, delta in result['delta'].items():
                    interval = result['paired_intervals']['micro'][metric]
                    row.update({metric+'_delta':delta, metric+'_low':interval['low'], metric+'_high':interval['high']})
                rows.append(row)
    save(RESULTS/'paired_comparisons.json',results)
    write_csv(RESULTS/'tables/paired_comparisons.csv',rows)
    return rows


def main():
    torch.set_num_threads(2)
    verified=read(RESULTS/'verification_final.json')
    assert verified['status']=='PASS'
    selection=read(RESULTS/'selection.json')
    groups={regime:{r['id']:r['group'] for r in read(WORK/dataset/'manifest.json')['rows']}
            for regime,dataset in (('strict','medicine'),('catalog','medicine_catalog'))}
    run_info={}
    for name,run in selection['runs'].items():
        folder=RUNS/name
        complete=read(folder/'complete.json')
        ckpt=torch.load(project_path(run['checkpoint']),weights_only=True)
        initial=read(folder/'initialization.json') if (folder/'initialization.json').exists() else {}
        parameters=initial.get('parameters',complete.get('parameters',0))+len(ckpt['theta'])
        config=ckpt['config']
        training_ids=config.get('train',config.get('training_rows',[]))
        run_info[name]=dict(regime=run['regime'],family=run['family'],parameters=parameters,
             training_queries=len(training_ids),training_groups=len({groups[run['regime']][qid] for qid in training_ids}),
             seed=config.get('seed'),location=config.get('location'),condition=config.get('condition','scalar'),
             arm=config.get('arm'),loss=config.get('loss'),budget=config.get('budget',100),
             epochs=config.get('epochs'),linear_fit_steps=config.get('steps'),
             bottleneck=config.get('bottleneck'),head_width=config.get('width'),train_seconds_per_update=complete.get('mean_update_seconds'),
             checkpoint=run['checkpoint'],checkpoint_sha256=run['checkpoint_sha256'])
    all_rows=[]
    for dataset in ('medicine','grozi','milk','forb32'):
        splits=('val','test') if dataset=='medicine' else ('transfer',)
        for split in splits:
            for method,result in read(RESULTS/'baselines'/dataset/f'{split}.json').items():
                all_rows.append(table_row(dataset,split,method,result['metrics'],kind=result.get('kind','existing_baseline'),
                               regime='mixed_region_diagnostic' if result.get('kind') == 'mixed_region_diagnostic' else 'frozen',
                               source=result['source']))
    for split in ('val','test'):
        for model,result in read(RESULTS/'baselines/medicine'/f'tuned_fusion_{split}.json').items():
            all_rows.append(table_row('medicine',split,model+'/bm25_val_calibrated',result['metrics'],
                                     kind='calibrated_baseline',regime='validation_calibrated',alpha=result['alpha']))
    for name,run in selection['runs'].items():
        metrics=validation(name,run)
        all_rows.append(table_row('medicine','val',name,metrics,kind='trained',k=run['candidate_k'],**run_info[name]))
    for split in ('val','test'):
        for name,result in read(RESULTS/'fusion'/f'{split}.json')['methods'].items():
            info=run_info.get(name,dict(regime='frozen',parameters=0))
            row=table_row('medicine',split,'RRF_BM25+'+name,result['metrics'],
                          kind='fused_trained' if name in run_info else 'rrf_baseline',alpha=result['alpha'],**info)
            comparison=result.get('comparison_to_calibrated_RAW_RRF',{}).get('paired_intervals',{}).get('micro',{}).get('R@1',{})
            row.update(delta_R1_vs_RAW_RRF_ci_low=comparison.get('low'),delta_R1_vs_RAW_RRF_ci_high=comparison.get('high'))
            all_rows.append(row)
    evaluations={}
    for path in sorted((RESULTS/'evaluation').glob('*/*/*/complete.json')):
        result=read(path);cfg=result['config'];name=cfg['run']
        data='medicine' if cfg['dataset'].startswith('medicine') else cfg['dataset']
        info=run_info.get(name,dict(regime='frozen',parameters=0))
        row=table_row(data,cfg['split'],name,result['metrics'],kind='trained' if name in run_info else 'control',
                      k=cfg['k'],intervention=cfg['intervention'],
                      cached_rerank_seconds=result['timing']['seconds_per_query'],**info)
        all_rows.append(row)
        evaluations[(cfg['dataset'],cfg['split'],name,cfg['k'],cfg['intervention'])]=row
    write_csv(RESULTS/'tables/all_results.csv',all_rows)
    write_csv(RESULTS/'tables/ablations.csv',[r for r in all_rows if r['kind']=='trained' and r['split']=='test'])
    transfer=[r for r in all_rows if r['split']=='transfer']
    write_csv(RESULTS/'tables/transfer.csv',transfer)
    grouped=defaultdict(list)
    for name,run in selection['runs'].items():
        info=run_info[name]
        if info['seed'] is None:
            continue
        seed_family=info['family']
        if seed_family=='external' and info['head_width'] is not None:
            seed_family=f"scalar_head_w{info['head_width']}_{info['loss']}"
        elif seed_family not in ('hidden','embedding','patch_hidden'):
            continue
        else:
            seed_family += '_' + info['loss']
        ds='medicine_catalog' if run['regime']=='catalog' else 'medicine'
        row=evaluations[(ds,'test',name,run['candidate_k'],None)]
        grouped[(run['regime'],seed_family)].append(row)
    seeds=[]
    for (regime,family),rows in grouped.items():
        row=dict(regime=regime,family=family,seeds=len(rows))
        for metric in ('micro_R@1','micro_R@5','micro_R@10','micro_MRR','macro_R@1','macro_MRR'):
            data=[r[metric] for r in rows]
            row[metric+'_mean']=float(np.mean(data))
            row[metric+'_std']=float(np.std(data,ddof=1)) if len(data)>1 else None
        seeds.append(row)
    write_csv(RESULTS/'tables/seeds.csv',seeds)
    timings=[]
    for path in sorted((RESULTS/'timing').glob('*/complete.json')):
        result=read(path)
        row=dict(method=result['run'],queries=result['queries'],k=result['k'],host=result['host'],
                 gpus='; '.join(result['gpus']),scope=result['scope'],
                 uses_roma=result['uses_roma'],support_cache=result['support_cache'],
                 model_loading_seconds=result['model_loading_seconds'],
                 torch_peak_allocated_gib='; '.join(f'{v/2**30:.3f}' for v in result['peak_memory_bytes']))
        row.update({stage:values['mean'] for stage,values in result['seconds'].items()})
        timings.append(row)
    write_csv(RESULTS/'tables/online_timing.csv',timings)
    comparisons(selection)
    from .figures import render
    render(selection)
    medicine_primary, medicine_head, forb_primary = {}, {}, {}
    for regime in ('strict','catalog'):
        dataset = 'medicine_catalog' if regime=='catalog' else 'medicine'
        primary = selection['primary'][regime]
        head = selection['selected'][regime+'/external']
        medicine_primary[regime] = 100*evaluations[(dataset,'test',primary,selection['runs'][primary]['candidate_k'],None)]['micro_R@1']
        medicine_head[regime] = 100*evaluations[(dataset,'test',head,selection['runs'][head]['candidate_k'],None)]['micro_R@1']
        forb_primary[regime] = 100*evaluations[('forb32','transfer',primary,32,None)]['micro_R@1']
    medicine_ocr = next(100*r['micro_R@1'] for r in all_rows if r['dataset']=='medicine' and r['split']=='test' and r['method']=='bm25/bm25')
    forb_fixed = 100*evaluations[('forb32','transfer','weighted_L',32,None)]['micro_R@1']
    lines=['# Rebut experiments','',
        '所有表格来自实际完成并通过检查的运行。模型和超参数只根据药盒验证集选择；跨数据集直接使用同一检查点，无重新训练或外部数据调参。',
        '', f"药盒测试 R@1：验证集选定的主适配器在严格训练／图库可见训练下分别为 {medicine_primary['strict']:.2f}%／{medicine_primary['catalog']:.2f}%；对应外部头为 {medicine_head['strict']:.2f}%／{medicine_head['catalog']:.2f}%，纯 OCR＋BM25 为 {medicine_ocr:.2f}%。比较训练收益时，需要同时考虑这些简单对照及配对置信区间。",
        '', f"FORB 上，无需训练的固定 weighted_L 规则为 {forb_fixed:.2f}%，两套主适配器分别为 {forb_primary['strict']:.2f}%／{forb_primary['catalog']:.2f}%。这些结果区分了几何匹配本身的收益与训练适配器的收益；不能只根据相对原始图像模型的提升来判断训练是否必要。",
        '', '## Protocol','',
        '- 药盒：706 张训练、147 张验证、147 张测试；75/16/16 个来源组；完整图库 5,405 行、5,330 个不同图像。所有照片存在历史开发接触，不能称为从未接触过的盲测。',
        '- Strict：训练只使用 80 个训练包装的正设计图作为候选。Catalog：完整固定设计图库参与训练负样本；只有训练照片标签参与参数拟合，不能称为未见设计图训练。',
        '- 两种训练范围也改变了负样本池的大小和难度；二者差异不能单独归因于是否见过验证／测试设计图。各项方法和消融都在各自相同训练范围内比较。',
        '- 候选来自原始 ColNomic 7B 自然排名，不插入正确答案。Catalog 中正确图不在 Top-128 的训练照片排除于重排损失；验证和测试全部保留。',
        '- 25%/50% 指从 75 个训练来源组选择 19/38 个完整组，实际使用 194/381 张训练照片，约为全部 706 张的 27.5%/54.0%。这是查询标注量消融，80 张训练设计图的负样本库保持不变，不能解释为同时缩小设计图库。',
        '- 主体裁剪补充对照复用已有完整结果：SAM3 自动选取最高置信度包装 mask；历史 human_boxes 实为 200 张人工确认框 + 800 张 SAM3 默认框的混合矩形裁剪。人工框在 train/val/test 分别为 120/22/58 张，其余为默认框。这是混合区域信息诊断，不是完整人工框上界，也不是纯自动方法；两路的区域选择与背景处理同时不同。均使用相同 ColNomic 7B 与完整设计图库。',
        '- 检索指标使用完整图库排名。只改排 Top-K，剩余名次不变；Top-K 外的正样本仍计入 MRR。区间按来源组配对重采样；FORB 的 23 对重复查询图像保留在指标中，并按相同 SHA 归为同一重采样组，宏平均按域。',
        '- GroZi：468 个官方 val-new-cl 包围框裁剪查询、1,063 张设计图。MILK：479 个测试病灶的临床图→皮肤镜图配对检索；是项目改编协议，不是官方诊断任务。MILK 未提供患者 ID，无法确认患者独立性，区间按病灶重采样。FORB：8 个域、8,995 个查询、43,585 张图库图像；固定 Top-32 重排，保留全部查询和多正样本。',
        '- 原始查询 embedding 一致性是缓存检查的一部分。FORB 的单卡编码路径在 91 张查询上与原双卡缓存不一致，已按原路径重算；原基线、候选、标签和训练权重保持不变。全部 FORB 查询的实际向量均与原缓存逐项核验，不通过检查的特征不进入报告。',
        '- FORB 评估曾有一次非有限评分中断和两次进程启动崩溃。失败查询在 CPU 上复算一次、在原 GPU 上复算 10 次，均通过，GPU 重复结果完全一致；随后三个原配置分片全部恢复完成，有限性检查未放宽。根因未在新进程中复现，原日志和跨设备对照见[恢复记录](../cache/intermediate/audits/evaluation_retry_20260925.json)。',
        '', '## What is trained','',
        'ColNomic 7B、原有检索投影及 RoMa 全部冻结。RoMa 双向可信度按图像 token 网格做面积池化，得到查询支持 u、参考支持 v，以及 M = sqrt(mean(u) × mean(v))。RAW 使用原缓存的全部有效 token；L0 使用归一化图像 token。INTERNAL3 对照控制 L0 分支本身的影响。',
        '', '无需训练的固定对照也在主表报告：M 只用整体匹配置信度；weighted_L 对参考 token 的相似度乘以 v，再对 MaxSim 按查询支持 u 加权平均；S = M × weighted_L。这些规则没有在测试或迁移数据上选权重。',
        '',
        '原始 Post 适配器把标准化 log(M) 与每个图像位置的 LayerNorm 隐状态拼接，经过 16 维瓶颈和 0.1 倍残差，再经冻结的检索投影得到该候选对应的查询向量。局部条件版本将整体 M 换成各位置的 u。隐藏层版本训练 118,304 个适配器参数及 3 个评分参数；128 维 embedding 版本训练 4,256 + 3 个参数。所有条件标准化统计量只来自训练对。',
        '', '适配器的 3 个评分输入是标准化 RAW 分差、相对内容分差和偏置；M 只通过条件化进入表示，不直接进入这个评分头。INTERNAL3 使用同样的评分输入但不训练图像表示，ADDITIVE4 则额外直接加入 M 的相对分差。常量条件适配器和 INTERNAL3 的推理不需要 RoMa。',
        '',
        '外部头也训练：INTERNAL3 / RAW_M / ADDITIVE4 / PRODUCT5 / WEIGHTED7 分别拟合 3 / 3 / 4 / 5 / 7 个评分参数。RoMa-free Reader 直接读取冻结的 token 对应关系，包含 COMPRESSED、SET 和 SPATIAL 三种对照，推理无需 RoMa。',
        '', '线性头每次拟合使用全训练集批次的 2,000 步 AdamW；ADDITIVE4、PRODUCT5 和 ADDITIVE_UV 先额外拟合 INTERNAL3 初始化。适配器和非线性头也用相应线性头初始化，再按单查询更新 8 轮。训练照片保持相同，线性头与非线性模型的优化步数不同。',
        '', '方向性对照分别使用 U = mean(u)、V = mean(v)：RAW_U / RAW_V 各 3 个参数，ADDITIVE_UV 用 5 个参数分别融合两侧覆盖率与视觉内容。这用于检查实拍局部视图与完整展开设计图的覆盖范围差异；U/V 本身的无训练排名也单独报告。',
        '', '空间打乱和 180° 旋转控制只对 SET/SPATIAL Reader 执行；COMPRESSED 没有对应的邻域消息传递，不把无操作当成空间消融。',
        '', '非线性外部头使用 RAW/M/L0 相对分数以及候选和原第一名的标准化 log(M)，在 ADDITIVE4 上增加两层 tanh MLP。宽度 61 和 340 分别对应 4,210 和 118,321 个 MLP 参数，另有 4 个基础评分参数，与两种 token 适配器的参数量接近；它不修改图像 token。它使用相同训练照片、8 轮训练、COST1/CE 对照及验证选择规则。',
        '',
        '候选分数相对原始第一名建立 HOLD 决策。COST1 对正确候选及最强错误候选约束，另有 CE 对照。适配器训练先计算完整候选前向，再对非零损失梯度对应部分重算反向，每步检查分数一致性。重算实现已增加原前向批次形状保持，避免低精度批形状变化；一次受保护中断从最后有效检查点恢复，修正及逐步校验记录见 protocol.json。冻结验证选择后才评估新模型的测试和迁移结果。',
        '', '## Medicine test','',
        '| Setting / validation-selected family | R@1 | R@5 | R@10 | MRR | Rescued / harmed vs image |',
        '|---|---:|---:|---:|---:|---:|']
    baseline_names=('colpali/image','colnomic3b/image','colnomic7b/image','Qwen3-VL-Embedding-2B/image',
                    'dinov3/image','siglip2/image','bm25/bm25','bm25_image_fusion/colnomic7b/alpha_0.5',
                    'bm25_image_fusion/Qwen3-VL-Embedding-2B/alpha_0.5','sam3_auto/colnomic7b','human_boxes/colnomic7b',
                    'colpali/bm25_val_calibrated','colnomic3b/bm25_val_calibrated',
                    'colnomic7b/bm25_val_calibrated','Qwen3-VL-Embedding-2B/bm25_val_calibrated')
    for row in all_rows:
        if row['dataset']=='medicine' and row['split']=='test' and row['method'] in baseline_names:
            label = 'mixed-box diagnostic' if row['regime'] == 'mixed_region_diagnostic' else 'baseline'
            lines.append(f"| {label}: {row['method']} | {100*row['micro_R@1']:.2f} | {100*row['micro_R@5']:.2f} | {100*row['micro_R@10']:.2f} | {row['micro_MRR']:.4f} | {row['rescued']} / {row['harmed']} |")
    for name in ('L0','M','weighted_L','S'):
        row=evaluations[('medicine','test',name,128,None)]
        lines.append(f"| fixed rule: {name} | {100*row['micro_R@1']:.2f} | {100*row['micro_R@5']:.2f} | {100*row['micro_R@10']:.2f} | {row['micro_MRR']:.4f} | {row['rescued']} / {row['harmed']} |")
    for role,name in selection['selected'].items():
        run=selection['runs'][name];ds='medicine_catalog' if run['regime']=='catalog' else 'medicine'
        row=evaluations[(ds,'test',name,run['candidate_k'],None)]
        lines.append(f"| {role} | {100*row['micro_R@1']:.2f} | {100*row['micro_R@5']:.2f} | {100*row['micro_R@10']:.2f} | {row['micro_MRR']:.4f} | {row['rescued']} / {row['harmed']} |")
    lines += ['', '## OCR fusion','',
        'RRF 在完整图库融合 BM25 排名与视觉重排后的排名，offset 固定为 60；权重仅在药盒验证集选择。相同的 RRF 基线也单独校准，避免只给新方法调权重。',
        '', '| Method | OCR weight | R@1 | R@5 | R@10 | MRR |', '|---|---:|---:|---:|---:|---:|']
    for label,name in [('RAW','RAW'),*selection['selected'].items()]:
        result=read(RESULTS/'fusion/test.json')['methods'][name];metrics=result['metrics']['micro']
        lines.append(f"| {label} | {result['alpha']:.2f} | {100*metrics['R@1']:.2f} | {100*metrics['R@5']:.2f} | {100*metrics['R@10']:.2f} | {metrics['MRR']:.4f} |")
    lines += ['', '## Frozen transfer','',
        '| Dataset | Checkpoint selected on medicine validation | K | R@1 | R@5 | R@10 | MRR |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in transfer:
        if row['kind']=='existing_baseline' and row['method'] in ('colpali','colpali/image','colnomic3b','colnomic3b/image',
                'colnomic7b','colnomic7b/image','Qwen3-VL-Embedding-2B','Qwen3-VL-Embedding-2B/image',
                'dinov3','dinov3/image','siglip2','siglip2/image','biomedclip','bm25/bm25'):
            lines.append(f"| {row['dataset']} | baseline: {row['method']} | full | {100*row['micro_R@1']:.2f} | {100*row['micro_R@5']:.2f} | {100*row['micro_R@10']:.2f} | {row['micro_MRR']:.4f} |")
    for ds in ('grozi','milk','forb32'):
        k=32 if ds=='forb32' else 128
        for name in ('L0','M','weighted_L','S'):
            row=evaluations[(ds,'transfer',name,k,None)]
            lines.append(f"| {ds} | fixed rule: {name} | {k} | {100*row['micro_R@1']:.2f} | {100*row['micro_R@5']:.2f} | {100*row['micro_R@10']:.2f} | {row['micro_MRR']:.4f} |")
    for role,name in selection['selected'].items():
        for ds in ('grozi','milk','forb32'):
            k=32 if ds=='forb32' else selection['runs'][name]['candidate_k']
            row=evaluations[(ds,'transfer',name,k,None)]
            lines.append(f"| {ds} | {role} | {k} | {100*row['micro_R@1']:.2f} | {100*row['micro_R@5']:.2f} | {100*row['micro_R@10']:.2f} | {row['micro_MRR']:.4f} |")
    lines += ['', '## Online timing','',
        '下表为每张查询的平均秒数，三个主要方法使用相同硬件和查询样本。ColNomic 图库向量已经建好并驻留，查询图片重新读取和编码。RoMa 图像描述子缓存在每张查询前清空，因此计入了参考图描述子的重复计算；同一查询内会复用查询描述子。模型加载时间与 PyTorch 峰值显存另列在 online_timing.csv 中。',
        '', '| Method | Queries | K | Encode + search | RoMa | Pool + rerank | Total seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    roles = {name:role for role,name in selection['selected'].items()}
    for timing in timings:
        lines.append(f"| {roles.get(timing['method'],timing['method'])} | {timing['queries']} | {timing['k']} | {timing['raw_end_to_end']:.3f} | {timing['roma']:.3f} | {timing['feature_pooling']+timing['learned_reranking']:.3f} | {timing['end_to_end']:.3f} |")
    lines += ['', '硬件：'+'；'.join(dict.fromkeys(t['host']+' / '+t['gpus'] for t in timings))+'.']
    lines += ['', '## Figures','',
        '![Paired R@1 changes](figures/paired_r1_changes.png)',
        '', '点为相对冻结 ColNomic 7B 的 R@1 差值，线为按来源组配对重采样的 95% 区间；包括无需训练的 weighted_L 对照。学习方法跨域均使用同一药盒验证选择的检查点。[PDF](figures/paired_r1_changes.pdf)。',
        '', '![Candidate sensitivity](figures/candidate_sensitivity.png)',
        '', '候选数敏感性使用冻结的验证最佳检查点，测试曲线只作诊断；正式 operating K 仍来自验证集。[PDF](figures/candidate_sensitivity.pdf)。']
    lines += ['', '## Files and reproduction','',
        '- `tables/all_results.csv`：全部基线、验证、测试、候选数和条件干预结果。R 指标为 0–1，正文表为百分数。',
        '- `tables/ablations.csv` / `seeds.csv`：消融和各次训练随机种子结果；每次运行的检查点及 K 均按验证集选择。',
        '- `tables/transfer.csv`：跨数据集全部结果；FORB 另含官方分母的 mAP 指标。',
        '- `tables/paired_comparisons.csv`：主适配器相对外部评分头、RoMa-free Reader、固定几何对照和主要基线的逐查询配对差值及 95% 区间；按数据集来源组重采样，不只比较纯图片基线。',
        '- `tables/online_timing.csv`：重新读取照片、编码、完整图库检索、必要的 RoMa 和重排的实测端到端耗时；固定图库已建库，模型初始化另计。缓存重排可来自不同 GPU worker，只用于说明计算范围；公平速度比较以注明硬件的端到端实测为准。',
        '- `selection.json`：冻结的选择依据、检查点路径和 SHA-256；`runs/` 保存训练输出；`evaluation/` 保存逐查询预测。',
        '- `verification_final.json`：数据、缓存、训练来源、完整图库指标与检查点一致性检查。',
        '', '基于现有正式图片、基础 embedding、OCR 和权重缓存复现；以下缓存 worker 可在独立 GPU 上重复启动，文件锁防止同一查询重复写入。训练队列可分别指定空闲 GPU，统计与选择命令应在相应输入完成后执行。',
        '', '```bash',
        'for dataset in medicine medicine_catalog grozi milk forb forb32; do',
        '  bash scripts/forb_python.sh -m drugpack.method.prepare --dataset "$dataset"',
        'done',
        'CUDA_VISIBLE_DEVICES=0,1 bash scripts/forb_python.sh -m drugpack.method.hidden --datasets medicine grozi milk forb',
        'CUDA_VISIBLE_DEVICES=0 bash scripts/forb_python.sh -m drugpack.method.roma --precision fp16 --datasets medicine medicine_catalog grozi milk forb32 --watch',
        'bash scripts/forb_python.sh -m drugpack.method.features --datasets medicine medicine_catalog grozi milk forb32 --watch',
        'bash scripts/forb_python.sh -m drugpack.method.queue --gpu 0',
        'bash scripts/forb_python.sh -m drugpack.method.queue --gpu cpu0 --jobs configs/method/head_jobs.json',
        'for dataset in medicine medicine_catalog; do',
        '  for loss in cost ce; do',
        '    for kind in INTERNAL3 RAW_M ADDITIVE4 PRODUCT5 WEIGHTED7; do',
        '      bash scripts/forb_python.sh -m drugpack.method.heads --dataset "$dataset" --loss "$loss" --kind "$kind" --wait',
        '    done',
        '  done',
        'done',
        'bash scripts/forb_python.sh -m drugpack.method.baselines --phase validation',
        'bash scripts/forb_python.sh -m drugpack.method.select',
        'bash scripts/forb_python.sh -m drugpack.method.fusion --phase validation',
        'bash scripts/forb_python.sh -m drugpack.method.dispatch',
        'bash scripts/forb_python.sh -m drugpack.method.queue --gpu 0 --jobs configs/method/evaluation_gpu_jobs.json',
        'bash scripts/forb_python.sh -m drugpack.method.queue --gpu cpu0 --jobs configs/method/evaluation_cpu_jobs.json',
        'bash scripts/forb_python.sh -m drugpack.method.baselines --phase final',
        'bash scripts/forb_python.sh -m drugpack.method.fusion --phase final',
        '# Online benchmark: two 24 GB GPUs, matching the measured RTX 6000 setup.',
        'CUDA_VISIBLE_DEVICES=0,1 bash scripts/forb_python.sh -m drugpack.method.benchmark --suite --single-encoder',
        'bash scripts/forb_python.sh -m drugpack.method.verify',
        'bash scripts/forb_python.sh -m drugpack.method.report',
        '```','']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines))
    save(RESULTS/'run_inventory.json',run_info)
    print('REPORT_COMPLETE',len(all_rows),'rows',flush=True)


if __name__=='__main__':
    main()
