#!/usr/bin/env python3
"""Export complete pipeline costs and paired fusion outcomes without fitting."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.data import read_json, write_json
from evaluation.grozi import DATA, IDS, MODELS


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    data=DATA
    costs={}
    for model in MODELS:
        for view,folder in [('image','image_embeddings'),('text','ocr_embeddings')]:
            if view=='text' and model not in MODELS[:4]:continue
            for side in ('photo','ref'):
                bank=torch.load(data/'cache'/folder/model/f'{side}.pt',map_location='cpu',weights_only=True)
                costs[model,view,side]=bank['encoding_seconds']
    ocr={side:read_json(data/'cache/ocr'/f'{side}.json')['inference_seconds'] for side in ('photo','ref')}
    configurations={(model,mode):read_json(data/'results'/model/mode/'config.json') for model,mode in IDS}
    rows=[];predictions={}
    for (model,mode),identifier in sorted(IDS.items(),key=lambda item:item[1]):
        photo_stages=set();ref_stages=set();base_modes=[]
        if mode.startswith('image'):
            photo_stages.add('image');ref_stages.add('image');base_modes=[(model,mode)]
        elif mode.startswith('ocr_to_image'):
            photo_stages.update(['ocr','text']);ref_stages.add('image');base_modes=[(model,mode)]
        elif mode=='ocr_to_ocr':
            photo_stages.update(['ocr','text']);ref_stages.update(['ocr','text']);base_modes=[(model,mode)]
        elif mode=='bm25':
            photo_stages.add('ocr');ref_stages.add('ocr');base_modes=[('bm25','bm25')]
        elif mode=='fusion':
            photo_stages.update(['image','ocr','text']);ref_stages.add('image')
            base_modes=[(model,'image'),(model,'ocr_to_image'),(model,mode)]
        elif mode=='ocr_fusion':
            photo_stages.update(['image','ocr','text']);ref_stages.update(['image','ocr','text'])
            base_modes=[(model,'image'),(model,'ocr_to_ocr'),(model,mode)]
        elif mode=='bm25_image_fusion':
            photo_stages.update(['image','ocr']);ref_stages.update(['image','ocr'])
            base_modes=[(model,'image'),('bm25','bm25'),(model,mode)]
        else:raise ValueError(mode)
        def cost(stages,side):
            return sum(ocr[side] if stage=='ocr' else costs[model,stage,side] for stage in stages)
        query=cost(photo_stages,'photo');reference=cost(ref_stages,'ref')
        scoring=sum(configurations[key]['scoring_seconds'] for key in base_modes)
        rows.append(dict(id=identifier,model=model,mode=mode,photo_inputs='+'.join(sorted(photo_stages)),
            ref_inputs='+'.join(sorted(ref_stages)),photo_count=468,ref_count=1063,
            photo_processing_seconds=query,photo_processing_ms_per_query=1000*query/468,
            ref_processing_seconds=reference,ref_processing_ms_per_image=1000*reference/1063,
            full_gallery_scoring_seconds=scoring,amortized_scoring_ms_per_query=1000*scoring/468,
            photo_processing_plus_scoring_seconds=query+scoring,
            query_pipeline_amortized_ms=1000*(query+scoring)/468,
            scoring_includes_bm25_index=any(key[0]=='bm25' for key in base_modes),
            scoring_includes_joint_maxsim_crisp=model in MODELS[:3],
            timing_scope='sum of measured batch stages; excludes model loading and cache I/O; not request latency'))
        path=data/'results'/model/mode/'predictions.jsonl'
        predictions[model,mode]=[json.loads(line) for line in path.read_text().splitlines()]
    write_csv(data/'results/pipeline_timing.csv',rows)
    comparisons=[]
    for model in MODELS:
        modes=['bm25_image_fusion']+(['fusion','ocr_fusion'] if model in MODELS[:4] else [])
        for mode in modes:
            fusion=predictions[model,mode]
            comparisons_to=[(model,'image')]
            if mode=='bm25_image_fusion':comparisons_to.append(('bm25','bm25'))
            for base_model,base_mode in comparisons_to:
                base=predictions[base_model,base_mode]
                assert [p['photo_id'] for p in fusion]==[p['photo_id'] for p in base]
                rescued=[a['photo_id'] for a,b in zip(fusion,base) if a['rank']==1 and b['rank']!=1]
                harmed=[a['photo_id'] for a,b in zip(fusion,base) if a['rank']!=1 and b['rank']==1]
                both=sum(a['rank']==b['rank']==1 for a,b in zip(fusion,base))
                comparisons.append(dict(id=IDS[model,mode],model=model,mode=mode,versus=IDS[base_model,base_mode],
                    rescued=len(rescued),harmed=len(harmed),both_correct=both,both_wrong=468-both-len(rescued)-len(harmed),
                    delta_R1_percentage_points=100*(len(rescued)-len(harmed))/468,
                    rescued_query_ids=rescued,harmed_query_ids=harmed))
    write_json(data/'results/fusion_comparisons.json',comparisons)
    write_csv(data/'results/fusion_comparisons.csv',[{k:v for k,v in row.items() if not k.endswith('_ids')} for row in comparisons])
    report=data/'results/REPORT.md'
    content=report.read_text().split('\n运行硬件为 NVIDIA GeForce RTX 2080 Ti',1)[0].rstrip()+'\n'
    readable=[]
    labels={'bm25':'OCR→OCR，BM25','image_crisp':'图像→图像，CRISP',
        'ocr_to_image_crisp':'OCR→图像，CRISP','fusion':'图像＋OCR→图像融合',
        'ocr_fusion':'图像＋OCR→OCR 融合','bm25_image_fusion':'图像＋BM25 融合'}
    for line in content.splitlines():
        if line.startswith('| B') or line.startswith('| F'):
            cells=line.split('|');model=cells[2].strip();mode=cells[3].strip()
            if mode in ('image','ocr_to_image','ocr_to_ocr'):
                score='MaxSim' if model in MODELS[:3] else '余弦'
                direction={'image':'图像→图像','ocr_to_image':'OCR→图像','ocr_to_ocr':'OCR→OCR'}[mode]
                cells[3]=f' {direction}，{score} ';line='|'.join(cells)
            elif mode in labels:
                cells[3]=' '+labels[mode]+' ';line='|'.join(cells)
        readable.append(line.replace('| 模式 |','| 输入及评分 |'))
    content='\n'.join(readable)+'\n'
    content=content.replace('缓存：../cache/image_embeddings、../cache/ocr_embeddings、../cache/ocr、../cache/scores；均按 photo/ref 分开。',
        '图像 embedding、OCR embedding 和 OCR 结果均按 photo/ref 分开缓存；../cache/scores 保存完整查询×图库分数矩阵。')
    report.write_text(content)
    with report.open('a') as stream:
        stream.write('\n运行硬件为 NVIDIA GeForce RTX 2080 Ti（每卡 11 GB）。ColNomic 7B 使用两张 GPU 分片，其余模型每次使用一张 GPU；ColNomic 在该硬件上使用 FP32 attention 兼容路径。\n')
        stream.write('\n完整链路耗时见 `pipeline_timing.csv`：已累计各方案所需的 OCR、图像/文字编码和基础评分成本。参考侧处理与查询侧处理分开；BM25 建索引时间包含在评分阶段，批量摊销时间不是单请求延迟。\n')
        stream.write('\n融合的改善/退化逐查询计数见 `fusion_comparisons.csv`；对应图片 ID 保存在 `fusion_comparisons.json`。所有融合均保持预设权重。\n')
        stream.write('\nColNomic 额外核对：图片顺序、抽样 FP64 分数以及固定官方版本的预处理 JSON 均通过检查。两侧 token 数中位数分别为 429（photo）和 146（ref）；这属于输入统计，不能单独用来解释性能差异。详见 `colnomic_input_diagnostics.json` 与 `colnomic_preprocessing_audit.json`。\n')
        stream.write('\n最终校验结果见 `verification.json`，数据重复与划分检查见 `data_audit.json`。\n')
        stream.write('\n融合精度检查：FP32 与独立 FP64 计算在全部 14 项融合中得到相同的 Top-1 和正确参考图排名。B11 的最大原始分数差约 5.3×10⁻⁵，来自低方差分数归一化时的均值舍入；去除查询内常量偏移后不足 10⁻⁶。该数值差异未改变实验结果，详见 `fusion_precision_audit.json`。\n')
    print(f'Exported {len(rows)} pipeline costs and {len(comparisons)} paired comparisons')


if __name__=='__main__':main()
