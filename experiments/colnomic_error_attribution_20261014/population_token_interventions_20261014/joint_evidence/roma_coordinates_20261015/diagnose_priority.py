"""Per-case contribution ledger for original54errors; no oracle model selection."""
from pathlib import Path
import sys,json
import numpy as np
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;OUT=D/'roma_coordinates';sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite
plan=read(OUT/'PLAN.json');parts={x['query']:x for x in read(D/'MECHANISM54.json')};provenance={x['query']:x['kind'] for x in read(OUT/'QUERY_PROVENANCE.json')}
cohort={x['query']:x for x in read(P/'cohort.json')}
(OUT/'token_deltas54').mkdir(exist_ok=True);rows=[];ledger=[]
for q in plan['queries']:
    if not q['target54']:continue
    qid=q['query'];pair=q['diagnostic_pairs']['joint'];files=[OUT/'pairs'/qid/f'{pair[k]:06}.npz' for k in ['positive','negative']]
    assert all(p.with_suffix('.json').exists() for p in files),qid
    records=[read(p.with_suffix('.json')) for p in files];a,b=[np.load(p) for p in files]
    assert np.array_equal(a['q_xy'],b['q_xy']) and np.array_equal(a['subject_mask'],b['subject_mask'])
    arrays=dict(q_xy=a['q_xy'],subject_mask=a['subject_mask'],positive_maxsim_arg=a['maxsim_arg'],negative_maxsim_arg=b['maxsim_arg'],positive_reference_xy=a['r_xy'],negative_reference_xy=b['r_xy'])
    deltas={};mask=a['subject_mask']
    for key,local in [('MaxSim','maxsim'),('S','visibility_local'),('D','dense_local'),('H','homography_local')]:
        ci=[]
        for z,rec in zip([a,b],records):
            if key=='MaxSim':v=z[local].astype(float)/len(mask)
            else:v=rec['features']['M']*z['u'].astype(float)*z[local].astype(float)/max(float(z['u'].sum(dtype=float)),1e-12)
            ci.append(v)
        delta=ci[0]-ci[1];gap=records[0]['features'][key]-records[1]['features'][key]
        assert abs(delta.sum()-gap)<2e-6,(qid,key)
        arrays[key+'_positive_contribution']=ci[0];arrays[key+'_negative_contribution']=ci[1];arrays[key+'_delta']=delta
        deltas[key]=dict(total=float(delta.sum()),inside=float(delta[mask].sum()),outside=float(delta[~mask].sum()),positive_token_count=int((delta>0).sum()),negative_token_count=int((delta<0).sum()))
    np.savez_compressed(OUT/'token_deltas54'/f'{qid}.npz',**arrays)
    before=parts[qid];evidence=[];directions=[]
    outside_overturn=deltas['MaxSim']['inside']>0 and deltas['MaxSim']['total']<0
    if outside_overturn:
        evidence.append('历史图像Token中主体内正负差额为正，但主体外贡献使该固定候选对总差额变负。')
        directions.append('以主体支持门控或区域归一化削弱主体外负贡献；必须同时核查成功样本误伤。')
    image_parts=sum(v for k,v in before['parts'].items() if k in ['original_image','subject_image','new_image']);ocr_parts=before['parts']['ocr']+before['parts']['code']
    if image_parts>0 and before['margin']<0:
        evidence.append('在最终固定候选对上，图像分支合计支持正确图，文字及编号分支合计使最终排序反转。')
        directions.append('优先核查OCR读数、主体归属、词与编号计分；比较自动重新识别及固定融合权重的全量误伤。')
    if image_parts<0 and ocr_parts>0 and before['margin']<0:
        evidence.append('文字及编号合计支持正确图，但图像负贡献更大。')
        directions.append('保留已有正向文字证据，优先修复图像匹配或融合标度；不把该错误简单归为OCR读错。')
    if records[0]['features']['M']>records[1]['features']['M'] and deltas['MaxSim']['total']<0:
        evidence.append('RoMa可见区域置信度支持正确图，历史ColNomic的无约束Patch MaxSim仍支持错误图。')
        directions.append('检验主体视觉支持重排；用置信度、几何单独评分及坐标打乱分开检验空间约束的增量。')
    if deltas['S']['total']<=0 and deltas['H']['total']>0:
        evidence.append('相同冻结特征和区域权重下，单一平面几何惩罚令该固定候选对由负差额转为正。')
        directions.append('优先检验几何约束匹配，须通过同参数坐标打乱、几何单独评分和完整图库对照。')
    absent={k:not any(plan['references'][str(j)]['sha256'] in set(cohort[qid]['positive_sha256']) for j in q['top'][k]) for k in ['image','joint']}
    if any(absent.values()):
        evidence.append('至少一条分支的自然Top10不含任何真值允许的参考SHA；诊断对已额外计算，但未插入重排候选。')
        directions.append('该分支应改善召回或使用统一、无真值的扩候选规则；Top10重排无法选中其集合外参考。')
    if not directions:
        evidence.append('已保存正负图的双向匹配、逐Token差额及现有各分支贡献；当前对照尚未产生该固定对的稳定反转证据。')
        directions.append('按现有差额最大的错误区域复核对应包装面、颜色或编号，验证局部视觉比较；不能据此宣称编码器完全丢失信息。')
    item=dict(query=qid,cache_source=provenance[qid],fixed_pair=pair,fixed_rank=before['rank'],current_branch_parts=before['parts'],current_margin=before['margin'],patch_contributions=deltas,outside_overturn=outside_overturn,selected_positive_outside_top10=absent,positive_features=records[0]['features'],negative_features=records[1]['features'],evidence=evidence,repair_directions=directions,causal_boundary='Score algebra and fixed-feature counterfactual evidence; not proof of a unique pixel/encoder root cause. Diagnostic positive is never inserted into inference candidates.')
    ledger.append(item);rows.append(dict(query=qid,baseline_rank=before['rank'],cache_source=provenance[qid],outside_overturn=outside_overturn,MaxSim_delta=deltas['MaxSim']['total'],subject_delta=deltas['MaxSim']['inside'],outside_delta=deltas['MaxSim']['outside'],visibility_delta=deltas['S']['total'],dense_delta=deltas['D']['total'],homography_delta=deltas['H']['total'],image_parts=image_parts,text_code_parts=ocr_parts,image_positive_outside10=absent['image'],joint_positive_outside10=absent['joint'],evidence='；'.join(evidence),repair_directions='；'.join(directions)))
assert len(ledger)==54
write(OUT/'MECHANISM54_COORDINATES.json',ledger);csvwrite(OUT/'mechanism54_coordinates.csv',rows)
print(json.dumps(dict(cases=len(rows),outside_token_overturn=sum(x['outside_overturn'] for x in rows),image_pair_positive_text_overturn=sum(x['image_parts']>0 and x['text_code_parts']<0 for x in rows),text_positive_image_overturn=sum(x['image_parts']<0 and x['text_code_parts']>0 for x in rows),selected_positive_outside_image10=sum(x['image_positive_outside10'] for x in rows),selected_positive_outside_joint10=sum(x['joint_positive_outside10'] for x in rows))),flush=True)
