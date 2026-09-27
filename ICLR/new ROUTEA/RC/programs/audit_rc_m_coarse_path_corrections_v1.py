#!/usr/bin/env python3
"""Independent full-C128 decision accounting of existing coarse J/P worlds.

No fitting, new inference, candidate change, or mutation of sealed experiments.
Reconstructs all frozen head logits and zero-threshold HOLD/SWITCH decisions.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RC=Path(__file__).resolve().parents[1]
BASE=RC/'results/rc_m_causal128_attribution_chain_v1/bridge'
OUT=RC/'results/rc_m_coarse_path_correction_audit_v1'
REPORT=RC/'reports/REPORT_M_COARSE_PATH_CORRECTION_ACCOUNTING_20260927.md'
MODELS=['NATIVE7','M_FREE','POST']
WORLDS=['ORIGINAL_HR1','A1J1P1','A1J1P0','A1J0P1','A1J0P0']


def read(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(b):assert bind(b['path'])==b;return Path(b['path'])
def write(p,d):
    text=json.dumps(d,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
def sym(a,b):return (a-b)/(abs(a)+abs(b)+1e-12)
def pick(z,w,c):
    best=max(c,key=lambda k:z[k])
    return int(best) if z[best]>0 else int(w)
def describe(indices,rows):
    return dict(count=len(indices),indices=sorted(indices),queries=[dict(index=i,query_id=rows[i]['query_id'],
        original_query_id=rows[i]['original_query_id'],component=rows[i]['component']) for i in sorted(indices)])


def metrics(rows,model,world,eligible):
    ids=set(eligible);raw={i for i in ids if rows[i]['raw_correct']}
    correct={i for i in ids if rows[i]['models'][model][world]['correct']}
    orig={i for i in ids if rows[i]['models'][model]['ORIGINAL_HR1']['correct']}
    coarse={i for i in ids if rows[i]['models'][model]['A1J1P1']['correct']}
    resc=correct-raw;brk=raw-correct;origresc=orig-raw;origbreak=raw-orig
    def desc(x):return describe(x,rows)
    return dict(total=len(ids),target_present=sum(rows[i]['target_present'] for i in ids),
        correct=len(correct),RAW_correct=len(raw),rescue_vs_RAW=desc(resc),break_vs_RAW=desc(brk),net_vs_RAW=len(resc)-len(brk),
        original_HR1_rescue_count=len(origresc),original_HR1_rescues_retained=desc(resc&origresc),
        original_HR1_rescues_lost=desc(origresc-resc),rescues_outside_original_HR1_set=desc(resc-origresc),
        original_HR1_breaks_retained=desc(origbreak&brk),original_HR1_breaks_repaired=desc(origbreak-brk),
        new_RAW_breaks_vs_original_HR1=desc(brk-origbreak),
        relative_to_HR1=dict(correct_to_wrong=desc(orig-correct),wrong_to_correct=desc(correct-orig),
            decisions_changed=sum(rows[i]['models'][model][world]['prediction_position']!=rows[i]['models'][model]['ORIGINAL_HR1']['prediction_position'] for i in ids)),
        relative_to_COARSE_NATIVE=dict(correct_to_wrong=desc(coarse-correct),wrong_to_correct=desc(correct-coarse)))


def main():
    validation=read(BASE/'validation.json');assert validation['status']=='CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS'
    result=read(checked(validation['result']));protocol=read(checked(validation['protocol']))
    inputs={r['index']:r for r in read(checked(protocol['inputs']))['rows']}
    labels={r['query_id']:r for r in read(checked(protocol['original_label_source']))['rows']}
    for source in protocol['code_sources']:checked(source)
    rrows={r['index']:r for r in result['rows']};rows={};sources=[];maximum=0.;feature_error=0.
    for binding in validation['rows']:
        q=read(checked(binding));i=q['index'];item=inputs[i];r=rrows[i];label=labels[q['query_id']]
        assert q['protocol']==validation['protocol'] and q['query_id']==r['query_id']==item['query_id']
        raw=np.asarray(item['post_row']['raw_scores'],np.float64);winner=int(item['post_row']['winner_index'])
        challengers=list(item['post_row']['challenger_positions']);ext=item['external'];ids=item['post_row']['candidate_identities']
        assert challengers==ext['challengers']==[k for k in range(128) if k!=winner]
        assert winner==ext['winner'] and winner==int(raw.argmax())
        targets=[k for k,v in enumerate(ids) if v==label['identity']]
        assert bool(targets)==r['target_present'] and len(targets)<=1
        assert (winner in targets)==r['raw_correct']
        models={m:{} for m in MODELS}
        for world in WORLDS:
            arm=q['worlds'][world];mass=np.asarray(arm['M']);content=np.asarray(arm['L'])
            assert len(mass)==len(content)==128 and np.isfinite(mass).all() and (mass>0).all()
            scaled=np.asarray(ext['native_scores'])*(mass/np.asarray(ext['original_M']))[:,None]
            sr,qr,rr=scaled.T;local=sr/np.maximum(mass,1e-12);free=np.asarray(ext['free_content'])
            computed={m:np.zeros(128) for m in MODELS}
            for j,pos in enumerate(challengers):
                d0=ext['native_X'][j][0]
                x=[d0,sym(sr[pos],sr[winner]),sym(mass[pos],mass[winner]),sym(local[pos],local[winner]),
                   sym(sr[pos]-qr[pos],sr[winner]-qr[winner]),sym(sr[pos]-rr[pos],sr[winner]-rr[winner])]
                y=[d0,sym(mass[pos]*free[pos],mass[winner]*free[winner]),sym(mass[pos],mass[winner]),
                   sym(free[pos],free[winner]),0.,0.]
                for model,features,theta in [('NATIVE7',x,ext['native_theta']),('M_FREE',y,ext['simple_theta'])]:
                    feature_error=max(feature_error,float(np.max(abs(np.asarray(features)-arm['external'][model]['features'][j]))))
                    computed[model][pos]=math.fsum(float(a*b) for a,b in zip(features,theta[:-1]))+theta[-1]
                post=[(raw[pos]-raw[winner])/max(float(raw.std()),1e-12),sym(content[pos],content[winner]),1.]
                computed['POST'][pos]=math.fsum(float(a*b) for a,b in zip(post,q['head']))
            for model in MODELS:
                old=arm['POST'] if model=='POST' else arm['external'][model]
                error=float(np.max(abs(computed[model]-np.asarray(old['logits128']))));maximum=max(maximum,error)
                assert error<2e-10
                prediction=pick(computed[model],winner,challengers)
                assert prediction==old['prediction_position']==r['arms'][world][model+'_prediction_position']
                correct=prediction in targets
                assert correct==r['arms'][world][model+'_correct']
                assert (prediction!=winner)==r['arms'][world][model+'_switched']
                models[model][world]=dict(prediction_position=prediction,prediction_physical_row=item['post_row']['candidate_ids'][prediction],
                    correct=correct,switched=prediction!=winner)
        rows[i]=dict(index=i,query_id=q['query_id'],original_query_id=label['original_query_id'],component=r['component'],fold=r['fold'],
            target_present=bool(targets),target_position=targets[0] if targets else None,raw_position=winner,raw_correct=winner in targets,models=models)
        sources.append(binding)
    assert sorted(rows)==list(range(128)) and feature_error<2e-10
    panels={'FULL128':list(rows),'TARGET_PRESENT120':[i for i,r in rows.items() if r['target_present']]}
    summary={panel:{model:{world:metrics(rows,model,world,ids) for world in WORLDS} for model in MODELS} for panel,ids in panels.items()}
    restoration={}
    comparisons={'restore_P_with_J':('A1J1P0','A1J1P1'),'restore_J_with_P':('A1J0P1','A1J1P1'),
        'restore_J_without_P':('A1J0P0','A1J1P0'),'restore_P_without_J':('A1J0P0','A1J0P1')}
    for model in MODELS:
        origresc={i for i,r in rows.items() if not r['raw_correct'] and r['models'][model]['ORIGINAL_HR1']['correct']}
        raw={i for i,r in rows.items() if r['raw_correct']};restoration[model]={}
        for name,(before,after) in comparisons.items():
            b={i for i,r in rows.items() if r['models'][model][before]['correct']};a={i for i,r in rows.items() if r['models'][model][after]['correct']}
            restoration[model][name]=dict(before=before,after=after,wrong_to_correct=describe(a-b,rows),correct_to_wrong=describe(b-a,rows),
                recovered_original_HR1_rescues=describe((a-b)&origresc,rows),
                recovered_RAW_correct=describe((a-b)&raw,rows),new_RAW_breaks=describe((b-a)&raw,rows),
                net_correct_change=len(a)-len(b))
    data=dict(status='COARSE_PATH_FULL128_CORRECTION_ACCOUNTING_INDEPENDENT_PASS',
        sources=dict(source_validation=bind(BASE/'validation.json'),source_result=bind(BASE/'result.json'),
            source_protocol=bind(BASE/'protocol.json'),source_inputs=protocol['inputs'],labels=protocol['original_label_source'],query_outputs=sources),
        reviewer_code=bind(__file__),queries=128,target_present=120,target_absent=8,groups=46,target_present_groups=45,
        RAW_correct=93,RAW_wrong_full=35,RAW_wrong_target_present=27,full_logit_vectors_recomputed=1920,
        max_independent_logit_error=maximum,max_independent_feature_error=feature_error,
        summary=summary,conditional_restorations=restoration,rows=[rows[i] for i in sorted(rows)],
        scope='Opened H593 front128 and original natural ColNomic C128, original held-fold frozen heads. Not legacy EVAL128, not all H593.',
        intervention='Coarse A/J/P input path deletion, propagated through candidate M and consistent descendants only; original local weighting shapes and content fixed. POST receives M via frozen internal adapter.',
        limitations=['Conditional pathway participation, not additive unique causal shares.',
            'ORIGINAL_HR1 anchor is separate from complete COARSE A1J1P1; do not attribute coarse replacement drift to deleting J/P.',
            'No GLOBAL/LOCAL fixed-pair count is treated as full-C128 correction accounting.',
            'Not the phase/structure/content-binding property-specific causal correction contribution.',
            'No extrapolation to the full 593-query net improvement.'],new_inference_or_training=False)
    write(OUT/'result.json',data)
    lines=['# 已有粗 J/P 通路：全128候选决策与原纠错集合独立核算','',
        '**粗通路对实际纠错的量化已经存在。尚未补齐的是更细的相位／幅度／上下文绑定性质，对全C128实际纠错的独立份额。**','',
        '面板是 H593 已打开的前128个query、46组，沿用原held折冻结头和自然ColNomic C128；不是历史旧EVAL128。RAW为93/128。',
        '其中120张target进入C128（45组），8张未进入；target-present部分RAW为93/120。可在C128内救回的RAW错误为27张，全128中的RAW错误为35张。','',
        '复算了三种模型×五个世界×128张的全部1920组128候选logits，并重新执行原0阈值HOLD/SWITCH；所有选择、正确性及切换均与封存结果一致。','',
        '| 冻结模型 | M世界 | 正确/128 | 正确/120（target在C128） | 救回RAW | 误伤RAW | 原HR1救回保留 |','|---|---|---:|---:|---:|---:|---:|']
    names={'ORIGINAL_HR1':'原HR1锚','A1J1P1':'COARSE完整A+J+P','A1J1P0':'COARSE去P，保留A+J','A1J0P1':'COARSE去J，保留A+P','A1J0P0':'COARSE仅A'}
    for model in MODELS:
        for world in WORLDS:
            m=summary['FULL128'][model][world]
            lines.append(f"| {model} | {names[world]} | {m['correct']}/128 | {m['correct']}/120 | {m['rescue_vs_RAW']['count']} | {m['break_vs_RAW']['count']} | {m['original_HR1_rescues_retained']['count']}/{m['original_HR1_rescue_count']} |")
    lines += ['', 'NATIVE7与M_FREE的原HR1锚各有10次救回、1次误伤；COARSE完整在这些正确集合上复现。POST的原HR1锚为8救0损，COARSE完整保住8次救回但新误伤索引84，必须把此漂移单列。','',
        '| 条件恢复 | NATIVE7恢复的原救回 | M_FREE恢复的原救回 | POST恢复的原救回 |','|---|---:|---:|---:|']
    for name in comparisons:
        lines.append('| '+name+' | '+' | '.join(str(restoration[m][name]['recovered_original_HR1_rescues']['count']) for m in MODELS)+' |')
    lines += ['', '“恢复”表示已有封存反事实世界间的对比，不是新增训练或另一次优化。不同条件的恢复集合重叠，不能把J/P数量相加当作唯一归因比例。','',
        '## 原救回与误伤集合的具体变化','',
        '- NATIVE7：去P丢失原救回21、116、118；原误伤84同时消失。去J或同时去J/P，原10次救回全部消失。',
        '- M_FREE：去P丢失原救回9、21、32、116，但新增救回104，故总救回仍为7；原误伤84消失。去J或同时去J/P，原10次救回全部消失。',
        '- POST：去P仅保留原救回44、55，误伤变为16；去J或同时去J/P，原8次救回全失，但另救回103、误伤16。COARSE完整的误伤84与去P/去J后的误伤16不同，不能仅看“都损1张”。','',
        '以上数字为H593执行索引。完整query_id、原query名称、各救回／误伤／恢复集合及所有来源SHA见附带JSON。','',
        '## 归因边界','',
        '这已经把粗J/P通路和实际target-free全C128决策联系起来，不能再概括为“只有M相关性、没有纠错量化”。但干预是粗通路删除后的M-only传播，尚不等于相位结构、幅度和上下文配准的唯一机制份额；原局部权重形状与内容仍固定。',
        '原HR1→COARSE替换和COARSE内删除必须分开；GLOBAL/LOCAL固定target/wrong的小面板只能说明该配对传播，不能替代本表的全候选决策核算。',
        '本表不外推到全H593的481或478正确数，也不证明J/P各自不可替代。','',
        f"独立logit最大误差：`{maximum:.17g}`；特征最大误差：`{feature_error:.17g}`。",'',
        f"数据：`{OUT.relative_to(RC)}/result.json`。",'']
    text='\n'.join(lines)
    if REPORT.exists():assert REPORT.read_text()==text
    else:REPORT.write_text(text)
    write(OUT/'validation.json',dict(status=data['status'],result=bind(OUT/'result.json'),report=bind(REPORT),
        source_validation=bind(BASE/'validation.json'),queries=128,target_present=120,vectors_recomputed=1920,
        maximum_logit_error=maximum,maximum_feature_error=feature_error,new_inference_or_training=False))
    print(json.dumps(dict(status=data['status'],queries=128,target_present=120,max_logit_error=maximum,report=str(REPORT)),ensure_ascii=False))


if __name__=='__main__':main()
