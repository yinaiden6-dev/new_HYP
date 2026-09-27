#!/usr/bin/env python3
"""Seal the attribution chain at its actual causal levels, with no new fitting.

The final join verifies declared validator/output seals, preserves every result,
then assembles continuous path effects and the property-specific four-step matrix.
It never promotes a path deletion to a phase-specific necessity experiment.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile

RC=Path(__file__).resolve().parents[1]
DEFAULT=RC/'results/rc_m_causal128_attribution_chain_v1'
STAGES={
    'cache':('paths/cache_validation.json','M_CAUSAL128_EXTRACT_ALL128_PASS'),
    'compensation':('paths/validation.json','M_CAUSAL128_COMPENSATION_ALL5_INDEPENDENT_PASS'),
    'bridge':('bridge/validation.json','CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS'),
    'restoration_bridge':('bridge/joint_pairs/validation.json','JOINT_RESTORATION_FIXED14_BRIDGE_PASS'),
    'phase_replication':('phase_replication/validation.json','PHASE_NEWGROUP_REPLICATION_JOIN_PASS'),
    'pair_structure':('pair_structure/validation.json','M_CAUSAL128_PAIR_STRUCTURE_PASS'),
}
ARMS=('A1J1P1','A1J1P0','A1J0P1','A1J0P0')
METRICS=('logM_gap','NATIVE7_action_gap','M_FREE_action_gap','POST_content_gap','POST_action_gap')
NEW_GROUP_INDICES=[78,82,93,96,100,101,104,107,108,112,114,118]


def read(path):return json.loads(Path(path).read_text())
def bind(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path),sha256=h.hexdigest())
def checked(record):
    assert isinstance(record,dict) and 'path' in record and 'sha256' in record, 'INVALID_BINDING'
    path=Path(record['path']);assert path.is_file(),('BOUND_ARTIFACT_MISSING',str(path))
    assert bind(path)['sha256']==record['sha256'],('BOUND_ARTIFACT_SHA_DRIFT',str(path))
    return path

def write(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n'
    if immutable and path.exists():
        assert path.read_text()==text,('FINAL_SUMMARY_INPUT_DRIFT',str(path));return
    temp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp');temp.write_text(text);os.replace(temp,path)

def bindings(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            yield value
        else:
            for child in value.values():yield from bindings(child)
    elif isinstance(value,list):
        for child in value:yield from bindings(child)

def check_direct_seals(value,verified):
    # The declared objects are hash-verified without recursively reopening raw
    # cache tensors; those were read by the already independent stage validators.
    for b in bindings(value):
        key=(b['path'],b['sha256'])
        if key not in verified:
            checked(b);verified.add(key)

def load_stages(root):
    missing=[str(root/file) for file,_ in STAGES.values() if not (root/file).is_file()]
    if missing:return None,missing
    values={};verified=set()
    for stage,(file,status) in STAGES.items():
        path=root/file;validation=read(path)
        assert validation['status']==status,('STAGE_STATUS',stage,validation.get('status'))
        check_direct_seals(validation,verified)
        item=dict(validation=validation,validation_binding=bind(path))
        if stage!='cache':
            result_binding=validation.get('result')
            assert result_binding is not None,('RESULT_BINDING_REQUIRED',stage)
            result=read(checked(result_binding));item.update(result=result,result_binding=result_binding)
        values[stage]=item
    return (values,sorted(verified)),[]

def quantile(values,p):
    values=sorted(values);x=(len(values)-1)*p;lo=math.floor(x);hi=math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)

def group_stats(rows):
    groups=defaultdict(list)
    for component,value in rows:
        if value is not None:
            assert math.isfinite(value),'NONFINITE_MECHANISM_METRIC'
            groups[component].append(float(value))
    values=[statistics.mean(groups[k]) for k in sorted(groups)]
    if not values:return dict(groups=0,queries=0,mean=None,exploratory_bootstrap95=None)
    rng=random.Random(20260927);n=len(values)
    samples=[sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000)]
    return dict(groups=n,queries=sum(map(len,groups.values())),mean=statistics.mean(values),
        positive_groups=sum(x>0 for x in values),negative_groups=sum(x<0 for x in values),
        exploratory_bootstrap95=[quantile(samples,.025),quantile(samples,.975)],
        uncertainty='Exploratory resampling of opened component groups; not an untouched confirmatory interval.')

def compensation_effects(result):
    assert result['models']==20 and result['queries']==128
    assert result.get('causal_level')=='DIRECT_PREDICTOR_INPUT_PATH'
    assert result.get('phase_or_amplitude_necessity_test') is False
    rows=result['rows'];assert [r['index'] for r in rows]==list(range(128))
    effects={}
    for panel,rs in [('FULL128',rows),('GROUP_DISJOINT12',[r for r in rows if r['index'] in NEW_GROUP_INDICES])]:
        effects[panel]={}
        for arm in ARMS[1:]:
            contrasts=[]
            for row in rs:
                native=row['target_vs_strongest_wrong_margin']['A1J1P1']
                deleted=row['target_vs_strongest_wrong_margin'][arm]
                assert (native is None)==(deleted is None)
                if native is not None:contrasts.append((row['component'],deleted-native))
            effects[panel][arm]=dict(baseline='A1J1P1 same-family refit',
                endpoint='OOF target action minus strongest other action, including HOLD=0',
                direction='Deleted-path head margin minus native-path head margin',**group_stats(contrasts))
    return effects

def check_lineage(values):
    c=values['cache']['validation'];r=values['compensation']['result'];b=values['bridge']['result']
    assert c['queries']==r['queries']==b['queries']==128,'PANEL128_COUNT'
    assert r['models']==20,'TWENTY_SAME_FAMILY_MODELS'
    cr,br=r['rows'],b['rows'];assert len(cr)==len(br)==128
    assert [x['index'] for x in cr]==[x['index'] for x in br]==list(range(128))
    for one,two in zip(cr,br):
        assert (one['query_id'],one['fold'],one['component'])==(two['query_id'],two['fold'],two['component']),('BRIDGE_REFIT_QUERY_BINDING',one['index'])
    oldgroups={row['component'] for row in br if row['index']<71}
    new=[row for row in br if row['component'] not in oldgroups]
    assert [row['index'] for row in new]==NEW_GROUP_INDICES and len({row['component'] for row in new})==9
    assert [row['index'] for row in br if row['mechanism_group_replication']]==NEW_GROUP_INDICES
    restored=values['restoration_bridge']['result']
    assert restored['pairs']==14 and restored['queries']==7
    assert restored.get('full_C128_predictions_computed') is False
    assert b.get('unique_causality_claim') is False
    phase=values['phase_replication']['result']
    assert phase['selected_queries']==9 and phase['accuracy_claim'] is False
    assert phase['unique_cause_claim'] is False
    assert len({row['group'] for row in phase['rows']})==len(phase['rows'])
    byindex={row['index']:row for row in br}
    for row in phase['rows']:
        assert row['index'] in NEW_GROUP_INDICES
        assert row['group']==byindex[row['index']]['component'] and row['group'] not in oldgroups
        assert row['query_id']==byindex[row['index']]['query_id']
    return dict(main_queries=128,natural_candidates=128,group_folds=5,
        same_family_models=20,locked_new_mechanism_queries=12,locked_new_mechanism_groups=9,
        restoration_fixed_pairs=14,restoration_effective_groups=7,
        phase_replication_selected_queries=phase['selected_queries'],
        phase_replication_target_present_groups=phase['target_present_groups'],
        same_query_fold_component_bridge_and_refit=True)

def four_step_matrix(values):
    return [
      dict(property='Direct correspondence-distribution input P',causal_level='PREDICTOR_ENTRY',
        step1='Measured on full128: frozen direct P deletion, with J present and absent, and same-intervention M-to-external/POST propagation.',
        step2='The full J-by-P factorial measures conditional entry effects and their interaction; this is not recovery of a uniquely identified geometric property.',
        step3='Completed: all128 original five folds, same PRODUCT5 family and COST1 recipe, including P-deleted inputs; interpret against recorded TRAIN fitting quality.',
        step4='Locked twelve queries/nine groups absent from prior F71 are reported separately for frozen propagation; data remain opened H593.',
        property_specific_removal_refit_complete=True,unique_necessity_proven=False),
      dict(property='Direct cross-image context input J',causal_level='PREDICTOR_ENTRY',
        step1='Measured on full128: frozen J deletion, P present/absent; P may still carry information produced by the native interaction.',
        step2='Conditional J-by-P effects measured; J deletion is not deletion of every cross-image computation.',
        step3='Completed: J-deleted and J/P-deleted same-family PRODUCT5 compensation with matched recipe.',
        step4='Same locked group-disjoint12 frozen-mechanism subset, not an untouched external dataset.',
        property_specific_removal_refit_complete=True,unique_necessity_proven=False),
      dict(property='Cross-patch phase consistency of P',causal_level='WITHIN_P_PROPERTY',
        step1='Historical balanced LOCAL/GLOBAL perturbations preserve patch norm/frequency power and match signed perturbation dose.',
        step2='Completed fixed14 conditional restoration: GLOBAL versus LOCAL with AMP_PERMUTE retained; propagated to the same frozen external and POST paths.',
        step3='NOT COMPLETED: deleting the entire P input and refitting PRODUCT5 does not test phase-specific deletion and compensation.',
        step4='Completed separate fixed-pair replication: minimum ordinal from each of nine groups absent from F71, eighteen same-CPU arms; exact target-present denominator retained below.',
        property_specific_removal_refit_complete=False,unique_necessity_proven=False),
      dict(property='Binding of patch amplitude/concentration to position',causal_level='WITHIN_P_PROPERTY',
        step1='Historical amplitude permutation preserves the norm distribution but changes which patch receives which amplitude.',
        step2='Completed fixed14 restoration with LOCAL damage retained; report measured direction and uncertainty rather than assuming stable recovery.',
        step3='NOT COMPLETED: whole-P entry removal does not isolate amplitude-binding removal and compensation.',
        step4='Completed amplitude-specific conditional restoration within the same eighteen-arm, nine-selected-group fixed-pair replication; not downstream full-C128 replication.',
        property_specific_removal_refit_complete=False,unique_necessity_proven=False),
    ]

def fmt(v):
    if v is None:return 'NA'
    if isinstance(v,(int,float)):return f'{v:+.6g}'
    return str(v)

def effect_text(value):
    if not value or value.get('mean') is None:return 'NA'
    ci=value.get('exploratory_bootstrap95');s=fmt(value['mean'])
    if ci is not None:s+=f' [{fmt(ci[0])}, {fmt(ci[1])}]'
    return s

def scientific_report(result):
    blocks=['# 全128归因链：因素、候选区分与外部／内部传播','',
      '本轮主问题：哪些计算输入／属性支持 M 对正确与错误候选的相对区分，这种区分如何通过原冻结外部头和 POST_REAL 内容调制传播；删除输入后，同结构读出能否补偿。',
      '', '**全128自然候选面板是主结果。正确数为次要机制终点，不以某一正确数增长代替归因。**',
      '', '三个层级分别是：直接 J/P 入口；P 内部相位与幅度属性；经 M 接口向下游传播。前两者不能相互替代。',
      '', '## 冻结传播的连续效应','',
      '下表是加入该通路的条件效应（例如 P_with_J = A1J1P1 − A1J1P0）。每个指标先在 query 上取固定 target−wrong 差，再按 component 等权汇总。方括号为探索性组重采样区间。',
      '', '| 面板／条件效应 | log M差 | 原外部NATIVE7 action差 | 简化M_FREE action差 | POST内容差 | POST action差 |',
      '|---|---|---|---|---|---|']
    bridge=result['complete_stage_results']['bridge']['summary']
    for panel in ('FULL128','GROUP_DISJOINT12'):
        for effect in ('P_with_J','P_without_J','J_with_P','J_without_P','J_P_interaction'):
            stats=bridge[panel]['conditional_path_effects']
            blocks.append('| '+panel+'/'+effect+' | '+' | '.join(effect_text(stats[m][effect]) for m in METRICS)+' |')
    blocks += ['', 'POST_REAL 接收真实 M 条件；其末端 INTERNAL3 头不直接读 M。外部路径所有依赖 M 的统计量同步更新，其余局部形状和自由内容固定。这是 **M-only中介传播**，不是重新运行整个 RoMa 与编码器的总干预。',
      '所有新消融相对于同次 COARSE A1J1P1；ORIGINAL_HR1仅为原模型重放锚点。COARSE与HR1的差异不混入 J/P删除效应。',
      '', '## 属性恢复及新机制组重复','',
      '旧14对固定 target/wrong 的联合恢复已接到同一外部／POST模型。若 RAW winner 是第三个候选，固定其原 HR1 M/L 锚点；因此只能解释两候选差值，不生成完整C128排序或准确率。',
      '', '| 条件恢复 | log M差 | NATIVE7 action差 | M_FREE action差 | POST内容差 | POST action差 |','|---|---|---|---|---|---|']
    joint=result['complete_stage_results']['restoration_bridge']['contrasts']
    for contrast in ('phase_restore_after_amplitude_damage','amplitude_restore_under_local_damage','interaction'):
        blocks.append('| '+contrast+' | '+' | '.join(effect_text(joint[m][contrast]) for m in METRICS)+' |')
    blocks += ['', '历史相位桥接使用 LOCAL−GLOBAL；本轮恢复使用 GLOBAL−LOCAL，符号已经明确记录。',
      '新组复验从此前F71以外的9组各选最小ordinal，共9张；每张仅预先固定的target／自由内容最强wrong。18臂全部在同一CPU后端计算；没有把旧救回案例9/21/32当作独立确认。',
      '', '| 新组条件恢复 | target logM效应 | 固定wrong logM效应 | target−wrong logM效应 |','|---|---|---|---|']
    phase=result['complete_stage_results']['phase_replication']
    for contrast in ('phase_restoration_after_amplitude_permutation','amplitude_restoration_under_local_phase','factorial_interaction','phase_restoration_native_amplitude'):
        stats=phase['summary']
        cells=[]
        for metric in ('target_logM','fixed_wrong_logM','fixed_target_wrong_logM_gap'):
            stat=dict(stats[metric][contrast]);stat['mean']=stat['group_equal_mean'];cells.append(effect_text(stat))
        blocks.append('| '+contrast+' | '+' | '.join(cells)+' |')
    blocks += ['',f"新组实际target-present分母为{phase['target_present_groups']}组；候选缺失不补入，也不从下一张替换。",
      '', '## 同结构补偿：直接入口层','',
      '20项训练仅比较删除 J/P入口后的 PRODUCT5补偿：原五折、相同COST1、步数、参数量与固定0阈值。它检验这套读出能否利用剩余信息，不是相位或幅度属性的删除后重训。',
      '下表是删除通路模型相对原通路重训模型的 OOF target−strongest-wrong action margin，负值表示当前配方补偿不完全；不能据此推出信息不存在。',
      '', '| 面板 | 删除输入 | 连续margin差及探索性区间 |','|---|---|---|']
    for panel,arms in result['same_family_continuous_compensation'].items():
        for arm,stats in arms.items():blocks.append(f'| {panel} | {arm} | {effect_text(stats)} |')
    blocks += ['', '每折的TRAIN拟合损失、拟合正确数及收敛轨迹仍保留在原结果中；未充分拟合时，不能将OOF失败解释成信息缺失或不可替代。',
      '', '## 四步因果矩阵','', '| 属性层级 | 干预 | 条件恢复 | 同族补偿 | 组隔离重复 |','|---|---|---|---|---|']
    for row in result['four_step_causal_matrix']:
        blocks.append('| '+row['property']+' | '+' | '.join(row[f'step{i}'] for i in range(1,5))+' |')
    blocks += ['', '## 单图可分性与配对项','',
      'pair_structure是无标签输入结构检查：检验query侧因子是否跨候选恒定，以及 log M(q,r) 是否能由单图加性项解释。若发现非零矩形循环残差，它排除了该面板上严格的单图可分表达；它本身不证明这些非可分项就是身份收益的唯一来源。',
      '完整arm统计和循环审计保存在 complete_stage_results.pair_structure，不将理论代数恒等式与测得的数值残差混为一谈。',
      '', '| 输入通路 | query侧跨候选极差最大值 | reference侧跨query极差最大值 | logM矩形循环RMS |','|---|---:|---:|---:|']
    structure=result['complete_stage_results']['pair_structure']
    for arm,stats in structure['arms'].items():
        blocks.append(f"| {arm} | {fmt(stats['query_side_candidate_range']['max_abs'])} | {fmt(stats['reference_side_query_range']['max_abs'])} | {fmt(stats['logM_cycle']['rms'])} |")
    blocks += ['',
      '', '## 次要决策结果','', '| 面板 | M来源 | POST | 原外部NATIVE7 | M_FREE |','|---|---|---:|---:|---:|']
    for panel in ('FULL128','GROUP_DISJOINT12'):
        for arm,v in bridge[panel]['worlds'].items():
            blocks.append(f"| {panel} | {arm} | {v['POST']['correct']}/{v['POST']['total']} | {v['NATIVE7']['correct']}/{v['NATIVE7']['total']} | {v['M_FREE']['correct']}/{v['M_FREE']['total']} |")
    blocks += ['', '## 可主张的范围与未闭合部分','',
      '本轮能够给出：特定计算入口／属性的条件作用、正确／错误候选相对证据的变化、这份证据经冻结外部和内部路径的传播，以及当前同结构读出的可补偿程度。',
      '**不能据此主张 M 是唯一原因、身份充分统计量、RoMa 普遍不可替代，或全部空间信息都由P入口承载。** A/J中仍可能包含空间信息；固定内部模型只读取M也是设计事实，不是充分性证明。',
      '相位／幅度属性的删除后补偿仍未由本链完成，必须与本轮已经完成的J/P入口补偿分开。',
      '数据是已打开H593的前128。12张／9组是相对于此前F71的机制组隔离复验，不是新外部数据，也没有重新定义旧593、32或128基准的最优成绩。',
      '', '独立阶段验收与其直接输出封存SHA已重新检查；没有再次运行编码器、RoMa或训练。所有完整summary、逐query结果及来源封存在同目录result.json。','']
    return '\n'.join(blocks)


def summarize(root):
    out=root/'final_summary';out.mkdir(parents=True,exist_ok=True)
    with (out/'summary.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        loaded,missing=load_stages(root)
        if missing:
            write(out/'waiting.json',dict(status='M_CAUSAL128_ATTRIBUTION_WAITING',missing=missing));return 75
        values,verified=loaded
        counts=check_lineage(values)
        results={stage:item['result'] for stage,item in values.items() if stage!='cache'}
        value=dict(status='M_CAUSAL128_ATTRIBUTION_SUMMARY_PASS',counts=counts,
            purpose='Computational factor/property -> target-vs-fixed-wrong M contrast -> frozen external/POST propagation -> same-family compensation -> locked mechanism-group replication.',
            primary_endpoint='Continuous, component-equal conditional effects and their propagation',
            accuracy_endpoint='Secondary; same natural C128 and explicit HOLD=0',
            same_family_continuous_compensation=compensation_effects(results['compensation']),
            four_step_causal_matrix=four_step_matrix(values),complete_stage_results=results,
            source_validations={name:record['validation_binding'] for name,record in values.items()},
            source_validation_payloads={name:record['validation'] for name,record in values.items()},
            source_results={name:record['result_binding'] for name,record in values.items() if name!='cache'},
            output_seals_verified=len(verified),verification_scope='Every artifact directly declared by the six independent stage validators; original raw tensors and internal stage implementations are not rerun by this final merger.',
            summary_code=bind(Path(__file__)),training=False,new_model_forwards=0,
            unique_causality_claim=False,identity_sufficient_statistic_claim=False,external_confirmation_claim=False,
            property_specific_phase_or_amplitude_refit_complete=False,
            data_boundary='Opened H593 front128; group-disjoint12 means absent from prior F71 components, not untouched or external.',
            remaining_gaps=['Phase-specific deletion followed by same-family compensation has not been performed.',
                'Amplitude-binding-specific deletion followed by compensation has not been performed.',
                'Negative compensation results constrain the tested readout/optimization only.',
                'A unique global causal decomposition and universal information sufficiency are not established.'])
        report=scientific_report(value)
        write(out/'result.json',value,immutable=True)
        reportpath=out/'report.md'
        if reportpath.exists():assert reportpath.read_text()==report,'FINAL_REPORT_DRIFT'
        else:reportpath.write_text(report)
        verification=[dict(path=p,sha256=s) for p,s in verified]
        write(out/'verified_output_seals.json',verification,immutable=True)
        write(out/'validation.json',dict(status=value['status'],result=bind(out/'result.json'),report=bind(reportpath),
            verified_output_seals=bind(out/'verified_output_seals.json'),source_validations=value['source_validations'],
            counts=counts,new_model_forwards=0,training=False,scientific_claims_preserve_property_level=True),immutable=True)
        (out/'waiting.json').unlink(missing_ok=True)
        print(json.dumps(dict(status=value['status'],counts=counts,verified_seals=len(verified))),flush=True)
        return 0


def self_test():
    with tempfile.TemporaryDirectory() as name:
        root=Path(name)
        assert summarize(root)==75 and read(root/'final_summary/waiting.json')['missing']
        value=dict(rows=[dict(index=i,component='g'+str(i%5),target_vs_strongest_wrong_margin={a:(float(i%3)+j*.1) for j,a in enumerate(ARMS)}) for i in range(128)],
            models=20,queries=128,causal_level='DIRECT_PREDICTOR_INPUT_PATH',phase_or_amplitude_necessity_test=False)
        effects=compensation_effects(value)
        assert abs(effects['FULL128']['A1J1P0']['mean']-.1)<1e-12
        value['phase_or_amplitude_necessity_test']=True
        try:compensation_effects(value)
        except AssertionError:pass
        else:raise AssertionError('PROPERTY_LEVEL_CONFUSION_NOT_REJECTED')
        path=root/'sealed.json';write(path,{'x':1},immutable=True);b=bind(path);checked(b)
        try:write(path,{'x':2},immutable=True)
        except AssertionError:pass
        else:raise AssertionError('IMMUTABLE_OUTPUT_DRIFT_NOT_REJECTED')
        path.write_text('tampered')
        try:checked(b)
        except AssertionError:pass
        else:raise AssertionError('INPUT_SHA_DRIFT_NOT_REJECTED')
    matrix=four_step_matrix({})
    assert all(not row['unique_necessity_proven'] for row in matrix)
    assert [r['property_specific_removal_refit_complete'] for r in matrix]==[True,True,False,False]
    print(json.dumps(dict(status='M_CAUSAL128_SUMMARY_SELF_TEST_PASS',missing_prerequisite_exit75=True,
        property_level_confusion_rejected=True,group_equal_continuous_effect=True,
        immutable_drift_rejected=True,source_tamper_rejected=True)),flush=True)
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:return self_test()
    try:return summarize(args.root)
    except (AssertionError,ValueError,KeyError,FileNotFoundError,TypeError) as exc:
        write(args.root/'final_summary/blocked.json',dict(status='M_CAUSAL128_ATTRIBUTION_BLOCKED',
            error_type=type(exc).__name__,error=str(exc)))
        print(json.dumps(dict(status='M_CAUSAL128_ATTRIBUTION_BLOCKED',error=str(exc))),flush=True)
        return 2

if __name__=='__main__':sys.exit(main())
