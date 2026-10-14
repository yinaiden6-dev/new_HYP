"""Descriptive, identity-balanced summaries of preselected intervention pairs.

No edited identity accuracy is defined and no candidate membership is changed.
The independently checked new and historical matrices remain authoritative.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.colnomic_mechanism.common import ROOT,read,write,rows,csvout,sha,workspace_path

FOLDER=ROOT/'results/colnomic_full_audit/20261013_resume/fresh_reference_interventions'
ARMS=['base','sham','numeric_mask','blank_control','both_masks']


def cluster_summary(values,groups,seed):
    unique=sorted(set(groups));means=np.array([np.mean([v for v,g in zip(values,groups) if g==u]) for u in unique])
    rng=np.random.default_rng(seed);boots=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
    return float(means.mean()),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))


def tables():
    source=FOLDER/'analysis';pairs=rows(source/'pair_effects.csv');comparisons=rows(source/'fresh_vs_frozen.csv')
    independent=read(FOLDER/'independent/validation.json')
    assert independent['status']=='passed' and independent['full_pointwise_matrices']==290 and independent['pair_token_rows']==127150
    assert independent['analysis_validation_sha256']==sha(source/'validation.json')
    summaries=[]
    for bank in ['fresh','frozen']:
        for arm in ARMS:
            selected=[r for r in pairs if r['bank']==bank and r['arm']==arm]
            groups=[r['package_id'] for r in selected];effects=[float(r['edit_minus_sham_gap']) for r in selected]
            mean,lo,hi=cluster_summary(effects,groups,2026101405)
            gap=[float(r['gap']) for r in selected]
            macro_positive=np.mean([np.mean([float(g)>0 for g,p in zip(gap,groups) if p==pid]) for pid in set(groups)])
            interaction=[float(r['interaction_gap']) for r in selected];intermean,interlo,interhi=cluster_summary(interaction,groups,2026101406)
            summaries.append(dict(bank=bank,arm=arm,preselected_pairs=len(selected),independent_identities=len(set(groups)),
                positive_gap_pairs=sum(g>0 for g in gap),mean_gap=float(np.mean(gap)),identity_macro_positive_gap_fraction=float(macro_positive),
                mean_edit_minus_sham_gap=float(np.mean(effects)),identity_macro_edit_minus_sham_gap=mean,
                identity_bootstrap_edit_effect_lower=lo,identity_bootstrap_edit_effect_upper=hi,
                identity_macro_interaction_gap=intermean,identity_bootstrap_interaction_lower=interlo,identity_bootstrap_interaction_upper=interhi,
                fraction_with_edit_minus_sham_positive=float(np.mean(np.array(effects)>0)),
                denominator_note='17 fixed candidate pairs from12 independent identities, not17 independent images; positive margin sign is not edited identity accuracy'))
    changes=[]
    for arm in ARMS:
        selected=[r for r in comparisons if r['arm']==arm]
        difference=[float(r['fresh_minus_frozen_edit_effect']) for r in selected]
        qids=[r['photo_id'] for r in selected];macro,lo,hi=cluster_summary(difference,qids,2026101407)
        changes.append(dict(arm=arm,preselected_pairs=len(selected),query_identities=len(set(qids)),
            frozen_positive_to_fresh_nonpositive=sum(float(r['frozen_gap'])>0 and float(r['fresh_gap'])<=0 for r in selected),
            frozen_nonpositive_to_fresh_positive=sum(float(r['frozen_gap'])<=0 and float(r['fresh_gap'])>0 for r in selected),
            mean_fresh_minus_frozen_gap=float(np.mean([float(r['fresh_minus_frozen_gap']) for r in selected])),
            mean_fresh_minus_frozen_edit_effect=float(np.mean(difference)),identity_macro_fresh_minus_frozen_edit_effect=macro,
            identity_bootstrap_difference_lower=lo,identity_bootstrap_difference_upper=hi))
    inputs=[source/'validation.json',source/'pair_effects.csv',source/'fresh_vs_frozen.csv',source/'reference_cache_parity.csv',
            FOLDER/'candidate_selection.json',FOLDER/'encoding_manifest.json',FOLDER/'independent/validation.json',FOLDER/'interpretation_corrections.json',
            FOLDER.parent/'detail_interventions/visual_numeric_verification.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    return summaries,changes,hashes


def summarize(output,verify_existing=None):
    assert not output.exists() or not any(output.iterdir()),'Use a fresh empty summary output'
    output.mkdir(parents=True,exist_ok=True)
    summaries,changes,hashes=tables()
    if verify_existing is not None:
        old=read(verify_existing/'validation.json');assert old['source_SHA256']==hashes
        for name,expected in [('pair_effect_summary.csv',summaries),('fresh_frozen_effect_summary.csv',changes)]:
            observed=rows(verify_existing/name);assert len(observed)==len(expected)
            for a,b in zip(observed,expected):
                assert set(a)==set(b)
                for key,value in b.items():
                    if isinstance(value,(float,int)):assert abs(float(a[key])-value)<1e-12,(name,key)
                    else:assert a[key]==value,(name,key)
        write(output/'validation.json',dict(status='passed',scope='Complete recomputation of all15 descriptive rows and their identity bootstrap intervals',source_SHA256=hashes,
            verified_summary_sha256=sha(verify_existing/'validation.json'),code_SHA256=sha(Path(__file__)),full_gallery_accuracy_computed=False,edited_identity_accuracy_computed=False))
        return
    csvout(output/'pair_effect_summary.csv',summaries);csvout(output/'fresh_frozen_effect_summary.csv',changes)
    validation=dict(status='passed',scope='Descriptive aggregates of independently pointwise-verified preselected intervention responses; not new causal classification',
        pair_summary_rows=len(summaries),fresh_frozen_summary_rows=len(changes),identities=12,pairs=17,bootstrap_clusters='package identity',bootstrap_resamples=2000,
        source_SHA256=hashes,code_SHA256=sha(Path(__file__)),full_gallery_accuracy_computed=False,edited_identity_accuracy_computed=False)
    write(output/'validation.json',validation)
    report=['# 新旧运行时：预选干预候选响应\n',
        '12个独立身份，17组预先固定正负候选，28张图库原图重新编码；60个Query编辑臂复用，另重新编码12张Query原图。全部12张原图Token与历史Query缓存逐位一致，28张新Reference均未逐位复现旧缓存。\n',
        '完整独立重放覆盖145个新Reference矩阵和145个历史矩阵、127150个逐Token记录、170个候选对×编辑臂记录、850个ROI分解与170个ROI交互分解。\n',
        '下表的差额为正候选mean MaxSim减错误候选mean MaxSim，编辑效应为edit减sham；并非编辑后身份准确率。每列分母是同一批17组候选对，来自12个身份。\n',
        '|臂|新Reference正差额|旧Reference正差额|新Reference平均编辑效应|旧Reference平均编辑效应|\n|---|---:|---:|---:|---:|']
    for arm in ARMS:
        a=next(r for r in summaries if r['bank']=='fresh' and r['arm']==arm);b=next(r for r in summaries if r['bank']=='frozen' and r['arm']==arm)
        report.append(f'|{arm}|{a["positive_gap_pairs"]}/17|{b["positive_gap_pairs"]}/17|{a["mean_edit_minus_sham_gap"]:.8f}|{b["mean_edit_minus_sham_gap"]:.8f}|')
    report+=['\n[按身份汇总及2000次身份bootstrap](pair_effect_summary.csv)保留小样本区间；[新旧参考差异](fresh_frozen_effect_summary.csv)包括方向变化。不能以平均响应直接确定某张原始错误的唯一原因。\n',
        '数字框来自先验OCR提议，已有独立视觉审核区分规格与公共说明；空白对照存在位置、颜色和纹理差异。旧worker源码快照缺失，新旧worker SHA不同，不能声称worker源码相同；数值encoder源码、版本、模型配置、torchvision处理后端一致，以及12张Query原图逐位复现，是复用Query编辑臂的兼容性证据。\n',
        '同Token数量或同有效序号轴不证明同物理patch格点；旧图库处理器无法追溯，新旧argmax变化仅作序号诊断。同一bank内edit与sham的匹配切换、切换/未切换贡献、ROI抵消与交互均已独立核查。\n',
        '全部候选来自旧145矩阵的冻结并集，未按新运行时响应选样。新旧各自使用完整选定候选，不是完整5405图库；不能报告新运行时Recall，亦不沿用编辑后的身份真值。原数字遮挡分支保留不变。\n']
    (output/'REPORT.md').write_text('\n'.join(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=FOLDER/'interpretation')
    parser.add_argument('--verify-existing',type=Path)
    args=parser.parse_args();summarize(args.output.resolve(),args.verify_existing.resolve() if args.verify_existing else None)
    print('Fresh-reference descriptive summaries passed',flush=True)


if __name__=='__main__':main()
