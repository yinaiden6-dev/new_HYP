#!/usr/bin/env python3
"""Frozen, cache-only property compensation on seven TRAIN / nine TEST groups.

This is an antisymmetric fixed-pair ranking diagnostic, never C128 action.
No encoder, DPT, token tensor, native-M bypass, held selection or tuning.
Design is frozen separately before implementation and new phase-result reads.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

RC = Path(__file__).resolve().parents[1]
BASE = RC / 'results/rc_m_causal128_attribution_chain_v1'
OUT = BASE / 'property_compensation_fixed_pairs_v1'
PLAN = RC / 'reports/REPORT_M_CAUSAL128_PROPERTY_COMPENSATION_DESIGN_20260927.md'
EPS = 1e-12


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(binding):
    p = Path(binding['path'])
    assert bind(p) == binding, ('SOURCE_CHANGED', str(p))
    return p


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == data, ('IMMUTABLE_OUTPUT_CHANGED', str(path))
        return
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(data)
    try:
        os.link(tmp, path)
    except FileExistsError:
        assert path.read_text() == data
    finally:
        tmp.unlink(missing_ok=True)


def guard(execution=True):
    p = read(OUT / 'protocol.json')
    assert p['status'] == 'PROPERTY_FIXED_PAIR_COMPENSATION_DESIGN_FROZEN'
    assert p['ridge_lambda'] == .001 and p['training_groups'] == 7 and p['test_groups'] == 9
    for b in p['sources'].values():
        checked(b)
    if execution:
        seal = read(OUT / 'execution_seal.json')
        assert seal['protocol'] == bind(OUT / 'protocol.json')
        for b in seal['sources']:
            checked(b)
    return p


def feature(row, masses):
    ra, rb = row['raw_scores_pair']; la, lb = row['free_content_pair']; ma, mb = masses
    assert all(math.isfinite(v) for v in (ra, rb, la, lb, ma, mb))
    assert 0 < ma <= 1 and 0 < mb <= 1
    def s(a, b): return (a-b) / (abs(a)+abs(b)+EPS)
    return np.asarray([(ra-rb)/row['raw_C128_population_std'], s(ma*la, mb*lb),
                       s(ma, mb), s(la, lb)], dtype=np.float64)


def scalar_feature(row, masses):
    """Independent literal arithmetic for persisted feature/score verification."""
    vals = []
    vals.append((float(row['raw_scores_pair'][0])-float(row['raw_scores_pair'][1])) /
                float(row['raw_C128_population_std']))
    ml = [float(m)*float(l) for m, l in zip(masses, row['free_content_pair'])]
    for a, b in [ml, masses, row['free_content_pair']]:
        vals.append((float(a)-float(b))/(math.fsum([abs(float(a)), abs(float(b))])+1e-12))
    return vals


def train_arrays(p):
    manifest = read(checked(p['sources']['pair_manifest']))
    old = read(checked(p['sources']['old_result']))
    rows = {r['index']: r for r in old['rows']}
    X = {}; inputs = []
    for cell, arms in p['cells'].items():
        values = []
        for row in manifest['train']:
            native = rows[row['index']]
            assert native['query_id'] == row['query_id'] and native['component'] == row['component']
            t, w = row['target_position'], row['wrong_position']
            assert native['axis'] == row['candidate_axis']
            group_values = []
            for arm in arms:
                masses = [native['mass_by_arm'][arm][str(k)] for k in (t,w)]
                x = feature(row, masses)
                assert np.max(np.abs(x-np.asarray(scalar_feature(row,masses)))) < 1e-13
                group_values.append(x)
                inputs.append(dict(split='TRAIN', index=row['index'], component=row['component'],
                                   cell=cell, arm=arm, masses=masses, features=x.tolist()))
            values.append(group_values)
        X[cell] = np.asarray(values, np.float64)
        assert X[cell].shape == (7,4,4)
    # SAME scale for all cells: only TRAIN GLOBAL features. No mean centering.
    scales = np.sqrt(np.mean(X[p['reference_cell']]**2, axis=(0,1)))
    scales = np.where(scales == 0, 1., scales)
    assert np.isfinite(scales).all() and (scales > 0).all()
    return X, scales, inputs


def objective(theta, X, ridge):
    z = X @ theta
    probability = np.exp(-np.logaddexp(0., z))
    loss = float(np.logaddexp(0., -z).mean() + ridge/2 * np.dot(theta,theta))
    grad = -(X.T @ probability)/len(X) + ridge*theta
    hess = (X.T * (probability*(1-probability))) @ X / len(X) + ridge*np.eye(X.shape[1])
    return loss, grad, hess


def independent_certificate(theta, X, ridge):
    """Scalar-loop gradient/Hessian/objective, independent of GEMM reduction."""
    n,d = X.shape
    gradient = [ridge*float(t) for t in theta]
    hessian = [[ridge if i==j else 0. for j in range(d)] for i in range(d)]
    losses = []
    for row in X:
        z = math.fsum(float(x)*float(t) for x,t in zip(row,theta))
        loss = max(0.,-z) + math.log1p(math.exp(-abs(z)))
        pr = math.exp(-max(0.,z)-math.log1p(math.exp(-abs(z))))
        losses.append(loss)
        for i in range(d):
            gradient[i] -= float(row[i])*pr/n
            for j in range(d):
                hessian[i][j] += float(row[i])*float(row[j])*pr*(1-pr)/n
    obj = math.fsum(losses)/n + ridge/2*math.fsum(float(t)**2 for t in theta)
    eigen = np.linalg.eigvalsh(np.asarray(hessian))
    return dict(objective=obj, gradient=gradient, gradient_inf=max(map(abs,gradient)),
        hessian_min_eigenvalue=float(eigen.min()),
        strong_convex_objective_gap_upper_bound=math.fsum(g*g for g in gradient)/(2*ridge))


def solve(X, p):
    opt = p['optimizer']; ridge = p['ridge_lambda']; theta = np.zeros(X.shape[1]); trace=[]
    for iteration in range(opt['max_iterations']):
        value,g,h = objective(theta,X,ridge)
        gn=float(np.abs(g).max()); trace.append(dict(iteration=iteration,objective=value,gradient_inf=gn))
        if gn <= opt['gradient_inf_tolerance']:
            break
        step = np.linalg.solve(h,g); dec=float(np.dot(g,step)); assert dec > 0
        rate=1.
        for bt in range(opt['max_backtracking']):
            proposed=theta-rate*step; next_value=objective(proposed,X,ridge)[0]
            # Machine-rounding allowance, fixed and unrelated to held outcomes.
            allowance=8*np.finfo(np.float64).eps*max(1.,abs(value))
            if next_value <= value-opt['armijo']*rate*dec+allowance:
                theta=proposed;break
            rate*=opt['backtracking_factor']
        else:raise AssertionError('NEWTON_LINESEARCH_FAILED')
    else:raise AssertionError('NEWTON_DID_NOT_CONVERGE')
    value,g,h=objective(theta,X,ridge); cert=independent_certificate(theta,X,ridge)
    assert cert['gradient_inf'] <= opt['gradient_inf_tolerance']*1.01
    assert cert['hessian_min_eigenvalue'] >= ridge-1e-12
    assert abs(cert['objective']-value)<1e-12 and max(abs(np.asarray(cert['gradient'])-g))<1e-12
    return dict(theta=theta.tolist(),certificate=cert,iterations=len(trace)-1,trace=trace)


def scores_summary(margins, metadata):
    # Four replicates are one group, not four independent samples.
    assert margins.shape == (len(metadata),4)
    losses=np.logaddexp(0.,-margins);rows=[]
    for row,z,loss in zip(metadata,margins,losses):
        rows.append(dict(index=row['index'],component=row['component'],query_id=row['query_id'],
            margins=z.tolist(),mean_margin=float(z.mean()),minimum_margin=float(z.min()),
            group_mean_loss=float(loss.mean()),positive_replicates=int((z>0).sum()),
            tied_replicates=int((z==0).sum())))
    means=margins.mean(1)
    return dict(groups=len(metadata),replicates_per_group=4,mean_group_loss=float(losses.mean()),
        mean_group_margin=float(means.mean()),positive_mean_margin_groups=int((means>0).sum()),
        negative_mean_margin_groups=int((means<0).sum()),tied_mean_margin_groups=int((means==0).sum()),
        mean_within_group_positive_fraction=float((margins>0).mean()),rows=rows)


def prepare():
    p=guard(False)
    assert read(checked(p['sources']['old_validation']))['status']=='PROPERTY_FACTORIAL_FIXED14_18_INDEPENDENT_PASS'
    assert read(checked(p['sources']['bridge_validation']))['status']=='CAUSAL128_FROZEN_BRIDGE_INDEPENDENT_ARITHMETIC_PASS'
    manifest=read(checked(p['sources']['pair_manifest']))
    assert not {r['component'] for r in manifest['train']} & {r['component'] for r in manifest['test']}
    # Does not open new phase result or any per-query phase output.
    sources=[bind(Path(__file__)),bind(PLAN)]
    tests=self_test()
    write(OUT/'self_test.json',tests)
    write(OUT/'execution_seal.json',dict(status='PROPERTY_FIXED_PAIR_EXECUTION_SEALED',
        protocol=bind(OUT/'protocol.json'),sources=sources,self_test=bind(OUT/'self_test.json'),
        test_M_values_read=False,new_forward_calls=0))
    print('PROPERTY_FIXED_PAIR_EXECUTION_SEALED')


def fit():
    p=guard();X,scales,inputs=train_arrays(p);manifest=read(checked(p['sources']['pair_manifest']))
    write(OUT/'training_inputs.json',dict(protocol=bind(OUT/'protocol.json'),scales=scales.tolist(),rows=inputs))
    models={}
    for name in p['fit_models']:
        cell=p['reference_cell'] if name=='RAW_L_ONLY' else name
        idx=p['baseline_features'] if name=='RAW_L_ONLY' else list(range(4))
        arr=(X[cell]/scales)[:,:,idx]
        saved=OUT/'fits'/f'{name}.json'
        if saved.exists():
            prior=read(saved)
            assert prior['protocol']==bind(OUT/'protocol.json')
            assert prior['training_inputs']==bind(OUT/'training_inputs.json')
            assert prior['name']==name and prior['cell']==cell and prior['feature_indices']==idx
            c=independent_certificate(prior['theta'],arr.reshape(-1,len(idx)),p['ridge_lambda'])
            assert c['gradient_inf']<=p['optimizer']['gradient_inf_tolerance']*1.01
            assert c['hessian_min_eigenvalue']>=p['ridge_lambda']-1e-12
            assert np.array_equal(scales,np.asarray(prior['scales']))
            models[name]=bind(saved);continue
        sol=solve(arr.reshape(-1,len(idx)),p)
        margin=arr@np.asarray(sol['theta'])
        rec=dict(status='PROPERTY_FIXED_PAIR_FIT_CERTIFIED',protocol=bind(OUT/'protocol.json'),
            training_inputs=bind(OUT/'training_inputs.json'),name=name,cell=cell,feature_indices=idx,
            scales=scales.tolist(),**sol,train=scores_summary(margin,manifest['train']),
            test_M_values_read=False)
        write(OUT/'fits'/f'{name}.json',rec);models[name]=bind(OUT/'fits'/f'{name}.json')
    write(OUT/'fit_validation.json',dict(status='PROPERTY_FIXED_PAIR_ALL5_FITS_CERTIFIED',
        protocol=bind(OUT/'protocol.json'),execution=bind(OUT/'execution_seal.json'),
        fits=models,training_inputs=bind(OUT/'training_inputs.json'),groups=7,models=5,
        test_M_values_read=False,held_parameters_selected=False))
    print('PROPERTY_FIXED_PAIR_ALL5_FITS_CERTIFIED')


def contrast(values,p):
    x=np.asarray(values,np.float64);rng=np.random.default_rng(p['group_bootstrap_seed'])
    bootstrap=x[rng.integers(0,len(x),size=(p['group_bootstrap_draws'],len(x)))].mean(1)
    return dict(groups=len(x),mean=float(x.mean()),positive_groups=int((x>0).sum()),
        negative_groups=int((x<0).sum()),exploratory_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
        per_group=x.tolist())


def evaluate():
    started=time.monotonic();p=guard()
    # Every parameter and scaler is fixed BEFORE opening any held phase M.
    fv=read(OUT/'fit_validation.json');assert fv['status']=='PROPERTY_FIXED_PAIR_ALL5_FITS_CERTIFIED'
    assert fv['protocol']==bind(OUT/'protocol.json')
    models={name:read(checked(b)) for name,b in fv['fits'].items()}
    manifest=read(checked(p['sources']['pair_manifest']))
    phase_root=Path(p['sources']['phase_protocol']['path']).parent
    vpath=phase_root/'validation.json'
    if not vpath.exists():print('WAITING_FOR_PHASE_NEW9_VALIDATION');return 75
    v=read(vpath);assert v['status']=='PHASE_NEWGROUP_REPLICATION_JOIN_PASS'
    assert v['protocol']==p['sources']['phase_protocol']
    result=read(checked(v['result']));assert result['target_present_groups']==9
    held={r['index']:r for r in result['rows']};assert set(held)==set(p['test_indices'])
    X={};inputs=[];max_feature_error=0.
    for cell,arms in p['cells'].items():
        samples=[]
        for row in manifest['test']:
            h=held[row['index']];assert h['query_id']==row['query_id'] and h['group']==row['component']
            vals=[]
            for arm in arms:
                masses=[h['masses']['target'][arm],h['masses']['fixed_wrong'][arm]]
                x=feature(row,masses);err=float(np.max(np.abs(x-np.asarray(scalar_feature(row,masses)))))
                max_feature_error=max(max_feature_error,err);assert err<1e-13
                vals.append(x);inputs.append(dict(index=row['index'],component=row['component'],cell=cell,
                    arm=arm,masses=masses,features=x.tolist()))
            samples.append(vals)
        X[cell]=np.asarray(samples);assert X[cell].shape==(9,4,4)
    # Recheck TRAIN stationary certificates, without any parameter update.
    trainX,scales,_=train_arrays(p);maximum_score_error=0.
    for name,m in models.items():
        idx=m['feature_indices'];arr=(trainX[m['cell']]/scales)[:,:,idx].reshape(-1,len(idx))
        c=independent_certificate(m['theta'],arr,p['ridge_lambda'])
        assert c['gradient_inf']<=1.01*p['optimizer']['gradient_inf_tolerance']
        assert np.array_equal(scales,np.asarray(m['scales']))
    evaluations={};ref=p['reference_cell']
    for cell in p['cells']:
        evaluations[cell]={}
        for method,name in [('FROZEN_GLOBAL',ref),('REFIT',cell),('RAW_L_ONLY','RAW_L_ONLY')]:
            m=models[name];idx=m['feature_indices'];features=(X[cell]/scales)[:,:,idx]
            theta=np.asarray(m['theta']);margins=features@theta
            independent=np.asarray([[math.fsum(float(a)*float(b) for a,b in zip(row,theta))
                                     for row in group] for group in features])
            err=float(np.max(np.abs(margins-independent)));maximum_score_error=max(maximum_score_error,err)
            assert err<1e-12
            evaluations[cell][method]=scores_summary(margins,manifest['test'])
    base=np.asarray([r['group_mean_loss'] for r in evaluations[ref]['FROZEN_GLOBAL']['rows']])
    contrasts={}
    for cell,val in evaluations.items():
        frozen=np.asarray([r['group_mean_loss'] for r in val['FROZEN_GLOBAL']['rows']])
        refit=np.asarray([r['group_mean_loss'] for r in val['REFIT']['rows']])
        plain=np.asarray([r['group_mean_loss'] for r in val['RAW_L_ONLY']['rows']])
        contrasts[cell]=dict(frozen_damage_loss_increase=contrast(frozen-base,p),
            refit_recovery_loss_reduction=contrast(frozen-refit,p),
            remaining_gap_loss_increase_vs_global=contrast(refit-base,p),
            benefit_over_RAW_L_loss_reduction=contrast(plain-refit,p))
    # RAW/L is invariant to M intervention; no hidden M bypass.
    raw0=evaluations[ref]['RAW_L_ONLY']
    assert all(evaluations[c]['RAW_L_ONLY']==raw0 for c in p['cells'])
    result_out=dict(status='PROPERTY_FIXED_PAIR_COMPENSATION_COMPLETE',protocol=bind(OUT/'protocol.json'),
        fit_validation=bind(OUT/'fit_validation.json'),phase_validation=bind(vpath),phase_result=v['result'],
        training_groups=7,test_groups=9,models=5,evaluations=evaluations,contrasts=contrasts,
        train={n:m['train'] for n,m in models.items()},scope=p['scope'],
        property_specific_readout_compensation_test=True,full_C128_action_test=False,
        all_information_necessity_test=False,unique_cause_claim=False,new_forward_calls=0,
        held_selection=False,threshold_selection=False,parameters_refit_on_test=False,
        no_claims=p['no_claims'],boundary=p['compensation_boundary'])
    write(OUT/'test_inputs.json',dict(protocol=bind(OUT/'protocol.json'),phase_result=v['result'],rows=inputs))
    write(OUT/'result.json',result_out)
    lines=['# 性质级补偿：旧7组训练、新9组固定配对检验','',
        '这是缓存上的4维反对称排序读出，不是原 COST1、不含 HOLD，也不是完整 C128 检索评测。',
        '所有参数及尺度在读取新9组相位结果前封存；四个轴／符号是同组重复，不增加组数。',
        'GLOBAL 是原幅度下的一致相位平移控制，不等于原生 NATIVE P。','',
        '| 性质单元 | 冻结 GLOBAL 读出损失 | 本单元重拟合损失 | RAW/L 损失 | 重拟合正均值分差组 |',
        '|---|---:|---:|---:|---:|']
    for cell,e in evaluations.items():
        lines.append(f"| {cell} | {e['FROZEN_GLOBAL']['mean_group_loss']:.8f} | {e['REFIT']['mean_group_loss']:.8f} | {e['RAW_L_ONLY']['mean_group_loss']:.8f} | {e['REFIT']['positive_mean_margin_groups']}/9 |")
    lines += ['', '| 性质单元 | 冻结损伤（损失增加） | 重拟合补偿（损失减少） | 相对 GLOBAL 的剩余损失差 |',
              '|---|---|---|---|']
    def stat(v):return f"{v['mean']:.8f} [{v['exploratory_bootstrap95'][0]:.8f}, {v['exploratory_bootstrap95'][1]:.8f}]"
    for cell,c in contrasts.items():
        lines.append('| '+cell+' | '+' | '.join(stat(c[k]) for k in ['frozen_damage_loss_increase','refit_recovery_loss_reduction','remaining_gap_loss_increase_vs_global'])+' |')
    lines += ['', '括号为按9个组重采样的探索性95%区间。区间含0不证明等价；负剩余差不等于普遍可替代。',
              '严格凸正则目标的数值最优已核对，但小样本泛化、模型族限制和计算干预分布仍构成边界。',
              '本轮补的是具体性质干预之后的有界读出补偿；不重训 RoMa 预测器，不宣称信息不可替代或唯一因果。']
    text='\n'.join(lines)+'\n';rp=OUT/'report.md'
    if rp.exists():assert rp.read_text()==text
    else:rp.write_text(text)
    assert time.monotonic()-started < p['runtime_budget_seconds']
    write(OUT/'validation.json',dict(status='PROPERTY_FIXED_PAIR_COMPENSATION_INDEPENDENT_PASS',
        protocol=bind(OUT/'protocol.json'),execution=bind(OUT/'execution_seal.json'),
        fit_validation=bind(OUT/'fit_validation.json'),phase_validation=bind(vpath),
        result=bind(OUT/'result.json'),report=bind(rp),test_inputs=bind(OUT/'test_inputs.json'),
        groups_train=7,groups_test=9,maximum_feature_error=max_feature_error,
        maximum_independent_score_error=maximum_score_error,all_fit_certificates_rechecked=True,
        RAW_L_invariant=True,forward_calls=0,full_C128_accuracy_claim=False))
    print('PROPERTY_FIXED_PAIR_COMPENSATION_INDEPENDENT_PASS');return 0


def self_test():
    row=dict(raw_scores_pair=[7.,3.],raw_C128_population_std=2.,free_content_pair=[.3,.5])
    f=feature(row,[.2,.7]);other=dict(row,raw_scores_pair=[3.,7.],free_content_pair=[.5,.3])
    assert np.array_equal(f,-feature(other,[.7,.2]))
    assert np.max(abs(f-np.asarray(scalar_feature(row,[.2,.7]))))<1e-14
    rng=np.random.default_rng(19);X=rng.normal(size=(28,4));t=rng.normal(size=4);lam=.001
    value,g,h=objective(t,X,lam);eps=1e-5;ng=[];nh=[]
    for j in range(4):
        step=np.eye(4)[j]*eps
        ng.append((objective(t+step,X,lam)[0]-objective(t-step,X,lam)[0])/(2*eps))
        nh.append((objective(t+step,X,lam)[1]-objective(t-step,X,lam)[1])/(2*eps))
    assert np.max(abs(g-np.asarray(ng)))<1e-8 and np.max(abs(h-np.asarray(nh).T))<1e-8
    p=dict(ridge_lambda=.001,optimizer=dict(max_iterations=100,gradient_inf_tolerance=1e-10,
        armijo=.0001,backtracking_factor=.5,max_backtracking=60))
    a=solve(X,p);b=solve(X,p);assert a==b
    reversed_fit=solve(-X,p)
    assert np.max(abs(np.asarray(a['theta'])+np.asarray(reversed_fit['theta'])))<1e-12
    return dict(status='PROPERTY_FIXED_PAIR_ALGEBRA_OPTIMIZER_PASS',
        antisymmetric_no_intercept=True,gradient_max_finite_difference_error=float(np.max(abs(g-np.asarray(ng)))),
        hessian_max_finite_difference_error=float(np.max(abs(h-np.asarray(nh).T))),
        independent_stationarity=a['certificate'],fit_repeat_exact=True,orientation_equivariance=True,
        actual_new9_M_read=False)


def main():
    global OUT
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','fit','evaluate','run','self-test'])
    a.add_argument('--root',type=Path,default=OUT);args=a.parse_args();OUT=args.root.resolve()
    if args.stage=='self-test':print(json.dumps(self_test(),indent=2));return 0
    if args.stage=='prepare':prepare();return 0
    if args.stage=='fit':fit();return 0
    if args.stage=='evaluate':return evaluate()
    fit();return evaluate()


if __name__=='__main__':
    raise SystemExit(main())
