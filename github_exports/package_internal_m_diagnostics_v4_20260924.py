#!/usr/bin/env python3
"""Snapshot text evidence for internal M; never runs or changes experiments."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil, subprocess, zipfile

ROOT = Path('/hkfs/work/workspace/scratch/ap7811-benchmark')
RC = ROOT / 'ICLR/new ROUTEA/RC'
DEST = ROOT / 'github_exports/new_HYP_20260919'
NAME = 'internal_m_diagnostics_v4_snapshot_20260924'
INDEX = DEST / 'backup' / NAME
LOCAL = RC / 'reports' / NAME

def sha(b): return hashlib.sha256(b).hexdigest()
def json_bytes(d): return (json.dumps(d, ensure_ascii=False, indent=2)+'\n').encode()
def main():
    start = datetime.now(timezone.utc).isoformat()
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=DEST,text=True).strip()
    payload, origin = {}, {}
    previous = DEST / 'backup/internal_m_v3_final_closure_20260924/files.json'
    sealed = json.loads(previous.read_text())
    for item in sealed['files']:
        rel = item['path']; blob = (DEST/rel).read_bytes()
        assert sha(blob) == item['sha256'], ('previous archive changed',rel)
        payload[rel] = blob; origin[rel] = 'verified_existing_V3_archive'
    selected = [
        RC/'reports/REPORT_COLNOMIC_INTERNAL_M_V3_CAUSE_AND_V4_REPAIR_20260924.md',
        RC/'programs/audit_rc_internal_m_v3_condition_scale.py',
        RC/'programs/audit_rc_internal_m_v3_readout_capacity.py',
        RC/'programs/run_rc_internal_m_condition_scale_v4.py',
        RC/'programs/submit_rc_internal_m_condition_scale_v4.py',
        RC/'programs/run_rc_prellm_m_pilot_v1.py',
        RC/'tests/test_internal_m_condition_scale_v4.py',
        RC/'slurm/rc_internal_m_condition_scale_v4.sbatch',
        RC/'registry/rc_internal_m_condition_scale_v4_authority_20260924.json',
        RC/'plan/RC_COLNOMIC_INTERNAL_M_CONDITION_SCALE_V4_EXECUTION_20260924.md',
        Path(__file__).resolve(),
    ]
    excluded=[]
    for dirname in ('rc_internal_m_v3_fit_diagnostics','rc_internal_m_condition_scale_v4'):
        for p in sorted((RC/'results'/dirname).rglob('*')):
            if not p.is_file(): continue
            if p.suffix in {'.json','.md','.csv'} and not any(s in p.name for s in ('.partial','.tmp','.lock')):
                selected.append(p)
            else:
                excluded.append({'path':str(p.relative_to(ROOT)), 'reason':'binary weights/traces, runtime or partial file excluded'})
    for p in sorted(set(selected)):
        assert p.is_file() and not p.is_symlink(), p
        rel=str(p.relative_to(ROOT)); blob=p.read_bytes()
        assert b'\0' not in blob,p
        if p.suffix=='.json': json.loads(blob)
        assert not re.search(rb'(?:ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',blob),('secret-like material',rel)
        payload[rel]=blob;origin[rel]='workspace_snapshot'
    rc_rel='ICLR/new ROUTEA/RC/'
    v4_rel=rc_rel+'results/rc_internal_m_condition_scale_v4/'
    progress={}
    for arm in ('PRE_REAL','PRE_SHUFFLED'):
        names=[p for p in payload if p.startswith(v4_rel+arm+'/steps/')]
        steps=[int(Path(p).stem) for p in names]
        progress[arm]={'captured_steps':len(steps),'last_captured_update':max(steps,default=0),'target_updates':128}
    endpoints=[]
    for rel,blob in payload.items():
        if rel.startswith(v4_rel) and '/endpoints/' in rel and rel.endswith('/validation.json'):
            d=json.loads(blob)
            for record in d['predictions']:
                name=str(Path(record['path']).relative_to(ROOT))
                assert name in payload and sha(payload[name])==record['sha256'],('endpoint binding',name)
            endpoints.append({'arm':d['arm'],'step':d['step'],'status':d['status'],'prediction_hashes_verified':len(d['predictions'])})
    final_present=v4_rel+'result.json' in payload
    files=[{'path':p,'bytes':len(b),'sha256':sha(b),'origin':origin[p]} for p,b in sorted(payload.items())]
    manifest={'capture_started_utc':start,'capture_finished_utc':datetime.now(timezone.utc).isoformat(),
              'base_git_commit':head,'source_root':str(ROOT),'files':files,'excluded':excluded,
              'scope':'Completed V3 text evidence, post-hoc TRAIN16 diagnostics, V4 code and in-progress text snapshot',
              'v4_progress':progress,'v4_endpoint_validation':endpoints,'v4_final_result_present':final_present,
              'restore_note':'Binary checkpoints, original images, pretrained models and token caches are external dependencies; this archive alone cannot resume training.',
              'scientific_boundary':'V3 and subsequent diagnostics are TRAIN16 fitting evidence, not held-out generalization. V4 is not represented as completed.'}
    assert not final_present, 'V4 completed during capture: update scope before publication'
    readme=f'''# 内部 M：最终结果、失败定位与 V4 阶段快照

采集窗口：{manifest['capture_started_utc']} 至 {manifest['capture_finished_utc']}。

[下载 ZIP](new_HYP_internal_m_diagnostics_v4_20260924.zip) · [逐文件 SHA256 清单](files.json)

本包承接 Git 提交 `{head}` 已发布的 V3 最终结果，补齐失败定位与 V4 的源码、协议、参数配置、已完成阶段的逐候选结果及验收。

- **V3 已完成**：同一 TRAIN16 / ColNomic 自然 C128，RAW 8/16；内部真实 M 与恒定 M 均 8/16；外部加性和乘积头均 12/16。优化预算不同，不能作为同预算优劣结论。
- **诊断已完成**：固定内容后统一重拟合，真实 M 9/16、恒定训练臂 10/16。读出优化有影响，但未建立真实 M 的特有收益。单张已保存 trace 显示 M 条件支路幅度偏弱，这是待检验解释。
- **V4 尚未完成**：只放大标准化 M 输入尺度，保留真实/错绑/恒定对照；首步工程验收已通过。快照记录 PRE_REAL 至第 {progress['PRE_REAL']['last_captured_update']}/128 次更新，PRE_SHUFFLED 至第 {progress['PRE_SHUFFLED']['last_captured_update']}/128 次更新。已完成的第16步终点已收录；没有最终 V4 或新留出结论。

主要入口（仓库浏览）：

- [V3 最终结果](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_FINAL_20260924.md)
- [失败定位与修复依据](../../ICLR/new%20ROUTEA/RC/reports/REPORT_COLNOMIC_INTERNAL_M_V3_CAUSE_AND_V4_REPAIR_20260924.md)
- [V4 冻结协议](../../ICLR/new%20ROUTEA/RC/plan/RC_COLNOMIC_INTERNAL_M_CONDITION_SCALE_V4_EXECUTION_20260924.md)
- [V4 阶段数据](../../ICLR/new%20ROUTEA/RC/results/rc_internal_m_condition_scale_v4/)

ZIP 内按 workspace 相对路径保留结构；上述相对链接供 GitHub 浏览，解压后请从根目录的 ICLR/new ROUTEA/RC 进入。原报告保留其记录时的状态；本 README 与 files.json 给出本次快照边界。

保存小头数值参数、完整候选分数和文本收据；排除模型/适配器二进制权重、token 缓存、原图、err/out/log、锁及 partial 文件。二进制依赖和原始封存 SHA 仍由源码/收据引用，因此本包并非可直接恢复训练的完整运行环境。历史 Qwen 乘积项和其他实验保留于仓库已有归档，本包专门收录内部 M 分支。
'''.encode()
    INDEX.mkdir(parents=True,exist_ok=True); LOCAL.mkdir(parents=True,exist_ok=True)
    changed=[]
    for rel,blob in payload.items():
        p=DEST/rel
        if not p.exists() or p.read_bytes()!=blob:
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob);changed.append(rel)
    metadata={'README.md':readme,'files.json':json_bytes(manifest)}
    for name,blob in metadata.items(): (INDEX/name).write_bytes(blob)
    archive=INDEX/'new_HYP_internal_m_diagnostics_v4_20260924.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,blob in sorted(payload.items()): z.writestr(rel,blob)
        for rel,blob in metadata.items(): z.writestr(rel,blob)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(payload)|set(metadata)
        for rel,blob in {**payload,**metadata}.items(): assert sha(z.read(rel))==sha(blob),rel
    receipt={'status':'ARCHIVE_HASH_AND_CRC_PASS','archive':archive.name,'sha256':sha(archive.read_bytes()),
             'bytes':archive.stat().st_size,'payload_files':len(files),'zip_entries':len(files)+len(metadata),
             'v4_progress':progress,'v4_endpoint_validation':endpoints}
    (INDEX/'validation.json').write_bytes(json_bytes(receipt))
    for p in INDEX.iterdir():
        if p.is_file(): shutil.copyfile(p,LOCAL/p.name)
    readme_path=DEST/'README.md';text=readme_path.read_text()
    line=f'- **[2026-09-24 内部M失败定位与V4阶段快照、ZIP下载](backup/{NAME}/README.md)**：补齐V3读出/条件尺度诊断，V4仍在训练，阶段数据不作为最终结论。\n\n'
    if line not in text: readme_path.write_text(text.replace('## 从这里阅读\n\n','## 从这里阅读\n\n'+line,1));changed.append('README.md')
    changed += [str(p.relative_to(DEST)) for p in INDEX.iterdir() if p.is_file()]
    Path('/tmp/new_hyp_internal_m_archive_paths_20260924.json').write_bytes(json_bytes(sorted(set(changed))))
    print(json.dumps({'validation':receipt,'changed_files':len(set(changed)),'local_download':str(LOCAL/archive.name)},ensure_ascii=False))

if __name__=='__main__': main()
