"""Render the complete B03 pure-image token audit and verify every contrast."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.report_colnomic_tokens import NAMES, focal_lines, fmt, read, table


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def diagnosis(s,ambiguous=False):
    if ambiguous:return '同一候选对的身份歧义待复核'
    if s['image_gap']==0:return '图像完全同分，图库顺序决定赢家'
    if s['subject_only_gap'] is None:return '主体框内没有有效网格中心'
    if s['subject_only_gap']>0:return '主体筛选可翻转这两个候选'
    return '主体匹配仍偏错误候选'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'results/token_mechanism_analysis/B03')
    args=p.parse_args();root,out=args.root,args.output
    data=read(out/'summary.json');assert data['configuration']=='B03'
    inventory=list(csv.DictReader((out/'B03_error_inventory.csv').open()));inv={r['photo_id']:r for r in inventory}
    primary={r['photo_id']:r for r in [json.loads(l) for l in (root/'results/colnomic7b/image/predictions.jsonl').read_text().splitlines()]}
    expected={id for id,r in primary.items() if not r['correct']}
    assert {t['photo_id'] for t in data['targets']}==set(inv)==expected
    query_ocr=read(root/'cache/ocr/photos.json');boxes=read(root/'dataset/annotations/subject_boxes.json')['records']
    contrasts={(s['target'],s['photo_id']):s for s in data['cases']}
    diagnoses=[];links=[];paired=[];target_data=[];token_count=0;focal_signal=[]
    for t in data['targets']:
        id=t['photo_id'];folder=out/id/id;s=contrasts[id,id];a=inv[id];target_data.append(s)
        assert s['negative']==primary[id]['selected_reference_id'] and s['image_gap']<=0
        rows=list(csv.DictReader((folder/'tokens.csv').open()));token_count+=len(rows)
        assert len(rows)==s['valid_tokens']
        assert abs(sum(float(r['contribution_to_mean_gap']) for r in rows)-s['image_gap'])<2e-6
        assert abs(s['positive_evidence']+s['negative_evidence']-s['image_gap'])<2e-6
        for name in ('correspondence.png','contribution.png','regions.csv','ocr_lines.csv','summary.json'):
            assert (folder/name).is_file(),(id,name)
        focal=focal_lines(folder,s,rows,query_ocr[id],boxes[id])
        for f in focal:focal_signal.append(dict(photo_id=id,**f))
        group=diagnosis(s,t['identity_ambiguity'])
        diagnoses.append(dict(photo_id=id,package_id=t['package_id'],split=t['split'],diagnostic_group=group,
            positive=s['positive'],image_top1_negative=s['negative'],image_rank=s['image_rank'],F03_comparison_rank=s['fusion_rank'],
            image_gap=s['image_gap'],subject_only_gap=s['subject_only_gap'],subject_tokens=s['subject_tokens'],
            positive_evidence=s['positive_evidence'],negative_evidence=s['negative_evidence'],control=t['control'],
            prior_other_candidate_ambiguity=t['prior_other_candidate_ambiguity']))
        identity=('本查询与相同正负图像已有身份歧义标记，不能直接作为模型失败证据。' if t['identity_ambiguity'] else
                  '此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。' if t['prior_other_candidate_ambiguity'] else
                  '按当前评测真值分析，未新增身份人工复核。')
        worst=min(s['groups'],key=lambda g:g['contribution_to_mean_gap'])
        parts=[f'# B03 · {id}\n',identity+'\n',
            f'包装 `{t["package_id"]}`，划分 `{t["split"]}`；用户主题：{NAMES[t["case_type"]]}。'
            f'正确参考 `{s["positive"]}`，**纯图像实际 Top-1 错误参考** `{s["negative"]}`。\n',
            '## 评分与检索\n',table(['检查','结果'],[['正确参考排名',s['image_rank']],
                ['正确参考 MaxSim',fmt(s['image_positive'])],['错误参考 MaxSim',fmt(s['image_negative'])],
                ['正确减错误的分差',fmt(s['image_gap'])],['F03 的正确参考排名（辅助对照）',s['fusion_rank']],['基于测量的诊断',group]]),
            '\nB03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。\n',
            '## Token 分差守恒\n',f'有效 token {s["valid_tokens"]} 个，主体框内视觉 token {s["subject_tokens"]} 个。'
            f'正向贡献 {fmt(s["positive_evidence"])}，负向贡献 {fmt(s["negative_evidence"])}，相加等于原图像分差。'
            f'与原评分缓存的最大误差 {s["replay_max_error"]:.2g}。\n',
            table(['区域','Token 数','对完整均值分差的贡献','区域内均值'],[
                [g['region'],g['n'],fmt(g['contribution_to_mean_gap']),fmt(g['within_region_mean'])] for g in s['groups']]),
            f'\n最主要负差额来自 `{worst["region"]}`，贡献 {fmt(worst["contribution_to_mean_gap"])}。'
            f'只保留主体 query token 后，本对分差为 {fmt(s["subject_only_gap"])}。'
            '这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。\n',
            '## 对应与独立几何\n',table(['候选','SIFT 内点','主体覆盖率','对齐通过阈值','残差中位数（参考网格单元）','超过3单元比例'],[
                [name,g['inliers'],f'{g["coverage"]:.1%}',g['accepted'],fmt(s[key+'_geometry_distance_median_patches']),
                 '不可评估' if s[key+'_geometry_over_3_patches_fraction'] is None else f'{s[key+"_geometry_over_3_patches_fraction"]:.1%}']
                for name,key,g in [('正确','positive',s['geometry'][0]),('错误','negative',s['geometry'][1])]]),
            f'\n主体中 {s["foreground_OCR_tokens"]} 个 token 与 OCR 行相交，{s["multiple_OCR_line_subject_tokens"]} 个与多行相交。'
            '视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。\n']
        for title,sign in [('最不利的8个 token',1),('最有利的8个 token',-1)]:
            parts.extend(['\n## '+title+'\n',table(['缓存索引','区域','Δi','Query OCR','正确参考 OCR','错误参考 OCR'],[
                [r['q_token_index'],r['region'],fmt(float(r['delta'])),r['q_ocr_text'],r['pos_ocr_text'],r['neg_ocr_text']]
                for r in sorted(rows,key=lambda r:sign*float(r['delta']))[:8]])])
        if focal:
            parts.extend(['\n## 数字单位／末尾词区域\n',table(['OCR 行','交叠 token 数','对完整均值分差的区域贡献'],[
                [f['text'],f['n'],fmt(f['contribution_to_full_mean_gap'])] for f in focal]),
                '\n区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。\n'])
        if t['control']:
            c=contrasts[id,t['control']];assert c['image_rank']==1;paired.append((s,c))
            parts.extend(['\n## 同包装成功对照\n',f'`{c["photo_id"]}` 为纯图像 Top-1 正确的照片，不要求融合也正确。'
                '按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。\n',
                table(['诊断','错误照片','成功照片'],[['图像分差',fmt(s['image_gap']),fmt(c['image_gap'])],
                    ['只保留主体分差',fmt(s['subject_only_gap']),fmt(c['subject_only_gap'])],['主体 token 数',s['subject_tokens'],c['subject_tokens']]]),
                f'\n[成功对照 token 表](../{c["photo_id"]}/tokens.csv) · [对照摘要](../{c["photo_id"]}/summary.json)\n'])
        else:parts.extend(['\n## 成功对照\n','此包装缺少纯图像 Top-1 正确的照片，未用其他包装冒充同实例对照。\n'])
        parts.extend(['\n## 图与原始数据\n','![对应连线](correspondence.png)\n','![贡献分解](contribution.png)\n',
            '[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)\n',
            '## 结论边界\n','本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。'
            '未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。\n'])
        if (folder/'focal_text.png').is_file():parts.append('![末尾说明文字局部](focal_text.png)\n')
        (folder/'REPORT.md').write_text('\n'.join(parts))
        links.append([f'[{id}]({id}/{id}/REPORT.md)',t['split'],s['image_rank'],fmt(s['image_gap']),fmt(s['subject_only_gap']),
                      group,t['control'] or '无'])
    flips=sum(s['image_gap']<=0 and s['subject_only_gap'] is not None and s['subject_only_gap']>0 for s in target_data)
    harmed=sum(c['subject_only_gap'] is not None and c['subject_only_gap']<0 for _,c in paired)
    stats=dict(configuration='B03',targets=len(target_data),comparisons=len(data['cases']),target_token_rows=token_count,
        paired_controls=len(paired),unique_control_photos=len({c['photo_id'] for _,c in paired}),missing_controls=len(target_data)-len(paired),
        identity_ambiguity_same_pair=data['identity_ambiguity'],prior_other_candidate_ambiguity=data['prior_other_candidate_ambiguity'],
        both_reference_geometry_accepted=sum(all(g['accepted'] for g in s['geometry']) for s in target_data),
        subject_pair_flips_to_positive=flips,subject_pair_flips_to_negative_on_success_controls=harmed,
        exact_image_score_ties=sum(s['image_gap']==0 for s in target_data),
        rescued_by_F03=sum(s['fusion_rank']==1 for s in target_data),still_wrong_in_F03=sum(s['fusion_rank']>1 for s in target_data),
        reused_contrasts=sum('reused_from' in s for s in data['cases']),
        max_score_replay_error=max(s['replay_max_error'] for s in data['cases']),
        validated_score_conservation=True,actual_image_top1_candidates_verified=True,all_target_reports_and_figures_verified=True)
    (out/'verification.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    write_csv(out/'diagnostics.csv',diagnoses)
    if focal_signal:write_csv(out/'focal_lines_all.csv',focal_signal)
    highlighted=[]
    for id in ['photo_41de3eb383934ec3','photo_1f48d45127db4129','photo_0e87e0717ab1c215']:
        if id in inv:
            s=contrasts[id,id];highlighted.append([f'[{NAMES[s["case_type"]]}]({id}/{id}/REPORT.md)',s['image_rank'],fmt(s['image_gap']),fmt(s['subject_only_gap'])])
    split_rows=[]
    for split in ('train','val','test'):
        rows=[s for s in target_data if s['split']==split]
        split_rows.append([split,len(rows),sum(s['subject_only_gap'] is not None and s['subject_only_gap']>0 for s in rows),sum(s['fusion_rank']==1 for s in rows)])
    report=[f'# B03：ColNomic 7B 纯图像检索全部 {len(target_data)} 个错误\n',
        '分析的是冻结图像 encoder 与原始 MaxSim 的检索错误。负候选均为纯图像实际 Top-1，逐案核对原错误清单和缓存分数。'
        '每例含完整 token 对应、贡献分解、独立几何诊断、焦点文字检查、匹配连线与单独报告。\n',
        '## 分析范围与对照\n',table(['检查','结果'],[['错误案例完成',f'{len(target_data)}/{len(expected)}'],
            ['目标 token 记录数',token_count],['同包装成功对照',len(paired)],['不同成功照片',stats['unique_control_photos']],
            ['缺少同包装成功对照',stats['missing_controls']],['相同候选对已有身份歧义',data['identity_ambiguity']],
            ['其他候选曾有身份歧义，未直接转移标签',data['prior_other_candidate_ambiguity']],
            ['正负几何均通过自动阈值',stats['both_reference_geometry_accepted']],['最大缓存评分复现误差',stats['max_score_replay_error']]]),
        '\n对照只要求纯图像 Top-1 正确，F03 成功与否不参与选择。重复对照不是独立样本，可见包装面没有逐对确认。\n',
        '## 根据测量得到的诊断分组\n',table(['诊断','数量'],Counter(r['diagnostic_group'] for r in diagnoses).items()),
        f'\n固定 token 的主体筛选使 {flips}/{len(target_data)} 组正负候选由非正翻正，但也使 {harmed}/{len(paired)} 组成功对照的分差变负。'
        '这是双候选诊断，不是完整图库 Recall@1 的提升，也不是裁剪图重新编码的效果。\n',
        '## 划分与 F03 辅助比较\n',table(['划分','B03 错误数','主体筛选双候选翻正','在 F03 中正确'],split_rows),
        f'\nB03 的 {len(target_data)} 例中，{stats["rescued_by_F03"]} 例在 F03 中检索正确，{stats["still_wrong_in_F03"]} 例仍错误。'
        '本文 token 匹配始终针对 B03 负候选；F03 仅提供已存在的结果对照。\n',
        '## 先前指定案例\n',table(['案例','B03 真值排名','图像分差','仅主体分差'],highlighted),
        '\n此前的颜色案例 photo_21bd7df9ed6795e7 在 B03 中正确，不属于这247例；它的 F03 错误不能计入纯图像失败。\n',
        '## 方法与证据边界\n',
        '- FP32 点积、有效 query token 的 MaxSim 均值，保留有效模板 token，去掉零向量，不再次归一化；每例验证贡献守恒。\n'
        '- 原始缓存网格按 EXIF 转到显示坐标再关联人工主体框与 OCR。OCR 交叠阈值为 patch 面积5%，同时记录多行歧义。\n'
        '- SIFT 独立估计几何：比率 .75、最大边1400、RANSAC 4px、至少8个内点且覆盖主体15%。未通过时不填空间错配结论。\n'
        '- 匹配图展示主体内最有利／最不利各8个 token，完整数据都在CSV。散点色阶按绝对分差的95%分位裁剪，曲线和表格保留完整数值。\n'
        '- 数字单位与末尾词区域可能共享同一 patch、包含邻近文字；上下文 token 和矩形框内遮挡也限制语义归因。\n'
        '- 完成本轮对应与贡献分析并不证明编码器丢失身份信息。编码可分性需要冻结特征 Probe，因果归因需要受控干预。\n',
        '## 原始数据与复现\n','[错误候选核查](B03_error_inventory.csv) · [逐案诊断](diagnostics.csv) · [全部对照摘要](cohort_summary.csv) · '
        '[焦点文字汇总](focal_lines_all.csv) · [数值与覆盖验证](verification.json) · [完整JSON及缓存签名](summary.json)\n',
        '```bash\npython -B scripts/analyze_colnomic_tokens.py --configuration B03 --all-errors --resume --reuse-from results/token_mechanism_analysis/F03\n'
        'python -B scripts/report_colnomic_b03_tokens.py\n```\n',
        '## 全部逐案报告\n',table(['案例','划分','排名','图像分差','仅主体分差','诊断','成功对照'],links)]
    (out/'REPORT.md').write_text('\n'.join(report));print(json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
