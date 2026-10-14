"""Summarize the already completed eight-case visual audit, without relabeling."""
from collections import Counter
from scripts.colnomic_mechanism.common import ROOT,read


def main():
    folder=ROOT/'results/colnomic_full_audit/20261013_resume/foreground_harms_review'
    records=read(folder/'join_records.json');validation=read(folder/'validation.json')
    assert len(records)==8 and validation['status']=='passed'
    assert all(r['image_correct'] and not r['fusion_correct'] for r in records)
    counts=Counter(r['pair_discriminative_spec_occlusion']['status'] for r in records)
    lines=['# 原前景遮挡组的8例融合新增错误', '',
        '这8张是原标签中32张前景遮挡照片的融合新增错，属于被选中的错误子集。全部8例已实际查看原图、主体裁切及正负候选的区分规格；22份查看材料和19张独立原图/参考的SHA见验收。该审核不代替1117张的总体随机抽样。', '',
        f'与该正负候选有关的区分规格：被挡 {counts.get("yes",0)}/8，未挡 {counts.get("none",0)}/8，不确定 {counts.get("unknown",0)}/8。一般身份区域与这组候选的区分规格分别保存；隐藏文字不会由图库反推成已看见。', '',
        '|照片|区分规格遮挡|正/负规格|图像项|文字项|融合差额|实审证据|',
        '|---|---|---|---:|---:|---:|---|']
    for r in records:
        spec=r['pair_discriminative_spec_occlusion'];scores=r['score_decomposition'];qid=r['photo_id']
        positive=str(spec.get('positive_spec','未知')).replace('|','/');negative=str(spec.get('negative_spec','未知')).replace('|','/')
        lines.append(f'|{qid}|{spec["status"]}|{positive} / {negative}|{float(scores["normalized_image_contribution"]):.6f}|{float(scores["normalized_text_contribution"]):.6f}|{float(scores["fusion_gap"]):.6f}|[记录](queries/{qid}/review.json)|')
    lines+=['', '图像项为正、文字项为更大的负值，解释了历史公式的排名改变。这是确定性评分分解。它不证明遮挡改变OCR，也不证明遮挡造成融合失败；片数可见时仍有融合新增错，是反对单一遮挡解释的直接比较证据。', '',
        '颜色/图案、公共文字、规格与排版可以同时不同。可观察差异、遮挡位置及程度、尚未辨清的细字分别保留在逐图记录中；本报告不新增视觉结论或唯一机制标签。', '',
        '[可浏览材料](index.html) · [全部关联记录](join_records.json) · [原始实际观看记录](manual_reviews.json) · [独立验收](validation.json)']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
