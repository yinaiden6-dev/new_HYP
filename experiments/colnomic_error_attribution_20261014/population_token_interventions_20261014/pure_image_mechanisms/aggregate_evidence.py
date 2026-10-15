"""Join completed image experiments without converting untested explanations into causes."""
from pathlib import Path
import csv,sys,collections
O=Path(__file__).resolve().parent;P=O.parent;sys.path.insert(0,str(P))
from experiment_utils import read,write,csvwrite
def cr(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
def main():
 for f in ['CAUSAL_VALIDATION_309.json','CAUSAL_VALIDATION_62.json','REFERENCE_CONTRAST_VALIDATION.json','PURE_MATCHING_VALIDATION.json']:assert read(O/f)['status']=='passed'
 original=cr(O/'pure_image_all1117.csv');c={r['query']:r for r in cr(O/'causal_ledger_309.csv')};c62={r['query']:r for r in cr(O/'causal_ledger_62.csv')};crop={r['query']:int(r['prior_crop_rank']) for r in cr(O/'prior_crop_link1117.csv')};matching={(r['query'],r['method']):int(r['rank']) for r in cr(O/'pure_matching_per_query.csv')};contrast={(r['query'],r['method']):int(r['rank']) for r in cr(O/'reference_contrast_per_query.csv')};ps={r['query']:r for r in read(P/'cohort.json')};out=[]
 for x in original:
  q=x['query'];fail=int(x['image_rank'])>1;subject=int(x['subject_rank']);cc=c.get(q);r=dict(query=q,package_id=x['package_id'],scene=x['scene'],split=x['split'],image_rank=int(x['image_rank']),subject_rank=subject,crop_rank=crop[q],corrected_fusion_rank=int(x['corrected_fusion_rank']),original_fusion_rank=int(x['original_fusion_rank']),numeric_fusion_rank=int(x['numeric_fusion_rank']),positive_reference=ps[q]['image_positive'],image_failed=fail,subject_repaired=fail and subject==1,subject_harmed=not fail and subject>1,readout_diagnostic=x['readout_diagnostic'])
  for m in ['reverse','balanced','visual_forward','visual_reverse','visual_balanced']:r['matching_'+m+'_rank']=matching[q,m]
  for m in ['global_std','hard20_std']:r['contrast_'+m+'_rank']=contrast[q,m]
  r['pixel_cohort']=cc['role'] if cc else 'not_selected';r['exact_old_query_parity']=cc['exact_old_query_parity'] if cc else ''
  for arm in ['far_background_gray','far_background_blur']:
   for scope in ['all','subject_template']:r[arm+'_'+scope+'_rank']=int(cc[arm+'_'+scope+'_rank']) if cc else ''
  r['context_both_full_repair']=bool(cc and cc['full_repaired_by_both_far_fills']=='True');r['context_both_subject_repair']=bool(cc and cc['subject_repaired_by_both_far_fills']=='True');r['context_foreground_input_unchanged']=bool(cc and all(int(cc[a+'_foreground_input_changed_count'])==0 for a in ['far_background_gray','far_background_blur']))
  tags=[];directions=[]
  if r['subject_repaired']:tags.append('冻结特征：主体计分救回');directions.append('自动主体定位＋保守权重；保留全图分数作回退')
  if fail and crop[q]==1:tags.append('既有裁剪输入救回（多因素）');directions.append('联合验证视角、分辨率和上下文，不能只归因于缩放')
  if r['context_both_full_repair']:tags.append('两种远背景编辑均修复全图检索');directions.append('背景扰动一致性训练／稳定性门控，先检查成功样本损害')
  if r['context_both_subject_repair']:tags.append('主体输入不变：两种背景编辑均修复主体读出');directions.append('区分视觉编码的上下文依赖与后续Token重匹配')
  if r['subject_harmed']:tags.append('主体筛选新增错误');directions.append('避免硬删除；检查边界覆盖、上下文与模板比例')
  if fail and any(r['matching_'+m+'_rank']==1 for m in ['reverse','balanced','visual_forward','visual_reverse','visual_balanced']):tags.append('冻结特征匹配对照可救回');directions.append('验证对应面与成分行匹配；通用反向匹配未普遍获益')
  if fail and any(r['contrast_'+m+'_rank']==1 for m in ['global_std','hard20_std']):tags.append('参考候选差异加权可救回');directions.append('用无标签竞争参考估计Token权重，验证随机权重对照')
  r['prior_image_controls']=x['other_successful_image_controls']
  if fail and subject>1 and r['prior_image_controls']:tags.append('既有图像敏感性对照可救回');directions.append('保留已完成边界、模板和参考Token范围对照；避免重复实验')
  r['any_tested_image_repair']=bool(fail and (subject==1 or bool(r['prior_image_controls']) or crop[q]==1 or any(r['matching_'+m+'_rank']==1 for m in ['reverse','balanced','visual_forward','visual_reverse','visual_balanced']) or any(r['contrast_'+m+'_rank']==1 for m in ['global_std','hard20_std']) or (cc and any(int(cc[a+'_'+s+'_rank'])==1 for a in ['far_background_gray','far_background_blur'] for s in ['all','subject_template']))))
  if fail and not r['any_tested_image_repair']:tags.append('本轮修复族未救回');directions.append('固定对应包装面与身份区域，补跨身份受控读出；编码与对应原因尚未分离')
  if fail and r['corrected_fusion_rank']==1:tags.append('既有OCR联合修正救回');directions.append('保持既有主体文字、计分和规格绑定对照')
  if fail and subject==1 and r['corrected_fusion_rank']>1:tags.append('图像修复独有');directions.append('OCR不应覆盖已有图像正证据；检查词项归属与融合权重')
  r['evidence']='；'.join(tags) if tags else '原图成功对照';r['repair_direction']='；'.join(dict.fromkeys(directions)) if directions else '保持成功，作为改动损害对照';r['causal_limit']='多标签证据，非互斥根因；候选级、输入级干预均不等于全部语义机制已证实';out.append(r)
 assert len(out)==1117 and sum(r['image_failed'] for r in out)==247
 csvwrite(O/'all1117_image_evidence.csv',out)
 failures=[r for r in out if r['image_failed']];sets={k:{r['query'] for r in failures if pred(r)} for k,pred in {'subject':lambda r:r['subject_rank']==1,'crop':lambda r:r['crop_rank']==1,'context_both_full':lambda r:r['context_both_full_repair'],'context_both_subject':lambda r:r['context_both_subject_repair'],'any_image_intervention':lambda r:r['any_tested_image_repair'],'corrected_fusion':lambda r:r['corrected_fusion_rank']==1}.items()}
 results=dict(total=1117,image_failures=247,repair_coverage={k:len(v) for k,v in sets.items()},image_repair_union_is_oracle_diagnostic=True,not_repaired_by_this_family=247-len(sets['any_image_intervention']),image_repair_outside_corrected_fusion=len(sets['any_image_intervention']-sets['corrected_fusion']),both_context_full_and_subject=len(sets['context_both_full']&sets['context_both_subject']),subject_crop_overlap=len(sets['subject']&sets['crop']),context_both_full_outside_subject=len(sets['context_both_full']-sets['subject']))
 write(O/'IMAGE_EVIDENCE_SUMMARY.json',results);print(results)
if __name__=='__main__':main()
