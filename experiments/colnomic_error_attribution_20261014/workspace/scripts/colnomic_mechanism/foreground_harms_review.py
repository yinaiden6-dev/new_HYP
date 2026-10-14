"""Read-only materialization of the eight original foreground fusion harms.

Manual observations are stored separately under foreground_harms_review/. The
figure renderer crops real pixels after EXIF transpose; it performs no model
encoding and does not infer occlusion or identity from colors/OCR alone.
"""
import argparse,csv,datetime,hashlib,html,json,os,shutil
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/colnomic_full_audit/20261013_resume'
OUT=BASE/'foreground_harms_review'
IDS=['photo_12b37e2d68711d62','photo_21bd7df9ed6795e7','photo_28d813f7c9451126','photo_3657bb9dffa08975',
     'photo_4920adb7ad190a5f','photo_7e0b721a9c25b555','photo_c0babaac8a7306cd','photo_daac4b2576b9e9f5']

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))
def evidence(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
def displayed(path):
    with Image.open(path) as im:return ImageOps.exif_transpose(im).convert('RGB')
def fit(canvas,im,box):
    im=ImageOps.contain(im,(box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
    canvas.paste(im,(box[0]+(box[2]-box[0]-im.width)//2,box[1]+(box[3]-box[1]-im.height)//2))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source_paths=[BASE/'scene35_reviewed/foreground_32_and_fusion_harms.csv',BASE/'visual_consolidated/records.json',
        BASE/'manual_box_review/subject_boxes_after_user_review.json',BASE/'reviewed_regions/manual_rois.json',ROOT/'dataset/lists/references.csv']
    source_hashes=[evidence(p) for p in source_paths]
    for path in source_paths:
        target=OUT/'source_snapshots'/path.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
    visuals={r['photo_id']:r for r in read(source_paths[1])};boxes=read(source_paths[2])['records']
    refs={r['reference_id']:r for r in rows(source_paths[4])};harms={r['photo_id']:r for r in rows(source_paths[0]) if r['fusion_introduced_error']=='True'}
    assert set(harms)==set(IDS) and len(harms)==8
    result=[]
    for qid in IDS:
        dossier_path=BASE/'full_tokens/queries'/qid/'dossier.json';d=read(dossier_path);v=visuals[qid]
        original=Path(d['original_path']);positive=d['primary_positive'];negative=d['fusion_strongest_wrong']
        gtpath=ROOT/'dataset'/refs[positive]['path'];negpath=ROOT/'dataset'/refs[negative]['path']
        assert positive==harms[qid]['positive_reference'] and negative==harms[qid]['fusion_compared_wrong_reference']
        folder=OUT/'queries'/qid;folder.mkdir(parents=True,exist_ok=True)
        im=displayed(original);bbox=boxes[qid]['bbox'];crop=im.crop(tuple(round(x) for x in bbox))
        crop.save(folder/'subject_crop.png');gt=displayed(gtpath);neg=displayed(negpath)
        gt.save(folder/'positive.png');neg.save(folder/'fusion_negative.png')
        overview=im.copy();ImageDraw.Draw(overview).rectangle(tuple(bbox),outline='#ff2020',width=9)
        canvas=Image.new('RGB',(3400,1400),'white');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
        draw.text((10,10),qid+' | original / reviewed subject crop / GT '+positive+' / fusion negative '+negative,fill='black',font=font)
        for picture,box in [(overview,(0,60,600,1390)),(crop,(600,60,1700,1390)),(gt,(1700,60,2550,1390)),(neg,(2550,60,3400,1390))]:fit(canvas,picture,box)
        canvas.save(folder/'overview.jpg',quality=95,subsampling=0)
        prior=[]
        for history in v.get('review_history',[]):
            raw=history.get('raw_record',{});key=raw.get('identity_region_obscured',raw.get('identity_key_region_occlusion'))
            if key:
                prior.append(dict(source_path=history['source_path'],source_sha256=history['source_sha256'],value=key,
                    fine_detail_verified=history.get('fine_detail_verified'),direct_observations=history.get('direct_observations'),uncertainty=history.get('uncertainty')))
        with Image.open(original) as raw:size=list(raw.size);orientation=int(raw.getexif().get(274,1))
        item=dict(photo_id=qid,package_id=d['package_id'],split=d['split'],positive_reference_id=positive,fusion_negative_reference_id=negative,
            image_correct=True,fusion_correct=False,score_decomposition=harms[qid],
            original=evidence(original),positive=evidence(gtpath),fusion_negative=evidence(negpath),
            raw_size=size,exif_orientation=orientation,display_size=list(im.size),subject_bbox=bbox,
            coordinate_system='EXIF-transposed original image pixels; exact current reviewed subject bbox',
            reviewed_box_snapshot=evidence(source_paths[2]),legacy_key_region_evidence=prior,
            consolidated_top_level_key_region_present='identity_region_obscured' in v,
            prior_observations=v['direct_observations'],prior_uncertainty=v['uncertainty'],
            generated_evidence={p.name:evidence(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in ['.png','.jpg']},
            review_status='awaiting_actual_view_image_detail_review',causal_mechanism_claimed=False)
        write(folder/'materials.json',item);result.append(item)
    write(OUT/'materials_all8.json',result);write(OUT/'source_manifest.json',source_hashes)
    print(json.dumps(dict(queries=len(result),legacy_top_level_key_region_present=sum(x['consolidated_top_level_key_region_present'] for x in result),
        legacy_nested_key_region_records=sum(len(x['legacy_key_region_evidence']) for x in result))),flush=True)

def render_details():
    for row in read(OUT/'detail_rois.json'):
        m=read(OUT/'queries'/row['photo_id']/'materials.json');im=displayed(ROOT/m['original']['path']).crop(row['roi'])
        if row['display_rotation_CCW']:im=im.rotate(row['display_rotation_CCW'],expand=True)
        im.save(ROOT/row['evidence']['path']);assert sha(ROOT/row['evidence']['path'])==row['evidence']['sha256']
    for row in read(OUT/'candidate_detail_rois.json'):
        pictures=[]
        for roi in row['rois']:
            im=displayed(ROOT/roi['source']['path']).crop(roi['roi'])
            if roi['rotation_CCW']:im=im.rotate(roi['rotation_CCW'],expand=True)
            pictures.append(im)
        m=read(OUT/'queries'/row['photo_id']/'materials.json');canvas=Image.new('RGB',(3200,1500),'white');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
        draw.text((10,10),row['photo_id']+' | GT '+m['positive_reference_id']+' / fusion negative '+m['fusion_negative_reference_id'],fill='black',font=font)
        fit(canvas,pictures[0],(0,60,1600,1490));fit(canvas,pictures[1],(1600,60,3200,1490));canvas.save(ROOT/row['evidence']['path'])
        assert sha(ROOT/row['evidence']['path'])==row['evidence']['sha256']

def finalize():
    manual=read(OUT/'manual_reviews.json');materials={r['photo_id']:r for r in read(OUT/'materials_all8.json')}
    assert {r['photo_id'] for r in manual['records']}==set(IDS)
    details={r['photo_id']:r for r in read(OUT/'detail_rois.json')};candidate_details={r['photo_id']:r for r in read(OUT/'candidate_detail_rois.json')}
    join=[];sources={};evidence_files={};score_errors=[]
    for source in read(OUT/'source_manifest.json'):
        assert sha(OUT/'source_snapshots'/Path(source['path']).name)==source['sha256'],'Consumed input snapshot SHA mismatch'
    for review in manual['records']:
        qid=review['photo_id'];m=materials[qid]
        assert review['review_method']=='actual_view_image' and review['actual_original_crop_and_GT_negative_detail_view_completed']
        assert review['pair_discriminative_spec_occlusion']['status'] in ['yes','none','unknown']
        assert review['causal_mechanism_claimed'] is False
        for role in ['original','positive','fusion_negative']:
            source=m[role];assert sha(ROOT/source['path'])==source['sha256'];sources[source['path']]=source['sha256']
        viewed=review['actually_viewed_evidence_paths']
        assert m['generated_evidence']['overview.jpg']['path'] in viewed and details[qid]['evidence']['path'] in viewed
        for path in viewed:
            assert (ROOT/path).is_file();evidence_files[path]=sha(ROOT/path)
        score=m['score_decomposition'];error=abs(float(score['normalized_image_contribution'])+float(score['normalized_text_contribution'])-float(score['fusion_gap']))
        assert error<=5e-7;score_errors.append(error)
        assert float(score['normalized_image_contribution'])>0 and float(score['normalized_text_contribution'])<0 and float(score['fusion_gap'])<0
        region_summary=BASE/'reviewed_regions/queries'/qid/'summary.json'
        if review.get('existing_token_evidence'):
            old_regions=read(region_summary);contrast=next(r for r in old_regions['contrasts'] if r['positive']==m['positive_reference_id'] and r['negative']==m['fusion_negative_reference_id'])
            pattern=next(r for r in contrast['regions'] if r['region']=='pattern_color')
            assert pattern['tokens']==review['existing_token_evidence']['manually_localized_pattern_color_tokens']
            assert abs(pattern['net_contribution']-review['existing_token_evidence']['pattern_color_net_image_contribution'])<1e-14
        item={**m,**review,'detail_roi':details[qid],
            'review_status':'actual_original_crop_and_candidate_spec_detail_reviewed',
            'fine_detail_verified':True,'fine_detail_scope':'Candidate pair distinguishing count/weight and actual occluder boundary; all other uninspected tiny text remains unknown.',
            'existing_region_evidence':evidence(region_summary),
            'visual_review_path':str((OUT/'queries'/qid/'review.json').relative_to(ROOT)),
            'occlusion_location':review['identity_region_obscured']['position'],
            'occlusion_extent':review['identity_region_obscured']['degree'],
            'identity_key_region_scope':'Generic medicine identity text/specs and pair-discriminative specs are separately reported; hidden characters are not reconstructed from GT.'}
        write(OUT/'queries'/qid/'review.json',item);join.append(item)
    write(OUT/'join_records.json',join)
    write(OUT/'join_by_photo_id.json',{r['photo_id']:r for r in join})
    flat=[]
    for item in join:
        spec=item['pair_discriminative_spec_occlusion'];score=item['score_decomposition']
        flat.append(dict(photo_id=item['photo_id'],package_id=item['package_id'],split=item['split'],occluder=item['occluder'],
            identity_region_obscured=item['identity_region_obscured']['status'],pair_discriminative_spec_occlusion=spec['status'],
            occlusion_location=item['occlusion_location'],occlusion_extent=item['occlusion_extent'],
            positive_spec=spec['positive_spec'],negative_spec=spec['negative_spec'],query_spec_observation=spec['query_observation'],
            normalized_image_contribution=float(score['normalized_image_contribution']),normalized_text_contribution=float(score['normalized_text_contribution']),
            fusion_gap=float(score['fusion_gap']),visual_review_path=item['visual_review_path']))
    with (OUT/'join_records_flat.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    page=['<!doctype html><meta charset="utf-8"><title>Eight foreground fusion harms: actual key-region review</title>',
        '<style>body{font:16px system-ui;max-width:1700px;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}img{max-width:100%}aside{background:#fff1cc;padding:15px}</style>',
        '<h1>8例前景遮挡新增融合错误：关键区实际核查</h1>',
        '<aside>全部8例原图像正确、原融合错误。泛身份文字遮挡与该正负候选对的区分规格遮挡分别报告；未恢复被挡字符，不作遮挡导致融合失败的因果推断。q4920是拇指及原图截出，数量缺失来源保留unknown。</aside>',
        '<p><a href="join_records.json">可join完整JSON</a> · <a href="join_records_flat.csv">平面CSV</a> · <a href="validation.json">实际验证与SHA</a></p>',
        '<table><tr><th>Query</th><th>遮挡物</th><th>泛身份区域</th><th>候选区分规格被遮</th><th>实际观察</th></tr>']
    for row in flat:
        qid=row['photo_id'];page.append('<tr><td><a href="#'+qid+'">'+qid+'</a></td>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in ['occluder','identity_region_obscured','pair_discriminative_spec_occlusion','query_spec_observation'])+'</tr>')
    page.append('</table>')
    for item in join:
        qid=item['photo_id'];detail_link=os.path.relpath(ROOT/item['actually_viewed_evidence_paths'][-1],OUT)
        original_link=os.path.relpath(ROOT/item['original']['path'],OUT)
        page.extend(['<h2 id="'+qid+'">'+qid+'</h2>',
            '<p><a href="queries/'+qid+'/review.json">完整逐图记录</a> · <a href="'+html.escape(detail_link)+'">GT/负候选规格放大</a> · <a href="'+html.escape(original_link)+'">原始文件</a></p>',
            '<p>'+html.escape(item['occlusion_extent'])+'</p>',
            '<img src="queries/'+qid+'/overview.jpg"><img src="queries/'+qid+'/key_detail.png">'])
    (OUT/'index.html').write_text('\n'.join(page))
    validation=dict(status='passed',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queries=8,
        actual_vision_review_records=8,actually_viewed_evidence_images=len(evidence_files),source_images_SHA_rechecked=len(sources),
        source_images_unchanged=True,source_image_SHA256=sources,evidence_SHA256=evidence_files,
        key_region_nested_history_entries=sum(len(r['legacy_key_region_evidence']) for r in join),
        key_region_present_at_consolidated_top_level=sum(r['consolidated_top_level_key_region_present'] for r in join),
        generic_identity_region_occlusion=dict(Counter(r['identity_region_obscured']['status'] for r in join)),
        pair_discriminative_spec_occlusion=dict(Counter(r['pair_discriminative_spec_occlusion']['status'] for r in join)),
        max_frozen_fusion_conservation_error=max(score_errors),model_encoding_rerun=False,causal_mechanism_claimed=False,
        manual_review_SHA256=sha(OUT/'manual_reviews.json'),join_records_SHA256=sha(OUT/'join_records.json'),
        review_scope='Eight selected original foreground fusion harms; original/crop and compared GT/negative identity details, not all tiny package text or a population experiment.',
        source_code_sha256=sha(Path(__file__)))
    write(OUT/'validation.json',validation);print(json.dumps({k:v for k,v in validation.items() if k not in ['source_image_SHA256','evidence_SHA256']}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--render-details',action='store_true');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.finalize:finalize()
    elif args.render_details:render_details()
    else:prepare()

if __name__=='__main__':main()
