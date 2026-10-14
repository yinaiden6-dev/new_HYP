"""CPU preparation of paired-view, same-word physical layout controls.

No image encoder, embedding response, ranking or correctness selects sources.
Every selected pair requires later actual word/face review before encoding.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts.colnomic_mechanism.common import ROOT, read, write, rows, csvout, sha, workspace_path
from scripts.colnomic_mechanism.display_pixels import display_image
from scripts.colnomic_mechanism.extended_layouts import prepare, to_raw, raw_bbox
from scripts.colnomic_mechanism.symmetric_layouts import variants
from scripts.colnomic_mechanism.geometry import anchors, estimate

BASE=ROOT/'results/colnomic_full_audit/20261013_resume'
ARMS=('base','sham','left','right','editor_left','editor_right')


def project(h,point):
    mapped=np.asarray(h,float)@np.r_[point,1.]
    if abs(mapped[2])<1e-10:raise ValueError('projection at infinity')
    return mapped[:2]/mapped[2]


def jacobian(h,point):
    h=np.asarray(h,float);point=np.asarray(point,float);den=h[2,:2]@point+h[2,2]
    if abs(den)<1e-10:raise ValueError('Jacobian at infinity')
    return (h[:2,:2]*den-np.outer(h[:2,:2]@point+h[:2,2],h[2,:2]))/den**2


def physical_correspondence(a,b):
    ha=np.asarray(a['alignment']['homography']);hb=np.asarray(b['alignment']['homography'])
    hab=np.linalg.inv(hb)@ha;hab=hab/hab[2,2]
    ca=np.asarray(a['center_normalized']);cb=np.asarray(b['center_normalized'])
    j=jacobian(hab,ca);direction=j@np.array([1.,0.])
    if abs(direction[1])>.16*np.linalg.norm(direction):
        raise ValueError('cross-view x direction not compatible with both .08-horizontal words')
    sign=1 if direction[0]>0 else -1
    center_error=float(np.linalg.norm(project(ha,ca)-project(hb,cb)))
    if center_error>.025:raise ValueError('same OCR word does not project to same GT position')
    ua=np.array([1/a['display_width'],0.]);ub=np.array([1/b['display_width'],0.])
    qa=np.linalg.norm(jacobian(ha,ca)@ua);qb=np.linalg.norm(jacobian(hb,cb)@ub)
    quantization_bound=float(.5*(qa+qb))
    moves=[]
    for offset in (-1,1):
        va=project(ha,ca+offset*a['shift_pixels']*ua)-project(ha,ca)
        vb=project(hb,cb+offset*sign*b['shift_pixels']*ub)-project(hb,cb)
        error=float(abs(np.linalg.norm(va)-np.linalg.norm(vb)))
        if float(va@vb)<=0:raise ValueError('cross-view projected word displacement signs disagree')
        moves.append(dict(A_image_sign=offset,B_corresponding_image_sign=offset*sign,
            A_GT_displacement=va.tolist(),B_GT_displacement=vb.tolist(),
            A_GT_magnitude=float(np.linalg.norm(va)),B_GT_magnitude=float(np.linalg.norm(vb)),
            magnitude_error=error,relative_magnitude_error=error/max(float(np.linalg.norm(va)),float(np.linalg.norm(vb))),
            source_halfpixel_quantization_bound=quantization_bound,
            crossview_magnitude_equality_required=False))
    return dict(H_AB=hab.tolist(),H_AB_definition='inv(H_BG) @ H_AG; all coordinates EXIF-displayed normalized',
        Jacobian_AB_at_A_word=j.tolist(),B_direction_label_swapped=sign<0,B_corresponding_sign=sign,
        same_GT_word_center_error=center_error,physical_displacement_checks=moves,
        physical_direction_scope='left/right along A displayed x; B labels correspond through H_AB')


def read_photo(path,spec):
    with Image.open(path) as im:
        raw=np.asarray(im.convert('RGB'));shown=np.asarray(display_image(im)).copy()
    if shown.shape[:2]!=(spec['display_height'],spec['display_width']):
        raise ValueError('display metadata shape mismatch')
    if not np.array_equal(to_raw(shown,spec['exif_orientation']),raw):
        raise ValueError('EXIF inverse differs from original raw RGB')
    return shown,raw


def candidate_words(photo,spec,ocr,box,positives,reference_anchors,rejections):
    path=ROOT/'dataset'/photo['path']
    if sha(path)!=photo['sha256']:raise ValueError('original image SHA mismatch')
    array,_=read_photo(path,spec);subject=np.asarray(box['bbox'],float)
    subject_norm=subject/np.tile([array.shape[1],array.shape[0]],2)
    qa=anchors(ocr,subject_norm);fits={}
    for rid in sorted(positives):
        ra=reference_anchors[rid];keys=qa.keys()&ra.keys()
        if len(keys)>=4:
            fit=estimate(qa,ra,keys,subject_norm)
            if fit is not None:fits[rid]=fit
    if not fits:
        rejections.append(dict(package_id=photo['package_id'],photo_id=photo['photo_id'],stage='alignment',reason='no accepted OCR homography to an identity positive reference'))
        return []
    result=[]
    for li,line in enumerate(ocr['lines']):
        try:
            bb,_,_=prepare(array,line,spec,subject,ocr['lines'],.08)
            edits,step,allowed,glyph,paper=variants(array,bb,subject,ocr['lines'])
        except ValueError as error:
            rejections.append(dict(package_id=photo['package_id'],photo_id=photo['photo_id'],OCR_line=li,stage='word_quality',reason=str(error)))
            continue
        word=line['text'].strip().rstrip('.!').lower()
        for rid,fit in fits.items():
            result.append(dict(photo_id=photo['photo_id'],package_id=photo['package_id'],word_key=word,
                OCR_text=line['text'],OCR_line=li,OCR_confidence=float(line['confidence']),
                bbox_display=list(map(int,bb)),shift_pixels=step,positive_reference=rid,alignment=fit,
                center_normalized=[(bb[0]+bb[2])/2/array.shape[1],(bb[1]+bb[3])/2/array.shape[0]],
                original_path=str(path.relative_to(ROOT)),image_sha256=photo['sha256'],source_sha256=photo['source_sha256'],
                display_width=array.shape[1],display_height=array.shape[0],exif_orientation=spec['exif_orientation'],
                background_RGB=paper.tolist()))
        del edits,allowed,glyph
    return result


def render_source(source,role,pair_id,folder,spec,box,ocr,sign):
    original=ROOT/source['original_path'];array,raw=read_photo(original,spec)
    bb,_,_=prepare(array,ocr['lines'][source['OCR_line']],spec,np.asarray(box['bbox']),ocr['lines'],.08)
    edits,step,allowed,glyph,paper=variants(array,bb,np.asarray(box['bbox']),ocr['lines'])
    if list(bb)!=source['bbox_display'] or step!=source['shift_pixels']:raise ValueError('source preparation replay differs')
    sham_colors,sham_counts=np.unique(edits['sham'][allowed],axis=0,return_counts=True)
    checks=[];manifest=[]
    for arm in ARMS:
        value=array if arm=='base' else edits[arm]
        offset=-step if arm=='left' else step if arm=='right' else 0
        word_box=(bb[0]+offset,bb[1],bb[2]+offset,bb[3])
        histogram_equal=None
        if arm in ('left','right'):
            colors,counts=np.unique(value[allowed],axis=0,return_counts=True)
            histogram_equal=bool(np.array_equal(colors,sham_colors) and np.array_equal(counts,sham_counts))
        glyph_equal=bool(np.array_equal(value[word_box[1]:word_box[3],word_box[0]:word_box[2]],glyph))
        outside=bool(np.array_equal(value[~allowed],array[~allowed]))
        if not glyph_equal or not outside or histogram_equal is False:raise ValueError('pixel conservation failure')
        rid=f'{pair_id}_{role}_{arm}'
        if arm=='base':output=original
        else:
            output=folder/'images'/(rid+'.png');output.parent.mkdir(parents=True,exist_ok=True)
            exif=Image.Exif();exif[274]=spec['exif_orientation'];Image.fromarray(to_raw(value,spec['exif_orientation'])).save(output,exif=exif)
            saved,_=read_photo(output,spec)
            if not np.array_equal(saved,value):raise ValueError('saved edit/EXIF image replay mismatch')
        image_sign=-1 if arm=='left' else 1 if arm=='right' else None
        physical_sign=image_sign*sign if image_sign is not None else None
        manifest.append(dict(id=rid,pair_id=pair_id,source_id=source['photo_id'],role=role,path=str(output),
            arm=arm,word_bbox=raw_bbox(word_box,array.shape,spec['exif_orientation']),
            image_direction=arm if image_sign is not None else None,
            physical_direction='left' if physical_sign==-1 else 'right' if physical_sign==1 else None,
            layout_class=0 if physical_sign==-1 else 1 if physical_sign==1 else None,
            label_scope='same-content paired layout compatibility; no natural identity correctness assigned',image_sha256=sha(output)))
        checks.append(dict(role=role,arm=arm,glyph_rectangle_RGB_exact=glyph_equal,outside_edit_band_exact=outside,
            sham_shift_pixel_histogram_equal=histogram_equal,saved_display_pixels_exact=True if arm=='base' else bool(np.array_equal(saved,value)),
            editor_control_word_unmoved=arm in ('editor_left','editor_right') and offset==0 if arm.startswith('editor_') else None))
    return manifest,checks,{arm:array if arm=='base' else edits[arm] for arm in ARMS}


def material(folder,pair,values_a,values_b,boxes):
    canvas=Image.new('RGB',(1800,1300),'white');draw=ImageDraw.Draw(canvas)
    for row,(role,source,values) in enumerate([('A',pair['A'],values_a),('B',pair['B'],values_b)]):
        im=Image.fromarray(values['base']);im.thumbnail((360,570));canvas.paste(im,(20,row*650+50))
        draw.text((10,row*650+10),f"{role} {source['photo_id']} word={source['OCR_text']}",fill='black')
        bb=source['bbox_display'];step=source['shift_pixels'];region=(max(0,bb[0]-step-20),max(0,bb[1]-60),min(source['display_width'],bb[2]+step+20),min(source['display_height'],bb[3]+60))
        for i,arm in enumerate(ARMS):
            patch=Image.fromarray(values[arm]).crop(region);patch.thumbnail((440,170));x=390+(i%3)*470;y=row*650+50+(i//3)*285
            canvas.paste(patch,(x,y+30));draw.text((x,y),arm,fill='black')
    path=folder/'manual_material'/(pair['pair_id']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return str(path.relative_to(ROOT)),sha(path)


def prepare_all(folder):
    folder.mkdir(parents=True,exist_ok=True)
    photos={r['photo_id']:r for r in rows(ROOT/'dataset/lists/photos.csv')};group_rows=rows(ROOT/'dataset/lists/groups.csv')
    groups={r['package_id']:r for r in group_rows};metadata={r['id']:r for r in read(ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json')['photos']}
    ocr=read(ROOT/'cache/ocr/photos.json');references=read(ROOT/'cache/ocr/references.json')
    boxes_path=BASE/'manual_box_review/subject_boxes_after_user_review.json';boxes=read(boxes_path)['records']
    positives=defaultdict(set)
    for row in rows(ROOT/'dataset/lists/positives.csv'):positives[row['package_id']].add(row['reference_id'])
    ra={rid:anchors(references[rid]) for rid in {r for rs in positives.values() for r in rs}}
    protocol=dict(schema='colnomic.controlled_crossview_layout.v1',source_population='all130 identities/all1117 original query images',
        selection='first deterministic quality-eligible two distinct source SHA photos, same identity/positive reference/isolated OCR word; no embedding/ranking/correctness',
        source_quality=dict(OCR_confidence=.95,maximum_tilt=.08,minimum_height_pixels=10,word_width_fraction_shift=.35,
            symmetric_uniform_blank_paper='existing symmetric_layouts.variants quantile<=24/maxdeviation<=35; no other OCR line collision',
            alignment='existing OCR RANSAC gates>=4inlierlines,coverage>=.15,positive Jacobian determinant,condition<=20',
            same_word_GT_center_tolerance=.025,crossview_x_tilt=.16,
            physical_equal_magnitude='within each original source +/-round(.35word_width) equal; across sources projected magnitude differences reported, not a rejection gate'),
        correspondence='H_AB=inv(H_BG)@H_AG; Jacobian horizontal sign swaps B direction labels when needed',
        arms=list(ARMS),arms_per_pair=12,expected='A left vs B corresponding left and A right vs B corresponding right consistent; opposite inconsistent',
        annotations='positive identity and confirmed subject boxes are source selection assistance; OCR supplies words/alignment; actual same word/same face review required',
        evaluation_scope='same-content manually edited layout matching across views; not natural packaging identity accuracy',
        historical_test_exposure=True,GPU_or_encoder_used_for_preparation=False,no_quality_relaxation_for_counts=True,
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'dataset/lists/photos.csv',ROOT/'dataset/lists/groups.csv',ROOT/'dataset/lists/positives.csv',boxes_path,ROOT/'cache/ocr/photos.json',ROOT/'cache/ocr/references.json',ROOT/'results/token_mechanism_analysis/F03/reproduction/token_metadata.json']})
    if (folder/'protocol.json').exists():
        if read(folder/'protocol.json')!=protocol:raise ValueError('Frozen preparation protocol changed')
    else:write(folder/'protocol.json',protocol)
    population=defaultdict(list)
    for qid,p in photos.items():population[p['package_id']].append(qid)
    sources=[];rejections=[];inventory=[]
    for package in sorted(population):
        checkpoint=folder/'source_checkpoints'/(package+'.json')
        if checkpoint.exists():
            old=read(checkpoint)
            for path,digest in old.get('output_sha256',{}).items():
                if sha(ROOT/path)!=digest:raise ValueError('Prepared source artifact changed')
            if old['source']:sources.append(old['source'])
            rejections.extend(old['rejections']);inventory.append(old['inventory']);continue
        rejected=[];candidates=[]
        for qid in sorted(population[package]):
            candidates.extend(candidate_words(photos[qid],metadata[qid],ocr[qid],boxes[qid],positives[package],ra,rejected))
        chosen=None
        for ai,a in enumerate(candidates):
            for b in candidates[ai+1:]:
                if a['photo_id']==b['photo_id'] or a['source_sha256']==b['source_sha256']:continue
                if a['word_key']!=b['word_key'] or a['positive_reference']!=b['positive_reference']:continue
                try:correspondence=physical_correspondence(a,b)
                except (ValueError,np.linalg.LinAlgError) as error:
                    rejected.append(dict(package_id=package,photo_id_A=a['photo_id'],photo_id_B=b['photo_id'],word=a['word_key'],stage='crossview_correspondence',reason=str(error)));continue
                chosen=dict(pair_id=package+'_crossview',package_id=package,split=groups[package]['split'],group_id=groups[package]['group_id'],
                    A=a,B=b,common_positive_reference=a['positive_reference'],word_key=a['word_key'],**correspondence,
                    visual_review_status='pending_actual_word_and_physical_face_check',natural_identity_accuracy_assigned=False)
                break
            if chosen:break
        hashes={}
        if chosen:
            ma,ca,va=render_source(chosen['A'],'A',chosen['pair_id'],folder,metadata[a['photo_id']],boxes[a['photo_id']],ocr[a['photo_id']],1)
            mb,cb,vb=render_source(chosen['B'],'B',chosen['pair_id'],folder,metadata[b['photo_id']],boxes[b['photo_id']],ocr[b['photo_id']],correspondence['B_corresponding_sign'])
            chosen['arms']=ma+mb;chosen['pixel_validation']=ca+cb
            chosen['evidence_image'],chosen['evidence_sha256']=material(folder,chosen,va,vb,boxes)
            hashes={str(Path(r['path']).relative_to(ROOT)):r['image_sha256'] for r in chosen['arms']}
            hashes[chosen['evidence_image']]=chosen['evidence_sha256'];sources.append(chosen)
        inv=dict(package_id=package,split=groups[package]['split'],original_photos=len(population[package]),eligible_word_reference_candidates=len(candidates),
            eligible_distinct_photos=len({r['photo_id'] for r in candidates}),qualified_pair=chosen is not None,
            rejection_reason_counts=dict(Counter(r['reason'] for r in rejected)))
        write(checkpoint,dict(source=chosen,rejections=rejected,inventory=inv,output_sha256=hashes));rejections.extend(rejected);inventory.append(inv)
        write(folder/'preparation_progress.json',dict(identities_completed=len(inventory),identities_total=len(population),qualified_pairs=len(sources)))
        print('Crossview source',package,bool(chosen),len(sources),flush=True)
    write(folder/'sources.json',sources);write(folder/'pending_encoding_manifest.json',[arm for source in sources for arm in source['arms']])
    write(folder/'encoding_manifest.json',[]);write(folder/'identity_inventory.json',inventory);csvout(folder/'rejections.csv',rejections)
    write(folder/'preparation_complete.json',dict(status='CPU_pixel_preparation_complete_actual_review_pending',identities_scanned=len(inventory),
        qualified_pairs=len(sources),pairs_by_split=dict(Counter(r['split'] for r in sources)),all_12arms_pixel_validated=sum(len(r['pixel_validation'])==12 for r in sources),
        quality_rules_frozen=True,encoder_invoked=False,embedding_response_used=False))


def finalize(folder):
    sources=read(folder/'sources.json');reviews=read(folder/'manual_reviews.json');byid={r['pair_id']:r for r in reviews}
    if {r['pair_id'] for r in sources}!={r['pair_id'] for r in reviews}:raise ValueError('Missing/extra actual pair reviews')
    accepted=[]
    for source in sources:
        review=byid[source['pair_id']]
        if sha(ROOT/source['evidence_image'])!=review['evidence_sha256']:raise ValueError('Actual view material SHA mismatch')
        for arm in source['arms']:
            if sha(workspace_path(arm['path']))!=arm['image_sha256']:raise ValueError('Prepared arm changed')
        source['actual_visual_review']=review
        source['visual_review_status']='confirmed_same_word_same_physical_face' if review['same_word_confirmed'] and review['same_face_confirmed'] and review['edit_quality_accepted'] else 'excluded_after_actual_visual_review'
        if source['visual_review_status'].startswith('confirmed'):accepted.append(source)
    manifest=[dict(arm,split=source['split'],package_id=source['package_id']) for source in accepted for arm in source['arms']]
    relations=[]
    for source in accepted:
        arms={(r['role'],r['physical_direction']):r['id'] for r in source['arms'] if r['physical_direction']}
        for role,other in [('A','B'),('B','A')]:
            for direction,opposite in [('left','right'),('right','left')]:
                relations.append(dict(pair_id=source['pair_id'],package_id=source['package_id'],split=source['split'],
                    query_id=arms[role,direction],consistent_id=arms[other,direction],inconsistent_id=arms[other,opposite],
                    label_scope='paired same-content physical layout agreement; not identity accuracy'))
    write(folder/'sources_reviewed.json',sources);write(folder/'encoding_manifest.json',manifest);write(folder/'expected_relations.json',relations)
    write(folder/'finalized.json',dict(status='CPU_preparation_and_actual_pair_review_complete',identities_scanned=len(read(folder/'identity_inventory.json')),
        prepared_pairs=len(sources),visually_accepted_pairs=len(accepted),excluded_pairs=len(sources)-len(accepted),
        accepted_pairs_by_split=dict(Counter(r['split'] for r in accepted)),encoding_rows=len(manifest),expected_relation_rows=len(relations),
        manual_review_sha256=sha(folder/'manual_reviews.json'),protocol_sha256=sha(folder/'protocol.json'),encoder_invoked=False,
        limitation='all fixed-quality qualifying pairs reported; no minimum identity count or quality relaxation'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE/'controlled_crossview_layouts');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args();cv2.setNumThreads(2)
    finalize(args.output.resolve()) if args.finalize else prepare_all(args.output.resolve())
