"""Scientific annotation safeguards and signed seven-region conservation."""
import numpy as np
from scripts.colnomic_mechanism.refine_token_regions import classify,summaries

SPEC=dict(layout=dict(start=0,count=1,width=1,height=1,total=1),exif_orientation=1,display_width=100,display_height=100)
BOX=dict(bbox=[0,0,100,100],width=100,height=100)


def row(**changes):
    return dict(q_type='visual',q_x='.5',q_y='.5',q_index='0',q_ocr_text='telephone 1800222',
                semantic_overlap_labels='[]',region='key_numbers_specs',delta='.6',**changes)


def test_unreviewed_ocr_digits_are_not_identity_key_numbers():
    region,status,_,labels,_,_,_=classify(row(),BOX,BOX,None,SPEC)
    assert region=='subject_other' and status=='unlocated_semantics_with_OCR_digits'
    assert 'OCR_digits_present_identity_relevance_not_automatically_assumed' in labels


def test_overlapping_manual_semantics_have_one_primary_region():
    manual=dict(rois=[dict(roi_id='layout',primary_region='difference_layout',bbox_xyxy_display_pixels=[0,0,100,100]),
                      dict(roi_id='dose',primary_region='key_numbers_specs',bbox_xyxy_display_pixels=[0,0,40,100])])
    region,status,_,labels,ids,_,_=classify(row(),BOX,BOX,manual,SPEC)
    assert region=='key_numbers_specs' and status=='manual_actual_image_ROI'
    assert set(ids)=={'layout','dose'}
    assert 'overlap_semantic:difference_layout' in labels
    result=summaries([dict(new_region=region,region_status=status,delta=.6)],1)
    assert sum(r['tokens'] for r in result)==1
    assert abs(sum(r['net_contribution'] for r in result)-.6)<1e-15


def test_bbox_revision_changes_membership_without_changing_signed_delta():
    old=dict(bbox=[0,0,20,20],width=100,height=100)
    source=row();region,_,_,_,_,before,after=classify(source,old,BOX,None,SPEC)
    assert not before and after and region=='subject_other'
    assert source['delta']=='.6'
    rs=summaries([dict(new_region='subject_other',region_status='unknown',delta=.6),
                  dict(new_region='background',region_status='box',delta=-.3),
                  dict(new_region='template',region_status='nonvisual',delta=.15)],3)
    assert len(rs)==7 and sum(r['tokens'] for r in rs)==3
    np.testing.assert_allclose(sum(r['net_contribution'] for r in rs),.15,atol=1e-15)
    np.testing.assert_allclose(sum(r['positive_contribution']+r['negative_contribution'] for r in rs),.15,atol=1e-15)
