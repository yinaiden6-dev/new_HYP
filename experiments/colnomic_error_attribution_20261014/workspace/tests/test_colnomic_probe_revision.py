"""Metrics and provenance must support the stated experimental interpretation."""
import numpy as np
import pytest
from PIL import Image,ImageOps
from scripts.colnomic_mechanism.probes import classification_metrics,paired_metrics,validation_threshold,pooled,feature_pair
from scripts.colnomic_mechanism.common import verify_split_records
from scripts.colnomic_mechanism.extended_layouts import to_raw,render_blocks
import torch


def test_half_threshold_accuracy_can_have_perfect_within_query_ordering():
    y=np.array([1,0]);scores=np.array([1.,.5]);packages=np.array(['a','a'])
    metrics=classification_metrics(y,scores,packages)
    pair,rows=paired_metrics(y,scores,['q','q'],packages)
    assert metrics['accuracy']==.5 and metrics['auc']==1.
    assert pair['pair_accuracy']==1. and rows[0]['gap']==.5
    threshold=validation_threshold(y,scores,packages)
    assert classification_metrics(y,scores,packages,threshold)['accuracy']==1.


def test_pairs_do_not_compare_candidates_from_unrelated_queries():
    y=np.array([1,0,1,0]);scores=np.array([10.,9.,1.,0.]);packages=['a','a','b','b']
    assert classification_metrics(y,scores,packages)['auc']==.75
    metric,rows=paired_metrics(y,scores,['qa','qa','qb','qb'],packages)
    assert metric['pair_accuracy']==1. and metric['pairs']==2
    assert all(r['gap']==1. for r in rows)
    with pytest.raises(AssertionError):paired_metrics(y,scores,['q','q','q','q'],packages)


def fixture_records():
    photos=[dict(photo_id=s,package_id=s,design_id='d'+s,identity='i'+s,sha256='h'+s,source_sha256='s'+s) for s in ['a','b','c']]
    splits={s:[r] for s,r in zip(['train','val','test'],photos)}
    groups=[dict(package_id=r['package_id'],group_id='g'+r['package_id'],split=s) for s,r in zip(['train','val','test'],photos)]
    return photos,splits,groups


def test_split_report_is_built_from_actual_sets_and_detects_sha_and_family_overlap():
    photos,splits,groups=fixture_records()
    result=verify_split_records(photos,splits,groups)
    assert result['status']=='passed' and result['checks']['group_id']['intersections']['train__test']==[]
    photos[1]['sha256']=photos[0]['sha256']
    with pytest.raises(AssertionError,match='sha256'):verify_split_records(photos,splits,groups)
    photos,splits,groups=fixture_records();groups[1]['group_id']=groups[0]['group_id']
    with pytest.raises(AssertionError,match='group_id'):verify_split_records(photos,splits,groups)


def test_summaries_and_maxsim_do_not_read_encoded_token_order():
    z=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    spec=dict(layout=dict(start=0,count=4));a=pooled(z,spec);b=pooled(z[[2,0,3,1]],spec)
    np.testing.assert_array_equal(feature_pair(a,a,1.),feature_pair(b,a,1.))
    assert float((z@z.T).max(1).values.mean())==float((z[[2,0,3,1]]@z.T).max(1).values.mean())


def test_inverse_exif_preserves_original_pixels_for_all_orientations():
    raw=np.arange(3*4*3,dtype=np.uint8).reshape(3,4,3)
    for orientation in range(1,9):
        im=Image.fromarray(raw);im.getexif()[274]=orientation
        displayed=np.asarray(ImageOps.exif_transpose(im))
        np.testing.assert_array_equal(to_raw(displayed,orientation),raw)


def test_full_glyph_block_and_editor_control_conserve_the_word_layout():
    a=np.full((16,48,3),190,np.uint8);a[6:10,5:11]=[20,40,70]
    box=(4,5,12,11);variants=render_blocks(a,box,np.array([190,190,190],np.uint8),[4,12])
    block=a[5:11,4:12]
    np.testing.assert_array_equal(variants['shift_1'][5:11,8:16],block)
    np.testing.assert_array_equal(variants['shift_2'][5:11,16:24],block)
    for arm in ['editor_1','editor_2']:
        np.testing.assert_array_equal(variants[arm][5:11,4:12],block)
    for value in variants.values():
        np.testing.assert_array_equal(value[:5],a[:5])
        np.testing.assert_array_equal(value[11:],a[11:])
