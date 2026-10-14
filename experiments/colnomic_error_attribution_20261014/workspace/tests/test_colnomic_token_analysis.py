"""Guard score conservation and the EXIF frame used by token diagnostics."""
import numpy as np
import torch
from PIL import Image, ImageOps

from scripts.analyze_colnomic_tokens import display_xy, trace_maxsim
from evaluation.scoring import multivector_scores
from scripts.report_colnomic_tokens import focal_lines, weight_diagnostic
from scripts.analyze_colnomic_tokens import primary_predictions, successful_control
from scripts.report_colnomic_b03_tokens import diagnosis
from scripts.analyze_colnomic_tokens import write_csv
import csv


def test_trace_matches_baseline_and_conserves_positive_negative_gap():
    q=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    p=torch.tensor([[0.,0.],[1.,0.],[0.,.6]])
    n=torch.tensor([[0.,.8],[.8,0.],[0.,0.]])
    qi,pm,pc=trace_maxsim(q,p)
    qi2,nm,nc=trace_maxsim(q,n)
    assert qi.tolist()==qi2.tolist()==[1,2,3]
    assert pm.tolist()==[1,2,1]
    assert nm.tolist()==[1,0,0]  # Stable first-token tie breaking.
    baseline,_=multivector_scores([q],[p,n],device='cpu',token_block=2)
    np.testing.assert_allclose([pc.mean(),nc.mean()],baseline[0].numpy(),atol=1e-7)
    delta=pc-nc
    np.testing.assert_allclose(delta.sum()/len(qi),(baseline[0,0]-baseline[0,1]).item(),atol=1e-7)
    positive=np.maximum(delta,0).sum()/len(delta)
    negative=np.minimum(delta,0).sum()/len(delta)
    np.testing.assert_allclose(positive+negative,delta.mean(),atol=1e-7)


def test_normalized_token_coordinates_follow_all_eight_exif_orientations():
    width,height=4,3
    pixels=np.arange(width*height,dtype=np.uint8).reshape(height,width)
    for orientation in range(1,9):
        im=Image.fromarray(pixels);im.getexif()[274]=orientation
        expected=np.asarray(ImageOps.exif_transpose(im))
        for row in range(height):
            for col in range(width):
                xy=display_xy([[(col+.5)/width,(row+.5)/height]],orientation)[0]
                c=int(xy[0]*expected.shape[1]);r=int(xy[1]*expected.shape[0])
                assert expected[r,c]==pixels[row,col]


def test_focal_region_keeps_small_line_despite_neighboring_ocr_assignment(tmp_path):
    rows=[dict(q_token_index='4',q_x_display='.375',q_y_display='.375',delta='.2'),
          dict(q_token_index='5',q_x_display='.625',q_y_display='.375',delta='-.1')]
    summary=dict(query_raw_grid=dict(width=4,height=4),query_exif_orientation=1,valid_tokens=17)
    ocr=dict(width=100,height=100,lines=[dict(text='Consult for details.',
        polygon=[[30,34],[48,34],[48,36],[30,36]])])
    focal=focal_lines(tmp_path,summary,rows,ocr,dict(bbox=[0,0,100,100]))
    assert focal[0]['n']==1
    assert focal[0]['token_indices']=='[4]'
    np.testing.assert_allclose(focal[0]['contribution_to_full_mean_gap'],.2/17)


def test_fusion_weight_threshold_has_correct_direction():
    condition,threshold=weight_diagnostic(dict(image_z_contribution='.6',bm25_z_contribution='-1'))
    assert 'w >' in condition
    np.testing.assert_allclose(threshold,.625)
    condition,threshold=weight_diagnostic(dict(image_z_contribution='-1',bm25_z_contribution='.6'))
    assert 'w <' in condition
    np.testing.assert_allclose(threshold,.375)


def test_b03_uses_image_winner_when_fusion_selects_a_different_reference():
    image={'q':dict(selected_reference_id='image_winner',correct=False)}
    fusion={'q':dict(selected_reference_id='fusion_winner',correct=False)}
    assert primary_predictions('B03',image,fusion)['q']['selected_reference_id']=='image_winner'
    assert primary_predictions('F03',image,fusion)['q']['selected_reference_id']=='fusion_winner'


def test_b03_control_does_not_require_a_correct_fusion_result():
    assert successful_control('B03',dict(correct=True),dict(correct=False))
    assert not successful_control('F03',dict(correct=True),dict(correct=False))
    assert not successful_control('B03',dict(correct=False),dict(correct=True))


def test_b03_diagnosis_distinguishes_ties_missing_foreground_and_pair_flips():
    assert '完全同分' in diagnosis(dict(image_gap=0.,subject_only_gap=.1))
    assert '没有' in diagnosis(dict(image_gap=-.1,subject_only_gap=None))
    assert '翻转' in diagnosis(dict(image_gap=-.1,subject_only_gap=.1))
    assert '身份歧义' in diagnosis(dict(image_gap=-.1,subject_only_gap=.1),True)


def test_summary_csv_keeps_provenance_only_present_on_reused_contrasts(tmp_path):
    path=tmp_path/'summary.csv'
    write_csv(path,[dict(photo_id='fresh'),dict(photo_id='reused',reused_from='prior/result')])
    with path.open() as f:rows=list(csv.DictReader(f))
    assert rows[0]['reused_from']==''
    assert rows[1]['reused_from']=='prior/result'
