import importlib.util
import tempfile
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location('visual', Path(__file__).parents[1]/'scripts/colnomic_mechanism/consolidate_visual_audit.py')
v=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(v)


class VisualConsolidationTests(unittest.TestCase):
    def test_package_only_count_is_not_scene_background(self):
        result=v.scene_values(Path('/tokens_resume/batches_141_144.json'),
            dict(object_count='single',capture='handheld',foreground_occlusion='absent'))
        self.assertNotIn('background',result)
        self.assertEqual(result,dict(placement='handheld',occlusion='none'))

    def test_actual_scene_recheck_wins_over_package_only_count(self):
        original=Path('/tokens_resume/batches_141_144.json')
        revised=Path('/tokens_resume/scene_rechecks_141_156.json')
        self.assertGreater(v.source_priority(revised),v.source_priority(original))
        self.assertEqual(v.scene_values(revised,dict(background='multi_object')),dict(background='multi_object'))

    def test_uncertainty_does_not_discard_known_independent_dimensions(self):
        record=dict(scene_labels=dict(background='unclear',placement='handheld',occlusion='foreground'),scene_labels_reviewed=False)
        self.assertEqual(v.scene_values(Path('/visual_audit/revisions/q.r1.json'),record),record['scene_labels'])

    def test_material_manifest_without_observation_is_not_review(self):
        record=dict(photo_id='q',original_path='prepared.jpg',candidate_paths=['right.jpg','wrong.jpg'])
        self.assertIsNone(v.normalize(Path('/prepared/manifest.json'),record,{},Path('/tmp'),Path('/tmp'),{},{}))

    def test_changed_visual_image_fails_instead_of_claiming_viewed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);image=root/'image.jpg';image.write_bytes(b'actual-image')
            source=root/'review.json';source.write_text('{}')
            with self.assertRaisesRegex(ValueError,'mismatched'):
                v.normalize(source,dict(photo_id='q',direct_observation='actual observation',
                    evidence_image='image.jpg',evidence_sha256='wrong-hash'),{},root,root,{}, {})

    def test_uncertain_face_never_matches_another_face_control(self):
        self.assertIsNone(v.face_key('unclear side or back'))
        self.assertIsNone(v.face_key('主正面可能不完整'))
        self.assertNotEqual(v.face_key('front'),v.face_key('front_and_top'))

    def test_quality_panel_crop_comparison_is_not_observed_truncation(self):
        record=dict(observable_difficulties=[],quality_observations=['crop is sharper','no blur observed'])
        v.assign_difficulty_labels(record)
        self.assertEqual(record['difficulty_labels'],['unclear_or_no_explicit_difficulty_tag'])

    def test_quality_dict_negative_unknown_and_zero_values_do_not_promote_keys(self):
        record=dict(observable_difficulties=[],quality_observations=[dict(reflection='none',
            perspective='unknown',rotation=0,severe_occlusion='not_observed',blur=False)])
        v.assign_difficulty_labels(record)
        self.assertEqual(record['quality_difficulty_labels'],[])

    def test_affirmative_dict_quality_is_kept_independently_of_negative_field(self):
        record=dict(observable_difficulties=[],quality_observations=[dict(perspective='moderate',
            reflection='none',subject_scale='small_in_original')])
        v.assign_difficulty_labels(record)
        self.assertIn('perspective_rotation_or_deformation',record['quality_difficulty_labels'])
        self.assertIn('blur_low_resolution_or_small',record['quality_difficulty_labels'])
        self.assertNotIn('reflection_or_low_contrast',record['quality_difficulty_labels'])

    def test_key_region_none_does_not_change_foreground_scene(self):
        r=dict(review_history=[dict(source_path='actual.json',source_sha256='h',scope='selected_fine_details_only',
            direct_observations=['thumb covers public text'],evidence=[],raw_record={'identity_region_obscured':{'status':'none','position':'public comparison line','degree':'local'}})],
            scene_labels={'occlusion':'foreground'})
        v.enrich_detail_fields(r)
        self.assertEqual(r['identity_region_obscured'],'none')
        self.assertEqual(r['scene_labels']['occlusion'],'foreground')


if __name__=='__main__':
    unittest.main()
