"""Scientific invariants for geometry, frozen probes, and controlled pixel edits."""
import itertools
import numpy as np
import torch

from scripts.colnomic_mechanism.geometry import canonical_polygon, estimate, constrained, grid
from scripts.colnomic_mechanism.interventions import render
from scripts.colnomic_mechanism.probes import shuffled_labels, regression_fit, regression_predict
from scripts.colnomic_mechanism.analyze_controls import sim


def test_control_contributions_keep_raw_reference_indices_after_zero_masking():
    q = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    r = torch.tensor([[0., 0.], [0., 1.], [0., 0.], [1., 0.]])
    score, values, winners = sim(q, r)
    assert score == 1.
    np.testing.assert_array_equal(values, [1., 1.])
    np.testing.assert_array_equal(winners, [3, 1])


def test_polygon_order_and_winding_do_not_change_correspondence():
    polygon = np.array([[.1, .2], [.6, .22], [.598, .26], [.098, .24]], np.float32)
    expected = canonical_polygon(polygon)
    for permutation in itertools.permutations(range(4)):
        np.testing.assert_allclose(canonical_polygon(polygon[list(permutation)]), expected)


def test_homography_recovers_known_transform_without_identity_labels():
    q, r = {}, {}
    h = np.array([[.6, .02, .15], [.01, .7, .08], [0., 0., 1.]])
    for i, (x, y) in enumerate([(.1, .1), (.5, .3), (.1, .6), (.5, .8)]):
        polygon = np.array([[x, y], [x+.2, y], [x+.2, y+.04], [x, y+.04]])
        transformed = (np.c_[polygon, np.ones(4)] @ h.T)[:, :2]
        q[str(i)] = canonical_polygon(polygon[[2, 1, 0, 3]])
        r[str(i)] = canonical_polygon(transformed[[1, 2, 3, 0]])
    fit = estimate(q, r, set(q), np.array([0., 0., 1., 1.]))
    assert fit is not None and fit['inlier_lines'] == 4
    np.testing.assert_allclose(fit['homography'], h, atol=1e-5)


def test_geometry_zero_replays_masked_mean_and_location_constraint_changes_matches():
    # Two visual tokens swap content; one valid template token has no coordinates.
    q = torch.tensor([[1., 0.], [0., 1.], [.5, .5], [0., 0.]])
    r = torch.tensor([[0., 1.], [1., 0.], [.5, .5], [0., 0.]])
    xy = np.array([[.1, .5], [.9, .5], [np.nan, np.nan], [np.nan, np.nan]])
    bonus, penalty, _, count, raw = constrained(q, r, xy, xy, np.array([0., 0., 1., 1.]), np.eye(3), [0., 1.], 'cpu')
    assert count == 2
    np.testing.assert_allclose(raw, (1+1+.5)/3)
    assert bonus[0] == penalty[0] == raw
    assert penalty[1] < raw
    np.testing.assert_allclose(bonus[1]-penalty[1], 2/3)


def test_grid_maps_only_visual_tokens_and_respects_orientation():
    spec = dict(layout=dict(count=6, width=3, height=2, start=2, total=9), exif_orientation=6)
    coords = grid(spec)
    assert np.isnan(coords[[0, 1, 8]]).all()
    np.testing.assert_allclose(coords[2], [.75, 1/6])


def test_pixel_relocation_preserves_foreground_histogram_and_untouched_region():
    a = np.full((18, 50, 3), 250, np.uint8)
    a[2, 2] = [10, 20, 30]  # Unrelated mark outside the edit band.
    bbox = (5, 7, 11, 11)
    mask = np.zeros((4, 6), bool); mask[1:3, 1:5] = True
    source = a[7:11, 5:11]; source[mask] = [42, 73, 19]
    variants = render(a, bbox, mask, [7, 14, 21, 28])
    allowed = np.zeros(a.shape[:2], bool); allowed[7:11, 5:39] = True
    for arm, image in variants.items():
        np.testing.assert_array_equal(image[~allowed], a[~allowed])
        offset = 0 if arm == 'sham' else 7*int(arm[-1])
        np.testing.assert_array_equal(image[7:11, 5+offset:11+offset][mask], source[mask])
    assert np.array_equal(variants['sham'][7:11, 5:11][mask], source[mask])


def test_probe_shuffle_is_within_identity_and_standardization_uses_only_training():
    packages = np.array(['a']*8 + ['b']*8)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1]*2)
    shuffled = shuffled_labels(labels, packages, 17)
    for identity in ['a', 'b']:
        np.testing.assert_array_equal(np.sort(shuffled[packages==identity]), np.sort(labels[packages==identity]))
    assert not np.array_equal(labels, shuffled)
    x = np.arange(16.).reshape(8, 2); y = x[:, :1]*.1
    model = regression_fit(x, y, .01)
    mean = model['mean'].copy(); _ = regression_predict(model, x+1000)
    np.testing.assert_array_equal(model['mean'], mean)
    np.testing.assert_array_equal(mean, x.mean(0))
