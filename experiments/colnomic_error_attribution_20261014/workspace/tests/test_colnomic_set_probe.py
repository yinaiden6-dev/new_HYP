"""Independent invariance and float32 pair-column checks for nonlinear set probes."""
import numpy as np
from scripts.colnomic_mechanism.set_probes import kernel_summary, gallery_score, pair_features, pair_moment_features
from scripts.colnomic_mechanism.probes import logistic_predict


def test_set_kernel_retains_distribution_beyond_identical_moments():
    a = np.array([-np.sqrt(2), 0, 0, np.sqrt(2)], dtype=np.float32)[:, None]
    b = np.array([-1, -1, 1, 1], dtype=np.float32)[:, None]
    kernel = {'landmarks': np.array([[-.7], [.8]], dtype=np.float32), 'sigma2': np.array(.2)}
    np.testing.assert_allclose(a.mean(0), b.mean(0), atol=1e-7)
    np.testing.assert_allclose(a.std(0), b.std(0), atol=1e-7)
    np.testing.assert_allclose(kernel_summary(a, kernel), kernel_summary(a[::-1], kernel))
    assert np.max(np.abs(kernel_summary(a, kernel) - kernel_summary(b, kernel))) > .05


def test_gallery_vectorization_replays_float32_pair_products_and_columns():
    rng = np.random.default_rng(113)
    qpool, rpool = rng.normal(size=256).astype(np.float32), rng.normal(size=(7, 256)).astype(np.float32)
    qset, rset = rng.normal(size=64).astype(np.float32), rng.normal(size=(7, 64)).astype(np.float32)
    scalar = rng.normal(size=7).astype(np.float32)
    pool = pair_moment_features(np.repeat(qpool[None], 7, axis=0), rpool, scalar)
    inputs = {'maxsim_only': scalar[:, None], 'pooled_only': pool[:, :-1], 'pooled_plus_maxsim': pool,
              'kernel_set_only': pair_features(np.repeat(qset[None], 7, axis=0), rset)}
    for kind, values in inputs.items():
        dim = values.shape[1]
        model = {'mean': rng.normal(size=dim), 'scale': np.exp(rng.normal(size=dim)), 'coefficient': rng.normal(size=dim+1)}
        expected = logistic_predict(model, values)
        actual = gallery_score(kind, model, qpool, rpool, qset, rset, scalar)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
