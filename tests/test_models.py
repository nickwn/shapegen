"""Acceptance tests for the basis layer and the flow-matching model.

These modules had no coverage: the pipeline was verified by running
``scripts/train_faces.py`` once and reading the output, which checks the whole
chain against real data but pins nothing. The failures that matter here are all
silent — a basis that quietly drops the whitening, a velocity field that ignores
``t`` and collapses to one average direction — so they are gated individually
against distributions whose answer is known in closed form, not against faces.
"""

import numpy as np
import pytest

from shapegen.models.basis import PCABasis, RhoBasis
from shapegen.models.flow import (VelocityField, gaussian_baseline,
                                   sample_flow, train_flow)


def _blobs(n=600, seed=0):
    """Anisotropic 2-D Gaussian: known mean, known per-axis std, correlated.

    Correlated on purpose — an independent-per-axis target would be reproduced by
    a model that had learned nothing but the marginals.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    A = np.array([[2.0, 0.0], [1.2, 0.6]])
    return z @ A.T + np.array([1.0, -3.0])


# --------------------------------------------------------------------------- #
# PCABasis                                                                    #
# --------------------------------------------------------------------------- #

def test_full_rank_round_trip_is_exact():
    """A rank-(N-1) basis spans the centred training set, so encode→decode must
    return it to machine precision. Catches the reshape, the centring and the
    whitening in one shot."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 15, 3))
    b = PCABasis.fit(X, k=19)
    assert np.abs(b.decode(b.encode(X)) - X).max() < 1e-10


def test_rank_is_capped_at_n_minus_one():
    """Asking for more components than a centred N-sample covariance has returns
    directions fitted to nothing — noise that looks like signal."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 40, 3))
    assert PCABasis.fit(X, k=500).k == 11


def test_components_are_nested():
    """``fit(k)`` must be the first k rows of ``fit(K)`` — the dry run slices a
    single fit to build its error-vs-k curve, so this layout is load-bearing."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 12, 3))
    big, small = PCABasis.fit(X, k=10), PCABasis.fit(X, k=4)
    # sign of a singular vector is arbitrary; compare up to it
    for a, b in zip(small.comps, big.comps[:4]):
        assert min(np.abs(a - b).max(), np.abs(a + b).max()) < 1e-8
    assert np.allclose(small.scale, big.scale[:4])


def test_codes_are_whitened():
    """Zero mean and unit variance per component on the training set.

    Not cosmetic: the flow-matching loss is an unweighted MSE in code space, so
    unwhitened codes make it fit whichever coordinates happen to be largest.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 20, 3)) * np.array([5.0, 0.2, 1.0])
    Z = PCABasis.fit(X, k=8).encode(X)
    assert np.abs(Z.mean(0)).max() < 1e-10
    # ddof=1 to match the (N-1) normalisation the basis uses
    assert np.abs(Z.std(0, ddof=1) - 1.0).max() < 1e-10


def test_explained_variance_is_monotone_and_reaches_one():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((25, 10, 3))
    e = PCABasis.fit(X, k=24).explained()
    assert np.all(np.diff(e) >= -1e-12)
    assert e[-1] == pytest.approx(1.0)


def test_truncation_loses_the_smallest_directions_only():
    """Reconstruction error must fall monotonically with k — the property the
    whole error-vs-k curve in the dry run rests on."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, 16, 3))
    errs = [np.abs(PCABasis.fit(X, k=k).decode(PCABasis.fit(X, k=k).encode(X)) - X).sum()
            for k in (2, 4, 8, 16)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


# --------------------------------------------------------------------------- #
# The velocity field                                                          #
# --------------------------------------------------------------------------- #

def test_velocity_field_actually_depends_on_t():
    """The documented failure mode: a raw appended scalar ``t`` is one input
    among K and the network learns to ignore it, collapsing the flow to a single
    average direction. The Fourier embedding is there to prevent that, so an
    untrained field must already separate two times."""
    import torch
    torch.manual_seed(0)
    m = VelocityField(6, width=64, depth=2)
    with torch.no_grad():
        z = torch.randn(32, 6)
        spread = (m(z, torch.zeros(32)) - m(z, torch.ones(32))).abs().mean().item()
    print(f"[t-dependence] mean |v(z,0) - v(z,1)| = {spread:.4f}")
    assert spread > 1e-3


# --------------------------------------------------------------------------- #
# Flow matching, against a distribution whose answer is known                 #
# --------------------------------------------------------------------------- #

def test_flow_reproduces_a_known_gaussian():
    """The end-to-end correctness gate, on 2-D data where the target moments are
    known exactly — so a failure is the model's, not the dataset's.

    Checks the covariance, not just the marginals: the target is correlated, and
    per-axis std alone would pass for a model that learned nothing but marginals.
    """
    X = _blobs(n=800, seed=0)
    model, hist = train_flow(X, steps=4000, batch=256, width=128, depth=3, seed=0)
    S = sample_flow(model, n=4000, k=2, steps=100, seed=0)

    print(f"[flow] mean {X.mean(0).round(3)} -> {S.mean(0).round(3)}")
    print(f"[flow] std  {X.std(0).round(3)} -> {S.std(0).round(3)}")
    assert np.abs(S.mean(0) - X.mean(0)).max() < 0.25
    assert np.abs(S.std(0) / X.std(0) - 1.0).max() < 0.20
    c_true = np.corrcoef(X.T)[0, 1]
    c_gen = np.corrcoef(S.T)[0, 1]
    print(f"[flow] corr {c_true:.3f} -> {c_gen:.3f}")
    assert abs(c_gen - c_true) < 0.15
    assert hist[-500:].mean() < hist[:500].mean()      # it did learn something


def test_flow_samples_are_deterministic_given_a_seed():
    """Sampling must be reproducible, or none of the dry-run checks are."""
    X = _blobs(n=200, seed=1)
    model, _ = train_flow(X, steps=200, batch=64, width=32, depth=2, seed=0)
    a = sample_flow(model, n=16, k=2, steps=10, seed=7)
    b = sample_flow(model, n=16, k=2, steps=10, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, sample_flow(model, n=16, k=2, steps=10, seed=8))


def test_gaussian_baseline_matches_the_data_moments():
    """The baseline the flow has to beat, so it must at least be the thing it
    claims to be."""
    X = _blobs(n=4000, seed=2)
    G = gaussian_baseline(X, n=20000, seed=0)
    assert np.abs(G.mean(0) - X.mean(0)).max() < 0.1
    assert np.abs(G.std(0) / X.std(0) - 1.0).max() < 0.05
    # ...and it is DIAGONAL, which is exactly why it is a weak baseline
    assert abs(np.corrcoef(G.T)[0, 1]) < 0.05


def test_variance_floor_blurs_by_the_amount_it_says():
    """§4(b): ``sigma`` convolves the target with N(0, sigma^2), so the samples
    should be wider than the data by that much and no more.

    The floor exists for near-singular targets, where flow matching is ill-posed
    rather than data-starved. Its cost has to be predictable to be worth paying.
    """
    X = _blobs(n=800, seed=3)
    sig = 0.5
    model, _ = train_flow(X, steps=3000, batch=256, width=128, depth=3,
                          sigma=sig, seed=0)
    S = sample_flow(model, n=4000, k=2, steps=100, seed=0)
    want = np.sqrt(X.std(0) ** 2 + sig ** 2)
    print(f"[floor] data {X.std(0).round(3)} + sigma {sig} -> want "
          f"{want.round(3)}, got {S.std(0).round(3)}")
    assert np.abs(S.std(0) / want - 1.0).max() < 0.25


def test_truncated_sampling_rescales_by_t_end_not_one_minus_t_end():
    """§4(c): the training path is ``(1-t) z0 + t z1``, so stopping at ``t_end``
    shrinks the sample by ``t_end`` and ``rescale`` must divide by that.

    Dividing by ``1 - t_end`` is the natural-looking mistake and inflates samples
    by 1/eps — 50x at eps=0.02. It is silent unless someone checks the std, which
    is what this does. Regression test for a bug that shipped for one commit.
    """
    X = _blobs(n=600, seed=4)
    model, _ = train_flow(X, steps=2500, batch=256, width=128, depth=3, seed=0)
    full = sample_flow(model, n=3000, k=2, steps=100, seed=0)
    for t_end in (0.98, 0.95):
        trunc = sample_flow(model, n=3000, k=2, steps=100, t_end=t_end, seed=0)
        ratio = trunc.std(0) / full.std(0)
        print(f"[t_end={t_end}] std ratio vs full integration {ratio.round(3)}")
        assert np.abs(ratio - 1.0).max() < 0.15
    raw = sample_flow(model, n=3000, k=2, steps=100, t_end=0.9, rescale=False)
    # unrescaled is shrunk by t_end, which is what rescale undoes
    assert np.abs(raw.std(0) / (0.9 * full.std(0)) - 1.0).max() < 0.15


# --------------------------------------------------------------------------- #
# RhoBasis                                                                    #
# --------------------------------------------------------------------------- #

def _bottles(n=6, subdiv=2):
    from shapegen import codec
    from shapegen.data.dataset import demo_bottle_meshes
    ms = demo_bottle_meshes(n=n, subdiv=subdiv, seed=0)
    return (np.stack([m.vertices for m in ms]), ms[0].faces,
            codec.template_sphere(subdiv))


def test_rho_codes_are_whitened_band_wise():
    """Each BAND must pool to unit variance — the invariant RhoBasis actually
    provides, and the one the flow's unweighted MSE needs.

    Regression test for a real bug: the scale was pooled over *uncentred*
    coefficients, which folds the spread of the per-coefficient means into it,
    over-estimates it, and left per-band std at 0.25 instead of 1. The flow then
    under-weights those bands for precisely the reason whitening exists. Unit std
    per *coefficient* is NOT the invariant — within a band the spread is signal.
    """
    from shapegen import codec
    X, F, S = _bottles()
    L = 6
    b = RhoBasis.fit(X, F, L, S)
    Z = b.encode(X)
    bi = codec.band_index(L)
    per_band = np.array([Z[:, bi == l].std() for l in range(L + 1)])
    print(f"[rho whitening] per-band std in "
          f"[{per_band.min():.4f}, {per_band.max():.4f}]")
    assert np.abs(Z.mean(0)).max() < 1e-10
    assert np.abs(per_band - 1.0).max() < 1e-10


def test_rho_fields_are_bandwidth_independent():
    """ρ_direct does not depend on L, so one field set serves every bandwidth.

    That is what makes an L-sweep affordable — otherwise each bandwidth repeats
    the per-mesh edge least squares, which is the expensive part.
    """
    X, F, S = _bottles()
    fields = RhoBasis(S, F, L=1).rho_fields(X)
    for L in (4, 8):
        a = RhoBasis.fit(X, F, L, S, fields=fields)
        b = RhoBasis.fit(X, F, L, S)
        assert np.allclose(a.encode(X, fields=fields), b.encode(X), atol=1e-12)


def test_rho_fields_refuse_the_adjoint():
    """The adjoint's ρ *is* fitted at a specific bandwidth, so reusing it across L
    would be silently wrong. Refuse rather than return something plausible."""
    X, F, S = _bottles(n=2)
    with pytest.raises(ValueError, match="(?i)direct"):
        RhoBasis(S, F, L=4, method="adjoint").rho_fields(X)


def test_rho_is_rotation_invariant():
    """ρ̂ on a shared template does not see rotation: H is rotation-invariant and
    e^u comes from edge lengths. This contradicts the warning in codec.py, which
    applies to unregistered data where each shape gets its own sphere — so it is
    worth pinning before anything relies on it."""
    X, F, S = _bottles(n=3)
    b = RhoBasis.fit(X, F, 6, S)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    z0, z1 = b.encode(X), b.encode(X @ Q)
    rel = np.linalg.norm(z0 - z1) / np.linalg.norm(z0)
    print(f"[rho rotation] rel change under a random rotation {rel:.2e}")
    assert rel < 1e-9
