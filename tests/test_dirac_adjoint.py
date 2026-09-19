"""Acceptance tests for the adjoint's null-space machinery.

The eigensolve's adjoint solves a singular system, and every failure mode here is
silent: deflating the wrong number of directions, or swapping ``P`` for ``Pᵀ``,
gives a plausible gradient that is simply wrong. These gates are what make the
composite finite-difference check interpretable when it eventually fails.
"""

import numpy as np
import pytest

from shapegen.geom import dirac
from shapegen.geom import dirac_adjoint as da
from shapegen.geom.dirac_adjoint import NullProjector, null_basis
from shapegen.geom.operators import edge_cotan_weights
from shapegen.geom.mesh import icosphere
from shapegen.geom.operators import vertex_areas
from shapegen.spectral.quaternion import qmul


def _sphere(sub):
    m = icosphere(sub)
    return m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True), m.faces


def _sh_rho(V, seed=0, lmax=6, amp=2.0):
    from scipy.special import sph_harm_y
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    ph = np.arctan2(V[:, 1], V[:, 0])
    rng = np.random.default_rng(seed)
    r = np.zeros(len(V))
    for l in range(2, lmax + 1):
        for m in range(-l, l + 1):
            r += rng.standard_normal() * np.real(sph_harm_y(l, m, th, ph)) / l ** 2
    return r * (amp / np.abs(r).max())


def _setup(sub=2, seed=0):
    V, F = _sphere(sub)
    rho = _sh_rho(V, seed=seed)
    lam, gamma = dirac.spin_transform(V, F, rho)
    A, M = dirac.dirac_operator(V, F, rho)
    return V, F, rho, lam, gamma, A, M


# --------------------------------------------------------------------------- #
# The basis itself                                                            #
# --------------------------------------------------------------------------- #

def test_null_basis_matches_quaternion_multiplication():
    """The hand-written ``(w,x,y,z)`` table is right multiplication by 1,i,j,k.

    Written out inline for speed, so it must be checked against ``qmul`` rather
    than trusted — a transposed sign here would deflate the wrong subspace and
    nothing downstream would raise.
    """
    rng = np.random.default_rng(0)
    lam = rng.standard_normal((17, 4))
    nb = null_basis(lam)
    for c in range(4):
        unit = np.zeros((1, 4))
        unit[0, c] = 1.0
        expected = qmul(lam, np.repeat(unit, len(lam), axis=0))
        assert np.max(np.abs(nb[c] - expected)) < 1e-14


def test_null_basis_spans_the_kernel():
    """All four of ``λ·{1,i,j,k}`` are annihilated by ``A − γM``.

    This is the claim that makes the null space 4-dimensional in the real 4×4
    embedding. A complex-2N encoding of the same operator sees only 2.
    """
    _, _, _, lam, gamma, A, M = _setup()
    K = (A - gamma * M).tocsr()
    scale = np.abs(A).max()
    nb = null_basis(lam)
    res = [np.linalg.norm(K @ nb[c].ravel()) / scale for c in range(4)]
    print(f"[kernel] residuals={['%.2e' % r for r in res]}  gamma={gamma:.2e}")
    assert max(res) < 1e-8


def test_basis_is_m_orthonormal_by_construction():
    """Gram–Schmidt is a safeguard here, not a construction: the four directions
    are already M-orthonormal by quaternion algebra."""
    _, _, _, lam, _, _, _ = _setup()
    V, F = _sphere(2)
    mass = vertex_areas(V, F)
    proj = NullProjector(lam, mass)
    assert proj.orthonormality_error() < 1e-12

    # ...and without any Gram-Schmidt at all, after only an overall M-normalisation
    raw = null_basis(lam)
    scale = np.sqrt((raw[0] ** 2 * mass[:, None]).sum())
    raw = raw / scale
    G = np.array([[(raw[a] * raw[b] * mass[:, None]).sum() for b in range(4)]
                  for a in range(4)])
    print(f"[orthonormality] no-GS error={np.abs(G - np.eye(4)).max():.2e}")
    assert np.abs(G - np.eye(4)).max() < 1e-12


# --------------------------------------------------------------------------- #
# P vs PT — the distinction the source document gets wrong                    #
# --------------------------------------------------------------------------- #

def test_PT_is_the_adjoint_of_P():
    """``⟨Pᵀp, z⟩ == ⟨p, Pz⟩`` for arbitrary p, z. This is the defining property,
    and the reason the input and output deflations cannot be the same object."""
    V, F = _sphere(2)
    _, _, _, lam, _, _, _ = _setup()
    proj = NullProjector(lam, vertex_areas(V, F))
    rng = np.random.default_rng(3)
    for _ in range(5):
        p = rng.standard_normal(lam.shape)
        z = rng.standard_normal(lam.shape)
        lhs = float((proj.PT(p) * z).sum())
        rhs = float((p * proj.P(z)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-11, abs=1e-11)


def test_P_and_PT_actually_differ():
    """They coincide only when M ∝ I. The lumped vertex areas of an icosphere
    span ~1.5×, so swapping them is a real error, not a cosmetic one — this pins
    that the two are meaningfully different on the meshes we actually use."""
    V, F = _sphere(2)
    _, _, _, lam, _, _, _ = _setup()
    mass = vertex_areas(V, F)
    proj = NullProjector(lam, mass)
    rng = np.random.default_rng(4)
    z = rng.standard_normal(lam.shape)
    rel = np.linalg.norm(proj.P(z) - proj.PT(z)) / np.linalg.norm(proj.P(z))
    print(f"[P vs PT] mass spread={mass.max()/mass.min():.3f}  rel diff={rel:.4f}")
    assert rel > 1e-3


def test_projectors_are_idempotent():
    _, _, _, lam, _, _, _ = _setup()
    V, F = _sphere(2)
    proj = NullProjector(lam, vertex_areas(V, F))
    rng = np.random.default_rng(5)
    z = rng.standard_normal(lam.shape)
    for f in (proj.P, proj.PT):
        once = f(z)
        assert np.max(np.abs(f(once) - once)) < 1e-11


# --------------------------------------------------------------------------- #
# The canary                                                                  #
# --------------------------------------------------------------------------- #

def test_canary_flags_contamination_and_clears_after_P():
    """A pure null vector reads ~1; anything through ``P`` reads ~0."""
    V, F = _sphere(2)
    _, _, _, lam, _, _, _ = _setup()
    proj = NullProjector(lam, vertex_areas(V, F))
    rng = np.random.default_rng(6)

    assert proj.canary(proj.n[0]) == pytest.approx(1.0, abs=1e-10)
    z = rng.standard_normal(lam.shape)
    dirty = z + 5.0 * proj.n[2]
    print(f"[canary] dirty={proj.canary(dirty):.4f}  clean={proj.canary(proj.P(dirty)):.2e}")
    assert proj.canary(dirty) > 0.5
    assert proj.canary(proj.P(dirty)) < 1e-12


# --------------------------------------------------------------------------- #
# Stage adjoints, each gated in isolation before the composite is trusted     #
# --------------------------------------------------------------------------- #

def _gauge_invariant_loss(Y, c):
    """``Σ c_v |Y_v|² / Σ |Y_v|²`` and its gradient.

    Invariant under ``Y → q̄Yq`` (each ``|Y_v|`` is) and under ``Y → sY`` (it is
    homogeneous of degree 0), so it is invariant under the full right-ℍ* gauge —
    which is exactly the condition :func:`stage_a_adjoint` requires of a loss.
    """
    sq = (Y ** 2).sum(1)
    S = sq.sum()
    J = float((c * sq).sum() / S)
    return J, 2.0 * Y * ((c - J) / S)[:, None]


def test_stage_c_adjoint_is_exact():
    """Stage C is linear, so its adjoint is pinned by a dot-product identity —
    no finite differences and no truncation error to hide behind."""
    V, F = _sphere(2)
    edges, _ = edge_cotan_weights(V, F)
    rng = np.random.default_rng(0)
    for _ in range(4):
        et = rng.standard_normal((len(edges), 3))
        dJdX = rng.standard_normal((len(V), 3))
        lhs = float((dJdX * dirac.integrate_transformed_edges(V, F, et)).sum())
        rhs = float((da.stage_c_adjoint(dJdX, V, F) * et).sum())
        assert lhs == pytest.approx(rhs, rel=1e-10)


def test_stage_b_adjoint_matches_fd():
    """Stage B in isolation. The edge rule is bilinear in λ, so FD is exact up to
    round-off and any tolerance here is measuring the derivation, not the step."""
    V, F = _sphere(2)
    edges, _ = edge_cotan_weights(V, F)
    i, j = edges[:, 0], edges[:, 1]
    rng = np.random.default_rng(1)
    lam = rng.standard_normal((len(V), 4))
    dEt = rng.standard_normal((len(edges), 3))

    def J(lm):
        return float((dEt * dirac.transformed_edges(V, lm, i, j)).sum())

    g = da.stage_b_adjoint(dEt, V, lam, i, j)
    h = 1e-6
    gfd = np.empty_like(lam)
    for v in range(len(V)):
        for c in range(4):
            lp, lm_ = lam.copy(), lam.copy()
            lp[v, c] += h
            lm_[v, c] -= h
            gfd[v, c] = (J(lp) - J(lm_)) / (2 * h)
    rel = np.linalg.norm(g - gfd) / np.linalg.norm(gfd)
    print(f"[stage B] rel={rel:.3e}  ratio={float((g*gfd).sum()/(gfd*gfd).sum()):.9f}")
    assert rel < 1e-7
    assert float((g * gfd).sum() / (gfd * gfd).sum()) == pytest.approx(1.0, abs=1e-7)


def test_stage_a_adjoint_matches_fd():
    """Stage A in isolation: ``ρ → λ`` against a gauge-invariant loss on λ."""
    V, F = _sphere(2)
    rho = _sh_rho(V, seed=0)
    rng = np.random.default_rng(2)
    c = rng.standard_normal(len(V))

    def J(r):
        lm, _ = dirac.spin_transform(V, F, r)
        return _gauge_invariant_loss(lm, c)[0]

    lam, gamma = dirac.spin_transform(V, F, rho)
    _, dlam = _gauge_invariant_loss(lam, c)
    stats = {}
    g = da.stage_a_adjoint(dlam, V, F, rho, lam, gamma, stats=stats)

    h = 1e-6
    gfd = np.array([(J(rho + h * np.eye(len(V))[v])
                     - J(rho - h * np.eye(len(V))[v])) / (2 * h)
                    for v in range(len(V))])
    rel = np.linalg.norm(g - gfd) / np.linalg.norm(gfd)
    ratio = float(g @ gfd / (gfd @ gfd))
    print(f"[stage A] rel={rel:.3e}  ratio={ratio:.9f}  canary={stats['canary']:.2e}")
    # 6.5e-8 is where the finite differences run out, not the adjoint; see the
    # shift sweep in stage_a_adjoint's docstring.
    assert rel < 1e-6
    assert ratio == pytest.approx(1.0, abs=1e-6)
    assert stats["canary"] < 1e-12


def test_decode_adjoint_matches_fd():
    """The composite ``dJ/dρ`` for a gauge-invariant shape loss — the gate the
    whole adjoint exists to pass."""
    V, F = _sphere(2)
    rho = _sh_rho(V, seed=0)
    rng = np.random.default_rng(3)
    c = rng.standard_normal(len(V))

    def J(r):
        X, _ = da.decode_forward(V, F, r)
        return _gauge_invariant_loss(X, c)[0]

    X, cache = da.decode_forward(V, F, rho)
    _, dJdX = _gauge_invariant_loss(X, c)
    g = da.decode_adjoint(dJdX, cache)

    h = 1e-6
    gfd = np.array([(J(rho + h * np.eye(len(V))[v])
                     - J(rho - h * np.eye(len(V))[v])) / (2 * h)
                    for v in range(len(V))])
    rel = np.linalg.norm(g - gfd) / np.linalg.norm(gfd)
    ratio = float(g @ gfd / (gfd @ gfd))
    cos = float(g @ gfd / (np.linalg.norm(g) * np.linalg.norm(gfd)))
    print(f"[composite] rel={rel:.3e}  ratio={ratio:.9f}  cos={cos:.10f}")
    assert rel < 1e-6
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_canary_raises_on_undeflated_solve():
    """The canary is load-bearing, so prove it actually fires: an absurd
    tolerance must raise rather than silently return a contaminated gradient."""
    V, F = _sphere(2)
    rho = _sh_rho(V, seed=0)
    lam, gamma = dirac.spin_transform(V, F, rho)
    rng = np.random.default_rng(4)
    dlam = rng.standard_normal(lam.shape)
    with pytest.raises(da.CanaryError):
        da.stage_a_adjoint(dlam, V, F, rho, lam, gamma, canary_tol=-1.0)
