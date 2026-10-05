"""Acceptance tests for the spectral core (SWSH + ð ladder), restored for
the quad-mesh port.

Backend: self-contained NumPy/SciPy (the Wigner small-d sum formula), zero
heavy deps. Independent oracle: the ``spherical`` package (Wigner / sYlm),
cross-checked against scipy ``sph_harm_y`` at spin 0.

Covers: SWSH round trip s∈{0,2,4} (tol 1e-10); ð ladder, Δ_s, q=ð²σ scaling
(tol 1e-9).
"""

import numpy as np
import pytest
import quaternionic
import spherical
from scipy.special import sph_harm_y

from shapegen.spectral import swsh
from shapegen.spectral.grids import gl_grid, solid_angle_weights

L = 16
SPINS = (0, 2, 4)


def _rand_coeffs(s, L, seed=0):
    """Random valid spin-s coefficients (zero where ℓ < |s| or |m| > ℓ)."""
    rng = np.random.default_rng(seed)
    C = swsh.empty_coeffs(L)
    for l in range(abs(s), L + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = rng.standard_normal() + 1j * rng.standard_normal()
    return C


def _oracle_sYlm(s, L):
    """Spin-s harmonics from `spherical` on gl_grid(L): dict (ℓ,m) -> (Nθ,Nφ)."""
    grid = gl_grid(L)
    TH, PH = np.meshgrid(grid.theta, grid.phi, indexing="ij")
    R = quaternionic.array.from_spherical_coordinates(TH.ravel(), PH.ravel())
    Y = np.asarray(spherical.Wigner(L).sYlm(s, R)).reshape(TH.shape + (-1,))
    out = {}
    for l in range(abs(s), L + 1):
        for m in range(-l, l + 1):
            out[(l, m)] = Y[..., spherical.Yindex(l, m)]
    return out


# --------------------------------------------------------------------------- #
# Test 1: round trip analy(synth(c,s),s) ≈ c                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", SPINS)
def test_round_trip(s):
    C = _rand_coeffs(s, L, seed=s)
    C2 = swsh.analy(swsh.synth(C, s, L), s, L)
    err = np.max(np.abs(C2 - C))
    print(f"[round trip] s={s}: max err = {err:.3e}")
    assert err < 1e-10


# --------------------------------------------------------------------------- #
# Oracle: synthesised field matches `spherical` sYlm reconstruction           #
# (validates the s2fft backend's convention end-to-end, every spin)           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", SPINS)
def test_synth_matches_spherical_oracle(s):
    C = _rand_coeffs(s, L, seed=100 + s)
    field = swsh.synth(C, s, L)
    Y = _oracle_sYlm(s, L)
    ref = np.zeros_like(field)
    for l in range(abs(s), L + 1):
        for m in range(-l, l + 1):
            ref += C[l, l + m] * Y[(l, m)]
    err = np.max(np.abs(field - ref))
    print(f"[oracle synth] s={s}: max err = {err:.3e}")
    assert err < 1e-10


@pytest.mark.parametrize("s", SPINS)
def test_single_harmonic_matches_oracle(s):
    Y = _oracle_sYlm(s, L)
    for (l, m) in [(max(2, abs(s)), 0), (abs(s) + 1, 1),
                   (L, -2), (L - 1, L - 1)]:
        if l < abs(s) or abs(m) > l:
            continue
        ours = swsh.swsh(s, l, m, L)
        assert np.max(np.abs(ours - Y[(l, m)])) < 1e-10


# --------------------------------------------------------------------------- #
# Anchor: spin 0 also matches scipy spherical harmonics                       #
# --------------------------------------------------------------------------- #

def test_spin0_matches_scipy():
    grid = gl_grid(L)
    th, ph = grid.theta[:, None], grid.phi[None, :]
    worst = 0.0
    for l in range(0, L + 1):
        for m in range(-l, l + 1):
            worst = max(worst, np.max(np.abs(swsh.swsh(0, l, m, L)
                                             - sph_harm_y(l, m, th, ph))))
    print(f"[s=0 vs scipy] max err = {worst:.3e}")
    assert worst < 1e-10


# --------------------------------------------------------------------------- #
# Orthonormality on the GL grid via the supplied quadrature weights           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", SPINS)
def test_orthonormal(s):
    grid = gl_grid(L)
    w = solid_angle_weights(grid)
    modes = [(l, m) for l in range(abs(s), L + 1) for m in range(-l, l + 1)]
    checks = [(modes[0], modes[0]), (modes[1], modes[1]),
              (modes[0], modes[1]), (modes[-1], modes[-1]),
              (modes[2], modes[-3])]
    for (l1, m1), (l2, m2) in checks:
        ip = np.sum(w * swsh.swsh(s, l1, m1, L) * np.conj(swsh.swsh(s, l2, m2, L)))
        expected = 1.0 if (l1, m1) == (l2, m2) else 0.0
        assert abs(ip - expected) < 1e-9


# --------------------------------------------------------------------------- #
# Test 2a: ð / ð̄ ladder factors                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", SPINS)
def test_ladder_factors(s):
    C = _rand_coeffs(s, L, seed=10 + s)
    up = swsh.edth(C, s)
    dn = swsh.edthbar(C, s)
    for l in range(L + 1):
        fu = np.sqrt(max((l - s) * (l + s + 1), 0))
        fd = -np.sqrt(max((l + s) * (l - s + 1), 0))
        assert np.max(np.abs(up[l] - fu * C[l])) < 1e-9
        assert np.max(np.abs(dn[l] - fd * C[l])) < 1e-9


# --------------------------------------------------------------------------- #
# Test 2b: spin-s Laplacian eigenvalues  ½(ðð̄+ð̄ð) = −[ℓ(ℓ+1)−s²]              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", SPINS)
def test_laplacian_eigenvalues(s):
    C = _rand_coeffs(s, L, seed=20 + s)
    lap = swsh.spin_laplacian(C, s)
    eig = swsh.laplacian_eig(L, s)
    for l in range(abs(s), L + 1):
        assert np.max(np.abs(lap[l] - eig[l] * C[l])) < 1e-9


# --------------------------------------------------------------------------- #
# Test 2c: Hopf scaling q = ð²σ = √((ℓ−1)ℓ(ℓ+1)(ℓ+2)) σ̂                       #
# --------------------------------------------------------------------------- #

def test_hopf_scaling():
    sigma = _rand_coeffs(0, L, seed=99)
    q = swsh.q_from_sigma(sigma)
    scale = swsh.hopf_scaling(L)
    for l in range(L + 1):
        assert np.max(np.abs(q[l] - scale[l] * sigma[l])) < 1e-9
    assert np.allclose(q[0], 0) and np.allclose(q[1], 0)


# --------------------------------------------------------------------------- #
# Consistency: ð maps to a genuine spin-(s+1) field (round-trips at s+1)        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", (0, 2))
def test_edth_produces_valid_higher_spin_field(s):
    C = _rand_coeffs(s, L, seed=30 + s)
    Cup = swsh.edth(C, s)
    back = swsh.analy(swsh.synth(Cup, s + 1, L), s + 1, L)
    assert np.max(np.abs(back - Cup)) < 1e-10


if __name__ == "__main__":
    for s in SPINS:
        C = _rand_coeffs(s, L, seed=s)
        C2 = swsh.analy(swsh.synth(C, s, L), s, L)
        print(f"spin {s}: max round-trip error = {np.max(np.abs(C2 - C)):.3e}")
