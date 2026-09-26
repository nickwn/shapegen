"""PLAN.md §P7 route 2: the spinor-to-quaternion dictionary and the
mesh-free lift built on it (:mod:`shapegen.quad.lift`).

The dictionary is the gate — and the round sphere alone cannot pin it (see
``test_kernel_gauge_is_a_right_constant``). ``geom/dirac_spectral`` returns ``λ`` as spin-±½
halves; every one of these tests exists because building an immersion on an
unverified spinor-to-quaternion map is how a wrong convention becomes a silent
geometric error (PLAN.md §2.4).
"""

from __future__ import annotations

import numpy as np
import pytest

from shapegen import codec
from shapegen.geom import dirac, dirac_spectral as ds
from shapegen.geom.encode import n_coeffs, real_sh_basis
from shapegen.geom.mesh import icosphere
from shapegen.quad import lift as ql
from shapegen.spectral.quaternion import qinv, qmul, qnorm

from tests.quad_fixtures import bundle_style_coeffs

L = 8


def _sphere(subdiv=3):
    m = icosphere(subdiv)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    return V, m.faces


def _edges(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def _scatter(q):
    """Median distance from the mean on sign-aligned unit quaternions.
    A random field scores ~0.91; a constant field scores 0."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    s = np.sign((q * q[0]).sum(1))
    s[s == 0] = 1.0
    qs = q * s[:, None]
    mu = qs.mean(0)
    mu /= np.linalg.norm(mu)
    return float(np.median(np.linalg.norm(qs - mu, axis=1)))


# --------------------------------------------------------------------------- #
# The dictionary                                                              #
# --------------------------------------------------------------------------- #

def test_frame_quaternion_carries_i_to_the_normal():
    """``R = q_z(φ) q_y(θ − π/2)`` is unit and rotates the quaternion **i**
    axis — the one ``ℍ = ℂ ⊕ ℂj`` quantises spin about — onto the outward
    normal, which is what makes left multiplication by it a frame."""
    from shapegen.spectral.quaternion import from_vec3, qconj, to_vec3
    rng = np.random.default_rng(0)
    u = rng.normal(size=(500, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    th = np.arccos(np.clip(u[:, 2], -1, 1))
    ph = np.arctan2(u[:, 1], u[:, 0])
    R = ql.frame_quaternion(th, ph)
    assert R.shape == (500, 4)
    assert np.allclose(qnorm(R), 1.0, atol=1e-13)
    i = from_vec3(np.tile([1.0, 0.0, 0.0], (500, 1)))
    assert np.abs(to_vec3(qmul(qmul(R, i), qconj(R))) - u).max() < 1e-12


def test_dictionary_is_exact_on_the_round_sphere():
    """At γ = 0 the spinor is constant up to the right gauge, and agrees with
    the mesh solver's ``λ ≡ 1``. Necessary, but **not sufficient**: any
    ``A(x)·raw·B(x)`` with ``A raw₀ B = 1`` passes here, which is how a wrong
    dictionary survived until 2026-09-26. The two tests below are the ones
    that can tell."""
    V, faces = _sphere(3)
    lam_mesh, _ = dirac.spin_transform(V, faces, np.zeros(len(V)))
    res = ds.solve_spinor(np.zeros(n_coeffs(L)), L)
    assert res.gamma < 1e-12
    assert res.multiplicity == 2
    lam = ql.spinor_quaternion(res, V)
    assert qnorm(lam).std() / qnorm(lam).mean() < 1e-12
    assert _scatter(qmul(qinv(lam), lam_mesh)) < 1e-10


def test_kernel_gauge_is_a_right_constant():
    """The spectral kernel is two complex dimensions — one quaternionic one —
    and under the dictionary it must act on ``λ`` as ``λ → λq``: a *right*
    constant, i.e. a rigid motion of the output. A *left* constant rotates
    the base sphere under a fixed ``λ`` and is a different surface.

    The pre-2026-09-26 map (``F`` on the right, no conjugation) failed exactly
    this, mirrored: left 1e-14, right 0.185."""
    import scipy.linalg as sla
    V, _faces = _sphere(3)
    rho_hat = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    Ls = ds.spinor_band(L)
    G = ds.dirac_matrix(ds.rho_on_grid(rho_hat, L, Ls), Ls, L)
    n = G.shape[0] // 2
    _w, vecs = sla.eigh(G.conj().T @ G, subset_by_index=[0, 1])
    l1, l2 = (ql.spinor_quaternion(
        ds.DiracSpectral(v[:n], v[n:], 0.0, 0.0, 2, Ls, L, 1.0), V)
        for v in vecs.T)
    assert _scatter(qmul(qinv(l1), l2)) < 1e-10          # right: constant
    assert _scatter(qmul(l2, qinv(l1))) > 0.05           # left: not


def test_dictionary_agrees_with_the_mesh_solver_off_the_sphere():
    """Away from γ = 0, ``λ_spec⁻¹ λ_mesh`` is constant to the mesh solver's
    own discretisation error — it falls with subdivision — at production-like
    amplitude. Measured (L=8, ``ℓ^-2``): 2.3e-4 → 1.9e-3 over amp 0.25 → 2 at
    subdivision 3, and 1.6e-3 at amp 2 on subdivision 4.

    The old map scored 0.014 → 0.080 here. It was guarded by a test that
    compared ``λ_mesh λ_spec⁻¹`` — the **left** side, the wrong gauge — where
    old and new score the same ~0.10, and that residual was read as the two
    solvers legitimately disagreeing."""
    rho_hat = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * 2.0
    res = ds.solve_spinor(rho_hat, L)
    out = []
    for subdiv in (3, 4):
        V, faces = _sphere(subdiv)
        lam_mesh, _ = dirac.spin_transform(V, faces, real_sh_basis(V, L) @ rho_hat)
        out.append(_scatter(qmul(qinv(ql.spinor_quaternion(res, V)), lam_mesh)))
    assert out[0] < 5e-3, out
    assert out[1] < out[0], out


def test_spinor_quaternion_is_single_valued():
    """``half_eval_basis`` is antiperiodic in φ and so is ``F``; the product
    is not. Checked across the φ = ±π cut and at both poles, where φ is
    undefined."""
    res = ds.solve_spinor(np.zeros(n_coeffs(L)), L)
    th = np.linspace(0.05, np.pi - 0.05, 64)
    eps = 1e-9

    def at(phi):
        return ql.spinor_quaternion(res, np.stack(
            [np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)], 1))

    assert np.abs(at(np.pi - eps) - at(-np.pi + eps)).max() < 1e-10

    for z in (+1.0, -1.0):
        t = 1e-7
        ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        d = np.stack([np.sin(t) * np.cos(ang), np.sin(t) * np.sin(ang),
                      np.full_like(ang, z * np.cos(t))], 1)
        q = ql.spinor_quaternion(res, d)
        assert np.abs(q - q.mean(0)).max() < 1e-9


# --------------------------------------------------------------------------- #
# The lift                                                                    #
# --------------------------------------------------------------------------- #

def test_integrate_edges_lsq_is_exact_on_a_consistent_field():
    """PLAN.md §1: the global least-squares solve is *exact*, not O(h²).
    On an integrable edge field it must return the generating positions."""
    V, faces = _sphere(2)
    E = _edges(faces)
    et = V[E[:, 1]] - V[E[:, 0]]              # exactly integrable by construction
    X, G = ql.integrate_edges_lsq(len(V), E, et)
    assert np.abs(G @ X - et).max() < 1e-10
    assert codec.shape_error(V, X) < 1e-10


def test_route2_reconstructs_the_round_sphere():
    """ρ = 0 ⇒ λ is constant ⇒ df̃ is a rotation of df₀ ⇒ the sphere back."""
    V, faces = _sphere(3)
    E = _edges(faces)
    P3, info = ql.lift_route2(V, E, np.zeros(n_coeffs(L)), L, faces=faces)
    assert info["gamma"] < 1e-12
    assert codec.shape_error(V, P3) < 1e-7
    assert info["edge_residual"] < 1e-7
    assert info["holonomy_rel"] < 1e-7      # Gate P7.A at γ = 0


def test_p7a_holonomy_grows_with_gamma():
    """**Gate P7.A.** ``df̃`` is closed only at γ = 0; the obstruction must be
    ~0 there and grow with γ. Measured before the least-squares solve, which
    would otherwise absorb it silently — and the post-solve edge residual is
    checked to grow too, so a solve that *did* hide it would fail here."""
    V, faces = _sphere(3)
    E = _edges(faces)
    base = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    hol, resid, gam = [], [], []
    for amp in (0.0, 0.5, 2.0):
        _P, info = ql.lift_route2(V, E, base * amp, L, faces=faces)
        hol.append(info["holonomy_rel"])
        resid.append(info["edge_residual"])
        gam.append(info["gamma"])
    assert gam[0] < 1e-12 < gam[1] < gam[2]
    assert hol[0] < 1e-7
    assert hol[0] < hol[1] < hol[2]
    assert resid[0] < 1e-7
    assert resid[0] < resid[1] < resid[2]
    # second order in amplitude, not the old map's ~0.28·√γ: measured 1.8e-3
    # at amp 2 here (the pre-2026-09-26 dictionary gave ~1.7e-2)
    assert hol[2] < 5e-3, hol


def test_route2_quadrature_converges():
    """The arc integration is Simpson, O(h⁵) per edge, so at ρ = 0 — where
    the only error left *is* the quadrature — the residual must fall sharply
    with the mesh.

    The *ratio* is asserted rather than an absolute tolerance: the floor
    depends on the band (measured 3.8e-8 → 2.3e-9 at L=8, and 4.5e-10 →
    1.4e-11 at L=12, both between icosphere subdivisions 3 and 4), so a fixed
    threshold would only encode which ``L`` this file happens to use.
    """
    out = []
    for subdiv in (3, 4):
        V, faces = _sphere(subdiv)
        _P3, info = ql.lift_route2(V, _edges(faces), np.zeros(n_coeffs(L)), L)
        out.append(info["edge_residual"])
    assert out[0] / out[1] > 8.0, out
    assert out[1] < 1e-8, out


@pytest.mark.parametrize("amp, tol", [(0.1, 1e-3), (2.0, 5e-3)])
def test_route2_matches_the_decode(amp, tol):
    """Route 2 and route 1's ``codec.decode`` agree, and the disagreement
    shrinks with the *decode's* mesh — route 2 never uses one. Measured at
    amp 2 (γ = 1.2e-2): 2.1e-3 at subdivision 3, 7.5e-4 at 4. The
    pre-2026-09-26 dictionary gave ~0.12 there, and only met a 0.05 bound at
    amp 0.1."""
    base = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)] * amp
    errs = []
    for subdiv in (2, 3):
        V, faces = _sphere(subdiv)
        P3, _info = ql.lift_route2(V, _edges(faces), base, L)
        errs.append(codec.shape_error(codec.decode(base, V, faces), P3))
    assert errs[1] < tol, errs
    assert errs[1] < errs[0], errs


def test_arc_rule_choice_does_not_matter():
    """Simpson vs 5-point Gauss on the same arcs: the quadrature is far below
    the γ term, so refining the rule must change nothing material."""
    V, faces = _sphere(3)
    E = _edges(faces)
    base = bundle_style_coeffs(L=32, power=-2.0, seed=1)[:n_coeffs(L)]
    res = ds.solve_spinor(base, L)
    a = ql.arc_transformed_edges(res, V, E, rule="simpson")
    b = ql.arc_transformed_edges(res, V, E, rule="gauss5")
    rel = np.abs(a - b).max() / np.linalg.norm(a, axis=1).mean()
    assert rel < 1e-4, rel


def test_lift_route2_rejects_an_unknown_rule():
    V, faces = _sphere(2)
    res = ds.solve_spinor(np.zeros(n_coeffs(L)), L)
    with pytest.raises(ValueError, match="unknown rule"):
        ql.arc_transformed_edges(res, V, _edges(faces), rule="nope")
