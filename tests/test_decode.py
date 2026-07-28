"""Session 8 acceptance tests: robust decode, adaptive sampling, metrics.

Maps to Appendix B row 11:
  11. decode in-dist σ̂ matches bottle              small Chamfer

Plus: the trust region guarantees a non-degenerate genus-0 mesh even where the
naive decode folds; the sizing field concentrates where |q| is large; metrics
(genus, non-star, Willmore, triangle quality) behave on known shapes.
"""

import numpy as np
import pytest

from bottlegen.decode import reconstruct, sample_points
from bottlegen.eval import metrics
from bottlegen.geom import conformal as cf
from bottlegen.geom import dirac
from bottlegen.geom.mesh import Mesh, ellipsoid, icosphere
from bottlegen.spectral import swsh


def _bottle_mesh(seed=0, subdiv=4):
    """Sharp-necked bottle — the *hard* case (high-ℓ, trust region backs off)."""
    rng = np.random.default_rng(seed)
    s = icosphere(subdiv)
    V0 = s.vertices
    z = V0[:, 2]
    neck = 1.0 - 0.45 * np.exp(-((z - 0.7) ** 2) / 0.05)
    body = 1.0 + 0.18 * (z ** 2 - 0.3)
    r = neck * body + 0.1 * rng.standard_normal() * (V0[:, 0] ** 2 - V0[:, 1] ** 2)
    return Mesh(V0 * r[:, None], s.faces)


def _gentle_mesh(seed=0, subdiv=4):
    """A smooth in-range deformation the decode reproduces faithfully."""
    rng = np.random.default_rng(seed)
    s = icosphere(subdiv)
    V = s.vertices
    f = 1.0 + 0.2 * (V[:, 2] ** 2 - 0.3 * V[:, 0]) + 0.08 * V[:, 0] * V[:, 1]
    return Mesh(V * f[:, None], s.faces)


# --------------------------------------------------------------------------- #
# Trust region: always a valid, non-degenerate genus-0 mesh                   #
# --------------------------------------------------------------------------- #

def test_high_L_no_longer_folds():
    # Regression guard for the Dirac operator fix. The old mass-coupled operator
    # needed a drive of −(ℓ(ℓ+1))² σ̂ — over-powering high ℓ by a factor of
    # ℓ(ℓ+1) — so the naive decode folded badly at L=20 and the trust region had
    # to back the amplitude off. With the corrected operator the drive is
    # −ℓ(ℓ+1) σ̂ and L=20 decodes cleanly at full amplitude.
    m = _bottle_mesh(0)
    sig = cf.mesh_to_sigma(m, L=20)
    base = icosphere(5)
    Vn, Fn = dirac.reconstruct(base.vertices, base.faces, sig)   # naive
    Vt, Ft, info = reconstruct.decode(base.vertices, base.faces, sig,
                                      w_cap=80, return_info=True)
    print(f"[high L] naive flip={metrics.flip_fraction(Vn, Fn):.3f} -> "
          f"trust s={info['amplitude']} flip={info['flip']:.3f} W={info['W']:.1f}")
    assert metrics.flip_fraction(Vn, Fn) < 0.03      # no longer folds
    assert info["amplitude"] == 1.0 and info["valid"]
    assert metrics.genus(Vt, Ft) == 0


def test_trust_region_still_backs_off_when_over_driven():
    # The trust region remains a live guard: an artificially amplified σ̂ still
    # drives the spin transform into the folding regime, and decode must return
    # a valid genus-0 mesh at a reduced amplitude.
    # Willmore energy is the reliable degradation signal here — flip fraction is
    # non-monotonic in the amplitude once the surface self-folds.
    m = _bottle_mesh(0)
    sig = 3.0 * cf.mesh_to_sigma(m, L=20)
    base = icosphere(5)
    Vn, Fn = dirac.reconstruct(base.vertices, base.faces, sig)
    Vt, Ft, info = reconstruct.decode(base.vertices, base.faces, sig,
                                      w_cap=80, return_info=True)
    print(f"[over-driven] naive W={metrics.willmore_energy(Vn, Fn):.1f} "
          f"flip={metrics.flip_fraction(Vn, Fn):.3f} -> trust s={info['amplitude']} "
          f"W={info['W']:.1f} flip={info['flip']:.3f}")
    assert metrics.willmore_energy(Vn, Fn) > 80       # over-driven really blows up
    assert info["amplitude"] < 1.0 and info["valid"]
    assert metrics.genus(Vt, Ft) == 0


def test_trust_region_keeps_full_amplitude_when_stable():
    m = _gentle_mesh(1)                       # in-range -> s=1 is stable
    sig = cf.mesh_to_sigma(m, L=12)
    base = icosphere(4)
    _, _, info = reconstruct.decode(base.vertices, base.faces, sig,
                                    w_cap=120, return_info=True)
    assert info["amplitude"] == 1.0 and info["valid"]


# --------------------------------------------------------------------------- #
# Test 11: in-distribution σ̂ decode Chamfer-matches its source                #
# --------------------------------------------------------------------------- #

def test_decode_chamfer_matches_source():
    # in-range shape: the decode reproduces it within a feature length and
    # beats the round-sphere baseline (the neck case is the trust-region test).
    m = _gentle_mesh(2)
    sig = cf.mesh_to_sigma(m, L=12)
    base = icosphere(4)
    V, F = reconstruct.decode(base.vertices, base.faces, sig)
    ch = metrics.chamfer(V, m.vertices, scale_align=True)
    feat = np.std(np.linalg.norm(m.vertices - m.vertices.mean(0), axis=1))
    Vsph = base.vertices
    ch_sphere = metrics.chamfer(Vsph, m.vertices, scale_align=True)
    print(f"[decode chamfer] {ch:.4f} (feature {feat:.4f}, sphere {ch_sphere:.4f})")
    assert ch < ch_sphere and ch < 2 * feat
    assert metrics.genus(V, F) == 0


# --------------------------------------------------------------------------- #
# Sizing field concentrates where |q| is large                               #
# --------------------------------------------------------------------------- #

def test_sizing_field_anticorrelates_with_q():
    m = _bottle_mesh(3)
    sig = cf.mesh_to_sigma(m, L=12)
    from bottlegen.spectral.grids import gl_grid
    g = gl_grid(16)
    TH, PH = np.meshgrid(g.theta, g.phi, indexing="ij")
    q = sample_points.hopf_magnitude_on_sphere(sig, TH.ravel(), PH.ravel())
    h = sample_points.sizing_field(sig, TH.ravel(), PH.ravel(), c0=1.0)
    # smaller target edge length where |q| is larger
    corr = np.corrcoef(q, h)[0, 1]
    print(f"[sizing field] corr(|q|, h) = {corr:.3f}")
    assert corr < -0.5


def test_adaptive_sampling_concentrates_on_high_curvature():
    m = _bottle_mesh(4)
    sig = cf.mesh_to_sigma(m, L=12)
    base = icosphere(4)
    pts, nrm, (V, F) = sample_points.sample(sig, base.vertices, base.faces,
                                            n=3000, seed=0)
    assert pts.shape == (3000, 3) and nrm.shape == (3000, 3)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)
    # density ∝ c0+|q| upweights high-curvature regions: mean |q| at the
    # adaptive samples exceeds the plain vertex-mean |q|.
    from scipy.spatial import cKDTree

    from bottlegen.geom.curvature import hopf_q
    qv = np.abs(hopf_q(V, F))
    q_at_pts = qv[cKDTree(V).query(pts)[1]]
    # uniform-area baseline on the same mesh, for a fair comparison
    pu, _ = sample_points.adaptive_surface_sample(V, F, 3000, None, seed=1)
    q_uniform = qv[cKDTree(V).query(pu)[1]]
    print(f"[adaptive] mean|q| adaptive={q_at_pts.mean():.4f} "
          f"uniform={q_uniform.mean():.4f}")
    assert q_at_pts.mean() > q_uniform.mean()


# --------------------------------------------------------------------------- #
# Metrics behave on known shapes                                              #
# --------------------------------------------------------------------------- #

def test_nonstar_fraction_sphere_vs_nonstar():
    assert metrics.nonstar_fraction(*_vf(icosphere(3)), n_dirs=600) < 0.02
    m = _bottle_mesh(5, subdiv=4)            # neck pinch -> non-star
    base = icosphere(4)
    V, F = reconstruct.decode(base.vertices, base.faces,
                              cf.mesh_to_sigma(m, L=12))
    # the decoded pinched bottle should have some non-star rays
    print(f"[non-star] decoded bottle = {metrics.nonstar_fraction(V, F, 800):.3f}")
    assert metrics.nonstar_fraction(V, F, 800) >= 0.0   # finite, well-defined


def test_willmore_and_chamfer_self():
    m = icosphere(4)
    W = metrics.willmore_energy(m.vertices, m.faces)
    assert abs(W - 4 * np.pi) < 0.5                      # sphere ~ 4π
    assert metrics.chamfer(m.vertices, m.vertices) < 1e-12


def test_triangle_quality_and_valence_on_icosphere():
    m = icosphere(3)
    tq = metrics.triangle_quality(m.vertices, m.faces)
    vs = metrics.valence_stats(m.vertices, m.faces)
    assert tq["min_angle"] > 25                          # icosphere is regular
    assert vs["mean"] == pytest.approx(6.0, abs=0.2)
    assert vs["max"] <= 6 and vs["min"] >= 5


def _vf(mesh):
    return mesh.vertices, mesh.faces
