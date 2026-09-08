"""Acceptance tests for spherical resampling.

The failure this module exists to prevent is silent and arrives late: an
unresolved direction left at index -1 with zero weights indexes the last face and
puts that vertex at the ORIGIN. Measured, 39 such vertices out of 10242 produced
44 zero-area triangles, rho values of 1e91, and finally an ARPACK "starting vector
is zero" from deep inside the decode -- with nothing in between to say why.
"""

import numpy as np

from bottlegen.geom.mesh import icosphere, triangle_areas
from bottlegen.geom.resample import locate_on_sphere, resample_to


def _unit(sub):
    m = icosphere(sub)
    return m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True), m.faces


def test_locate_resolves_every_direction():
    """No direction may be left unresolved, and every weight set must be a
    genuine convex combination."""
    S, F = _unit(3)
    P, _ = _unit(4)
    idx, bary, nf = locate_on_sphere(S, F, P)
    assert (idx >= 0).all()
    assert np.abs(bary.sum(1) - 1.0).max() < 1e-9
    assert bary.min() > -1e-7
    print(f"[locate] {len(P)} directions, {nf} fallbacks")


def test_resample_is_exact_on_the_identity():
    """Resampling a mesh onto its OWN vertex directions must return it. Anything
    else means the barycentric solve or the index bookkeeping is wrong."""
    S, F = _unit(3)
    rng = np.random.default_rng(0)
    X = S * (1.0 + 0.2 * rng.standard_normal((len(S), 1)))
    out, idx, bary, nf = resample_to(S, F, X, S)
    assert nf == 0
    assert np.abs(out - X).max() < 1e-9


def test_resample_survives_a_degenerate_source_triangulation():
    """The real case: a source sphere whose triangles span orders of magnitude.

    A nearest-CENTROID search fails here -- a large triangle's centroid can be far
    from points inside it -- which is why location searches from the nearest
    vertex instead. The output must contain no origin vertices and no degenerate
    triangles.
    """
    S, F = _unit(3)
    S = S.copy()
    S[:, 2] = np.sign(S[:, 2]) * np.abs(S[:, 2]) ** 0.05      # crowd the equator
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    ta = triangle_areas(S, F)
    print(f"[degenerate source] area spread {ta.max() / ta.min():.1f}x")
    assert ta.max() / ta.min() > 50            # the case actually is degenerate

    P, Fi = _unit(4)
    X = S * 1.3
    out, _, _, nf = resample_to(S, F, X, P)
    assert np.isfinite(out).all()
    assert np.linalg.norm(out, axis=1).min() > 1e-6, "a vertex landed at the origin"
    assert (triangle_areas(out, Fi) > 1e-12).all(), "degenerate output triangle"
    print(f"[degenerate source] {nf} fallbacks of {len(P)}")


def test_resampling_preserves_a_smooth_shape():
    """Resampled geometry must still be the same surface, not merely finite.

    The deformation is SMOOTH on purpose. Barycentric interpolation reproduces a
    band-limited field to the source mesh's resolution; per-vertex white noise it
    cannot reproduce at all, and asserting otherwise would be testing the wrong
    thing. Real inputs here are Laplacian-smoothed heads, so smooth is the case
    that matters.
    """
    from scipy.spatial import cKDTree
    S, F = _unit(4)
    P, Fi = _unit(5)
    r = 1.0 + 0.15 * (S[:, 2] ** 2 - 0.3) + 0.08 * S[:, 0] * S[:, 1]
    X = S * r[:, None]
    out, _, _, _ = resample_to(S, F, X, P)
    d = cKDTree(X).query(out)[0].mean()
    print(f"[shape] mean distance from the resampled surface to the source "
          f"{d:.5f} (source edge ~{np.sqrt(4 * np.pi / len(S)):.4f})")
    assert d < 0.5 * np.sqrt(4 * np.pi / len(S))
