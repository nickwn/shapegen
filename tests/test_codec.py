"""Acceptance tests for the public codec façade.

The maths is gated in test_dirac_spectral / test_dirac_quotient / test_eigen_sphere /
test_encode. What is checked here is the contract downstream code depends on: a
round trip through the façade reproduces the error it reports.
"""

import numpy as np
import pytest

from shapegen import codec
from shapegen.data.dataset import demo_bottle_meshes


def test_roundtrip_reproduces_the_reported_error():
    """``decode(encode(m))`` must actually reproduce the error ``encode`` claims —
    the one contract everything downstream leans on."""
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    res = codec.encode(mesh, L=8)
    X = codec.decode(res.coeffs, res.sphere)
    err = codec.shape_error(mesh.vertices, X)
    print(f"[codec] {res}  decode-check {err:.5f}")
    assert err == pytest.approx(res.err, rel=1e-9)
    assert res.coeffs.shape == (codec.n_coeffs(8),)


def test_shape_error_is_similarity_invariant():
    """The decode fixes the immersion only up to a similarity, so the metric must
    not see one."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 3))
    X = A + 0.2 * rng.standard_normal((30, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    assert codec.shape_error(A, 3.7 * X @ Q + np.array([1.0, -2.0, 0.5])) == \
        pytest.approx(codec.shape_error(A, X), rel=1e-12)


def test_base_sphere_is_gauge_fixed_and_round():
    """The parameterisation must land on the unit sphere with the conformal
    barycenter at the origin — that is what makes ρ̂ comparable across shapes.

    The barycenter is weighted by the **mesh's** conformal density, not the
    sphere's; centring the sphere's own vertices instead would leave the ℓ=1
    Möbius freedom alive and the codes incomparable.
    """
    from shapegen.geom.operators import vertex_areas
    mesh = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    w = vertex_areas(mesh.vertices, mesh.faces)
    w = w / w.sum()
    S = codec.base_sphere(mesh)
    assert np.abs(np.linalg.norm(S, axis=1) - 1.0).max() < 1e-9
    assert np.linalg.norm((w[:, None] * S).sum(0)) < 1e-6


def test_default_method_is_polyhedral_and_refuses_a_sphere():
    """The default computes its own sphere map; a given sphere needs a method that
    encodes against one, and the error says which."""
    mesh = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    assert codec.encode(mesh, L=4).meta["solver"] in ("exact", "fast")
    with pytest.raises(ValueError, match="quotient"):
        codec.encode(mesh, L=4, sphere=codec.base_sphere(mesh))


def test_unknown_method_raises():
    mesh = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    with pytest.raises(ValueError):
        codec.encode(mesh, L=4, method="magic")


def test_template_sphere_is_canonical_and_round():
    """One sphere per subdivision, unit, deterministic — codes are only
    comparable if every shape was encoded against the identical object."""
    for sub in (2, 3):
        t1, t2 = codec.template_sphere(sub), codec.template_sphere(sub)
        assert np.array_equal(t1, t2)
        assert np.abs(np.linalg.norm(t1, axis=1) - 1.0).max() < 1e-12


def test_encode_dataset_can_share_one_sphere():
    """A registered dataset encoded against ONE sphere: every code is defined
    relative to the identical object."""
    meshes = demo_bottle_meshes(n=2, subdiv=2, seed=0)
    T = codec.template_sphere(2)
    coeffs, results = codec.encode_dataset(meshes, L=4, sphere=T, method="quotient")
    assert coeffs.shape == (2, codec.n_coeffs(4))
    assert all(np.array_equal(r.sphere, T) for r in results)


def test_encode_dataset_defaults_to_per_mesh_spheres():
    """With no ``sphere``, each mesh is encoded on its own conformal sphere, and
    each code decodes on it."""
    meshes = demo_bottle_meshes(n=2, subdiv=2, seed=0)
    coeffs, results = codec.encode_dataset(meshes, L=4, method="quotient")
    assert coeffs.shape == (2, codec.n_coeffs(4))
    assert not np.array_equal(results[0].sphere, results[1].sphere)
    assert np.isfinite(codec.decode(coeffs[0], results[0].sphere)).all()


def test_parallel_encode_matches_serial():
    """``n_jobs`` must not change the answer, and must not scramble the order.

    Order is the real risk: collecting results as they complete rather than by
    index would misalign every code with its mesh, and nothing downstream would
    notice.
    """
    meshes = demo_bottle_meshes(n=4, subdiv=2, seed=0)
    T = codec.template_sphere(2)
    a, _ = codec.encode_dataset(meshes, L=4, sphere=T, method="quotient", n_jobs=1)
    b, _ = codec.encode_dataset(meshes, L=4, sphere=T, method="quotient", n_jobs=2)
    assert np.abs(a - b).max() < 1e-8
    # order preserved: each row must match ITS OWN mesh, not merely be present
    for i, m in enumerate(meshes):
        solo = codec.encode(m, L=4, sphere=T, method="quotient").coeffs
        assert np.abs(b[i] - solo).max() < 1e-8


def test_chamfer_scale_align_is_vertex_count_independent():
    """``scale_align`` must normalise by RMS radius, not by the raw sum.

    Regression: it used ``sqrt(sum(B**2)/sum(A**2))``, which scales with the
    VERTEX COUNT. Two unit spheres at 10242 and 40962 vertices scored 0.99999 --
    the smaller was scaled by 2.0 -- instead of 0.00708. It went unnoticed because
    with equal counts the two forms agree, and that is the only case the codec
    itself uses. It breaks precisely where chamfer is otherwise the right tool:
    comparing decodes on differently-sized meshes.
    """
    from shapegen.eval.metrics import chamfer
    from shapegen.geom.mesh import icosphere
    u = lambda s: (lambda m: m.vertices / np.linalg.norm(  # noqa: E731
        m.vertices, axis=1, keepdims=True))(icosphere(s))
    A, B = u(4), u(5)
    plain = chamfer(A, B)
    assert chamfer(A, B, scale_align=True) == pytest.approx(plain, rel=1e-9)
    # and it still does its job: a pure rescale must be removed
    assert chamfer(A * 3.0, A, scale_align=True) < 1e-12


def test_decode_is_sampling_independent():
    """The contract behind "encode to SH, decode from SH": the decoded surface is
    a band-limited function on S², so decoding on ANY set of sphere points gives
    the same surface -- whatever γ is (the old mesh decoder lost this at
    γ ≈ 2e-2, where two meshes disagreed by 0.057)."""
    from scipy.spatial import cKDTree
    from shapegen.geom.mesh import icosphere
    from shapegen.geom.operators import vertex_areas

    def norm_shape(V, F):
        w = vertex_areas(V, F)
        w = w / w.sum()
        V = V - (w[:, None] * V).sum(0)
        return V / np.sqrt((w * (V ** 2).sum(1)).sum())

    def cham(A, B):
        return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())

    a, b = icosphere(3), icosphere(4)
    Pa = a.vertices / np.linalg.norm(a.vertices, axis=1, keepdims=True)
    Pb = b.vertices / np.linalg.norm(b.vertices, axis=1, keepdims=True)
    floor = cham(norm_shape(Pa, a.faces), norm_shape(Pb, b.faces))

    rng = np.random.default_rng(0)
    c = np.zeros(codec.n_coeffs(4))
    c[1:9] = 0.5 * rng.standard_normal(8)           # far from realisable: γ ≫ 0
    Xa = norm_shape(codec.decode(c, Pa), a.faces)
    Xb = norm_shape(codec.decode(c, Pb), b.faces)
    print(f"[mesh-independence] floor {floor:.5f}, with rho {cham(Xa, Xb):.5f}")
    assert cham(Xa, Xb) < floor * 1.15


def test_nose_tip_picks_the_dense_end_not_the_far_end():
    """Both ends of the depth axis are extreme; the face is the populated one.

    Regression: choosing by |z| alone picked the occiput, giving a "face region"
    of 1% of vertices and a 17% improvement that measured the back of the skull.
    """
    import numpy as np
    from shapegen.eval.metrics import nose_tip

    rng = np.random.default_rng(0)
    # A dense cluster near z=+1 and a lone spike further out at z=-1.4.
    dense = np.c_[rng.normal(0, .15, 400), rng.normal(0, .15, 400),
                  1.0 + rng.normal(0, .05, 400)]
    sparse = np.array([[0., 0., -1.4]])
    V = np.vstack([dense, sparse])
    assert nose_tip(V, r=0.8)[2] > 0     # the dense end, not the extreme one


def test_chamfer_region_is_restricted_and_rejects_an_empty_region():
    import numpy as np
    import pytest
    from shapegen.eval.metrics import chamfer, chamfer_region

    rng = np.random.default_rng(1)
    A = rng.normal(size=(500, 3))
    B = A.copy()
    far = np.linalg.norm(A, axis=1) > 1.2
    B[far] += 5.0                        # all the error lives outside the region

    c = np.zeros(3)
    assert chamfer_region(A, B, c, 0.5) < 0.05 * chamfer(A, B)
    with pytest.raises(ValueError, match="region is empty"):
        chamfer_region(A, B, np.array([50., 0., 0.]), 0.5)
