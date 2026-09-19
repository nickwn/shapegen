"""Acceptance tests for the public codec façade.

The maths is gated in test_dirac / test_dirac_adjoint / test_encode. What is
checked here is the contract downstream code depends on: a round trip through the
façade reproduces the error it reports, and the convergence flag is honest.
"""

import numpy as np
import pytest

from shapegen import codec
from shapegen.data.dataset import demo_bottle_meshes


def test_roundtrip_reproduces_the_reported_error():
    """``decode(encode(m))`` must actually reproduce the error ``encode`` claims —
    the one contract everything downstream leans on."""
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    res = codec.encode(mesh, L=8, maxiter=40, method="adjoint")
    X = codec.decode(res.coeffs, res.sphere, mesh.faces)
    err = codec.shape_error(mesh.vertices, X)
    print(f"[codec] {res}  decode-check {err:.5f}")
    assert err == pytest.approx(res.err, rel=1e-9)
    assert res.coeffs.shape == (codec.n_coeffs(8),)
    assert res.err < res.err0                       # optimising beat the initialiser


def test_encode_beats_its_initialiser():
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=1)[0]
    res = codec.encode(mesh, L=8, maxiter=40, method="adjoint")
    assert res.err < 0.6 * res.err0


def test_converged_flag_is_honest():
    """A one-iteration budget must report itself as capped, not as converged.

    An iteration-capped encode is an upper bound on the error. This flag is the
    only thing standing between that and a silently overstated result.
    """
    mesh = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    capped = codec.encode(mesh, L=4, maxiter=1, method="adjoint")
    assert not capped.converged


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


def test_direct_and_adjoint_are_both_available_and_differ():
    """The two encoders sit at opposite ends of an accuracy / mesh-independence
    trade, so both must be reachable and the adjoint must actually be better on a
    fixed mesh (which is the axis it wins on)."""
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    loc = codec.encode(mesh, L=8, method="direct")
    adj = codec.encode(mesh, L=8, method="adjoint", maxiter=60, sphere=loc.sphere)
    print(f"[direct] {loc.err:.5f}   [adjoint] {adj.err:.5f}")
    assert adj.err < loc.err
    assert loc.niter == 0                      # a measurement, not an optimisation
    assert loc.coeffs.shape == adj.coeffs.shape
    # the direct result must still decode to what it claims
    X = codec.decode(loc.coeffs, loc.sphere, mesh.faces)
    assert codec.shape_error(mesh.vertices, X) == pytest.approx(loc.err, rel=1e-9)


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


def test_encode_dataset_shares_one_sphere():
    """Every code in a dataset must be defined against the same sphere, else the
    coefficients do not share a meaning and the model learns noise."""
    meshes = demo_bottle_meshes(n=2, subdiv=2, seed=0)
    coeffs, results, dec = codec.encode_dataset(meshes, L=4, maxiter=20,
                                               method="adjoint")
    assert coeffs.shape == (2, codec.n_coeffs(4))
    assert np.array_equal(results[0].sphere, results[1].sphere)
    assert np.array_equal(results[0].sphere, codec.template_sphere(2))
    assert np.array_equal(dec, codec.template_sphere(2))


def test_encode_dataset_direct_uses_per_mesh_spheres():
    """With no ``sphere`` given, ``direct`` measures each mesh on its OWN cMCF
    sphere and returns a canonical decode sphere separately."""
    meshes = demo_bottle_meshes(n=2, subdiv=2, seed=0)
    coeffs, results, dec = codec.encode_dataset(meshes, L=4, method="direct")
    assert coeffs.shape == (2, codec.n_coeffs(4))
    # per-mesh spheres genuinely differ ...
    assert not np.array_equal(results[0].sphere, results[1].sphere)
    # ... but the decode sphere is one shared object
    assert dec.shape == meshes[0].vertices.shape
    # and the codes decode on it
    X = codec.decode(coeffs[0], dec, meshes[0].faces)
    assert np.isfinite(X).all()


def test_encode_dataset_direct_accepts_a_shared_sphere():
    """``direct`` against ONE template is the mode a generative pipeline needs —
    at sample time there is no mesh, so there is no cMCF sphere to recover.

    This was previously rejected as a category error. It is not: measured, a
    direct code decodes on a sphere it never saw at 0.5-0.9x the error it gets on
    its own, so a shared template is portable rather than meaningless.
    """
    meshes = demo_bottle_meshes(n=2, subdiv=2, seed=0)
    T = codec.template_sphere(2)
    coeffs, results, dec = codec.encode_dataset(meshes, L=4, method="direct",
                                                sphere=T)
    assert np.array_equal(dec, T)
    assert all(np.array_equal(r.sphere, T) for r in results)
    assert coeffs.shape == (2, codec.n_coeffs(4))


def test_parallel_encode_matches_serial():
    """``n_jobs`` must not change the answer, and must not scramble the order.

    Order is the real risk: collecting results as they complete rather than by
    index would misalign every code with its mesh, and nothing downstream would
    notice. Tolerance is 1e-8 rather than exact because ARPACK's eigensolve is not
    bit-reproducible — two *serial* runs differ at 6e-11 — so demanding equality
    would be testing the wrong thing.
    """
    meshes = demo_bottle_meshes(n=4, subdiv=2, seed=0)
    T = codec.template_sphere(2)
    a, _, _ = codec.encode_dataset(meshes, L=4, sphere=T, method="direct", n_jobs=1)
    b, _, _ = codec.encode_dataset(meshes, L=4, sphere=T, method="direct", n_jobs=2)
    assert np.abs(a - b).max() < 1e-8
    # order preserved: each row must match ITS OWN mesh, not merely be present
    for i, m in enumerate(meshes):
        solo = codec.encode(m, L=4, sphere=T, method="direct").coeffs
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


def test_decode_is_mesh_independent_when_rho_is_realisable():
    """The contract behind "encode to SH, decode from SH": ρ̂ is a function on S²,
    so decoding it on ANY sphere mesh gives the same surface.

    That holds exactly while ρ is **realisable**. When γ > 0 no immersion has that
    ρ, the eigensolve returns a least-squares fit, and which fit depends on the
    quadrature — so the decode stops being a function of the coefficients alone.
    Measured on a face at L=16: γ=4.6e-13 gives cross-mesh agreement at the
    vertex-spacing floor, while γ=2.2e-2 puts it 0.057 above.
    """
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
    c[1:9] = 0.02 * rng.standard_normal(8)          # small => nearly realisable
    Xa = norm_shape(codec.decode(c, Pa, a.faces), a.faces)
    Xb = norm_shape(codec.decode(c, Pb, b.faces), b.faces)
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
