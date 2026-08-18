"""Acceptance tests for the public codec façade.

The maths is gated in test_dirac / test_dirac_adjoint / test_encode. What is
checked here is the contract downstream code depends on: a round trip through the
façade reproduces the error it reports, and the convergence flag is honest.
"""

import numpy as np
import pytest

from bottlegen import codec
from bottlegen.data.dataset import demo_bottle_meshes


def test_roundtrip_reproduces_the_reported_error():
    """``decode(encode(m))`` must actually reproduce the error ``encode`` claims —
    the one contract everything downstream leans on."""
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    res = codec.encode(mesh, L=8, maxiter=40)
    X = codec.decode(res.coeffs, res.sphere, mesh.faces)
    err = codec.shape_error(mesh.vertices, X)
    print(f"[codec] {res}  decode-check {err:.5f}")
    assert err == pytest.approx(res.err, rel=1e-9)
    assert res.coeffs.shape == (codec.n_coeffs(8),)
    assert res.err < res.err0                       # optimising beat the initialiser


def test_encode_beats_its_initialiser():
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=1)[0]
    res = codec.encode(mesh, L=8, maxiter=40)
    assert res.err < 0.6 * res.err0


def test_converged_flag_is_honest():
    """A one-iteration budget must report itself as capped, not as converged.

    An iteration-capped encode is an upper bound on the error. This flag is the
    only thing standing between that and a silently overstated result.
    """
    mesh = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    capped = codec.encode(mesh, L=4, maxiter=1)
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
    """Both parameterisations must land on the unit sphere with the conformal
    barycenter at the origin — that is what makes ρ̂ comparable across shapes."""
    from bottlegen.geom.operators import vertex_areas
    mesh = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    w = vertex_areas(mesh.vertices, mesh.faces)
    w = w / w.sum()
    for method in ("cmcf", "willmore"):
        S = codec.base_sphere(mesh, method=method)
        assert np.abs(np.linalg.norm(S, axis=1) - 1.0).max() < 1e-9
        assert np.linalg.norm((w[:, None] * S).sum(0)) < 1e-6


def test_local_and_adjoint_are_both_available_and_differ():
    """The two encoders sit at opposite ends of an accuracy / mesh-independence
    trade, so both must be reachable and the adjoint must actually be better on a
    fixed mesh (which is the axis it wins on)."""
    mesh = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    loc = codec.encode(mesh, L=8, method="local")
    adj = codec.encode(mesh, L=8, method="adjoint", maxiter=60, sphere=loc.sphere)
    print(f"[local] {loc.err:.5f}   [adjoint] {adj.err:.5f}")
    assert adj.err < loc.err
    assert loc.niter == 0                      # a measurement, not an optimisation
    assert loc.coeffs.shape == adj.coeffs.shape
    # the local result must still decode to what it claims
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
    coeffs, results = codec.encode_dataset(meshes, L=4, maxiter=20)
    assert coeffs.shape == (2, codec.n_coeffs(4))
    assert np.array_equal(results[0].sphere, results[1].sphere)
    assert np.array_equal(results[0].sphere, codec.template_sphere(2))
