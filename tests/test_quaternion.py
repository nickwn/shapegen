"""Session 1 quaternion-algebra tests (storage order [w,x,y,z], ij=k)."""

import numpy as np

from bottlegen.spectral import quaternion as Q

ONE = np.array([1.0, 0, 0, 0])
I = np.array([0.0, 1, 0, 0])
J = np.array([0.0, 0, 1, 0])
K = np.array([0.0, 0, 0, 1])


def test_basis_relations():
    assert np.allclose(Q.qmul(I, J), K)      # ij = k
    assert np.allclose(Q.qmul(J, K), I)      # jk = i
    assert np.allclose(Q.qmul(K, I), J)      # ki = j
    assert np.allclose(Q.qmul(I, I), -ONE)   # i^2 = -1
    assert np.allclose(Q.qmul(J, I), -K)     # ji = -k (non-commutative)


def test_identity_and_conjugate():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 4))
    assert np.allclose(Q.qmul(a, ONE), a)
    assert np.allclose(Q.qmul(ONE, a), a)
    # a * conj(a) = |a|^2 (real)
    aa = Q.qmul(a, Q.qconj(a))
    assert np.allclose(aa[..., 0], (a * a).sum(-1))
    assert np.allclose(aa[..., 1:], 0)


def test_multiplicativity_and_inverse():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((7, 4))
    b = rng.standard_normal((7, 4))
    # |ab| = |a||b|
    assert np.allclose(Q.qnorm(Q.qmul(a, b)), Q.qnorm(a) * Q.qnorm(b))
    # a * a^-1 = 1
    assert np.allclose(Q.qmul(a, Q.qinv(a)), ONE)


def test_associativity():
    rng = np.random.default_rng(2)
    a, b, c = (rng.standard_normal((4, 4)) for _ in range(3))
    assert np.allclose(Q.qmul(Q.qmul(a, b), c), Q.qmul(a, Q.qmul(b, c)))


def test_vec3_roundtrip():
    rng = np.random.default_rng(3)
    v = rng.standard_normal((6, 3))
    assert np.allclose(Q.to_vec3(Q.from_vec3(v)), v)
    assert np.allclose(Q.from_vec3(v)[..., 0], 0)


def test_sandwich_conformal_scaling():
    """λ̄ v λ scales lengths by |λ|^2 and reduces to a rotation for unit λ."""
    rng = np.random.default_rng(4)
    v = rng.standard_normal((8, 3))
    lam = rng.standard_normal((8, 4))
    out = Q.sandwich(lam, v)
    scale = Q.qnorm(lam) ** 2
    assert np.allclose(np.linalg.norm(out, axis=-1), scale * np.linalg.norm(v, axis=-1))
    # identity quaternion leaves v unchanged
    assert np.allclose(Q.sandwich(np.broadcast_to(ONE, v.shape[:-1] + (4,)), v), v)


def test_sandwich_is_rotation_for_unit_quaternion():
    rng = np.random.default_rng(5)
    v = rng.standard_normal((10, 3))
    lam = Q.qnormalize(rng.standard_normal((10, 4)))
    out = Q.sandwich(lam, v)
    # pure rotation: norm preserved and angles preserved (dot products)
    assert np.allclose(np.linalg.norm(out, axis=-1), np.linalg.norm(v, axis=-1))
    out2 = Q.sandwich(lam, np.roll(v, 1, axis=0))
    assert np.allclose((out * out2).sum(-1), (v * np.roll(v, 1, axis=0)).sum(-1))
