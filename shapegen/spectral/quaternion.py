"""Vectorized quaternion algebra (NumPy, float64).

Storage order is ``[w, x, y, z]`` with basis ``1, i, j, k`` and ``ij = k``.
ℝ³ is identified with the imaginary part: ``(x, y, z) <-> x i + y j + z k``.

All ops broadcast over leading dimensions; the last axis has length 4.
"""

import numpy as np

_CONJ = np.array([1.0, -1.0, -1.0, -1.0])


def qmul(a, b):
    """Hamilton product ``a * b`` (shapes ``(..., 4)``, broadcasting)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(a):
    """Quaternion conjugate ``w - xi - yj - zk``."""
    return a * _CONJ


def qnorm(a):
    """Euclidean norm ``|a|`` over the last axis."""
    return np.linalg.norm(a, axis=-1)


def qnormalize(a, eps=1e-12):
    """Return ``a / |a|`` (safe near zero)."""
    n = qnorm(a)[..., None]
    return a / np.maximum(n, eps)


def qinv(a, eps=1e-12):
    """Multiplicative inverse ``conj(a) / |a|^2``."""
    n2 = (a * a).sum(-1, keepdims=True)
    return qconj(a) / np.maximum(n2, eps)


def from_vec3(v):
    """Embed ℝ³ as pure-imaginary quaternions: ``(x,y,z) -> [0,x,y,z]``."""
    return np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)


def to_vec3(q):
    """Project to ℝ³ by dropping the real (``w``) part."""
    return q[..., 1:]


def sandwich(lam, v):
    """Conformal spin action ``λ̄ v λ`` for a pure-imaginary ``v`` (given as ℝ³).

    Returns ℝ³. For unit ``λ`` this is a rotation; in general it scales lengths
    by ``|λ|²`` (the conformal factor of the spin transform ``df̃ = λ̄ df λ``).
    """
    out = qmul(qmul(qconj(lam), from_vec3(v)), lam)
    return to_vec3(out)
