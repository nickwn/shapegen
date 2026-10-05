"""Exact conformal sphere map and spinor of a polyhedron: ``codec.encode(method="polyhedral")``.

A closed genus-0 triangle mesh, read literally, is a polyhedron: flat faces whose
intrinsic metric is flat except at cone points, the vertices with angle defect
``κ_v ≠ 0``. Vertices added by subdividing a face or an edge are flat (``κ_v = 0``)
and are not cones. So the shape, not its tessellation, fixes everything below, and
a 12-triangle cube encodes to the same code as a 12288-triangle one.

**Sphere map.** In a stereographic chart a flat cone metric is

    c² |W(ζ)|² |dζ|²,   W(ζ) = Π_cones (ζ − p_k)^{a_k},   a_k = −κ_k / 2π

(Troyanov: unique up to Möbius). The unknowns are the chart positions of all
vertices plus ``log c``. Each face is developed in its own chart, centred at ζ = 0
(:func:`face_rotations`), with ``D_k = K_f ∫_0^{ζ_k} W dζ`` along rays from the centre.
Gauss–Jacobi handles a cone endpoint. ``K_f = Π (1+|p|²)^{-a/2}`` is the chordal
normalisation shared by all charts. The developed triangle must match the 3D one in
two log edge lengths and one angle.

This is solved by Levenberg–Marquardt with an analytic sparse Jacobian (μ₀ = 1). The
derivatives are integrals at the same nodes: ``dI/dζ = W`` at a flat endpoint, and
``−a∫W/(ζ−P)`` for another cone. Flat vertices touch only their own faces; cones touch
every face. Gauss order is 8, or 24 for rays within 3 ray lengths of another cone.

**Gauge.** The ``|κ|``-weighted cone centroid is moved to 0 by a Möbius map. That is
exact and tessellation-free. The area centroid the other encoders use would need
surface quadrature, which leaks the tessellation back in. Rotation is left free.

**Spinor.** Each 3D face is sampled with a 6-point degree-4 rule on k² sub-triangles,
and each sample is mapped to the sphere exactly. The map is inverted by RK4 on
``dζ/dD = 1/W`` from the face centre, then Newton polish, with damped Newton for the
few samples near sharp corners. On each face the spinor is closed form:
``dX = M_f(c W dζ)``, with ``M_f`` taking the developed plane onto the face plane.
Samples carry the face's area, divided by ``s²`` to give sphere area. They are
sign-aligned over a k-NN spanning tree and projected by
:func:`~shapegen.geom.dirac_direct._project_faces`. No quadrature cell straddles an
edge: a fixed-grid projection of the edge-discontinuous spinor was O(h) and
rotation-dependent (6e-3 at Lg = 2L).

**Code.** :func:`~shapegen.geom.dirac_quotient.rho_quotient` with ``over = 3``. The
pointwise quotient is not band-limited, and on the minimal grid it aliased by ~1e-3,
depending on orientation. Then :func:`~shapegen.geom.dirac_direct.close_code`.

**Measured** (2026-10-05). Rotation-invariant code-spectrum difference between
tessellations of one polyhedron, against the remesh codec:

| shape | tessellations | this encoder | remesh codec |
|---|---|---|---|
| cube | 12 → 12288 triangles, including flipped diagonals | ~1e-5 | 5e-4 – 5e-3 |
| 1×2×3 box | 12, 768 | 1e-5 – 8e-5 | 3e-4 – 2e-3 |
| 9-vertex convex hull | 14 → 3584 | ~2e-5 | 9e-4 – 3e-3 |

These hold at L = 16 and 32. Time is 0.6–11 s. The decoded error against the true
polyhedron is ~8% lower than the remesh codec on cubes, and 1.7× lower on the hull.

**Limits:**
- Cost grows as faces × cones. A dense smooth mesh (every vertex a cone, e.g. a scan)
  is better served by ``method="quotient"`` on the eigen sphere; see ``max_cones``.
- Needs a closed, consistently outward-oriented, genus-0 mesh without degenerate faces.
- Codes use the cone gauge, so they are not interchangeable with the other methods'.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from . import dirac_spectral as ds

__all__ = ["angle_defects", "PolyMap", "solve_map", "encode_polyhedral"]

NQ = 24            # Gauss order for rays near another cone (8 elsewhere)


# ------------------------------------------------------------------ polyhedron data
def angle_defects(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """κ_v = 2π − Σ face angles at v."""
    k = np.full(len(V), 2 * np.pi)
    for c in range(3):
        i, j, l = F[:, c], F[:, (c + 1) % 3], F[:, (c + 2) % 3]
        u, w = V[j] - V[i], V[l] - V[i]
        np.add.at(k, i, -np.arctan2(np.linalg.norm(np.cross(u, w), axis=1),
                                    np.einsum("ij,ij->i", u, w)))
    return k


def _normalize(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def face_rotations(C: np.ndarray) -> np.ndarray:
    """``R_f`` with ``R_f c_f = (0, 0, −1)``: face centre ``c_f`` → ζ = 0 of its chart."""
    C = _normalize(C)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(C, b)
    cth = C @ b
    K = np.zeros((len(C), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    return np.eye(3) + K + K @ K / (1.0 + cth)[:, None, None]


def chart(R: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Stereographic coordinate (pole at +z) of points ``X`` (…,3) after rotation ``R``."""
    Y = np.einsum("...ij,...j->...i", R, X)
    return (Y[..., 0] + 1j * Y[..., 1]) / (1.0 - Y[..., 2])


def _unstereo(z: np.ndarray) -> np.ndarray:
    r2 = np.abs(z) ** 2
    return np.stack([2 * z.real / (r2 + 1), 2 * z.imag / (r2 + 1), (r2 - 1) / (r2 + 1)], -1)


# ------------------------------------------------------------------ W and its integrals
@lru_cache(maxsize=None)
def _gj(a: float, nq: int):
    """Nodes/weights on [0, 1] for ``∫ (1−t)^a g(t) dt``."""
    from scipy.special import roots_jacobi
    x, w = roots_jacobi(nq, a, 0.0)
    return (1 + x) / 2, w * 2.0 ** (-a - 1)


def _nq_for(end, Pf):
    """8 nodes when every other cone is ≥ 3 ray lengths from the segment 0→end
    (Gauss error ~(ρ+√(ρ²+1))^{-2n} < 1e-12), else ``NQ``."""
    e = end[:, None]
    t = np.clip(np.real(Pf * np.conj(e)) / np.maximum(np.abs(e) ** 2, 1e-300), 0, 1)
    d = np.abs(Pf - t * e)
    d = np.where(d < 1e-12 * (1 + np.abs(e)), np.inf, d)          # the endpoint's own cone
    return np.where(d.min(1) / np.maximum(np.abs(end), 1e-300) >= 3.0, 8, NQ)


def _logW(d, P, a):
    """``Σ a log(ζ − P)`` with ``d = ζ − P``, on the branch fixed at ζ = 0."""
    return ((np.log(-P) + np.log(d / -P)) * a).sum(-1)


def _ray_groups(end, aend, Pf):
    key = np.round(aend, 12) + 1j * _nq_for(end, Pf)
    for kk in np.unique(key):
        yield np.flatnonzero(key == kk), float(kk.real), int(kk.imag)


def _rays(end, aend, Pf, a):
    """``∫_0^{end} W dζ`` per ray; ``aend`` the endpoint's exponent (0 if flat)."""
    out = np.empty(len(end), complex)
    for sel, av, nq in _ray_groups(end, aend, Pf):
        t, w = _gj(av, nq)
        zz = t[None, :] * end[sel][:, None]
        P = Pf[sel][:, None, :]
        L = _logW(zz[..., None] - P, P, a)
        if av != 0.0:
            L = L - av * np.log1p(-t)
        out[sel] = end[sel] * (np.exp(L) @ w)
    return out


def _rays_d(end, aend, Pf, a, endcone):
    """:func:`_rays` and its holomorphic derivatives: ``dI/dP_m`` for every cone (an
    endpoint cone moves the endpoint too) and ``dI/d(end)`` for flat endpoints."""
    m_, nc = len(end), Pf.shape[1]
    I = np.empty(m_, complex)
    dP = np.zeros((m_, nc), complex)
    dE = np.zeros(m_, complex)
    for sel, av, nq in _ray_groups(end, aend, Pf):
        t, w = _gj(av, nq)
        e = end[sel][:, None]
        P = Pf[sel][:, None, :]
        d = t[None, :, None] * e[..., None] - P
        L = _logW(d, P, a)
        if av != 0.0:
            L = L - av * np.log1p(-t)
        Wn = np.exp(L)
        I[sel] = e[:, 0] * (Wn @ w)
        own = endcone[sel]
        inv = 1.0 / d
        if av != 0.0:
            inv[np.arange(len(sel)), :, own] = 0.0
        dP[sel] = -a * (e[:, 0, None] * np.einsum("snm,sn,n->sm", inv, Wn, w))
        if av != 0.0:     # endpoint cone: dI/dp = (a+1) I/p + Σ_m a_m ∫ (ζ/p) W/(ζ−P_m)
            Sm = e[:, 0, None] * np.einsum("snm,sn,n,n->sm", inv, Wn, w, t)
            dP[sel, own] = (av + 1) * I[sel] / e[:, 0] + Sm @ a
        else:
            dE[sel] = np.exp(_logW(end[sel][:, None] - Pf[sel], Pf[sel], a))
    return I, dP, dE


# ------------------------------------------------------------------ the map
class PolyMap:
    """The polyhedron's sphere map. ``solve()`` fits it; afterwards ``sphere`` (per
    vertex), ``fit`` (residual rms/max), ``D`` (developed faces), ``AB``/``logc``
    (developed plane → 3D) and the face charts ``Rf``, ``Pf``, ``Kf`` are set."""

    def __init__(self, V, F, S0=None, cone_tol: float = 1e-9):
        from ..quad.cones import choose_pole, rotation_to_north

        self.V, self.F = np.asarray(V, float), np.asarray(F)
        kap = angle_defects(self.V, self.F)
        self.cone = np.abs(kap) > cone_tol
        self.a_v = np.where(self.cone, -kap / (2 * np.pi), 0.0)
        self.ci = np.flatnonzero(self.cone)
        E1, E2 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        self.E1, self.E2 = E1, E2
        self.ll1, self.ll2 = np.log(np.linalg.norm(E1, axis=1)), np.log(np.linalg.norm(E2, axis=1))
        self.phi = np.arctan2(np.linalg.norm(np.cross(E1, E2), axis=1), np.einsum("ij,ij->i", E1, E2))
        if S0 is None:                                             # radial start
            S0 = _normalize(self.V - self.V.mean(0))
        self.Rm = rotation_to_north(choose_pole(S0, n_samples=2000)[0])
        self.z = chart(self.Rm, S0)
        zf = chart(face_rotations(S0[self.F].mean(1))[:, None], S0[self.F])
        o = np.sign(np.imag(np.conj(zf[:, 1] - zf[:, 0]) * (zf[:, 2] - zf[:, 0])))
        self.sigma = 1.0 if np.sum(o > 0) >= np.sum(o < 0) else -1.0   # chart orientation
        self.logc = 0.0

    @property
    def sphere(self) -> np.ndarray:
        return _unstereo(self.z) @ self.Rm

    def develop(self, z):
        S = _unstereo(z) @ self.Rm
        Rf = face_rotations(S[self.F].mean(1))
        Pf = chart(Rf[:, None], S[self.ci][None])
        a = self.a_v[self.ci]
        Kf = np.exp(-0.5 * np.sum(a * np.log1p(np.abs(Pf) ** 2), -1))
        zeta = chart(Rf[:, None], S[self.F])
        D = np.stack([_rays(zeta[:, k], self.a_v[self.F[:, k]], Pf, a) for k in range(3)], 1)
        self._chart = (Rf, Pf, Kf)
        return D * Kf[:, None]

    def residual(self, x):
        n = len(self.V)
        D = self.develop(x[:n] + 1j * x[n:2 * n])
        e1, e2 = D[:, 1] - D[:, 0], D[:, 2] - D[:, 0]
        return np.concatenate([np.log(np.abs(e1)) + x[-1] - self.ll1,
                               np.log(np.abs(e2)) + x[-1] - self.ll2,
                               np.angle(e2 / e1 * np.exp(-1j * self.sigma * self.phi)),
                               1e-7 * (x[:-1] - self.x0[:-1])])

    def jacobian(self, x):
        """Analytic sparse Jacobian of :meth:`residual`. The residual is invariant to
        the face charts, so they are held fixed: only the holomorphic dependence on
        vertex and cone positions (through the Möbius chart maps) and ``K_f`` enter."""
        n, m, nc = len(self.V), len(self.F), len(self.ci)
        F = self.F
        z = x[:n] + 1j * x[n:2 * n]
        Rf = face_rotations((_unstereo(z) @ self.Rm)[F].mean(1))
        a = self.a_v[self.ci]

        def phi(zz, R):
            return chart(R, _unstereo(zz) @ self.Rm)

        def dphi(zz, R):
            h = 1e-6 * (1 + np.abs(zz))
            return (phi(zz + h, R) - phi(zz - h, R)) / (2 * h)

        zc = np.broadcast_to(z[self.ci], (m, nc))
        Pf, dPf = phi(zc, Rf[:, None]), dphi(zc, Rf[:, None])
        zeta, dzeta = phi(z[F], Rf[:, None]), dphi(z[F], Rf[:, None])
        conepos = -np.ones(n, int)
        conepos[self.ci] = np.arange(nc)
        I = np.empty((m, 3), complex)
        dP = np.empty((m, 3, nc), complex)
        dE = np.empty((m, 3), complex)
        for k in range(3):
            I[:, k], dP[:, k], dE[:, k] = _rays_d(zeta[:, k], self.a_v[F[:, k]], Pf, a, conepos[F[:, k]])
        g1, g2 = 1.0 / (I[:, 1] - I[:, 0]), 1.0 / (I[:, 2] - I[:, 0])
        rows, cols, vals = [], [], []
        fr = np.arange(m)

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        def emit(G1, G2, u, dq):
            # G = ∂log(I_j − I_0)/∂q (holomorphic); q = chart position of vertex u, dq = dq/dz
            A1, A2 = G1 * dq, G2 * dq
            add(fr, u, A1.real); add(fr, n + u, -A1.imag)
            add(m + fr, u, A2.real); add(m + fr, n + u, -A2.imag)
            add(2 * m + fr, u, (A2 - A1).imag); add(2 * m + fr, n + u, (A2 - A1).real)

        for k in range(3):                                         # flat endpoints
            s1, s2 = (k == 1) - (k == 0), (k == 2) - (k == 0)
            emit(s1 * dE[:, k] * g1, s2 * dE[:, k] * g2, F[:, k], dzeta[:, k])
        for j in range(nc):                                        # cones
            u = np.full(m, self.ci[j])
            emit((dP[:, 1, j] - dP[:, 0, j]) * g1, (dP[:, 2, j] - dP[:, 0, j]) * g2, u, dPf[:, j])
            P, q = Pf[:, j], dPf[:, j]                             # log K_f: real gradient
            gx, gy = -a[j] * P.real / (1 + np.abs(P) ** 2), -a[j] * P.imag / (1 + np.abs(P) ** 2)
            for off in (0, m):
                add(off + fr, u, gx * q.real + gy * q.imag)
                add(off + fr, n + u, -gx * q.imag + gy * q.real)
        for off in (0, m):
            add(off + fr, np.full(m, 2 * n), np.ones(m))
        reg = np.arange(2 * n)
        add(3 * m + reg, reg, np.full(2 * n, 1e-7))
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(3 * m + 2 * n, 2 * n + 1)).tocsr()

    def solve(self, max_iter: int = 40):
        n, m = len(self.V), len(self.F)
        self.x0 = np.r_[self.z.real, self.z.imag, 0.0]
        D = self.develop(self.z)
        self.x0[-1] = np.mean(self.ll1 - np.log(np.abs(D[:, 1] - D[:, 0])))
        r0 = self.residual(self.x0)[:3 * m]
        x, it = _lm(self.residual, self.jacobian, self.x0, max_iter)
        self.z, self.logc = x[:n] + 1j * x[n:2 * n], x[-1]
        r = self.residual(x)[:3 * m]
        self.fit = dict(rms0=float(np.sqrt(np.mean(r0 ** 2))), rms=float(np.sqrt(np.mean(r ** 2))),
                        max=float(np.abs(r).max()), iterations=it)
        self._faces()
        return self

    def _faces(self):
        self.D = self.develop(self.z)
        self.Rf, self.Pf, self.Kf = self._chart
        e1, e2 = self.D[:, 1] - self.D[:, 0], self.D[:, 2] - self.D[:, 0]
        Mx = np.stack([np.stack([e1.real, e1.imag], -1), np.stack([e2.real, e2.imag], -1)], 1)
        self.AB = np.linalg.solve(Mx, np.stack([self.E1, self.E2], 1) / np.exp(self.logc))

    def gauge_cones(self):
        """Möbius-normalise: the ``|κ|``-weighted cone centroid to 0 (exact)."""
        from .conformal import ball_automorphism

        w = np.abs(self.a_v[self.ci])
        X, Sv = self.sphere[self.ci], self.sphere
        for _ in range(200):
            cvec = (w[:, None] * X).sum(0) / w.sum()
            if np.linalg.norm(cvec) < 1e-13:
                break
            step = cvec / max(1.0, 2.0 * np.linalg.norm(cvec))
            X, Sv = ball_automorphism(X, step), ball_automorphism(Sv, step)
        self.z = chart(self.Rm, Sv)
        D = self.develop(self.z)
        self.logc = float(np.mean(self.ll1 - np.log(np.abs(D[:, 1] - D[:, 0]))))
        self._faces()

    def _W(self, zeta, Pf, Kf):
        return np.exp(_logW(zeta[:, None] - Pf, Pf, self.a_v[self.ci])) * Kf

    def surface_samples(self, n_total: int = 20000, rk_steps: int = 12):
        """Area quadrature on the polyhedron, each sample mapped exactly to the sphere.
        Returns ``(sphere points, area weights, 3D points)``; the faces are left in
        ``self.sample_faces``. Samples that could not be inverted (a few, within
        round-off of a sharp corner) get weight 0; ``self.inv_fail`` counts them."""
        A = 0.5 * np.linalg.norm(np.cross(self.E1, self.E2), axis=1)
        kf = np.maximum(1, np.ceil(np.sqrt(A / (A.sum() / (n_total / 6.0))))).astype(int)
        g1, g2, w1, w2 = 0.445948490915965, 0.091576213509771, 0.223381589678011, 0.109951743655322
        Q = np.array([[g1, g1], [1 - 2 * g1, g1], [g1, 1 - 2 * g1],
                      [g2, g2], [1 - 2 * g2, g2], [g2, 1 - 2 * g2]])      # Dunavant degree 4
        QW = np.array([w1] * 3 + [w2] * 3)
        a = self.a_v[self.ci]
        xs, ws, Xs, fsl = [], [], [], []
        self.inv_fail = 0
        for k in np.unique(kf):
            fs = np.flatnonzero(kf == k)
            tri = []
            for i in range(k):
                for j in range(k - i):
                    tri.append([(i, j), (i + 1, j), (i, j + 1)])
                    if i + j < k - 1:
                        tri.append([(i + 1, j), (i + 1, j + 1), (i, j + 1)])
            tri = np.array(tri, float) / k
            pts = (tri[:, 0, None] + Q[None, :, 0, None] * (tri[:, 1] - tri[:, 0])[:, None]
                   + Q[None, :, 1, None] * (tri[:, 2] - tri[:, 0])[:, None]).reshape(-1, 2)
            pw = np.tile(QW, len(tri)) / len(tri)
            f = np.repeat(fs, len(pts))
            b = np.tile(pts, (len(fs), 1))
            T = self.D[f]
            Dstar = T[:, 0] + b[:, 0] * (T[:, 1] - T[:, 0]) + b[:, 1] * (T[:, 2] - T[:, 0])
            Pf, Kf = self.Pf[f], self.Kf[f]
            zeta, err = self._invert(f, b, Dstar, Pf, Kf, a, rk_steps)
            bad = err > 1e-8
            self.inv_fail += int(bad.sum())
            wk = np.repeat(A[fs], len(pts)) * np.tile(pw, len(fs))
            wk[bad] = 0.0
            V0, V1, V2 = (self.V[self.F[f, c]] for c in range(3))
            Xs.append(V0 + b[:, :1] * (V1 - V0) + b[:, 1:] * (V2 - V0))
            xs.append(np.einsum("nji,nj->ni", self.Rf[f], _unstereo(zeta)))
            ws.append(wk)
            fsl.append(f)
        self.sample_faces = np.concatenate(fsl)
        return np.concatenate(xs), np.concatenate(ws), np.concatenate(Xs)

    def _invert(self, f, b, Dstar, Pf, Kf, a, rk_steps):
        """ζ with ``K_f ∫_0^ζ W = D*`` (the face centre is ζ = 0 ↔ D = 0)."""
        def W(z_, idx):
            return self._W(z_, Pf[idx], Kf[idx])

        def Dof(z_, idx):
            return _rays(z_, np.zeros(len(idx)), Pf[idx], a) * Kf[idx]

        zeta = np.zeros(len(f), complex)
        err = np.full(len(f), np.inf)
        idx = np.arange(len(f))
        for nsteps in (rk_steps, 4 * rk_steps, 16 * rk_steps, 64 * rk_steps):
            zi = np.zeros(len(idx), complex)                       # RK4 on dζ/dD = 1/W
            h = Dstar[idx] / nsteps
            for _ in range(nsteps):
                k1 = h / W(zi, idx); k2 = h / W(zi + k1 / 2, idx)
                k3 = h / W(zi + k2 / 2, idx); k4 = h / W(zi + k3, idx)
                zi = zi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            for _ in range(3):                                     # Newton polish
                zn = zi - (Dof(zi, idx) - Dstar[idx]) / W(zi, idx)
                zi = np.where(np.isfinite(zn), zn, zi)
            ei = np.abs(Dof(zi, idx) - Dstar[idx])
            better = ei < err[idx]
            zeta[idx[better]], err[idx[better]] = zi[better], ei[better]
            idx = idx[err[idx] > 1e-11]
            if len(idx) == 0:
                return zeta, err
        # leftovers next to a sharp corner: damped Newton from the chart-linear guess
        zf = chart(self.Rf[f[idx]][:, None], self.sphere[self.F[f[idx]]])
        zi = zf[:, 0] + b[idx, 0] * (zf[:, 1] - zf[:, 0]) + b[idx, 1] * (zf[:, 2] - zf[:, 0])
        Dz, Wz = Dof(zi, idx), W(zi, idx)
        dc = np.abs(zi[:, None] - Pf[idx]).min(1)
        ei = np.abs(Dz - Dstar[idx])
        for _ in range(80):
            st = (Dz - Dstar[idx]) / Wz
            st = np.where(np.abs(st) > 0.5 * dc, st * 0.5 * dc / np.abs(st), st)
            lam = np.ones(len(zi))
            for _ in range(10):
                zt = zi - lam * st
                Dt, Wt = Dof(zt, idx), W(zt, idx)
                et = np.abs(Dt - Dstar[idx])
                ok = et < ei
                zi, Dz, Wz, ei = (np.where(ok, zt, zi), np.where(ok, Dt, Dz),
                                  np.where(ok, Wt, Wz), np.where(ok, et, ei))
                dc = np.where(ok, np.abs(zi[:, None] - Pf[idx]).min(1), dc)
                if ok.all():
                    break
                lam = np.where(ok, 0.0, lam * 0.5)
            if ei.max() < 1e-12:
                break
        better = ei < err[idx]
        zeta[idx[better]], err[idx[better]] = zi[better], ei[better]
        return zeta, err

    def spinor_at(self, x, faces):
        """λ (quaternion) and surface area density ``s²`` at sphere points ``x`` lying
        on ``faces``: from ``dX = M_f(c W dζ)``, the face's similarity per point."""
        from ..spectral.quaternion import qconj
        from .spin import rotation_to_quaternion

        f = faces
        Rf = self.Rf[f]
        Wz = self._W(chart(Rf, x), self.Pf[f], self.Kf[f])
        y = np.einsum("nij,nj->ni", Rf, x)
        t1 = _normalize(np.cross(x, np.where(np.abs(x[:, 2:3]) < 0.9, [0, 0, 1.0], [1.0, 0, 0])))
        t2 = np.cross(x, t1)
        c = np.exp(self.logc)

        def dX(t):
            dy = np.einsum("nij,nj->ni", Rf, t)
            dz = (dy[:, 0] + 1j * dy[:, 1]) / (1 - y[:, 2]) + (y[:, 0] + 1j * y[:, 1]) * dy[:, 2] / (1 - y[:, 2]) ** 2
            zt = Wz * dz
            return c * (zt.real[:, None] * self.AB[f, 0] + zt.imag[:, None] * self.AB[f, 1])

        d1, d2 = dX(t1), dX(t2)
        s = np.linalg.norm(d1, axis=1)
        B = np.stack([d1, d2, s[:, None] * _normalize(np.cross(d1, d2))], 2)
        R = B @ np.linalg.inv(np.stack([t1, t2, x], 2)) / s[:, None, None]
        return np.sqrt(s)[:, None] * qconj(rotation_to_quaternion(R)), s ** 2


def _lm(fun, jac, x0, max_iter):
    """Levenberg–Marquardt, sparse direct solves of the damped normal equations."""
    import scipy.sparse.linalg as spla

    x, r = x0.copy(), fun(x0)
    f, mu, hist = r @ r, 1.0, []
    for it in range(max_iter):
        J = jac(x)
        A, g = (J.T @ J).tocsc(), J.T @ r
        dA = A.diagonal()
        while True:
            dx = -spla.spsolve(A + sp.diags(mu * np.maximum(dA, 1e-12)), g)
            rt = fun(x + dx)
            if rt @ rt < f:
                x, r, f = x + dx, rt, rt @ rt
                mu = max(mu / 5, 1e-12)
                break
            mu *= 8
            if mu > 1e8:
                break
        hist.append(np.sqrt(f / len(r)))
        if mu > 1e8 or hist[-1] < 1e-11 or (len(hist) > 4 and hist[-1] > 0.98 * hist[-5]):
            break
    return x, it + 1


def solve_map(V, F, tol: float = 1e-8) -> PolyMap:
    """:meth:`PolyMap.solve` with restarts. Meshes over 150 vertices start from the
    eigen sphere (up to 3 tries: its rotation is random, and a bad one can leave a
    folded local minimum), then the radial projection; the best fit is kept."""
    from .eigen_sphere import conformal_sphere

    starts = (["eigen"] * 3 if len(V) > 150 else []) + ["radial"]
    best = None
    for st in starts:
        try:
            S0 = conformal_sphere(V, F) if st == "eigen" else None
        except Exception:
            continue
        pm = PolyMap(V, F, S0=S0).solve()
        pm.fit["start"] = st
        if best is None or pm.fit["rms"] < best.fit["rms"]:
            best = pm
        if pm.fit["rms"] < tol:
            break
    return best


def _align_signs(lam, X, k: int = 8):
    """Consistent ± over scattered sphere points: MST on ``2 − |cos|`` over a k-NN graph."""
    from scipy.spatial import cKDTree

    m = len(X)
    _, nb = cKDTree(X).query(X, k + 1)
    a, b = np.repeat(np.arange(m), k), nb[:, 1:].ravel()
    u = _normalize(lam)
    cosv = np.einsum("ij,ij->i", u[a], u[b])
    tree = csg.minimum_spanning_tree(sp.coo_matrix((2.0 - np.abs(cosv), (a, b)), shape=(m, m)).tocsr())
    order, pred = csg.breadth_first_order(tree + tree.T, 0, directed=False, return_predecessors=True)
    if len(order) < m:
        raise ValueError("the samples' k-NN graph is disconnected")
    child = order[1:]
    odd = np.zeros(m, bool)
    odd[child] = np.einsum("ij,ij->i", lam[child], lam[pred[child]]) < 0
    anc = pred.copy()
    anc[order[0]] = order[0]
    while np.any(anc != order[0]):
        odd, anc = odd ^ odd[anc], anc[anc]
    return lam * np.where(odd, -1.0, 1.0)[:, None]


def encode_polyhedral(V, F, L: int, n_samples: int | None = None, iters: int = 3,
                      over: int = 3, max_cones: int = 2000, lift=None, Ls: float | None = None) -> dict:
    """Polyhedron ``(V, F)`` → closed ρ̂ at band ``L`` (module docstring).

    Returns ``{"coeffs", "sphere"`` (per vertex), ``"gamma", "f_hat",
    "samples": (sphere points, area weights, 3D points), "map": PolyMap, "info"}``.
    ``n_samples`` defaults to ``max(12000, 30 (L+1)²)``; L=32 needs ~30k for 1e-5
    tessellation invariance.
    """
    from ..quad.lift import frame_quaternion
    from ..spectral.quaternion import qconj, qmul
    from .dirac_direct import _project_faces, close_code
    from .dirac_quotient import rho_quotient
    from .spectral_adjoint import SpectralLift

    V, F = np.asarray(V, float), np.asarray(F)
    ncones = int(np.sum(np.abs(angle_defects(V, F)) > 1e-9))
    if ncones > max_cones:
        raise ValueError(
            f"{ncones} cone vertices (nonzero angle defect): the exact map costs faces x cones. "
            f"For a dense smooth mesh use method='quotient' (eigen sphere), or raise max_cones.")
    pm = solve_map(V, F)
    pm.gauge_cones()
    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    Lm = L + 0.5
    n_samples = max(12000, 30 * (L + 1) ** 2) if n_samples is None else n_samples
    xs, wA, Xs = pm.surface_samples(n_samples)
    lam, s2 = pm.spinor_at(xs, pm.sample_faces)
    lam = _align_signs(lam, xs)
    w = wA / s2                                                    # sphere-area weights
    th, ph = np.arccos(np.clip(xs[:, 2], -1, 1)), np.arctan2(xs[:, 1], xs[:, 0])
    raw = qmul(qconj(frame_quaternion(th, ph)), lam)
    v = _project_faces(Lm, th, ph, w * (raw[:, 0] + 1j * raw[:, 1]), w * (raw[:, 2] - 1j * raw[:, 3]))
    rho = rho_quotient(v, L, Ls, Lm, iters=iters, over=over)
    n, k = len(ds.sw.half_modes(0.5, Ls)), len(v) // 2
    rho, res, gamma = close_code(rho, L, Ls, v0=np.r_[v[:k], np.zeros(n - k), v[k:], np.zeros(n - k)])
    lift = SpectralLift(Ls) if lift is None else lift
    info = dict(fit=pm.fit, ncones=len(pm.ci), n_samples=len(xs), inv_fail=pm.inv_fail,
                sphere_area=float(w.sum()))
    return {"coeffs": rho, "sphere": pm.sphere, "gamma": gamma,
            "f_hat": lift.forward(np.r_[res.psi1, res.psi2])[0],
            "samples": (xs, wA, Xs), "map": pm, "info": info}
