"""Exact conformal sphere map and spinor of a polyhedron: ``codec.encode(method="polyhedral")``.

A closed polyhedron is flat except at its vertices, cones of angle defect κ, so
its conformal map to the sphere has a closed form: in a stereographic chart the
metric is ``c |W(ζ)| |dζ|`` with ``W = Π (ζ − p_v)^{a_v}``, ``a_v = −κ_v/2π``. The
unknowns are the vertices' sphere positions ``p_v`` and the scale ``c``. The mesh
is used as given (no remesh, no decimation).

1. **Start** — the mesh's discrete uniformization
   (:func:`~shapegen.geom.uniformize.uniformize`): 0.13° median face-angle error
   on a 26k-vertex head.
2. **Fit** — :class:`~shapegen.geom.edge_periods.EdgeSystem`: one row per edge
   (its period's modulus along the great arc must be its 3D length), square in
   the positions and a per-vertex log scale; Newton–GMRES preconditioned by the
   chord model (discrete uniformization's own equations).
3. **Gauge** — :meth:`PolyMap.gauge_cones`: the ``|κ|``-weighted cone centroid to
   the origin, then the rotation that best matches the cones to their 3D
   directions, so codes are deterministic.
4. **Quadrature** — :meth:`PolyMap.sphere_quadrature`, in closed form: each face
   in its own chart as 6 curved Duffy patches bounded by its true edges (one
   collocated curve per edge, shared by its two faces), the spinor
   ``λ = √s conj(Q_f e^{kα/2} k û r_f)`` at every node, signs from the exact jump
   across edges. Nothing is inverted and no sign is guessed.
5. **Code** — λ projected onto spin-weighted harmonics, fitted (``rho_quotient``)
   and closed (``close_code``).

This module keeps the face charts (per-face cone lists, a far field per face
chart, adaptive rays) that stages 3–4 use once per encode (:meth:`PolyMap.solve`
leaves the development to the gauge).

Cost on the converged 26k-vertex FaceScape head (52k faces, all vertices cones,
L = 32): the development ~25 s once (charts, far field and 3 rays per face 14 s;
re-centring 478 sliver/turned-over faces whose default chart centre lies outside
them, 11 s), the quadrature ~10 s for 1.6M nodes (edge curves, 3.7k extra panels
at cones within 1e-4 of an edge); the whole encode ~40 s, decoded vertex error
0.083 (L = 16) / 0.068 (L = 32). Codes are tessellation-invariant to ~1e-6 (a hull
vs its flat subdivisions; the projection's own floor is ~3e-7).

Scans break three things a clean polyhedron does not: rays from the vertex-mean
centre that leave the face (:meth:`PolyMap.develop` re-centres), edges that pass
clusters of cones at 1e-4 of their length (panelled curves,
:meth:`PolyMap._edge_curves`), and crumpled vertices whose star folds back so the
minimal hinge lifts compose to −1 (a seam of sign flips, :meth:`PolyMap._seam`).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from . import dirac_spectral as ds

__all__ = ["angle_defects", "PolyMap", "solve_map", "encode_polyhedral"]

NQ = 8             # Gauss nodes per ray segment
SEP = 1.0          # ... once every cone not at the segment's end is ≥ SEP segment lengths away
NEAR = 2.0         # cones within NEAR face radii enter W one by one, the rest as a Taylor series
NT = 64            # Taylor terms of the far cones (ratio ≤ 1.15 / NEAR at the nodes: < 1e-15)
CURVE_TOL = 1e-6   # edge curves: (residual angle ≈ relative position error) × dihedral angle
SEPC = 0.1         # edge curves: panels end where a cone comes closer than SEPC panel lengths
RC = 1.2           # their check circle, in face radii: just outside the nodes (|ζ| ≤ R_f), so the
                   # tree's block (≥ the check circle) stays small; no loss as RC > |ζ|/R_f
HALF = 3.0         # far-field tree: the square |Re w|, |Im w| ≤ HALF of each global chart
P_FMM = 56         # its expansion order (error ~0.64^P at a face's check circle)


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


def chart(R: np.ndarray, X: np.ndarray, inverse=False) -> np.ndarray:
    """Stereographic coordinate ζ (pole at +z) of unit vectors ``X`` (…,3) after
    rotation ``R``; ``1/ζ`` where ``inverse`` (a bool or mask). Near the pole
    ``1 − Y_z`` cancels, so there it is ``(Y_x² + Y_y²)/(1 + Y_z)``: a cone opposite
    a face (a regular tetrahedron's) would otherwise get 1e-4 relative error."""
    Y = np.einsum("...ij,...j->...i", R, X)
    w, north = Y[..., 0] + 1j * Y[..., 1], Y[..., 2] > 0
    up, down = 1.0 + Y[..., 2], 1.0 - Y[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(north, up / np.conj(w), w / down)
        if np.any(inverse):
            z = np.where(inverse, np.where(north, np.conj(w) / up, down / w), z)
    return z


def _unstereo(z: np.ndarray) -> np.ndarray:
    r2 = np.abs(z) ** 2
    return np.stack([2 * z.real / (r2 + 1), 2 * z.imag / (r2 + 1), (r2 - 1) / (r2 + 1)], -1)


# ------------------------------------------------------------------ the cones a face sees
def _ranges(ptr, rows):
    """For CSR row pointers ``ptr``: the entries of ``rows``, concatenated, and which
    position in ``rows`` each belongs to."""
    cnt = ptr[rows + 1] - ptr[rows]
    owner = np.repeat(np.arange(len(rows)), cnt)
    return owner, np.arange(cnt.sum()) + np.repeat(ptr[rows] - (np.cumsum(cnt) - cnt), cnt)


class _Cones:
    """The cones as each face's chart sees them, one row per face (or per point,
    after :meth:`take`). ``log(K_f W(ζ)) = C + Σ_near a log(1 − ζq) + Σ_n G_n ζⁿ``:

    - near cones (within ``NEAR`` face radii, CSR ``ptr``; cone ``idx``, ``q = 1/P``,
      exponent ``an``, phase form ``pn``) enter one by one. Every row has at least
      one entry; empty rows hold a dummy with ``a = 0``.
    - far cones enter through the Taylor coefficients ``G`` (rows, ``NT``) of
      their ``Σ a log(1 − ζq)``, and their part of the constant ``C``."""

    def __init__(self, ptr, idx, q, an, pn, C, G):
        self.ptr, self.idx, self.q, self.an, self.pn, self.C, self.G = ptr, idx, q, an, pn, C, G

    def take(self, rows):
        _, pos = _ranges(self.ptr, rows)
        ptr = np.r_[0, np.cumsum(self.ptr[rows + 1] - self.ptr[rows])]
        return _Cones(ptr, self.idx[pos], self.q[pos], self.an[pos], self.pn[pos],
                      self.C[rows], self.G[rows])

    def replace(self, rows, other):
        """A copy with rows ``rows`` replaced by the rows of ``other`` (in order)."""
        n = len(self.ptr) - 1
        src = np.zeros(n, int)                                        # 0: self, 1: other
        src[rows] = 1
        pos_in = np.zeros(n, int)
        pos_in[rows] = np.arange(len(rows))
        cnt = np.where(src == 1, np.diff(other.ptr)[pos_in], np.diff(self.ptr))
        ptr = np.r_[0, np.cumsum(cnt)]
        out = {}
        keep = np.flatnonzero(src == 0)
        dst_k = _ranges(ptr, keep)[1]
        dst_o = _ranges(ptr, rows)[1]
        src_k = _ranges(self.ptr, keep)[1]
        src_o = np.arange(other.ptr[-1])
        for name in ("idx", "q", "an", "pn"):
            a_, b_ = getattr(self, name), getattr(other, name)
            v = np.zeros(ptr[-1], a_.dtype)
            v[dst_k], v[dst_o] = a_[src_k], b_[src_o]
            out[name] = v
        C, G = self.C.copy(), self.G.copy()
        C[rows], G[rows] = other.C, other.G
        return _Cones(ptr, out["idx"], out["q"], out["an"], out["pn"], C, G)

    @property
    def owner(self):
        return np.repeat(np.arange(len(self.ptr) - 1), np.diff(self.ptr))

    def logW(self, z, deriv: bool = False, arg: bool = False):
        """``log(K_f W(z))`` at points ``z`` (rows, n), the branch fixed at z = 0, and
        with ``deriv`` also its ζ-derivative ``Σ −a q/(1 − ζq) + Σ (n+1) G_n ζⁿ``.
        Real log and ``arctan2`` separately: half the cost of a complex log; ``arg``
        returns only the imaginary part (``arg K_f W``), without the log."""
        u = 1.0 - z[self.owner] * self.q[:, None]
        if arg:
            t = np.arctan2(u.imag, u.real) * self.an[:, None]
        else:
            t = (0.5 * np.log(u.real ** 2 + u.imag ** 2) + 1j * np.arctan2(u.imag, u.real)) * self.an[:, None]
        acc = np.zeros_like(z)
        for n in range(self.G.shape[1] - 1, -1, -1):              # Horner: Σ G_n zⁿ⁺¹
            acc = (acc + self.G[:, n, None]) * z
        lw = np.add.reduceat(t, self.ptr[:-1], axis=0) + ((self.C[:, None] + acc).imag if arg else self.C[:, None] + acc)
        if not deriv:
            return lw
        d = np.zeros_like(z)
        for n in range(self.G.shape[1] - 1, 0, -1):
            d = (d + (n + 1) * self.G[:, n, None]) * z
        d += self.G[:, 0, None]
        return lw, d + np.add.reduceat(-(self.an * self.q)[:, None] / u, self.ptr[:-1], axis=0)

    def rows(self, z, rows=None, deriv: bool = False, budget: int = 4_000_000):
        """:meth:`logW` of rows ``rows`` (all if None) at ``z`` (len(rows), n), in
        chunks of ≤ ``budget`` (entry × point) terms."""
        rows = np.arange(len(self.ptr) - 1) if rows is None else rows
        cnt = np.cumsum((self.ptr[rows + 1] - self.ptr[rows]) * z.shape[1])
        cuts = np.r_[0, np.searchsorted(cnt, np.arange(budget, cnt[-1] if len(cnt) else 0, budget)), len(rows)]
        out = [self.take(rows[a:b]).logW(z[a:b], deriv) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
        if deriv:
            return (np.concatenate([o[0] for o in out]) if out else np.zeros_like(z),
                    np.concatenate([o[1] for o in out]) if out else np.zeros_like(z))
        return np.concatenate(out) if out else np.zeros_like(z)


def _cone_terms(q, a, pn):
    """Each cone's constant ``log(−P) − ½log(1+|P|²)`` (``pn``: near the chart centre,
    ``|P| ≤ 1``, or a corner of the face), else ``−½log(1+|q|²)``. The second drops a
    phase (a rotation of the face's developed plane, which nothing depends on), so no
    term grows with ``|P|``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return a * np.where(pn, -np.log(-q) - 0.5 * np.log1p(1.0 / np.abs(q) ** 2),
                            -0.5 * np.log1p(np.abs(q) ** 2))


def _face_cones(S, F, Rf, ci, a, conepos, C=None):
    """:class:`_Cones` of every face, from the vertices' sphere positions ``S``; ``C``:
    the faces' chart centres (default the normalised vertex means)."""
    from scipy.spatial import cKDTree

    m, X = len(F), S[ci]
    C = _normalize(S[F].mean(1)) if C is None else _normalize(C)
    rho = np.arccos(np.clip(np.einsum("fki,fi->fk", S[F], C), -1, 1)).max(1)    # face radius
    reach = np.minimum(NEAR * rho, np.pi)
    lists = cKDTree(X).query_ball_point(C, 2 * np.sin(reach / 2) + 1e-12)
    cnt = np.array([max(len(l), 1) for l in lists])
    ptr = np.r_[0, np.cumsum(cnt)]
    idx = np.zeros(ptr[-1], int)
    an = np.zeros(ptr[-1])
    for f, l in enumerate(lists):                                  # dummies keep a = 0
        if l:
            idx[ptr[f]:ptr[f + 1]] = l
            an[ptr[f]:ptr[f + 1]] = a[l]
    owner = np.repeat(np.arange(m), cnt)
    q = chart(Rf[owner], X[idx], inverse=True)
    own = np.zeros(len(idx), bool)
    for k in range(3):
        own |= idx == conepos[F[owner, k]]
    pn = own | (np.abs(q) >= 1)
    Cf = np.add.reduceat(_cone_terms(q, an, pn), ptr[:-1])
    far_C, G = _far_field(X, a, C, Rf, np.tan(rho / 2), ptr, idx, an)
    Cf += far_C
    return _Cones(ptr, idx, q, an, pn, Cf, G)


def _dsigma(y, v):
    """Differential of the stereographic chart at the rotated point ``y`` along ``v``."""
    return ((v[..., 0] + 1j * v[..., 1]) / (1 - y[..., 2])
            + (y[..., 0] + 1j * y[..., 1]) * v[..., 2] / (1 - y[..., 2]) ** 2)


def _far_field(X, a, C, Rf, R, ptr, idx, an):
    """The far cones' part of every face's ``log(K_f W)``: its constant (real part)
    and Taylor coefficients ``G`` (faces, ``NT``) in the face chart.

    The cones split, per face, into near ones (entered exactly elsewhere), a *tree*
    part and *series* ones. Each face is served by one of two global stereographic
    charts ``w`` (the one whose centre is in its hemisphere). If its check circle
    (radius ``RC R_f`` in the face chart) lies in the chart's core ``|w| ≤ 1.5``,
    :func:`~shapegen.geom.log_fmm.far_taylor` sums the cones outside a 3 × 3 block of
    boxes around it, at the level whose boxes cover its near cones and are ≥ 4 × its
    check circle, and the cones near the chart's pole (``|w| > HALF``) are one series
    about the chart centre. That tree part is taken to the face chart on the check
    circle, where ``Σ a [log(ζ−ζ_k) − ½log(1+|ζ_k|²)] = Σ a [log(w−w_k) −
    ½log(1+|w_k|²)] + (A/2) log(dζ/dw)`` up to an imaginary constant, and read off by
    FFT. Every other cone (the rest of the block; all cones for a face outside the
    core) is a series ``−a qⁿ/n`` in the face chart itself."""
    from .log_fmm import far_taylor

    m, PT = len(C), NT + 32                                        # aliasing ≤ Σ|a| (RC/NEAR)^PT
    owner = np.repeat(np.arange(m), np.diff(ptr))
    nearent = np.flatnonzero(an != 0)
    Cfar = np.zeros(m)
    G = np.zeros((m, NT), complex)
    th = 2 * np.pi * np.arange(PT) / PT
    pf_all, pk_all = [], []
    for Rg, sel in ((np.eye(3), C[:, 2] <= 0), (np.diag([1.0, -1.0, -1.0]), C[:, 2] > 0)):
        fs = np.flatnonzero(sel)
        if len(fs) == 0:
            continue
        fpos = np.full(m, -1)
        fpos[fs] = np.arange(len(fs))
        w = chart(Rg, X)
        c = chart(Rg, C[fs])
        zc = (RC * R[fs])[:, None] * np.exp(1j * th)               # check circle, face chart
        xc = np.einsum("fji,fpj->fpi", Rf[fs], _unstereo(zc))
        wc = chart(Rg, xc)
        ent = nearent[sel[owner[nearent]]]
        ef, ek = fpos[owner[ent]], idx[ent]                         # near cones of these faces
        r_near = np.zeros(len(fs))
        np.maximum.at(r_near, ef, np.abs(w[ek] - c[ef]))
        r_check = np.abs(wc - c[:, None]).max(1)
        dom = np.abs(w) <= HALF
        core = np.abs(wc).max(1) <= 1.5
        # finest level whose box covers the near cones and puts the check circle within
        # 0.6 of the local expansion's radius (1.5 box widths from the box centre)
        lev = np.zeros(len(fs), int)
        for L_ in range(20, 0, -1):
            hb = 2 * HALF / 2 ** L_
            ctr = (-HALF + (np.floor((c.real + HALF) / hb) + 0.5) * hb) + 1j * (-HALF + (np.floor((c.imag + HALF) / hb) + 0.5) * hb)
            ok = (lev == 0) & (hb >= r_near) & (np.abs(c - ctr) + r_check <= 0.9 * hb)
            lev[ok] = L_
        lev[~core] = 0
        di = np.flatnonzero(dom)
        T, bptr, bidx = far_taylor(w[di], a[di], np.where(core, c, 0.0), lev, HALF, P_FMM)
        T[~core] = 0.0
        # series cones: the block minus the near ones (core), else every cone but the near ones
        nc_ = len(X)
        chw = -0.5 * a * np.log1p(np.abs(w) ** 2)
        polar = np.flatnonzero(~dom)
        bf = np.repeat(np.arange(len(fs)), np.diff(bptr))
        bk = di[bidx]
        nearkey = ef * nc_ + ek
        keep = core[bf] & ~np.isin(bf * nc_ + bk, nearkey)
        of = np.flatnonzero(~core)
        of_f, of_k = np.repeat(of, nc_), np.tile(np.arange(nc_), len(of))
        okeep = ~np.isin(of_f * nc_ + of_k, nearkey)
        pf_all.append(fs[np.r_[bf[keep], of_f[okeep]]])
        pk_all.append(np.r_[bk[keep], of_k[okeep]])
        npol = ~dom[ek]
        A_tree = (a[di].sum() - np.bincount(bf, a[bk], len(fs)) + a[polar].sum()
                  - np.bincount(ef, a[ek] * npol, len(fs)))
        ch_tree = (chw[di].sum() - np.bincount(bf, chw[bk], len(fs)) + chw[polar].sum()
                   - np.bincount(ef, chw[ek] * npol, len(fs)))
        # the tree part on the check circle (core faces)
        cf = np.flatnonzero(core)
        if len(cf):
            dw = wc[cf] - c[cf, None]
            val = np.zeros(dw.shape, complex)
            for n_ in range(P_FMM, -1, -1):
                val = val * dw + T[cf, n_, None]
            if len(polar):                                           # polar cones about the chart centre
                pc_ = np.zeros(P_FMM + 1, complex)
                inv = 1.0 / w[polar]
                pc_[0] = np.sum(a[polar] * np.log(np.abs(w[polar])))
                pwp = np.ones(len(polar), complex)
                for n_ in range(1, P_FMM + 1):
                    pwp = pwp * inv
                    pc_[n_] = -np.sum(a[polar] * pwp) / n_
                pv = np.zeros(dw.shape, complex)
                for n_ in range(P_FMM, -1, -1):
                    pv = pv * wc[cf] + pc_[n_]
                val += pv
                cpos = np.full(len(fs), -1)
                cpos[cf] = np.arange(len(cf))
                pol = (~dom[ek]) & core[ef]                          # the near ones out again
                for e_f, e_k in zip(ef[pol].tolist(), ek[pol].tolist()):
                    t_ = np.full(PT, a[e_k] * np.log(np.abs(w[e_k])), complex)
                    pwp = np.ones(PT, complex)
                    for n_ in range(1, P_FMM + 1):
                        pwp = pwp * wc[e_f] / w[e_k]
                        t_ -= a[e_k] * pwp / n_
                    val[cpos[e_f]] -= t_
            tv = np.cross(xc[cf], np.where(np.abs(xc[cf][..., 2:3]) < 0.9, [0, 0, 1.0], [1.0, 0, 0]))
            ratio = (_dsigma(np.einsum("fij,fpj->fpi", Rf[fs[cf]], xc[cf]), np.einsum("fij,fpj->fpi", Rf[fs[cf]], tv))
                     / _dsigma(np.einsum("ij,fpj->fpi", Rg, xc[cf]), np.einsum("ij,fpj->fpi", Rg, tv)))
            lr = np.log(np.abs(ratio)) + 1j * np.unwrap(np.angle(ratio), axis=1)
            val += ch_tree[cf, None] + 0.5 * A_tree[cf, None] * lr
            f_ = np.fft.fft(val, axis=1) / PT
            Cfar[fs[cf]] = f_[:, 0].real
            G[fs[cf]] = f_[:, 1:NT + 1] / ((RC * R[fs[cf]])[:, None] ** np.arange(1, NT + 1))
    # series cones, in each face's own chart: powers by cumulative products, summed
    # per face over the face-sorted pairs. A cone at |P| contributes (1.25 R/|P|)ⁿ at
    # the nodes (|ζ| ≤ R, measured), so it gets only the terms that reach 1e-16:
    # most of a block is many face radii away and needs a fraction of NT.
    pf, pk = np.concatenate(pf_all), np.concatenate(pk_all)
    srt = np.argsort(pf, kind="stable")
    pf, pk = pf[srt], pk[srt]
    inv_n = -1.0 / np.arange(1, NT + 1)
    for s0 in range(0, len(pf), 500_000):
        f_, k_ = pf[s0:s0 + 500_000], pk[s0:s0 + 500_000]
        q = chart(Rf[f_], X[k_], inverse=True)
        Cfar += np.bincount(f_, -0.5 * a[k_] * np.log1p(np.abs(q) ** 2), m)
        with np.errstate(divide="ignore"):
            nt = np.ceil(-37.0 / np.log(np.minimum(1.25 * R[f_] * np.abs(q), 0.99)))
        nt = np.minimum(8 * np.ceil(np.maximum(nt, 1) / 8), NT).astype(int)
        for n_ in np.unique(nt):
            sel = np.flatnonzero(nt == n_)
            fsel = f_[sel]
            v = np.cumprod(np.broadcast_to(q[sel, None], (len(sel), n_)), axis=1) * (a[k_[sel]][:, None] * inv_n[:n_])
            st = np.flatnonzero(np.r_[True, fsel[1:] != fsel[:-1]])
            G[fsel[st], :n_] += np.add.reduceat(v, st, axis=0)
    return Cfar, G


# ------------------------------------------------------------------ W along rays
@lru_cache(maxsize=None)
def _gj(a: float, nq: int):
    """Nodes/weights on [0, 1] for ``∫ (1−t)^a g(t) dt``."""
    from scipy.special import roots_jacobi
    x, w = roots_jacobi(nq, a, 0.0)
    return (1 + x) / 2, w * 2.0 ** (-a - 1)


def _segments(end, cones, own):
    """Split rays ``0 → end`` (one per row of ``cones``) into ``(ray, t0, t1)``
    segments, bisecting until every near cone but the endpoint's own (``own``, −1 if
    none) is ≥ ``SEP`` segment lengths away: ``NQ``-point Gauss then errs < ~1e-13
    (Bernstein ellipse ρ ≥ 5.8). Far cones are that far by construction."""
    r, t0, t1 = np.arange(len(end)), np.zeros(len(end)), np.ones(len(end))
    out = []
    for _ in range(50):
        c = cones.take(r)
        o = c.owner
        with np.errstate(divide="ignore", invalid="ignore"):
            P = 1.0 / c.q
        P = np.where(np.isfinite(P), P, 1e300)
        e = end[r][o]
        s = np.clip(np.real(P * np.conj(e)) / np.maximum(np.abs(e) ** 2, 1e-300), t0[o], t1[o])
        d = np.abs(P - s * e)
        d[(t1[o] == 1.0) & (c.idx == own[r][o]) | (c.an == 0)] = np.inf
        ok = np.minimum.reduceat(d, c.ptr[:-1]) >= SEP * (t1 - t0) * np.abs(end[r])
        out.append((r[ok], t0[ok], t1[ok]))
        r, t0, t1 = r[~ok], t0[~ok], t1[~ok]
        if len(r) == 0:
            break
        mid = 0.5 * (t0 + t1)
        r, t0, t1 = np.r_[r, r], np.r_[t0, mid], np.r_[mid, t1]
    out.append((r, t0, t1))                                        # a cone on the ray: give up
    return (np.concatenate(c) for c in zip(*out))


def _gj_batch(alpha, n):
    """Gauss–Jacobi nodes/weights on [0, 1] for ``∫ (1−t)^α g(t) dt``, one rule per
    ``alpha`` (Golub–Welsch on the stacked Jacobi matrices: a scan has a different
    cone exponent at every vertex). Solved once per distinct exponent."""
    ua, inv = np.unique(np.asarray(alpha, float), return_inverse=True)
    x, w = _gj_rules(ua, n)
    return x[inv], w[inv]


def _gj_rules(alpha, n):
    al = np.asarray(alpha, float)[:, None]
    k = np.arange(n, dtype=float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -al ** 2 / ((2 * k + al) * (2 * k + al + 2))
        d[:, 0] = -al[:, 0] / (al[:, 0] + 2)
        kk = k[:, 1:]
        off = np.sqrt(4 * kk * (kk + al) * kk * (kk + al) / ((2 * kk + al) ** 2 * (2 * kk + al + 1) * (2 * kk + al - 1)))
    J = np.zeros((len(al), n, n))
    i = np.arange(n)
    J[:, i, i] = d
    J[:, i[:-1], i[1:]] = J[:, i[1:], i[:-1]] = off
    x, vec = np.linalg.eigh(J)
    w = (2.0 ** (al[:, 0] + 1) / (al[:, 0] + 1))[:, None] * vec[:, 0, :] ** 2
    return (1 + x) / 2, w * 2.0 ** (-al - 1)


def _quadrature(end, aend, cones, own):
    """Nodes and weights of ``∫_0^{end} W dζ``: ``(ray, t (s, NQ), w (s, NQ), a_end (s,))``
    per segment. A segment ending at a cone uses Gauss–Jacobi with that cone's
    exponent, and its weights carry ``(1−t)^a``."""
    r, t0, t1 = _segments(end, cones, own)
    av = np.where(t1 == 1.0, aend[r], 0.0)
    x, w = _gj_batch(av, NQ)
    h = (t1 - t0)[:, None]
    return r, t0[:, None] + h * x, h ** (1 + av[:, None]) * w, av


def _chunks(r, cones, budget=4_000_000):
    """Index slices of the segments ``r`` with ≤ ``budget`` (entry × node) terms each."""
    cnt = np.diff(cones.ptr)[r] * NQ
    cs = np.cumsum(cnt)
    cuts = np.searchsorted(cs, np.arange(budget, cs[-1] if len(cs) else 0, budget))
    edges = np.r_[0, cuts, len(r)]
    return [slice(a_, b_) for a_, b_ in zip(edges[:-1], edges[1:]) if b_ > a_]


def _rays(end, aend, cones, own):
    """``K_f ∫_0^{end} W dζ`` per ray (one per row of ``cones``). ``aend`` is the
    endpoint's exponent (0 if flat) and ``own`` its cone (−1 if flat)."""
    out = np.zeros(len(end), complex)
    R_, T_, W_, A_ = _quadrature(end, aend, cones, own)
    for sl in _chunks(R_, cones):
        r, t, w, av = R_[sl], T_[sl], W_[sl], A_[sl]
        L = cones.take(r).logW(t * end[r][:, None]) - av[:, None] * np.log1p(-t)
        v = end[r] * np.sum(np.exp(L) * w, 1)
        out += np.bincount(r, v.real, len(end)) + 1j * np.bincount(r, v.imag, len(end))
    return out


def _curve_basis(t, n):
    """``φ_i(t) = t(1 − t) T_i(2t − 1)`` and ``φ_i'(t)``, i < n, at ``t`` (…): (…, n)."""
    from numpy.polynomial import chebyshev as ch

    t = np.asarray(t, float)
    T = ch.chebvander(2 * t - 1, n - 1)
    dT = (ch.chebvander(2 * t - 1, n - 2) @ ch.chebder(np.eye(n), axis=0)) if n > 1 else np.zeros_like(T)
    b = (t * (1 - t))[..., None]
    return b * T, (1 - 2 * t)[..., None] * T + 2 * b * dT


# ------------------------------------------------------------------ the map
class PolyMap:
    """The polyhedron's sphere map. ``solve()`` fits it; afterwards ``sphere`` (per
    vertex), ``fit`` (residual rms/max), ``D`` (developed faces), ``AB``/``logc``
    (developed plane → 3D) and the face charts are set: rotations ``Rf`` and the
    cones as each chart sees them (``cones``, :class:`_Cones`)."""

    def __init__(self, V, F, S0=None, cone_tol: float = 1e-9):
        from ..quad.cones import choose_pole, rotation_to_north

        self.V, self.F = np.asarray(V, float), np.asarray(F)
        kap = angle_defects(self.V, self.F)
        self.cone = np.abs(kap) > cone_tol
        self.a_v = np.where(self.cone, -kap / (2 * np.pi), 0.0)
        self.ci = np.flatnonzero(self.cone)
        self.conepos = np.full(len(self.V), -1)                    # vertex → cone index
        self.conepos[self.ci] = np.arange(len(self.ci))
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
        self._cache = None
        self.recentred = self.recentre_failed = 0

    @property
    def sphere(self) -> np.ndarray:
        return _unstereo(self.z) @ self.Rm

    def develop(self, z):
        """The developed corners ``D`` (m, 3) of every face, by rays from its chart
        centre; sets ``self._chart = (Rf, cones)``. Cached for the last ``z``.

        A ray from the chart centre to a corner develops the face only if it stays in
        the face. The default centre (the normalised vertex mean) is outside the face
        for slivers and for faces whose great-arc triangle is turned over (on a
        converged 26k-vertex head: 367 faces, 23 of them with rays passing a nearby
        cone on the wrong side, which put their developed sides off by up to 0.8 in
        log). Faces whose developed triangle does not have the 3D shape, or whose
        centre develops outside it, are re-centred (:meth:`_recentre`)."""
        if self._cache is not None and np.array_equal(self._cache[0], z):
            Rf, cones, D = self._cache[1]
            self._chart = (Rf, cones)
            return D.copy()
        m = len(self.F)
        S = _unstereo(z) @ self.Rm
        Rf, cones, D = self._develop_at(S, np.arange(m), S[self.F].mean(1))
        bad = self._misdeveloped(D)
        if 0 < len(bad) <= max(16, 0.05 * m):
            Rf, cones, D = self._recentre(S, bad, Rf, cones, D)
        self.recentred = int(len(bad))
        self._cache = (z.copy(), (Rf, cones, D))
        self._chart = (Rf, cones)
        return D.copy()

    def _develop_at(self, S, faces, C):
        """Charts, cones and developed corners of ``faces`` with chart centres ``C``."""
        Rf = face_rotations(_normalize(C))
        cones = _face_cones(S, self.F[faces], Rf, self.ci, self.a_v[self.ci], self.conepos, C=C)
        zeta = chart(Rf[:, None], S[self.F[faces]])
        D = np.stack([_rays(zeta[:, k], self.a_v[self.F[faces, k]], cones, self.conepos[self.F[faces, k]])
                      for k in range(3)], 1)
        return Rf, cones, D

    def _shape_err(self, D, faces=None):
        """Per face: how far the developed triangle is from the 3D one (log side ratio,
        angle; scale-free), and the smallest barycentric coordinate of the chart
        centre (developed at 0) in it."""
        f = np.arange(len(self.F)) if faces is None else faces
        e1, e2 = D[:, 1] - D[:, 0], D[:, 2] - D[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.maximum(np.abs(np.log(np.abs(e1) / np.abs(e2)) - self.ll1[f] + self.ll2[f]),
                             np.abs(np.angle(e2 / e1 * np.exp(-1j * self.sigma * self.phi[f]))))
            A = np.imag(np.conj(e1) * e2)
            b = np.stack([np.imag(np.conj(D[:, 1]) * D[:, 2]), np.imag(np.conj(D[:, 2]) * D[:, 0]),
                          np.imag(np.conj(D[:, 0]) * D[:, 1])], 1) / A[:, None]
        return np.where(np.isfinite(err), err, np.inf), np.where(np.isfinite(b), b, -np.inf).min(1)

    def _misdeveloped(self, D):
        err, inner = self._shape_err(D)
        tol = max(1e-9, 100 * float(np.median(err)))
        return np.flatnonzero((err > tol) | (inner < 0.05))

    def _recentre(self, S, bad, Rf, cones, D):
        """New chart centres for faces ``bad``. A centre is good when the face develops
        from it (rays to the corners) to its 3D shape, and the better the deeper
        inside the developed triangle it lands (smallest barycentric coordinate,
        accepted at ≥ 0.05). Tried in turn while faces are left: Newton towards the
        point that develops to the developed centroid (``ζ ← ζ + (g − D(ζ))/K W(ζ)``,
        rays from the current centre), from the default centre; for faces still with
        no centre inside, a polar grid around each corner (radii 1e-3–0.8 of its
        shorter adjacent chord, coarse then fine); Newton again from the best centre
        so far (whose rays are in class). Returns ``(Rf, cones, D)``."""
        err0, _ = self._shape_err(D)
        tol = max(1e-9, 100 * float(np.median(err0)))
        nb = len(bad)
        best_s = np.full(nb, -np.inf)
        best = [None] * nb                                            # (batch, position)
        batches = []
        cur_R, cur_cones, cur_D = Rf[bad], cones.take(bad), D[bad]    # chart of each face's best

        def newton(rows):
            cb = cur_cones.take(rows)
            g = cur_D[rows].mean(1)
            zc = np.zeros(len(rows), complex)
            with np.errstate(all="ignore"):
                for _ in range(8):
                    Dz = _rays(zc, np.zeros(len(rows)), cb, np.full(len(rows), -1))
                    zc = zc + (g - Dz) / np.exp(cb.logW(zc[:, None])[:, 0])
            return np.einsum("nji,nj->ni", cur_R[rows], _unstereo(zc))

        def trial(rows, Cs):
            nonlocal cur_R, cur_cones, cur_D
            ok = np.all(np.isfinite(Cs), 1)
            rows, Cs = rows[ok], Cs[ok]
            if not len(rows):
                return
            Rc, cc, Dc = self._develop_at(S, bad[rows], Cs)
            err, inner = self._shape_err(Dc, bad[rows])
            score = np.where(err <= tol, inner, -np.inf)
            o = np.lexsort((score, rows))                             # per row, best last
            last = np.flatnonzero(np.r_[rows[o][1:] != rows[o][:-1], True])
            rb, pb = rows[o][last], o[last]
            up = score[pb] > best_s[rb]
            rb, pb = rb[up], pb[up]
            if not len(rb):
                return
            batches.append((Rc[pb], cc.take(pb), Dc[pb]))
            best_s[rb] = score[pb]
            for i, r_ in enumerate(rb):
                best[r_] = (len(batches) - 1, i)
            cur_R = cur_R.copy(); cur_R[rb] = Rc[pb]
            cur_cones = cur_cones.replace(rb, cc.take(pb))
            cur_D = cur_D.copy(); cur_D[rb] = Dc[pb]

        trial(np.arange(nb), newton(np.arange(nb)))
        for radii, nang in (((0.01, 0.1, 0.4), 12), (np.geomspace(1e-3, 0.8, 8), 24)):
            left = np.flatnonzero(best_s < 0)                          # no centre inside yet
            if not len(left):
                break
            P = chart(Rf[bad[left]][:, None], S[self.F[bad[left]]])
            ang = np.exp(2j * np.pi * (np.arange(nang) + 0.5) / nang)
            rows, zs = [], []
            for k in range(3):
                lk = np.minimum(np.abs(P[:, (k + 1) % 3] - P[:, k]), np.abs(P[:, (k + 2) % 3] - P[:, k]))
                for r in radii:
                    for a_ in ang:
                        rows.append(left); zs.append(P[:, k] + r * lk * a_)
            rows, zs = np.concatenate(rows), np.concatenate(zs)
            trial(rows, np.einsum("nji,nj->ni", Rf[bad[rows]], _unstereo(zs)))
        left = np.flatnonzero((best_s < 0.05) & np.isfinite(best_s))
        if len(left):                                                  # from a centre whose rays are in class
            trial(left, newton(left))
        fix = np.flatnonzero(np.isfinite(best_s))
        self.recentre_failed = int(nb - len(fix))
        if len(fix):
            Rf, D = Rf.copy(), D.copy()
            Rf[bad[fix]], D[bad[fix]] = cur_R[fix], cur_D[fix]
            cones = cones.replace(bad[fix], cur_cones.take(fix))
        return Rf, cones, D

    def residual(self, x):
        n = len(self.V)
        D = self.develop(x[:n] + 1j * x[n:2 * n])
        e1, e2 = D[:, 1] - D[:, 0], D[:, 2] - D[:, 0]
        return np.concatenate([np.log(np.abs(e1)) + x[-1] - self.ll1,
                               np.log(np.abs(e2)) + x[-1] - self.ll2,
                               np.angle(e2 / e1 * np.exp(-1j * self.sigma * self.phi))])

    def solve(self, max_iter: int = 30):
        """Fit the map by its edge periods (:class:`~shapegen.geom.edge_periods.EdgeSystem`:
        one row per edge, Newton–GMRES). The faces are developed on first use."""
        from .edge_periods import EdgeSystem

        es = EdgeSystem(self.V, self.F)
        X, u, info = es.solve(self.sphere, max_iter=max_iter)
        self.z = chart(self.Rm, X)
        self.logc = float(np.mean(u))
        self.fit = dict(rms0=info["rms0"], rms=info["rms"], max=info["max"], iterations=info["newton"],
                        gmres=info["gmres"], evals=info["evals"], u_spread=info["u_spread"])
        for name in ("D", "AB", "Rf", "cones"):                         # developed on first use
            self.__dict__.pop(name, None)
        return self

    def __getattr__(self, name):
        # D, AB, Rf, cones: developed when first used after solve(); gauge_cones (which
        # moves the map) develops anyway, so an encode develops once, not twice
        if name in ("D", "AB", "Rf", "cones") and "z" in self.__dict__:
            self._faces()
            return self.__dict__[name]
        raise AttributeError(name)

    def _faces(self, scale: bool = False):
        """Develop the faces at ``z``; ``scale`` also sets ``log c`` from the developed
        first sides."""
        self.D = self.develop(self.z)
        self.Rf, self.cones = self._chart
        e1, e2 = self.D[:, 1] - self.D[:, 0], self.D[:, 2] - self.D[:, 0]
        if scale:
            self.logc = float(np.mean(self.ll1 - np.log(np.abs(e1))))
        Mx = np.stack([np.stack([e1.real, e1.imag], -1), np.stack([e2.real, e2.imag], -1)], 1)
        self.AB = np.linalg.solve(Mx, np.stack([self.E1, self.E2], 1) / np.exp(self.logc))

    def gauge_cones(self):
        """Möbius-normalise: the ``|κ|``-weighted cone centroid to 0 (exact). Then
        rotate so the cones best match (``|κ|``-weighted) their directions from the
        surface centroid: the rotation follows the input's orientation instead of
        the start's, so the code is deterministic."""
        from .conformal import ball_automorphism

        w = np.abs(self.a_v[self.ci])
        X, Sv = self.sphere[self.ci], self.sphere
        for _ in range(200):
            cvec = (w[:, None] * X).sum(0) / w.sum()
            if np.linalg.norm(cvec) < 1e-13:
                break
            step = cvec / max(1.0, 2.0 * np.linalg.norm(cvec))
            X, Sv = ball_automorphism(X, step), ball_automorphism(Sv, step)
        A = 0.5 * np.linalg.norm(np.cross(self.E1, self.E2), axis=1)
        Y = _normalize(self.V[self.ci] - A @ self.V[self.F].mean(1) / A.sum())
        U, _, Vt = np.linalg.svd((w[:, None] * Y).T @ X)                # Kabsch: R X ≈ Y
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        self.z = chart(self.Rm, Sv @ R.T)
        self._faces(scale=True)

    # -------------------------------------------------------------- the code's quadrature
    def _spinor_frame(self, zeta, f, lw=None):
        """The sign-continuous part of λ at chart points ``zeta`` of faces ``f``:
        ``(q, s, logKW)`` with ``λ = √s · conj(Q_f q)``, ``q = e^{kα/2} k û r_f``
        (``α = arg K_f W``, ``û`` the unit vector from the chart pole to the point,
        ``r_f`` the chart rotation), ``s = c |A_f| |K_f W| (1 + |ζ|²)/2``. ``lw``:
        ``log K_f W`` at the points, if already known."""
        from ..spectral.quaternion import qmul
        from .spin import rotation_to_quaternion

        if lw is None:
            lw = self.cones.take(f).logW(zeta[:, None])[:, 0]
        al = lw.imag
        y = _unstereo(zeta)
        u = _normalize(y - [0.0, 0.0, 1.0])
        ek = np.c_[-np.sin(al / 2), np.zeros((len(al), 2)), np.cos(al / 2)]       # e^{kα/2} k
        uf, inv = np.unique(f, return_inverse=True)
        q = qmul(qmul(ek, np.c_[np.zeros(len(u)), u]), rotation_to_quaternion(self.Rf[uf])[inv])
        s = np.exp(self.logc + lw.real) * np.linalg.norm(self.AB[f, 0], axis=1) * (1 + np.abs(zeta) ** 2) / 2
        return q, s, lw

    def _edge_curves(self, fe, ke, n0: int, tol=1e-9, nmax: int = 32, iters: int = 50):
        """The true edges ``k → k+1`` of faces ``fe`` in their charts, as
        ``ζ(t) = P_k + t e + δ(t) n`` (chord ``e``, unit normal ``n``). The edge is a
        straight side of the development, so its tangent maps to one direction θ:
        ``arg K_f W(ζ) + arg ζ'(t) = θ`` (mod π). Angles instead of positions need no
        integral: a residual is one :meth:`_Cones.logW`.

        ``δ`` is piecewise: ``[0, 1]`` is bisected until every near cone but the
        edge's ends is ``SEPC`` panel lengths from the chord piece (≤ 24 levels; a
        scan has cones within 1e-3 of an edge, which a single polynomial cannot pass);
        on a panel ``δ = (1−τ)d_L + τ d_R + τ(1−τ) Σ c_i T_i(2τ−1)``, continuous, 0 at
        the edge's ends. Collocated at ``n + 1`` Chebyshev points per panel with θ
        unknown (:meth:`_curve_newton`), ``n`` doubling from ``n0`` while the
        residual between the collocation points (radians ≈ relative position error)
        exceeds ``tol``.

        The class: from the chord, Newton can settle on a geodesic that passes a cone
        on the wrong side (another direction θ). The developed side (rays from a
        centre inside the face, :meth:`develop`) fixes θ; such edges restart from the
        parabola through the chart centre, which runs inside the face.

        Returns ``(panels, nrm, est)``: ``panels`` a dict of per-panel arrays
        (``row, t0, h, dL, dR, coef``, sorted by row then t0)."""
        S = self.sphere
        E = len(fe)
        P0 = chart(self.Rf[fe], S[self.F[fe, ke]])
        e = chart(self.Rf[fe], S[self.F[fe, (ke + 1) % 3]]) - P0
        P = chart(self.Rf[fe][:, None], S[self.F[fe]])
        o = np.sign(np.imag(np.conj(P[:, 1] - P[:, 0]) * (P[:, 2] - P[:, 0])))
        nrm = -1j * o * e / np.abs(e)
        th = np.angle(self.D[fe, (ke + 1) % 3] - self.D[fe, ke])
        ends = np.c_[self.conepos[self.F[fe, ke]], self.conepos[self.F[fe, (ke + 1) % 3]]]
        cones = self.cones.take(fe)
        tol = np.broadcast_to(tol, (E,))
        # panels
        r, t0, t1 = np.arange(E), np.zeros(E), np.ones(E)
        pr, p0, p1 = [], [], []
        for depth in range(25):
            c = cones.take(r)
            o_ = c.owner
            with np.errstate(divide="ignore", invalid="ignore"):
                Pc = 1.0 / c.q
            Pc = np.where(np.isfinite(Pc), Pc, 1e300)
            A0 = P0[r][o_] + t0[o_] * e[r][o_]
            d_ = (t1 - t0)[o_] * e[r][o_]
            sp_ = np.clip(np.real((Pc - A0) * np.conj(d_)) / np.maximum(np.abs(d_) ** 2, 1e-300), 0.0, 1.0)
            dist = np.abs(Pc - A0 - sp_ * d_)
            dist[(c.idx == ends[r][o_, 0]) | (c.idx == ends[r][o_, 1]) | (c.an == 0)] = np.inf
            ok = (np.minimum.reduceat(dist, c.ptr[:-1]) >= SEPC * (t1 - t0) * np.abs(e[r])) | (depth == 24)
            pr.append(r[ok]); p0.append(t0[ok]); p1.append(t1[ok])
            r, t0, t1 = r[~ok], t0[~ok], t1[~ok]
            if not len(r):
                break
            mid = 0.5 * (t0 + t1)
            r, t0, t1 = np.r_[r, r], np.r_[t0, mid], np.r_[mid, t1]
        pr, p0, p1 = np.concatenate(pr), np.concatenate(p0), np.concatenate(p1)
        o_ = np.lexsort((p0, pr))
        pr, p0, p1 = pr[o_], p0[o_], p1[o_]
        st = dict(pr=pr, p0=p0, p1=p1, coef=np.zeros((len(pr), nmax)), jd=np.zeros(len(pr) + 1))
        est, th_s = np.zeros(E), th.copy()

        def solve(todo, n):
            pr_ = st["pr"]
            npan = np.bincount(pr_, minlength=E)
            geo = (P0, e, nrm, th, ends, cones, pr_, st["p0"], st["p1"] - st["p0"],
                   np.r_[0, np.cumsum(npan)[:-1]], npan)
            for np_ in np.unique(npan[todo]):
                rows = todo[npan[todo] == np_]
                nn = n
                while len(rows):
                    est[rows], th_s[rows] = self._curve_newton(rows, nn, np_, st["coef"], st["jd"], geo, iters)
                    rows = rows[(est[rows] > tol[rows]) & (nn < nmax)]
                    nn = min(2 * nn, nmax)

        def delta(rows, t):
            """δ of the current curves of edges ``rows`` at ``t`` (r,)."""
            pr_, p0_, p1_ = st["pr"], st["p0"], st["p1"]
            i = np.searchsorted(pr_ + p0_, rows + np.minimum(t, 1 - 1e-9), side="right") - 1
            first_ = np.r_[True, pr_[1:] != pr_[:-1]]
            last_ = np.r_[pr_[1:] != pr_[:-1], True]
            dL = np.where(first_, 0.0, st["jd"][:len(pr_)])[i]
            dR = np.where(last_, 0.0, st["jd"][1:len(pr_) + 1])[i]
            tau = (t - p0_[i]) / (p1_ - p0_)[i]
            B, _ = _curve_basis(tau, nmax)
            return (1 - tau) * dL + tau * dR + np.einsum("rn,rn->r", B, st["coef"][i])

        solve(np.arange(E), int(min(n0, nmax)))
        wrong = np.flatnonzero(np.abs(np.angle(np.exp(2j * (th_s - th)))) > np.maximum(1e-6, 100 * est))
        if len(wrong):
            pr_, p0_, p1_ = st["pr"], st["p0"], st["p1"]
            ts = np.clip(np.real(-P0[wrong] * np.conj(e[wrong])) / np.abs(e[wrong]) ** 2, 0.05, 0.95)
            Af = np.zeros(E)
            Af[wrong] = np.real(np.conj(nrm[wrong]) * (-P0[wrong] - ts * e[wrong])) / (ts * (1 - ts))
            pw = np.flatnonzero(np.isin(pr_, wrong))
            st["coef"][pw] = 0.0
            st["coef"][pw, 0] = Af[pr_[pw]] * (p1_ - p0_)[pw] ** 2
            st["jd"][pw] = Af[pr_[pw]] * p0_[pw] * (1 - p0_[pw])
            solve(wrong, int(min(max(n0, 8), nmax)))
            still = wrong[np.abs(np.angle(np.exp(2j * (th_s[wrong] - th[wrong])))) > np.maximum(1e-6, 100 * est[wrong])]
            est[still] = np.inf
        self.curve_class_fixes = getattr(self, "curve_class_fixes", 0) + len(wrong)
        # still above tolerance at nmax: halve those edges' panels, start from the
        # current curve (its values at the new breakpoints and midpoints), solve again
        for _ in range(3):
            bad = np.flatnonzero(np.isfinite(est) & (est > tol))
            if not len(bad):
                break
            pr_, p0_, p1_ = st["pr"], st["p0"], st["p1"]
            sel = np.isin(pr_, bad)
            mid = 0.5 * (p0_ + p1_)
            npr = np.r_[pr_[~sel], pr_[sel], pr_[sel]]
            np0 = np.r_[p0_[~sel], p0_[sel], mid[sel]]
            np1 = np.r_[p1_[~sel], mid[sel], p1_[sel]]
            ncoef = np.r_[st["coef"][~sel], np.zeros((2 * sel.sum(), nmax))]
            njd = np.r_[st["jd"][:len(pr_)][~sel], delta(npr[(~sel).sum():], np0[(~sel).sum():])]
            new_mid = delta(npr[(~sel).sum():], 0.5 * (np0 + np1)[(~sel).sum():])
            new_end = delta(npr[(~sel).sum():], np1[(~sel).sum():])
            ncoef[(~sel).sum():, 0] = 4 * (new_mid - 0.5 * (njd[(~sel).sum():] + new_end))
            o_ = np.lexsort((np0, npr))
            old_st, old_est, old_th = dict(st), est.copy(), th_s.copy()
            st.update(pr=npr[o_], p0=np0[o_], p1=np1[o_], coef=ncoef[o_], jd=np.r_[njd[o_], 0.0])
            solve(bad, int(min(max(n0, 8), nmax)))
            worse = bad[~(est[bad] < old_est[bad])]                    # keep the coarser curve
            if len(worse):
                keep_new = ~np.isin(st["pr"], worse)
                keep_old = np.isin(old_st["pr"], worse)
                jn, jo = st["jd"][:len(st["pr"])], old_st["jd"][:len(old_st["pr"])]
                mp = np.r_[st["pr"][keep_new], old_st["pr"][keep_old]]
                m0 = np.r_[st["p0"][keep_new], old_st["p0"][keep_old]]
                o2 = np.lexsort((m0, mp))
                st.update(pr=mp[o2], p0=m0[o2],
                          p1=np.r_[st["p1"][keep_new], old_st["p1"][keep_old]][o2],
                          coef=np.r_[st["coef"][keep_new], old_st["coef"][keep_old]][o2],
                          jd=np.r_[np.r_[jn[keep_new], jo[keep_old]][o2], 0.0])
                est[worse], th_s[worse] = old_est[worse], old_th[worse]
                bad = np.setdiff1d(bad, worse)
            if not len(bad):
                break
        pr, p0, p1, coef, jd = st["pr"], st["p0"], st["p1"], st["coef"], st["jd"]
        nc = max(1, int(np.max(np.flatnonzero(np.any(coef != 0, 0)), initial=0)) + 1)
        last = np.r_[pr[1:] != pr[:-1], True]
        dR = np.where(last, 0.0, np.r_[jd[1:len(pr)], 0.0])
        dL = np.where(np.r_[True, pr[1:] != pr[:-1]], 0.0, jd[:len(pr)])
        return dict(row=pr, t0=p0, h=p1 - p0, dL=dL, dR=dR, coef=coef[:, :nc]), nrm, est

    @staticmethod
    def _curve_newton(rows, n, np_, coef, jd, geo, iters):
        """Newton for :meth:`_edge_curves` on edges ``rows`` (each with ``np_``
        panels), ``n`` coefficients per panel: unknowns the coefficients, the
        ``np_ − 1`` interior junction values and the side's direction θ (with θ
        fixed, two pinned ends and a first-order condition overdetermine the curve,
        and the collocation's near-null mode amplifies any inconsistency by ~n²),
        collocated at ``n + 1`` Chebyshev points per panel. Steps are capped at ¼ of
        the way from the curve to the nearest cone not at the edge's ends. Updates
        ``coef``/``jd`` in place; returns the largest residual between the
        collocation points (∞ if Newton did not converge) and θ."""
        P0, e, nrm, th0, ends, cones, pr, p0, h, first, npan = geo
        E, Pn = len(rows), np_
        pan = first[rows][:, None] + np.arange(Pn)                    # (E, P) panel indices
        tN = (1 - np.cos(np.pi * (np.arange(n + 1) + 0.5) / (n + 1))) / 2
        tM = 0.5 * (tN[1:] + tN[:-1])
        el = np.abs(e[rows])
        th = th0[rows].copy()
        U = Pn * (n + 1)
        # junction j (1..P-1) is the left value of panel j: jd[pan[:, j]]
        c = coef[pan, :n].copy()                                      # (E, P, n)
        d = np.zeros((E, Pn + 1))
        if Pn > 1:
            d[:, 1:Pn] = jd[pan[:, 1:]]

        def curve(sel, cc, dd, tt):
            B, dB = _curve_basis(tt, n)
            hh = h[pan[sel]][:, :, None]
            t = p0[pan[sel]][:, :, None] + tt * hh
            dl = (1 - tt) * dd[:, :-1, None] + tt * dd[:, 1:, None] + np.einsum("jn,epn->epj", B, cc)
            ddl = (dd[:, 1:, None] - dd[:, :-1, None] + np.einsum("jn,epn->epj", dB, cc)) / hh
            r = rows[sel]
            z = P0[r, None, None] + t * e[r, None, None] + dl * nrm[r, None, None]
            zp = e[r, None, None] + ddl * nrm[r, None, None]
            return z.reshape(len(sel), -1), zp.reshape(len(sel), -1), B, dB

        def resid(sel, cc, dd, tt, deriv):
            z, zp, B, dB = curve(sel, cc, dd, tt)
            ca = cones.take(rows[sel])
            out = ca.logW(z, deriv, arg=True)
            al, dlw = out if deriv else (out, None)
            phi = al + np.angle(zp) - th[sel, None]
            return ca, z, zp, phi - np.pi * np.round(phi / np.pi), dlw, B, dB

        def room(ca, r, z):
            """¼ of the distance from each curve point to the nearest cone not at the
            edge's ends (rows, points)."""
            o_ = ca.owner
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = np.abs(z[o_] - (1.0 / ca.q)[:, None])
            dist[(ca.idx == ends[r][o_, 0]) | (ca.idx == ends[r][o_, 1]) | (ca.an == 0)] = np.inf
            dist[~np.isfinite(dist)] = np.inf
            return 0.25 * np.minimum(np.abs(e[r])[:, None], np.minimum.reduceat(dist, ca.ptr[:-1], axis=0))

        act = np.arange(E)
        for it in range(iters):
            ca, z, zp, phi, dlw, B, dB = resid(act, c[act], d[act], tN, True)
            Ea = len(act)
            a_ = np.imag(dlw * nrm[rows[act], None]).reshape(Ea, Pn, n + 1)
            b_ = np.imag(nrm[rows[act], None] / zp).reshape(Ea, Pn, n + 1)
            hh = h[pan[act]][:, :, None]
            J = np.zeros((Ea, Pn, n + 1, U))
            for p in range(Pn):
                J[:, p, :, p * n:(p + 1) * n] = a_[:, p, :, None] * B[None] + (b_[:, p, :] / hh[:, p])[:, :, None] * dB[None]
                if p >= 1:                                            # left junction d_p
                    J[:, p, :, Pn * n + p - 1] = a_[:, p] * (1 - tN) - b_[:, p] / hh[:, p]
                if p <= Pn - 2:                                       # right junction d_{p+1}
                    J[:, p, :, Pn * n + p] = a_[:, p] * tN + b_[:, p] / hh[:, p]
            J = J.reshape(Ea, U, U)
            J[:, :, U - 1] = -1.0
            fin = np.all(np.isfinite(J), (1, 2)) & np.all(np.isfinite(phi), 1)
            J[~fin], phi[~fin] = np.eye(U), 0.0
            try:
                step = np.linalg.solve(J, -phi[..., None])[..., 0]
            except np.linalg.LinAlgError:                              # a singular row: least squares, row by row
                step = np.stack([np.linalg.lstsq(J_, -p_, rcond=None)[0] for J_, p_ in zip(J, phi)])
            step[~fin] = np.nan
            ok = np.all(np.isfinite(step), 1)
            step[~ok] = 0.0
            sc_ = step[:, :Pn * n].reshape(Ea, Pn, n)
            sd = np.zeros((Ea, Pn + 1))
            sd[:, 1:Pn] = step[:, Pn * n:Pn * n + Pn - 1]
            dl_step = np.abs(np.einsum("jn,epn->epj", B, sc_) + (1 - tN) * sd[:, :-1, None]
                             + tN * sd[:, 1:, None]).reshape(Ea, -1)
            mx = dl_step.max(1)
            # trust region, pointwise: no curve point moves more than ¼ of its way to
            # the nearest cone (only rows whose step is not already negligible)
            sc = np.ones(Ea)
            big = np.flatnonzero(mx > 1e-6 * el[act])
            if len(big):
                lim = room(ca.take(big), rows[act[big]], z[big])
                sc[big] = np.minimum(1.0, (lim / np.maximum(dl_step[big], 1e-300)).min(1))
            c[act] += sc[:, None, None] * sc_
            d[act] += sc[:, None] * sd
            th[act] += sc * step[:, U - 1]
            # quadratic convergence: what is left after a step of mx is ~mx², far below
            act = act[ok & ~((sc == 1.0) & (mx < 1e-10 * el[act]) | (np.abs(phi).max(1) < 1e-14))]
            if not len(act):
                break
        est = np.abs(resid(np.arange(E), c, d, tM, False)[3]).max(1)
        est[act] = np.inf                                             # unconverged: never accepted silently
        coef[pan, :n] = c
        coef[pan, n:] = 0.0
        if Pn > 1:
            jd[pan[:, 1:]] = d[:, 1:Pn]
        return est, th

    def _shared_edges(self, qf):
        """One curve per edge of the mesh (:meth:`_edge_curves`, in the chart of its
        first face, ``n0`` = the larger of its faces' orders + 2), so the two faces
        tile the sphere exactly along it: an error of the curve only moves area
        between them, weighted by the spinor's jump (the dihedral angle), not lost
        or counted twice. Sets ``self._edge`` (per face side: owner face, owner side,
        reversed, edge) and ``self._curves`` (panels, per edge ``nrm``), and
        ``self.curve_err``."""
        m = len(self.F)
        fe, ke = np.repeat(np.arange(m), 3), np.tile(np.arange(3), m)
        i_, j_ = self.F[fe, ke], self.F[fe, (ke + 1) % 3]
        key = np.minimum(i_, j_) * len(self.V) + np.maximum(i_, j_)
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        of, ok_ = fe[first], ke[first]                                # owner side of each edge
        ne = len(first)
        n0 = np.zeros(ne, int)
        np.maximum.at(n0, inv, qf[fe] + 2)
        nrm, est = np.zeros(ne, complex), np.zeros(ne)
        # tolerance: a curve error ε moves ~ε|e|² of area across the edge, where the
        # spinor jumps by the dihedral angle, so ε · dihedral ≤ CURVE_TOL. Over a mesh
        # that bounds the code's change by ~4 CURVE_TOL; measured far less (head, L = 32:
        # v moves 2e-8 between CURVE_TOL 1e-10 and 1e-6, which saves a third of the time)
        N = _normalize(np.cross(self.E1, self.E2))
        fo2 = np.zeros(ne, int)
        np.maximum.at(fo2, inv, fe)                                   # the other face (the owner is the first)
        dih = np.arccos(np.clip(np.einsum("ij,ij->i", N[of], N[fo2]), -1, 1))
        tol = np.minimum(1e-4, CURVE_TOL / np.maximum(dih, 1e-12))
        parts = []
        for n_ in np.unique(n0):
            r = np.flatnonzero(n0 == n_)
            pan, nrm[r], est[r] = self._edge_curves(of[r], ok_[r], n_, tol[r])
            pan["row"] = r[pan["row"]]
            parts.append(pan)
        nc = max(p_["coef"].shape[1] for p_ in parts)
        pan = {k: np.concatenate([p_[k] for p_ in parts]) if k != "coef" else
               np.concatenate([np.pad(p_[k], ((0, 0), (0, nc - p_[k].shape[1]))) for p_ in parts]) for k in parts[0]}
        o_ = np.lexsort((pan["t0"], pan["row"]))
        pan = {k: v[o_] for k, v in pan.items()}
        self.curve_err = float((est * dih).max(initial=0.0))
        self.curve_panels = int(len(pan["row"]) - ne)
        rev = self.F[of[inv], ok_[inv]] != i_
        self._edge = (of[inv].reshape(m, 3), ok_[inv].reshape(m, 3), rev.reshape(m, 3), inv.reshape(m, 3))
        self._curves = (pan, nrm)

    def _curve_at(self, edge, t):
        """``δ(t), δ'(t)`` of edges ``edge`` (r,) at parameters ``t`` (r, k)."""
        pan, _ = self._curves
        keys = pan["row"] + pan["t0"]
        i = np.searchsorted(keys, edge[:, None] + np.minimum(t, 1 - 1e-9), side="right") - 1
        h = pan["h"][i]
        tau = (t - pan["t0"][i]) / h
        B, dB = _curve_basis(tau, pan["coef"].shape[1])
        cf = pan["coef"][i]
        dl = (1 - tau) * pan["dL"][i] + tau * pan["dR"][i] + np.einsum("rkn,rkn->rk", B, cf)
        ddl = (pan["dR"][i] - pan["dL"][i] + np.einsum("rkn,rkn->rk", dB, cf)) / h
        return dl, ddl

    def _side_curve(self, f, k, t):
        """Points ``C`` and ``dC/dt`` of the true side ``k → k+1`` of faces ``f``
        (r,) at parameters ``t`` (r, n), in each face's own chart: the owner's
        curve, reversed if the owner runs the edge the other way, and carried to
        this chart by the chart change (a Möbius map, differentiated by the
        chain rule through the sphere)."""
        of, ok_, rev, ed = (a[f, k] for a in self._edge)
        nrm = self._curves[1][ed]
        S = self.sphere
        t2 = np.where(rev[:, None], 1.0 - t, t)
        P0 = chart(self.Rf[of], S[self.F[of, ok_]])
        e = chart(self.Rf[of], S[self.F[of, (ok_ + 1) % 3]]) - P0
        dl, ddl = self._curve_at(ed, t2)
        C = P0[:, None] + t2 * e[:, None] + dl * nrm[:, None]
        dC = (e[:, None] + ddl * nrm[:, None]) * np.where(rev, -1.0, 1.0)[:, None]
        mv = np.flatnonzero(of != f)
        if len(mv):
            z, h = C[mv], dC[mv]
            u = 1 + np.abs(z) ** 2
            du = 2 * np.real(np.conj(z) * h)
            Y = np.stack([2 * z.real / u, 2 * z.imag / u, 1 - 2 / u], -1)
            dY = np.stack([2 * h.real / u - 2 * z.real * du / u ** 2, 2 * h.imag / u - 2 * z.imag * du / u ** 2,
                           2 * du / u ** 2], -1)
            M = self.Rf[f[mv]] @ np.transpose(self.Rf[of[mv]], (0, 2, 1))        # owner chart → this chart
            Yg, dYg = np.einsum("rij,rkj->rki", M, Y), np.einsum("rij,rkj->rki", M, dY)
            C[mv] = (Yg[..., 0] + 1j * Yg[..., 1]) / (1 - Yg[..., 2])
            dC[mv] = _dsigma(Yg, dYg)
        return C, dC

    def sphere_quadrature(self, Ls: float, qmin: int = 1):
        """Sphere quadrature of the code's projection, in closed form: ``(x, w, λ, f)``
        — nodes on the sphere, sphere-area weights, the spinor there (signs consistent
        over the whole sphere) and the face each node lies in.

        Each face is integrated in its own chart, with no inversion of the map, over
        its *curved* chart triangle: the true edges (:meth:`_shared_edges`, one
        curve per edge so neighbours tile the sphere exactly). 6 Duffy patches,
        corner ``P_k`` → (half side, centroid), each mapped from the unit square by
        ``z = P_k + s[(1−x)(A − P_k) + x(B − P_k)] + (side weight) Δ(t(s))``, ``Δ`` the
        side's offset from its chord (∝ ``t(1 − t)``): the patch side on the edge
        *is* the edge, and the Jacobian keeps its factor ``s``, so the corner rule
        stays Gauss–Jacobi with weight ``s^{1+a/2}`` (``λ ~ |ζ − p|^{a/2}``). This
        replaced straight triangles plus fitted "lunes" to the edges: 2/3 of the
        nodes on a dense mesh, and the lune fit (``q + 1`` points) was the error
        floor on coarse ones.

        ``q`` nodes per direction: ``q ≥ qmin``, ``≥ Ls·ang/3 + 1`` (the band across
        the face, ``ang`` its longest side), ``≥ 6 per radian``, and
        ``≥ 1 + 10 σ_f`` (:meth:`_cone_strength`: strong cones at or near the
        face; capped at 12). Measured per-face errors fall ~7× per order at
        ``σ ≈ 0.5``; without the cone term small faces next to strong cones (a
        flat-subdivided hull) held codes at ~1e-4, with it ~1e-6. A dense scan
        stays at ``q = 2`` almost everywhere (``σ`` median 0.03): v errs 7e-6
        against ``q = 4`` everywhere (3.3× the nodes), 6e-7 at ``q = 3`` (1.9×).

        λ is ``√s conj(Q_f e^{kα/2} k û r_f)`` (:meth:`_spinor_frame`): continuous in a
        face, ``Q_f`` fixed by one :meth:`spinor_at` per face. Across an edge the
        spinor jumps by the rotation between the two faces' normals,
        ``λ_g = λ_f · conj(q_fg)``, which fixes the faces' relative signs over a
        spanning tree of the dual graph (``self.sign_defects`` counts the edges off
        the tree where it disagrees: 0 on an exact map)."""
        from ..spectral.quaternion import qconj, qmul

        m = len(self.F)
        S = self.sphere
        P = chart(self.Rf[:, None], S[self.F])
        aF = self.a_v[self.F]
        ang = np.arccos(np.clip(np.einsum("fki,fki->fk", S[self.F], S[self.F[:, [1, 2, 0]]]), -1, 1)).max(1)
        qf = np.maximum.reduce([np.full(m, qmin), np.ceil(Ls * ang / 3) + 1, np.ceil(6 * ang),
                                np.minimum(np.ceil(1 + 10 * self._cone_strength()), 12)]).astype(int)
        groups = [(q, np.flatnonzero(qf == q)) for q in np.unique(qf)]
        self._shared_edges(qf)
        zc, _ = self._side_curve(np.repeat(np.arange(m), 3), np.tile(np.arange(3), m), np.full((3 * m, 1), 0.5))
        zc = zc.reshape(m, 3)
        sf, Q = self._face_signs(zc)
        X, Wt, Lam, Fo = [], [], [], []
        for q, fs in groups:
            z, w = self._patches(q, fs, P[fs], aF[fs])
            lw = self.cones.rows(z, fs)
            fo = np.repeat(fs, z.shape[1])
            z, w, lw = z.ravel(), w.ravel(), lw.ravel()
            qv, s, _ = self._spinor_frame(z, fo, lw)
            X.append(np.einsum("nji,nj->ni", self.Rf[fo], _unstereo(z)))
            Wt.append(w * 4.0 / (1 + np.abs(z) ** 2) ** 2)
            Lam.append(np.sqrt(s)[:, None] * qconj(qmul(Q[fo], qv)) * sf[fo, None])
            Fo.append(fo)
        return np.concatenate(X), np.concatenate(Wt), np.concatenate(Lam), np.concatenate(Fo)

    def _cone_strength(self):
        """σ_f: how hard the cones make face ``f``'s integrand, ``max |a|`` over its
        corners (``λ ~ |ζ − p|^{a/2}`` is exact under Gauss–Jacobi, but its angular
        factor ``e^{k a θ/2}`` and the patch's other corners are not),
        ``max |a| R_f / |P|`` over its other near cones (a branch point ``|P|/R_f``
        face radii from the centre) and the far cones' variation ``Σ |G_n| R_fⁿ⁺¹``
        of ``log W`` across the face."""
        c = self.cones
        o = c.owner
        R = np.abs(chart(self.Rf[:, None], self.sphere[self.F])).max(1)
        corner = (c.idx[:, None] == self.conepos[self.F[o]]).any(1)
        v = np.abs(c.an) * np.where(corner, 1.0, R[o] * np.abs(c.q))
        far = np.sum(np.abs(c.G) * R[:, None] ** np.arange(1, c.G.shape[1] + 1), 1)
        return np.maximum(np.maximum.reduceat(v, c.ptr[:-1]), far)

    def _patches(self, q, fs, P, aF):
        """Nodes and chart-area weights ``(nf, 6q²)`` of the curved Duffy patches
        (:meth:`sphere_quadrature`) of faces ``fs`` with chart corners ``P`` (nf, 3)."""
        nf = len(P)
        xr, wr = np.polynomial.legendre.leggauss(q)
        xr, wr = (xr + 1) / 2, wr / 2
        G = np.zeros(nf, complex)                                     # the chart centre: inside the face
        Z, W = [], []
        for k in range(3):
            k1, k2 = (k + 1) % 3, (k + 2) % 3
            Pk = P[:, k]
            xs_, ws_ = _gj_batch(1 + aF[:, k] / 2, q)                 # weight s^β, s = 1 − x
            s_ = 1.0 - xs_
            # patch (P_k, mid k→k+1, centroid): side k from its start, t = s/2, at x = 0;
            # patch (P_k, centroid, mid k+2→k): side k+2 from its end, t = 1 − s/2, at x = 1.
            # The side's offset from its chord, Δ(t) = C(t) − chord(t), is ∝ t(1 − t).
            for A_, B_, ed, t_, dt, bx, dbx in ((0.5 * (Pk + P[:, k1]), G, k, s_ / 2, 0.5, 1 - xr, -1.0),
                                                (G, 0.5 * (Pk + P[:, k2]), k2, 1 - s_ / 2, -0.5, xr, 1.0)):
                C, dC = self._side_curve(fs, np.full(nf, ed), t_)
                ech = P[:, (ed + 1) % 3] - P[:, ed]
                dl = C - P[:, ed, None] - t_ * ech[:, None]           # (nf, q)
                ddl = dC - ech[:, None]
                base = ((1 - xr) * (A_ - Pk)[:, None] + xr * (B_ - Pk)[:, None])[:, None, :]
                z = Pk[:, None, None] + s_[:, :, None] * base + bx * dl[:, :, None]
                zs = base + bx * (dt * ddl)[:, :, None]
                zx_s = (B_ - A_)[:, None, None] + dbx * (dl / s_)[:, :, None]         # ∂z/∂x / s
                J = self.sigma * np.imag(np.conj(zs) * zx_s)           # true faces all turn like sigma
                w = (ws_ * s_ ** (-aF[:, k, None] / 2))[:, :, None] * wr * J
                Z.append(z.reshape(nf, -1)); W.append(w.reshape(nf, -1))
        return np.concatenate(Z, 1), np.concatenate(W, 1)

    def _face_signs(self, zc):
        """Face signs and ``Q_f`` from curve points ``zc`` (m, 3) of every face edge:
        ``Q_f`` by one :meth:`spinor_at` at each chart centre (inside the face, see
        :meth:`develop`); the signs by the jump rule, with the seam of
        :meth:`_seam`, over a spanning tree of the dual graph (sets ``jump_err``,
        ``sign_defects``)."""
        from ..spectral.quaternion import qconj, qmul

        m = len(self.F)
        P = chart(self.Rf[:, None], self.sphere[self.F])
        G = np.zeros(m, complex)
        xg = np.einsum("nji,nj->ni", self.Rf, _unstereo(G))
        lg, _ = self.spinor_at(xg, np.arange(m))
        qg, _, _ = self._spinor_frame(G, np.arange(m))
        Q = qmul(qconj(lg / np.linalg.norm(lg, axis=1, keepdims=True)), qconj(qg))
        fe_, ke_ = np.repeat(np.arange(m), 3), np.tile(np.arange(3), m)
        zc = zc.ravel()
        i_, j_ = self.F[fe_, ke_], self.F[fe_, (ke_ + 1) % 3]
        key = np.minimum(i_, j_) * len(self.V) + np.maximum(i_, j_)
        order = np.argsort(key, kind="stable")
        f1, f2 = fe_[order][0::2], fe_[order][1::2]
        z1 = zc[order][0::2]
        xe = np.einsum("nji,nj->ni", self.Rf[f1], _unstereo(z1))
        q1, _, _ = self._spinor_frame(z1, f1)
        z2 = chart(self.Rf[f2], xe)
        q2, _, _ = self._spinor_frame(z2, f2)
        l1, l2 = qconj(qmul(Q[f1], q1)), qconj(qmul(Q[f2], q2))
        N = _normalize(np.cross(self.E1, self.E2))
        n1, n2 = N[f1], N[f2]
        qj = np.c_[1 + np.einsum("ij,ij->i", n1, n2), np.cross(n1, n2)]
        qj = qj / np.linalg.norm(qj, axis=1, keepdims=True)
        dot = np.einsum("ij,ij->i", l2, qmul(l1, qconj(qj)))
        self.jump_err = float(np.max(np.abs(np.abs(dot) - 1)))
        sgn_e = np.where(dot >= 0, 1.0, -1.0)
        ei, ej = np.minimum(i_, j_)[order][0::2], np.maximum(i_, j_)[order][0::2]
        crease = np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1, 1))
        sgn_e = sgn_e * self._seam(ei, ej, sgn_e, np.where(np.abs(np.abs(dot) - 1) < 1e-6, np.pi - crease, 0.0))
        Gd = sp.coo_matrix((np.r_[sgn_e, sgn_e] + 2.0, (np.r_[f1, f2], np.r_[f2, f1])), shape=(m, m)).tocsr()
        tree = csg.minimum_spanning_tree(sp.coo_matrix((np.r_[2.0 - np.abs(dot), 2.0 - np.abs(dot)] + 1e-3,
                                                        (np.r_[f1, f2], np.r_[f2, f1])), shape=(m, m)).tocsr())
        order_, pred = csg.breadth_first_order(tree + tree.T, 0, directed=False, return_predecessors=True)
        child = order_[1:]
        odd = np.zeros(m, bool)                                       # parity to the root, by pointer jumping
        odd[child] = np.asarray(Gd[pred[child], child]).ravel() < 2.0
        anc = pred.copy()
        anc[order_[0]] = order_[0]
        while np.any(anc != order_[0]):
            odd, anc = odd ^ odd[anc], anc[anc]
        sf = np.where(odd, -1.0, 1.0)
        self.sign_defects = int(np.sum(sf[f1] * sf[f2] * sgn_e < 0))
        self._jumps = (f1, f2, dot, sf)
        return sf, Q

    def _seam(self, ei, ej, sgn, w):
        """Which edges take the other lift of their hinge rotation: ``±1`` per edge
        ``(ei, ej)``.

        The jump rule lifts each hinge rotation by its minimal angle (< π). Around a
        vertex the lifts compose to ``±`` the rotation by its angle defect; ``−`` at a
        crumpled vertex whose star folds back on itself (a crease at 177–179.5°: a
        scan's faces that actually pass through each other turn the long way, by
        > π, and only the vertex can tell). Such *odd* vertices (sign product of the
        measured jumps around them −1; 100 on the FaceScape head, 0 on clean meshes)
        cannot all be satisfied: the signs need a seam of edges joining them in
        pairs. It is the cheapest such set (a T-join, by greedy pairing over
        shortest paths) with edge cost ``w`` = π − crease angle: the near-π creases,
        where the minimal lift is the doubtful one (an edge whose jump did not
        measure costs nothing). Sets ``sign_seam`` (edges flipped) and
        ``odd_vertices``."""
        n = len(self.V)
        par = np.ones(n)
        np.multiply.at(par, ei, sgn)
        np.multiply.at(par, ej, sgn)
        T = np.flatnonzero(par < 0)
        self.odd_vertices, self.sign_seam = int(len(T)), 0
        flip = np.ones(len(ei))
        if not len(T) or len(T) > 4000:
            return flip
        G = sp.coo_matrix((np.r_[w, w] + 1e-9, (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()
        dist, pred = csg.dijkstra(G, indices=T, return_predecessors=True)
        DT = dist[:, T]
        iu = np.triu_indices(len(T), 1)
        order_ = np.argsort(DT[iu], kind="stable")
        used = np.zeros(len(T), bool)
        ekey = ei * n + ej
        srt = np.argsort(ekey)
        on = np.zeros(len(ei), bool)
        for a_, b_ in zip(iu[0][order_], iu[1][order_]):
            if used[a_] or used[b_] or not np.isfinite(DT[a_, b_]):
                continue
            used[a_] = used[b_] = True
            v = T[b_]
            while v != T[a_]:
                u_ = pred[a_, v]
                k = np.minimum(u_, v) * n + np.maximum(u_, v)
                on[srt[np.searchsorted(ekey[srt], k)]] ^= True
                v = u_
        self.sign_seam = int(on.sum())
        return np.where(on, -1.0, 1.0)

    @staticmethod
    def _W(zeta, cones):
        """``K_f W(ζ)`` at points ``ζ`` (s,), one per row of ``cones``."""
        return np.exp(cones.logW(zeta[:, None])[:, 0])

    def surface_points(self, zeta, f):
        """3D points of chart points ``zeta`` of faces ``f``, by the developing map
        (a ray from the chart centre): ``V_0 + c (Re δD · A + Im δD · B)``."""
        D = _rays(zeta, np.zeros(len(f)), self.cones.take(f), np.full(len(f), -1))
        dD = (D - self.D[f, 0]) * np.exp(self.logc)
        return self.V[self.F[f, 0]] + dD.real[:, None] * self.AB[f, 0] + dD.imag[:, None] * self.AB[f, 1]

    def spinor_at(self, x, faces):
        """λ (quaternion) and surface area density ``s²`` at sphere points ``x`` lying
        on ``faces``: from ``dX = M_f(c W dζ)``, the face's similarity per point."""
        from ..spectral.quaternion import qconj
        from .spin import rotation_to_quaternion

        f = faces
        Rf = self.Rf[f]
        Wz = self._W(chart(Rf, x), self.cones.take(f))
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


def solve_map(V, F) -> PolyMap:
    """The polyhedron's sphere map (:meth:`PolyMap.solve`), started from its discrete
    uniformization (:func:`~shapegen.geom.uniformize.uniformize`)."""
    from .uniformize import uniformize

    return PolyMap(V, F, S0=uniformize(V, F)).solve()


def encode_polyhedral(V, F, L: int, iters: int = 3, over: int = 3, lift=None,
                      Ls: float | None = None, n_check: int = 6000) -> dict:
    """Polyhedron ``(V, F)`` → closed ρ̂ at band ``L`` (module docstring).

    Returns ``{"coeffs", "sphere"`` (per vertex), ``"gamma", "f_hat",
    "samples": (sphere points, weights, 3D points), "map": PolyMap, "info"}``;
    ``samples`` are ``n_check`` points inside faces with their exact surface
    points, to score a decode over the surface.
    """
    from ..quad.lift import frame_quaternion
    from ..spectral.quaternion import qconj, qmul
    from .dirac_direct import _project_faces, close_code
    from .dirac_quotient import rho_quotient
    from .spectral_adjoint import SpectralLift

    V, F = np.asarray(V, float), np.asarray(F)
    pm = solve_map(V, F)
    pm.gauge_cones()
    Ls = ds.spinor_band(L, ds.potential_band_extra(L)) if Ls is None else Ls
    Lm = L + 0.5
    xs, w, lam, _ = pm.sphere_quadrature(Ls)
    th, ph = np.arccos(np.clip(xs[:, 2], -1, 1)), np.arctan2(xs[:, 1], xs[:, 0])
    raw = qmul(qconj(frame_quaternion(th, ph)), lam)
    v = _project_faces(Lm, th, ph, w * (raw[:, 0] + 1j * raw[:, 1]), w * (raw[:, 2] - 1j * raw[:, 3]))
    rho = rho_quotient(v, L, Ls, Lm, iters=iters, over=over)
    n, k = len(ds.sw.half_modes(0.5, Ls)), len(v) // 2
    rho, res, gamma = close_code(rho, L, Ls, v0=np.r_[v[:k], np.zeros(n - k), v[k:], np.zeros(n - k)])
    lift = SpectralLift(Ls) if lift is None else lift
    # check points: halfway from the chart centre (inside the face) to each corner, on up to n_check / 3 faces
    fs = np.unique(np.linspace(0, len(F) - 1, max(1, min(len(F), n_check // 3))).astype(int))
    P = chart(pm.Rf[fs][:, None], pm.sphere[F[fs]])
    zc = (0.5 * P).ravel()
    fc = np.repeat(fs, 3)
    xc = np.einsum("nji,nj->ni", pm.Rf[fc], _unstereo(zc))
    Xc = pm.surface_points(zc, fc)
    info = dict(fit=pm.fit, ncones=len(pm.ci), nodes=len(xs), sphere_weight=float(w.sum()),
                jump_err=pm.jump_err, sign_defects=pm.sign_defects, curve_err=pm.curve_err,
                recentred=pm.recentred, odd_vertices=pm.odd_vertices, sign_seam=pm.sign_seam)
    return {"coeffs": rho, "sphere": pm.sphere, "gamma": gamma,
            "f_hat": lift.forward(np.r_[res.psi1, res.psi2])[0],
            "samples": (xc, np.ones(len(xc)), Xc), "map": pm, "info": info}
