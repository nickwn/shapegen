"""Periods and the lattice solve, ported from
``s5_grid.py``, ``core.py`` and ``conegate.py``.

Given cones in the flat chart :mod:`shapegen.quad.cones` produces (complex
positions ``C``, indices ``idx``), this finds a cell size ``s`` and a small
perturbation of ``C`` such that developing the flat metric around every loop
of a spanning tree lands on the integer/half-integer lattice ``ℤ[i]·s`` —
the condition a square grid needs to close up into quads without seams.

The developing map's derivative is ``w' = ψ^{-1/4}`` with
``ψ = prod_k (z - C_k)^{-4·idx_k}``, so a cone-to-cone period integral has
branch points at its endpoints; :func:`cut_and_grad` uses Gauss-Jacobi
quadrature with the actual endpoint exponents (machine precision at 12-16
nodes, where ordinary quadrature never converges).

Target selection is naive rounding (:func:`round_to_lattice`), not LLL/Babai.
"""

from __future__ import annotations

import functools
import heapq
import time
from collections import deque

import numpy as np

__all__ = [
    "cut_and_grad", "euclidean_mst", "loop_edges", "periods_and_grad",
    "round_to_lattice",
    "realify", "select_independent_loops", "node_count_for", "solve_lattice",
    "conegate", "integrand_along", "integrate", "period", "rotation",
    "enclosed_index", "clearance", "gauss_bonnet",
]


# --------------------------------------------------------------------------- #
# P5.1 -- cut integrals, value and gradient in one pass                       #
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=None)
def _gj_nodes(ai_r: float, aj_r: float, n: int):
    from scipy.special import roots_jacobi
    return roots_jacobi(n, aj_r, ai_r)


def gj_nodes(ai: float, aj: float, n: int):
    """Gauss-Jacobi nodes/weights for the segment's two endpoint exponents,
    cached by ``(ai, aj, n)`` (rounded to 9 places, matching the source, so
    float noise doesn't defeat the cache)."""
    return _gj_nodes(round(float(ai), 9), round(float(aj), 9), n)


def cut_and_grad(i: int, j: int, C: np.ndarray, a: np.ndarray, n: int = 16):
    """The cut integral ``∫_i^j w' dz`` between cones ``i`` and ``j``, and its
    gradient w.r.t. every cone position in ``C`` (needed for Newton's method
    in :func:`solve_lattice`).

    ``C`` are complex chart positions, ``a = -idx`` the per-cone exponents,
    ``n`` the Gauss-Jacobi node count. Returns ``(value, grad)`` with ``grad``
    complex of shape ``(len(C),)``. Matches central finite differences to
    1e-7 relative.
    """
    x, w = gj_nodes(a[i], a[j], n)
    s = (1.0 + x) / 2.0
    L = C[j] - C[i]
    z = C[i] + L * s
    mid = C[i] + L / 2.0
    msk = np.ones(len(C), dtype=bool)
    msk[i] = False
    msk[j] = False
    Ck = C[msk]
    ak = a[msk]

    ww = z[:, None] - Ck[None, :]
    w0 = mid - Ck
    rel = ww / w0[None, :]
    th = np.angle(w0)[None, :] + np.arctan2(rel.imag, rel.real)
    Q = np.exp((ak[None, :] * (np.log(np.abs(ww)) + 1j * th)).sum(axis=1))
    pref = (L * np.exp(a[i] * np.log(L)) * np.exp(a[j] * np.log(-L))
           * 2.0 ** (-a[i] - a[j]) / 2.0)

    wQ = w * Q
    S = wQ.sum()
    inv = 1.0 / ww
    g = np.zeros(len(C), dtype=complex)
    g[msk] = pref * (-ak) * (wQ[:, None] * inv).sum(axis=0)
    Qp = Q * (ak[None, :] * inv).sum(axis=1)
    e = 1.0 + a[i] + a[j]
    g[i] = -pref * e / L * S + pref * np.sum(w * Qp * (1.0 - s))
    g[j] = pref * e / L * S + pref * np.sum(w * Qp * s)
    return pref * S, g


# --------------------------------------------------------------------------- #
# P5.2 -- the condition set                                                   #
# --------------------------------------------------------------------------- #

def euclidean_mst(C: np.ndarray):
    """Euclidean minimum spanning tree over complex points ``C`` (Prim's
    algorithm); returns a list of ``(u, v)`` edges.

    Trap: freeze it once per solve. Recomputing it per iteration and
    rejecting when it changes throws away valid solutions -- the loop set is
    defined by cone pairs and is complete for any spanning tree."""
    n = len(C)
    seen = {0}
    E = []
    hq = [(abs(C[0] - C[q]), 0, q) for q in range(1, n)]
    heapq.heapify(hq)
    while hq and len(seen) < n:
        d, u, v = heapq.heappop(hq)
        if v in seen:
            continue
        seen.add(v)
        E.append((u, v))
        for q in range(n):
            if q not in seen:
                heapq.heappush(hq, (abs(C[v] - C[q]), v, q))
    return E


def loop_edges(idx: np.ndarray, mst_edges):
    """One loop per MST edge: same index -> difference loop (``sign=-1``),
    opposite index -> sum loop (``sign=+1``)."""
    return [(i, j, (-1 if abs(idx[i] - idx[j]) < 1e-12 else 1))
           for (i, j) in mst_edges]


def periods_and_grad(C: np.ndarray, a: np.ndarray, idx: np.ndarray, loops,
                     n: int = 16):
    """Periods and gradients for every loop in ``loops``.

    ``R_all[j] = e^{-2πi·idx_j}`` is the monodromy factor at cone ``j``; the
    loop's actual period is the cut integral times ``f = 1 - R_all[j]**(-sign)``
    -- the difference (same-index loops) or sum (opposite-index loops) across
    the two sheets the loop connects.
    """
    R_all = np.exp(-2j * np.pi * np.asarray(idx, dtype=float))
    t = np.empty(len(loops), dtype=complex)
    G = np.empty((len(loops), len(C)), dtype=complex)
    for li, (i, j, sg) in enumerate(loops):
        v, g = cut_and_grad(i, j, C, a, n)
        f = 1.0 - R_all[j] ** (-sg)
        t[li] = f * v
        G[li] = f * g
    return t, G


def realify(G: np.ndarray) -> np.ndarray:
    """Complex Jacobian ``(L loops, M cones)`` -> real ``(2L, 2M)`` block
    form, treating each ``d(period) = D * dz`` as the real-linear map it is
    for a holomorphic ``D``."""
    Lrows, M = G.shape
    J = np.zeros((2 * Lrows, 2 * M))
    for li in range(Lrows):
        for m in range(M):
            D = G[li, m]
            J[2 * li, 2 * m] = D.real
            J[2 * li, 2 * m + 1] = -D.imag
            J[2 * li + 1, 2 * m] = D.imag
            J[2 * li + 1, 2 * m + 1] = D.real
    return J


def select_independent_loops(loops, G: np.ndarray, tol: float = 1e-8):
    """Greedily keep loops whose (realified) rows are independent of every
    loop kept so far. Known to be complete (not re-derived here): the
    ``n-1`` MST edge loops span the full period lattice, and exactly one
    real-2-dimensional (one complex) redundancy is expected -- the affine
    (scale+rotation) freedom the cut integrals are blind to by
    construction -- leaving ``n-2`` independent loops."""
    rows = []
    sel = []
    for li in range(len(loops)):
        row1 = np.concatenate([[G[li, m].real, -G[li, m].imag]
                               for m in range(G.shape[1])])
        row2 = np.concatenate([[G[li, m].imag, G[li, m].real]
                               for m in range(G.shape[1])])
        c = rows + [row1, row2]
        if np.linalg.matrix_rank(np.array(c), tol=tol) == len(rows) + 2:
            rows = c
            sel.append(li)
    return sel


# --------------------------------------------------------------------------- #
# P5.3 -- target selection (naive rounding) and the homotopy/Newton solve     #
# --------------------------------------------------------------------------- #

def _point_segment_distance(p: complex, a: complex, b: complex) -> float:
    ab = b - a
    denom = (ab.conjugate() * ab).real
    if denom == 0.0:
        return abs(p - a)
    t = np.clip(((p - a).conjugate() * ab).real / denom, 0.0, 1.0)
    return abs(p - (a + t * ab))


def node_count_for(cones0: np.ndarray, loops, base: int = 16, cap_n: int = 48,
                   ratio_lo: float = 0.15, ratio_hi: float = 2.0) -> int:
    """Gauss-Jacobi node count from the configuration's worst segment
    clearance: over all loop segments, the minimum distance from any *other*
    cone to the segment divided by the segment's length. That ratio maps
    linearly from ``base`` nodes (ratio >= ``ratio_hi``) to ``cap_n`` (ratio
    <= ``ratio_lo``).

    Trap (silent): with a fixed 16 nodes, a segment passing within ~0.06-0.09x
    its length of a third cone (P0 corpus) lets Newton report convergence
    (< 1e-12) at a point that is only 2.3e-6 accurate at 48 nodes; 48 nodes
    recover 2.9e-14.
    """
    n = len(cones0)
    worst = np.inf
    for (i, j, _sg) in loops:
        length = abs(cones0[j] - cones0[i])
        if length == 0.0:
            continue
        others = [k for k in range(n) if k not in (i, j)]
        if not others:
            continue
        clearance = min(_point_segment_distance(cones0[k], cones0[i], cones0[j])
                        for k in others)
        worst = min(worst, clearance / length)
    if worst >= ratio_hi:
        return base
    if worst <= ratio_lo:
        return cap_n
    frac = (ratio_hi - worst) / (ratio_hi - ratio_lo)
    return int(round(base + frac * (cap_n - base)))


def round_to_lattice(q: np.ndarray, sublattice: str = "Z[i]") -> np.ndarray:
    """Nearest lattice point to each ``q``, in ``Z[i]`` or in its index-2
    sublattice ``(1+i)Z[i] = {a+bi : a+b even}``.

    With ``Z[i]`` periods, cone-to-cone developed offsets land on
    ``((1-i)/2)Z[i]``: half the cones sit at cell centres, and
    :func:`shapegen.quad.grid.quad_grid` renders those as triangles rather
    than valence-3 vertices. ``(1+i)Z[i]`` puts every cone on a grid vertex
    (pure quads) but is harder: on the P0 corpus it solves at cell 0.025 vs
    0.070, ~8x the faces. Hence ``Z[i]`` is the default.
    """
    a = np.round(q.real)
    b = np.round(q.imag)
    if sublattice == "Z[i]":
        return a + 1j * b
    if sublattice != "(1+i)Z[i]":
        raise ValueError(f"unknown sublattice {sublattice!r}")
    odd = ((a + b) % 2) != 0
    ra, rb = q.real - a, q.imag - b
    # fix whichever coordinate was rounded furthest -- the cheapest repair
    take_a = np.abs(ra) >= np.abs(rb)
    step = np.where(np.signbit(np.where(take_a, ra, rb)), -1.0, 1.0)
    a = np.where(odd & take_a, a + step, a)
    b = np.where(odd & ~take_a, b + step, b)
    return a + 1j * b


def _attempt(cones0, a, idx, loops, sel, t0, cap, s, goal, n_solve: int = 16,
            n_check: int = 48, maxeval: int = 250, tmax: float = 20.0):
    n_check = max(n_check, n_solve)
    C = cones0.copy()
    tau, dtau, nit = 0.0, 0.3, 0
    T = time.time()
    while tau < 1 - 1e-12:
        if nit > maxeval or time.time() - T > tmax:
            return None
        st = min(dtau, 1 - tau)
        tg = t0 + (tau + st) * (goal * s - t0)
        X = C.copy()
        ok = False
        for _ in range(30):
            tt, GG = periods_and_grad(X, a, idx, loops, n=n_solve)
            nit += 1
            r = tt[sel] - tg
            if np.linalg.norm(r) < 1e-12:
                ok = True
                break
            if np.abs(X - cones0).max() > cap:
                return None
            sol = np.linalg.lstsq(GG[sel], -r, rcond=1e-9)[0]
            al, good = 1.0, False
            for _ in range(20):
                Y = X + al * sol
                if np.abs(Y - cones0).max() > cap:
                    al *= 0.5
                    continue
                t2, _ = periods_and_grad(Y, a, idx, loops, n=n_solve)
                nit += 1
                if np.linalg.norm(t2[sel] - tg) < np.linalg.norm(r):
                    X = Y
                    good = True
                    break
                al *= 0.5
            if not good:
                break
        if ok:
            C = X
            tau += st
            dtau = min(dtau * 1.8, 0.5)
        else:
            dtau *= 0.45
            if dtau < 1e-5:
                return None
    tf, _ = periods_and_grad(C, a, idx, loops, n=n_check)
    q = tf / s
    if max(abs(q[k] - np.round(q[k])) for k in range(len(loops))) > 1e-6:
        return None
    return C


def solve_lattice(cones0: np.ndarray, idx: np.ndarray,
                  cell_sizes=(0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025,
                              0.018, 0.012, 0.008),
                  skip_factor: float = 1.6,
                  sublattice: str = "Z[i]") -> dict:
    """Find a cell size ``s`` (the largest of ``cell_sizes`` that works) and a
    perturbation of ``cones0`` within ``0.25 x`` the minimum cone gap such
    that every loop's period lands on ``ℤ[i]·s``.

    ``sublattice="(1+i)Z[i]"`` asks for the index-2 sublattice instead, which
    puts every cone on a *grid vertex* rather than letting half of them land
    on cell centres -- a pure quad mesh downstream, at the cost of a smaller
    cell. See :func:`round_to_lattice` for the arithmetic and the measured
    price.

    ``cones0`` are complex chart positions (:func:`~shapegen.quad.cones.project_cones`'s
    ``z``). Sizes whose linearised predicted motion exceeds ``skip_factor x
    cap`` are skipped without a Newton attempt.

    Returns a dict: ``cell`` (``None`` if nothing in ``cell_sizes`` solved),
    ``cones`` (the solved positions, or ``None``), ``gap``, ``cap`` (the
    ``0.25 x gap`` displacement budget), ``loops``, ``sel`` (the independent
    loop indices), ``sweep`` (per-cell-size diagnostics) and ``n_solve``
    (the node count from :func:`node_count_for`).
    """
    cones0 = np.asarray(cones0, dtype=complex)
    idx = np.asarray(idx, dtype=float)
    a = -idx

    E = euclidean_mst(cones0)
    loops = loop_edges(idx, E)
    n_solve = node_count_for(cones0, loops)
    n_check = max(48, n_solve)
    t, G = periods_and_grad(cones0, a, idx, loops, n=n_solve)
    sel = select_independent_loops(loops, G)
    t0 = t[sel]

    d = np.abs(cones0[:, None] - cones0[None, :])
    np.fill_diagonal(d, np.inf)
    gap = float(d.min())
    cap = 0.25 * gap
    Jp = np.linalg.pinv(realify(G[sel]), rcond=1e-9)

    def predicted_motion(goal, s):
        r = goal * s - t0
        real_r = np.concatenate([[x.real, x.imag] for x in r])
        return float(np.abs(Jp @ real_r).max())

    sweep = []
    for s in cell_sizes:
        goal = round_to_lattice(t0 / s, sublattice)
        pred = predicted_motion(goal, s)
        if pred > skip_factor * cap:
            sweep.append({"s": s, "predicted": pred, "status": "skipped"})
            continue
        C = _attempt(cones0, a, idx, loops, sel, t0, cap, s, goal,
                    n_solve=n_solve, n_check=n_check)
        sweep.append({"s": s, "predicted": pred,
                      "status": "solved" if C is not None else "no"})
        if C is not None:
            return {"cell": s, "cones": C, "gap": gap, "cap": cap,
                    "loops": loops, "sel": sel, "sweep": sweep,
                    "n_solve": n_solve}
    return {"cell": None, "cones": None, "gap": gap, "cap": cap,
            "loops": loops, "sel": sel, "sweep": sweep, "n_solve": n_solve}


# --------------------------------------------------------------------------- #
# P5.4 -- the cheap gate                                                      #
# --------------------------------------------------------------------------- #

def conegate(C: np.ndarray, idx: np.ndarray, n: int = 40) -> np.ndarray:
    """Cheap independent check (``conegate.py``): the developing map ``w`` at
    every cone, accumulated along the spanning tree one cut integral per
    edge, using the single-valued ``ψ = prod (z-p)^{±1}`` with ``arg(ψ)``
    unwrapped (no branch choice).

    Requires every index to be exactly ``±1/4``. Returns ``w`` per cone
    (``w[0] = 0``); the caller checks ``max|2w/s - round|`` (1.37e-13 on the
    bundle's 17-edge case).
    """
    idx = np.asarray(idx, dtype=float)
    a = -idx
    num = [j for j in range(len(C)) if a[j] < 0]
    den = [j for j in range(len(C)) if a[j] > 0]

    def cut_int_psi(i, j):
        x, w = gj_nodes(a[i], a[j], n)
        L = C[j] - C[i]
        z = C[i] + L * (1.0 + x) / 2.0
        nn = np.ones_like(z, dtype=complex)
        dd = np.ones_like(z, dtype=complex)
        for k in num:
            if k not in (i, j):
                nn = nn * (z - C[k])
        for k in den:
            if k not in (i, j):
                dd = dd * (z - C[k])
        core = nn / dd
        ph = np.unwrap(np.angle(core))
        Q = np.abs(core) ** (-0.25) * np.exp(-1j * ph / 4.0)
        pref = (L * np.exp(a[i] * np.log(L)) * np.exp(a[j] * np.log(-L))
               * 2.0 ** (-a[i] - a[j]) / 2.0)
        return pref * np.sum(w * Q)

    tree = euclidean_mst(C)
    adj = {i: [] for i in range(len(C))}
    for (u, v) in tree:
        adj[u].append(v)
        adj[v].append(u)
    wv = {0: 0j}
    Qd = deque([0])
    vals = {}
    while Qd:
        u = Qd.popleft()
        for v in adj[u]:
            if v in wv:
                continue
            e = (min(u, v), max(u, v))
            if e not in vals:
                vals[e] = cut_int_psi(*e)
            wv[v] = wv[u] + (vals[e] if u < v else -vals[e])
            Qd.append(v)
    return np.array([wv[i] for i in range(len(C))])


# --------------------------------------------------------------------------- #
# Diagnostics (ported from core.py)                                           #
# --------------------------------------------------------------------------- #

def integrand_along(path: np.ndarray, cones: np.ndarray, idx: np.ndarray):
    """``prod (z-p_i)^(-k_i)`` along ``path``, with continuous branch
    tracking via ``np.unwrap`` (not a discrete choice)."""
    expo = -np.asarray(idx, dtype=float)
    v = np.ones(len(path), dtype=complex)
    for k, p in enumerate(cones):
        d = path - p
        ang = np.unwrap(np.angle(d))
        v = v * np.maximum(np.abs(d), 1e-9) ** expo[k] * np.exp(1j * expo[k] * ang)
    return v


def integrate(path: np.ndarray, cones: np.ndarray, idx: np.ndarray):
    """Cumulative trapezoidal integral of :func:`integrand_along` over
    ``path``, starting at 0."""
    v = integrand_along(path, cones, idx)
    seg = 0.5 * (v[1:] + v[:-1]) * np.diff(path)
    return np.concatenate([[0.0 + 0j], np.cumsum(seg)])


def _circle(ctr: complex, r: float, n: int = 2000):
    return ctr + r * np.exp(1j * np.linspace(0, 2 * np.pi, n))


def period(ctr: complex, r: float, cones: np.ndarray, idx: np.ndarray,
          n: int = 2000):
    """The developing map's period around a circle of radius ``r`` centered
    at ``ctr`` -- a Gauss-Bonnet-style diagnostic, not part of the solve."""
    return integrate(_circle(ctr, r, n), cones, idx)[-1]


def rotation(ctr: complex, r: float, cones: np.ndarray, idx: np.ndarray,
            n: int = 2000):
    """Holonomy (rotation) of the frame around that same circle."""
    v = integrand_along(_circle(ctr, r, n), cones, idx)
    return v[-1] / v[0]


def enclosed_index(ctr: complex, r: float, cones, idx) -> float:
    return sum(k for p, k in zip(cones, idx) if abs(p - ctr) < r)


def clearance(ctr: complex, r: float, cones: np.ndarray) -> float:
    """Distance from the circle to the nearest cone -- how much room a
    diagnostic loop actually has before it grazes a singularity."""
    return float(min(abs(abs(p - ctr) - r) for p in cones))


def gauss_bonnet(idx: np.ndarray) -> float:
    """Total index -- must be 2 for a genus-0 field (Poincare-Hopf)."""
    return float(np.sum(idx))
