"""Extraction (PLAN.md §P6): march the developing map's grid from the solved
cones and assemble it into a quad mesh, ported from ``s6b_extract.py`` +
``04_extraction/{polyfield,simpler2,wavefront,phase5}.py``.

.. note::
   **Superseded by :mod:`shapegen.quad.grid`, and kept only as the P0
   reproduction.** This module is the faithful port of the bundle's own
   single-chart extraction, and ``test_quad_extract.py`` pins it to the
   saved ``q2_pos.npy``/``q2_F.npy``/``q2_adj.npy``, which is worth keeping.
   It is not what anything should call: on that same P0 configuration it
   reaches 91% coverage with 98 vertices of valence 2, 275 of valence 3
   where there are only 8 cones, and 77 of valence 5, while
   :func:`shapegen.quad.grid.quad_grid` closes the sphere exactly (Euler
   characteristic 2, coverage 1.000000000, no irregular vertex that is not a
   cone).

   Two claims below are wrong and are left in place, corrected here, because
   they are what the replacement had to disprove. §P6.4's "bound the traced
   region" and ``extract_quad_mesh``'s analysis of ``adaptive_tol`` both rest
   on the chart's excluded pole carrying an implicit extra singularity. It
   does not: that argument needs ``sum_k a_k != 0`` and Gauss-Bonnet fixes
   ``sum_k idx_k = 2`` exactly, which makes ``omega = W dz`` regular *and
   non-vanishing* at infinity -- the one value that cancels the order-2 pole
   of ``dz`` there. What does blow up approaching the pole is the chart
   *coordinate*, so a fixed developed step becomes an ever larger step in
   ``z``; that is an artifact of using one chart, and the measured 0.911
   ceiling is not asymptotic at all but ``r^2/(r^2+1)`` at this module's own
   ``refuse_radius = 3.2``.

**§P6.1 — the identity, and why there is nothing to track.**

    (w')^-4 = prod (z - p_i)^(sign_i)          <-- single-valued

``|w'| = |ψ|^(-1/4)`` and the four-fold direction set are *global*; only
*which* of the four is locally forward is a discrete choice, and that choice
is exactly the grid's own 90° symmetry — never a genuine global branch. This
module never picks among four values; it tracks ``log(w')`` continuously
(§P6.2's unwrap) instead, which sidesteps the choice entirely.

**§P6.2 — magnitude by log-sum, phase by unwrapping.** ``wavefront.py``'s
``psi_arg`` multiplies raw magnitudes (``np.prod(np.abs(D))``) and overflows
far from the cones; ``simpler2.py``'s log-sum version doesn't. Likewise the
phase is tracked by unwrapping against a running reference, not by
``s6b_extract.py``'s nearest-of-four search (unwrapping is both faster and
more accurate — it never makes a wrong discrete choice).

**§P6.3 — nsub=1, but only validated at the P0 corpus's own cone spacing.**
The cone solver already moves cones by ~2% of the cell spacing (§P5's
``cap``), so vertex errors below ~1e-3 of a cell are invisible downstream,
and RK4 with 1 substep (4 evaluations/cell) reaches ~9e-7 *there* --
``s6b_extract.py``'s own default of ``nsub=14`` looked like 14x the work for
accuracy the pipeline cannot see. That accuracy figure is a property of the
step size actually taken, though, not of ``nsub=1`` itself: a cell fixed at
arc length ``s`` in the *developed* plane is a chart-space step of
``s / |w'|``, and far from every cone (where ``|w'|`` is small) that step
can be large enough for one RK4 step to be a bad approximation regardless of
how small ``s`` is. Confirmed on a real face at bandwidth 32 (not the P0
corpus): ``nsub=1`` left a visibly self-overlapping (bowtie) quad region far
from the cones and ~2.6% degenerate (zero-area) quads; ``nsub=8`` resolved
it. Callers whose cones are more asymmetrically spread than the reference
configuration should raise ``nsub`` rather than trust this default --
:func:`~shapegen.quad.lift`'s caller in ``scripts/quad_demo.py`` does.

**§P6.5 — merge geometrically, never by index.** Integer ``(i,j)`` grid
labels assume trivial holonomy; going around a cone rotates the local frame
by a quarter turn, so one plane point is reached via different routes under
different labels. Vertices here are merged by position (within
``tol_factor * s``), and quads are 4-cycles of the resulting adjacency graph,
not index quadruples.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .lift import chart_coverage

__all__ = ["psi_log", "psi_arg", "velocity", "rk4_cell_step", "find_seed",
           "extract_quad_mesh", "assemble_quads", "closure_defects",
           "quad_anisotropy"]


# --------------------------------------------------------------------------- #
# The field: log(w') tracked continuously                                     #
# --------------------------------------------------------------------------- #

def _exponent_quantum(a: np.ndarray) -> float:
    """The common factor ``q`` such that every ``a_k = n_k * q`` for an
    integer ``n_k`` -- for this pipeline's cones (every ``|idx| = 1/4``),
    ``q = 1/4`` and ``ψ = prod_k (z-C_k)^{n_k}`` is a genuine single-valued
    meromorphic function (integer exponents, no branch cut at all), with the
    ordinary period-``2π`` phase ambiguity. ``w' = ψ^q``, so unwrapping must
    happen on ``arg(ψ)`` *before* scaling by ``q`` -- see :func:`psi_log`'s
    docstring for the bug this fixes.
    """
    nz = np.abs(a[np.abs(a) > 1e-12])
    return float(nz.min()) if len(nz) else 1.0


def psi_log(Z: np.ndarray, cones: np.ndarray, a: np.ndarray):
    """Raw ``log(ψ) = Σ_k n_k·(log|Z-C_k| + i·angle(Z-C_k))`` at every point
    in ``Z`` (``n_k = a_k/q``, ``a = -idx`` matching
    :mod:`shapegen.quad.periods`'s convention), phase at its principal value
    (not yet unwrapped). ``log(w') = q·log(ψ)``.

    **Why the raw (unscaled) phase, not ``arg(w')`` directly.** An earlier
    version of this function summed ``a_k·angle(D_k)`` directly, producing
    ``arg(w')`` with its *true* ambiguity of ``q·2π`` (``π/2`` for this
    pipeline's cones) -- but then unwrapped it by snapping to the nearest
    multiple of a full ``2π``, a grid four times too coarse to see that
    smaller ambiguity. The result silently picked the wrong branch after a
    few marching steps (confirmed against the P0 corpus: it diverged from
    the reference trace by exactly a factor of ``i`` -- a 90° rotation --
    starting at the third step, causing the extraction to never close up and
    hit the vertex cap instead of the expected ~2500). Unwrapping ``arg(ψ)``
    itself first, at its own genuine ``2π`` period, then scaling by ``q``,
    fixes this -- see ``test_quad_extract.py``'s reproduction of the P0
    corpus's ``q2_pos.npy``.
    """
    q = _exponent_quantum(a)
    n = a / q
    D = Z[:, None] - cones[None, :]
    lm = (n[None, :] * np.log(np.abs(D))).sum(axis=1)
    ph = (n[None, :] * np.angle(D)).sum(axis=1)
    return lm, ph, q


def psi_arg(Z: np.ndarray, PH: np.ndarray, cones: np.ndarray, a: np.ndarray):
    """``log(w') = lm_w + i·ph_w`` at every point in ``Z``, with the
    *unscaled* ``arg(ψ)`` unwrapped against a running reference ``PH`` (also
    unscaled -- carry this, not ``ph_w``, between calls). Returns
    ``(lm_w, ph_w, ph_raw)``: the first two already scaled by the exponent
    quantum (i.e. ready to use as ``log(w')``), the third the updated
    reference to pass to the next call.
    """
    lm, ph, q = psi_log(Z, cones, a)
    ph = ph + 2.0 * np.pi * np.round((PH - ph) / (2.0 * np.pi))
    return q * lm, q * ph, ph


def velocity(Z: np.ndarray, PH: np.ndarray, d: complex, cones: np.ndarray,
            a: np.ndarray):
    """``dz/dstep = d / w'``, and the updated (unscaled) phase reference."""
    lm_w, ph_w, ph_raw = psi_arg(Z, PH, cones, a)
    v = np.exp(np.clip(-lm_w, -60.0, 60.0)) * np.exp(-1j * ph_w) * d
    return v, ph_raw


def rk4_cell_step(Z: np.ndarray, PH: np.ndarray, d: complex, cones: np.ndarray,
                  a: np.ndarray, s: float, nsub: int = 1):
    """Advance every point in ``Z`` one full cell (arc length ``s`` in the
    developed plane) in direction ``d``, via RK4 with ``nsub`` substeps.
    Default ``nsub=1``: see the module docstring's §P6.3. ``PH`` (in and
    out) is the unscaled ``arg(ψ)`` reference -- see :func:`psi_arg`.
    """
    h = s / nsub
    cur = Z.copy()
    ph = PH.copy()
    for _ in range(nsub):
        k1, p1 = velocity(cur, ph, d, cones, a)
        k2, _p2 = velocity(cur + 0.5 * h * k1, p1, d, cones, a)
        k3, _p3 = velocity(cur + 0.5 * h * k2, p1, d, cones, a)
        k4, _p4 = velocity(cur + h * k3, p1, d, cones, a)
        cur = cur + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        _, _, ph = psi_arg(cur, p1, cones, a)
    return cur, ph


# --------------------------------------------------------------------------- #
# Seed and the BFS extraction                                                 #
# --------------------------------------------------------------------------- #

def find_seed(cones: np.ndarray, grid_radius: float = 1.1, grid_n: int = 41):
    """The chart point farthest from every cone -- a safe place to start
    marching. Ported as a dense grid search, matching the source exactly."""
    xs = np.linspace(-grid_radius, grid_radius, grid_n)
    ys = np.linspace(-grid_radius, grid_radius, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = (X + 1j * Y).ravel()
    d = np.abs(Z[:, None] - cones[None, :]).min(axis=1)
    return complex(Z[np.argmax(d)])


def extract_quad_mesh(cones: np.ndarray, idx: np.ndarray, s: float,
                      nsub: int = 1, tol_factor: float = 0.30,
                      refuse_near: float = 0.40, refuse_radius: float = 3.2,
                      max_vertices: int = 8000, seed: complex | None = None,
                      adaptive_tol: bool = False):
    """The BFS march: from a seed, step in all 4 directions, merging any new
    point that lands within ``tol_factor * s`` of an already-known vertex
    (geometric merge, §P6.5) instead of by any coordinate label. Stops a
    branch when it would land too close to a cone (``refuse_near * s``) or
    outside the traced region (``|z| > refuse_radius``) -- that radius *is*
    the chart, reported back as ``coverage``.

    **Why coverage plateaus well short of ``refuse_radius``, and
    ``adaptive_tol``.** ``s`` is arc length in the *developed* plane; a
    step's actual size in chart coordinates is ``s`` scaled by the local
    chart velocity, which shrinks continuously approaching the chart's own
    excluded pole (that point necessarily carries an implicit extra
    singularity -- for any finite cone set with ``Σaₖ ≠ 0``, a standard fact
    about meromorphic functions on the sphere forces one of order ``-Σaₖ``
    wherever the pole is put, independent of *which* point is chosen).
    ``tol = tol_factor * s`` is a *fixed* absolute distance, so once the
    local step shrinks below it, every further outward step lands within
    ``tol`` of the vertex it started from and gets treated as "arrived back
    where it began" -- a real vertex merged away as a phantom self-merge,
    not a genuine refusal. This isn't a boundary case: on a real face at
    L=32, steps at the reached frontier (max |z| ≈ 2.22, well under
    ``refuse_radius=3.2``) already measured 0.056-0.065 against
    ``tol=0.06`` -- the two were already the same order of magnitude, and
    every further step keeps shrinking while ``tol`` does not.

    ``adaptive_tol=True`` uses ``tol_factor * (this step's own size)``
    instead -- scaling the merge threshold to the *local* cell size lets the
    trace keep resolving genuinely smaller (but real) cells instead of
    conflating them with a return to the start. Measured on the same face:
    coverage 0.832 -> 0.846/0.876/0.896/0.911 at max_vertices
    2000/5000/10000/20000 (never fully converging -- the true limit is
    100% only asymptotically, since cells keep shrinking without bound
    approaching that point), with *zero* closure-defect-rejected quads at
    every budget tried (vs. 11% at the fixed tolerance), because the
    resulting cap of shrinking quads is smooth and well-formed rather than
    self-overlapping. Left opt-in (default ``False``) because it changes
    which vertices merge everywhere the local step size varies, not only
    near the pole -- this module's own P0-corpus reproduction test pins an
    exact merge count at the fixed-tolerance behavior.

    ``max_vertices`` is a safety cap, not a target: the P0 reference
    configuration converges (the BFS queue empties on its own) at 2519
    vertices, but a denser cell size or a larger chart extent needs more --
    hitting the cap truncates the mesh mid-BFS with an unclosed boundary,
    which is silent unless checked. ``converged`` in the return value is
    exactly that check -- with ``adaptive_tol=True`` expect it to stay
    ``False`` at any practical budget, by design (see above); raise
    ``max_vertices`` to trade compute for coverage, not to chase
    ``converged=True``.

    Returns a dict: ``pos`` (complex, chart positions), ``adj`` (list of
    neighbor-index sets), ``merged``, ``refused`` (counts), ``converged``
    (``False`` means ``max_vertices`` was hit -- raise it and re-run), and
    ``coverage`` (fraction of the sphere's area within the traced region,
    from the actual farthest point reached -- see
    :func:`shapegen.quad.lift.chart_coverage`).
    """
    cones = np.asarray(cones, dtype=complex)
    a = -np.asarray(idx, dtype=float)
    if seed is None:
        seed = find_seed(cones)
    tol = tol_factor * s

    pos = [seed]
    _lm0, ph0, _q0 = psi_log(np.array([seed]), cones, a)
    refph = [float(ph0[0])]
    adj = [set()]
    Q = deque([0])
    refused = 0
    merged = 0
    converged = True
    while Q:
        vi = Q.popleft()
        if len(pos) > max_vertices:
            converged = False
            break
        for d in (1.0 + 0j, -1.0 + 0j, 1j, -1j):
            z0 = np.array([pos[vi]])
            p0 = np.array([refph[vi]])
            try:
                zn_a, phn_a = rk4_cell_step(z0, p0, d, cones, a, s, nsub=nsub)
            except Exception:
                refused += 1
                continue
            zn, phn = complex(zn_a[0]), float(phn_a[0])
            dmin = np.abs(zn - cones).min()
            if not np.isfinite(zn) or dmin < refuse_near * s or abs(zn) > refuse_radius:
                refused += 1
                continue
            step_tol = tol_factor * abs(zn - z0[0]) if adaptive_tol else tol
            P = np.array(pos)
            dd = np.abs(P - zn)
            k = int(np.argmin(dd))
            if dd[k] < step_tol:
                j = k
                merged += 1
            else:
                pos.append(zn)
                refph.append(phn)
                adj.append(set())
                j = len(pos) - 1
                Q.append(j)
            adj[vi].add(j)
            adj[j].add(vi)

    pos = np.array(pos)
    coverage = chart_coverage(pos)
    return {"pos": pos, "adj": adj, "merged": merged, "refused": refused,
            "coverage": coverage, "seed": seed, "converged": converged}


# --------------------------------------------------------------------------- #
# Quads: 4-cycles of the adjacency graph, filtered by closure defect          #
# --------------------------------------------------------------------------- #

def assemble_quads(pos: np.ndarray, adj) -> list[list[int]]:
    """Quads as 4-cycles ``v -> a -> c -> b -> v`` of the adjacency graph
    (never index quadruples -- see the module docstring's §P6.5)."""
    quads = set()
    for v in range(len(pos)):
        nb = sorted(adj[v])
        for ia in range(len(nb)):
            for ib in range(ia + 1, len(nb)):
                A, B = nb[ia], nb[ib]
                common = (adj[A] & adj[B]) - {v}
                for c in common:
                    if len(adj[c]) >= 2:
                        q = tuple(sorted([v, A, c, B]))
                        quads.add((q, (v, A, c, B)))
    seen = set()
    ql = []
    for q, order in quads:
        if q in seen:
            continue
        seen.add(q)
        ql.append(list(order))
    return ql


def closure_defects(pos: np.ndarray, quads) -> np.ndarray:
    """How far each candidate quad is from being a consistent parallelogram
    in the developed plane (``|d1-d2|`` and ``|e1-e2|``, scaled by the
    quad's own size) -- large values are path-reconvergence artifacts to
    reject, not real quads."""
    defect = []
    for q in quads:
        p = pos[q]
        d1, d2 = p[1] - p[0], p[2] - p[3]
        e1, e2 = p[3] - p[0], p[2] - p[1]
        sc = max(abs(d1), abs(e1), 1e-12)
        defect.append(max(abs(d1 - d2), abs(e1 - e2)) / sc)
    return np.array(defect)


def quad_anisotropy(P3: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Per-quad edge-length anisotropy (longest / shortest edge) on the
    lifted 3D positions ``P3`` -- the quality number Gate P6 reports a
    median for."""
    out = []
    for q in quads:
        p = P3[q]
        lens = [np.linalg.norm(p[k] - p[(k + 1) % 4]) for k in range(4)]
        out.append(max(lens) / max(min(lens), 1e-12))
    return np.array(out)
