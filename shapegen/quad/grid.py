"""Grid tracing on the whole sphere.

Given cones that :mod:`shapegen.quad.periods` has placed on the lattice, walk
the developing map's square grid over the *entire* sphere and assemble a
**closed** quad(-dominant) mesh. :func:`quad_grid` is the entry point; it
returns a :class:`QuadGrid` carrying the diagnostics (``euler``,
``coverage``, ``merge_residual``, ``frame_residual``) that say whether to
trust it.

**Two charts.** The developing map's derivative is the meromorphic function

    W(z) = prod_k (z - C_k)^(-idx_k)

and the object integrated is the 1-form ``omega = W dz``. Gauss-Bonnet forces
``sum_k idx_k = 2``, so ``W ~ z^-2`` and in ``zeta = 1/z``

    omega = -prod_k (1 - C_k zeta)^(-idx_k) dzeta

is regular and non-vanishing at ``zeta = 0``: the chart pole is an ordinary
point of the flat structure, with no hidden singularity. Each chart is used
only on its own hemisphere, which removes the coordinate blow-up of the step
size near the other chart's pole.

**Adaptive substep.** Each cell is integrated with RK4 in developed arc
length, the substep capped by a fraction of the distance to the nearest cone
and an absolute fraction of the chart (:func:`trace_step`). There is no
fixed substep count to tune, and a step that ends on a cone (velocity
``~ dist^(1/4)``) refines itself.

**Connectivity from a rotation system.** Every traced edge records its
outgoing tangent at both ends; faces are the orbits of ``phi = sigma . alpha``
on darts (:func:`faces_from_rotation`). Every dart lies in exactly one orbit,
so the output is a closed surface by construction and ``V - E + F`` is a
measurement. Cones need no special case: a cone on a lattice point becomes a
valence-3 (or 5) vertex, one on a cell centre a triangular (or pentagonal)
face.

**Where the cones land.** :func:`~shapegen.quad.periods.solve_lattice` puts
every loop period on ``s.Z[i]``, which places cone offsets on
``((1-i)/2).s.Z[i]``: half on lattice points, half on cell centres (the
latter come out as triangles). ``sublattice="(1+i)Z[i]"`` puts every cone on
a lattice point for a pure quad mesh, at a smaller cell (0.025 vs 0.070 on
the P0 corpus, ~8x the faces). :func:`quad_grid` reports which happened in
``cone_vertices`` and ``face_degrees`` rather than choosing.

It replaced a single-chart extractor whose coverage was capped at 0.911.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .periods import gj_nodes

__all__ = ["QuadGrid", "quad_grid", "chart_of", "to_chart", "from_chart",
           "chart_tangent", "form_log_abs", "form_arg", "trace_step",
           "seed_vertex", "faces_from_rotation", "spherical_area",
           "quad_isotropy", "quad_areas", "developed_cell_area",
           "developed_area", "cell_for_faces"]


# --------------------------------------------------------------------------- #
# The two charts                                                              #
# --------------------------------------------------------------------------- #
#
# chart 0:  z    = (X0 + i X1) / (1 - X2)      used where X2 <= 0, |z|    <= 1
# chart 1:  zeta = (X0 - i X1) / (1 + X2)      used where X2 >= 0, |zeta| <= 1
#
# and zeta = 1/z exactly, so chart 1 is the coordinate in which chart 0's
# point at infinity sits at the origin. Both are conformal, and both carry
# the same orientation (each has the *inward* normal as d/dx cross d/dy), so
# a single sign convention for "counter-clockwise" works in both.

def chart_of(X: np.ndarray) -> np.ndarray:
    """Which chart to use at each sphere point: 0 below the equator, 1 above
    -- always the one whose coordinate has modulus ``<= 1``."""
    return (np.asarray(X)[..., 2] > 0.0).astype(np.int8)


def to_chart(X: np.ndarray, c) -> np.ndarray:
    """Sphere points ``(n,3)`` -> the complex coordinate of chart ``c``
    (a scalar, or one chart id per point)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    c = np.broadcast_to(np.asarray(c), (len(X),))
    out = np.empty(len(X), dtype=complex)
    m0 = c == 0
    if m0.any():
        out[m0] = (X[m0, 0] + 1j * X[m0, 1]) / (1.0 - X[m0, 2])
    if (~m0).any():
        m1 = ~m0
        out[m1] = (X[m1, 0] - 1j * X[m1, 1]) / (1.0 + X[m1, 2])
    return out


def from_chart(zeta: np.ndarray, c) -> np.ndarray:
    """Chart coordinate -> sphere points ``(n,3)``. Regular everywhere in
    both charts, which is the point of having two."""
    z = np.atleast_1d(np.asarray(zeta, dtype=complex))
    c = np.broadcast_to(np.asarray(c), z.shape)
    r2 = z.real ** 2 + z.imag ** 2
    den = r2 + 1.0
    sgn = np.where(c == 0, 1.0, -1.0)
    return np.stack([2.0 * z.real / den,
                     sgn * 2.0 * z.imag / den,
                     sgn * (r2 - 1.0) / den], axis=1)


def chart_tangent(zeta: np.ndarray, c, delta: np.ndarray) -> np.ndarray:
    """The sphere tangent vector a complex chart displacement ``delta``
    generates at ``zeta`` -- the differential of :func:`from_chart`.

    Two things need directions in a chart-free form: comparing the developed
    frames of two paths that reach the same vertex (they may differ only by
    a power of ``i``), and sorting the darts around a vertex for the
    rotation system. Both would be meaningless in a per-point chart.
    """
    z = np.atleast_1d(np.asarray(zeta, dtype=complex))
    c = np.broadcast_to(np.asarray(c), z.shape)
    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    x, y = z.real, z.imag
    n2 = x * x + y * y + 1.0
    den = (n2 * n2)[:, None]
    sgn = np.where(c == 0, 1.0, -1.0)
    dx = np.stack([2.0 * n2 - 4.0 * x * x, sgn * (-4.0 * x * y), sgn * 4.0 * x],
                  axis=1) / den
    dy = np.stack([-4.0 * x * y, sgn * (2.0 * n2 - 4.0 * y * y), sgn * 4.0 * y],
                  axis=1) / den
    return d.real[:, None] * dx + d.imag[:, None] * dy


def _cone_coords(cones: np.ndarray, c: int) -> np.ndarray:
    """Cone positions in chart ``c``. A cone at chart 0's origin has no
    chart-1 coordinate; it is then as far from chart 1's disk as a point can
    be, and ``inf`` is the right answer for the only use here (distance to
    the nearest cone, which caps the substep)."""
    if c == 0:
        return cones
    out = np.full(len(cones), np.inf + 0j, dtype=complex)
    nz = cones != 0
    out[nz] = 1.0 / cones[nz]
    return out


# --------------------------------------------------------------------------- #
# The form, in either chart                                                   #
# --------------------------------------------------------------------------- #

def _quantum(a: np.ndarray) -> float:
    """``q`` such that every ``a_k = n_k q`` with integer ``n_k``. For
    ``|idx| = 1/4`` cones ``q = 1/4`` and ``psi = prod f_k^{n_k}`` is
    single-valued, so its argument has an honest ``2*pi`` period: unwrap
    *there*, then scale by ``q``.

    Trap: unwrapping ``arg W`` itself at ``2*pi`` is four times too coarse to
    see its real ``pi/2`` ambiguity and silently rotates the trace by ``i``."""
    nz = np.abs(a[np.abs(a) > 1e-12])
    return float(nz.min()) if len(nz) else 1.0


def _factors(zeta: np.ndarray, c: np.ndarray, cones: np.ndarray) -> np.ndarray:
    """``(m, k)`` linear factors of ``psi`` at each point, in that point's own
    chart: ``z - C_k`` in chart 0, ``1 - C_k zeta`` in chart 1."""
    out = np.empty((len(zeta), len(cones)), dtype=complex)
    m0 = c == 0
    if m0.any():
        out[m0] = zeta[m0][:, None] - cones[None, :]
    if (~m0).any():
        out[~m0] = 1.0 - cones[None, :] * zeta[~m0][:, None]
    return out


def form_log_abs(zeta: np.ndarray, c: np.ndarray, cones: np.ndarray,
                 a: np.ndarray) -> np.ndarray:
    """``log|W|`` in each point's own chart, by log-sum -- never a product of
    magnitudes, which overflows in the far field."""
    F = _factors(zeta, c, cones)
    return (a[None, :] * np.log(np.abs(F))).sum(axis=1)


def form_arg(zeta: np.ndarray, c: np.ndarray, cones: np.ndarray,
             a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``arg W``, continued from the running reference ``ref``.

    The additive constant is ``pi`` in chart 1 (``omega = -prod(...) dzeta``)
    and 0 in chart 0. It never has to be right in an absolute sense, only
    self-consistent, because a chart change transports ``arg W`` exactly
    (:func:`_switch_chart`) rather than re-deriving it.
    """
    F = _factors(zeta, c, cones)
    q = _quantum(a)
    n = a / q
    base = q * (n[None, :] * np.angle(F)).sum(axis=1) + np.pi * (c == 1)
    per = 2.0 * np.pi * q
    return base + per * np.round((ref - base) / per)


def _switch_chart(zeta: np.ndarray, c: np.ndarray, arg: np.ndarray,
                  which: np.ndarray):
    """Move the selected points to the other chart, exactly.

    ``zeta_new = 1/zeta_old``, and the 1-form's component picks up the
    Jacobian ``W_new = W_old . d(old)/d(new) = W_old . (-1/zeta_new^2)``, so
    ``arg W`` gains ``pi - 2 arg(zeta_new)`` -- *added*, not reduced modulo
    anything, which is what keeps the branch continuous across the equator.
    """
    zn, cn, an = zeta.copy(), c.copy(), arg.copy()
    zn[which] = 1.0 / zeta[which]
    cn[which] = 1 - c[which]
    an[which] = arg[which] + np.pi - 2.0 * np.angle(zn[which])
    return zn, cn, an


# --------------------------------------------------------------------------- #
# One cell of marching, with an adaptive substep                              #
# --------------------------------------------------------------------------- #

_SWITCH_AT = 1.15        # hysteresis: |coordinate| past which we flip charts


def _velocity(zeta, c, arg, d, cones, a):
    """``dzeta/dt = d / W`` at unit developed speed, and the continued
    ``arg W`` to carry into the next stage."""
    lm = form_log_abs(zeta, c, cones, a)
    ag = form_arg(zeta, c, cones, a, arg)
    return d * np.exp(-lm) * np.exp(-1j * ag), ag


def _cone_distance(zeta, c, cones):
    """Distance from each point to the nearest cone, in that point's chart."""
    out = np.empty(len(zeta))
    for cc in (0, 1):
        m = c == cc
        if m.any():
            P = _cone_coords(cones, cc)
            out[m] = np.abs(zeta[m][:, None] - P[None, :]).min(axis=1)
    return out


def trace_step(zeta0: np.ndarray, c0: np.ndarray, arg0: np.ndarray,
               d: np.ndarray, cones: np.ndarray, a: np.ndarray, s: float,
               near_frac: float = 0.20, abs_cap: float = 0.05,
               pos_tol: float = 1e-13, max_iter: int = 4000):
    """Advance a batch of points one cell (developed arc length ``s``) along
    developed direction ``d`` (one unit complex number per point).

    Each point picks its own substep, each iteration, as the largest that
    moves no more than ``near_frac`` of its distance to the nearest cone and
    no more than ``abs_cap`` in chart coordinates -- clipped to whatever
    developed length it has left. The cone cap is what makes a step
    that *ends* on a cone work: the velocity there vanishes like
    ``dist^(1/4)``, so the cap shrinks geometrically and the point approaches
    from one side instead of stepping across.

    A point is finished when its remaining developed length would move it
    less than ``pos_tol`` *in the chart*, not when that length reaches zero.
    Trap: approaching a cone the remaining arc length is a geometric series
    that never reaches zero, so a length criterion runs out ``max_iter``
    while sitting ~1e-16 from the cone; the position criterion terminates in
    a few hundred iterations at the same answer and does not affect regular
    steps.

    Returns ``(zeta, chart, argW, iterations, ok)``. ``ok`` is false only for
    a point that ran out of iterations; :func:`quad_grid` counts those as
    refused rather than letting them through as silent inaccuracies.
    """
    zeta = np.array(zeta0, dtype=complex, copy=True)
    c = np.array(c0, dtype=np.int8, copy=True)
    arg = np.array(arg0, dtype=float, copy=True)
    d = np.asarray(d, dtype=complex)
    rem = np.full(len(zeta), float(s))
    act = np.arange(len(zeta))

    it = 0
    while len(act) and it < max_iter:
        it += 1
        z, cc, ag, dd = zeta[act], c[act], arg[act], d[act]
        speed = np.exp(-form_log_abs(z, cc, cones, a))            # |dzeta/dt|
        cap = np.minimum(near_frac * _cone_distance(z, cc, cones), abs_cap)
        h = np.minimum(rem[act], np.maximum(cap / np.maximum(speed, 1e-300),
                                            1e-300))

        k1, a1 = _velocity(z, cc, ag, dd, cones, a)
        k2, _ = _velocity(z + 0.5 * h * k1, cc, a1, dd, cones, a)
        k3, _ = _velocity(z + 0.5 * h * k2, cc, a1, dd, cones, a)
        k4, _ = _velocity(z + h * k3, cc, a1, dd, cones, a)
        zn = z + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        an = form_arg(zn, cc, cones, a, a1)

        far = np.abs(zn) > _SWITCH_AT
        if far.any():
            zn, cc, an = _switch_chart(zn, cc, an, far)

        zeta[act], c[act], arg[act] = zn, cc, an
        new_rem = np.maximum(rem[act] - h, 0.0)
        rem[act] = new_rem
        # how far the remaining arc length would still move the point, using
        # this substep's speed -- an over-estimate approaching a cone, where
        # the speed only falls, so stopping on it is conservative
        act = act[new_rem * speed > pos_tol]
    ok = np.ones(len(zeta), dtype=bool)
    ok[act] = False                       # only whatever ran out of iterations
    return zeta, c, arg, it, ok


# --------------------------------------------------------------------------- #
# Seeding: one grid vertex, aligned so cone ``anchor`` lands on the lattice   #
# --------------------------------------------------------------------------- #

def _cut_from_cone(k: int, target: complex, cones: np.ndarray, a: np.ndarray,
                   n: int = 64):
    """``(w(target) - w(C_k), arg W(target))`` along the straight chart-0 ray
    from cone ``k`` to ``target``.

    The integrand carries an algebraic endpoint singularity ``rho^{a_k}`` at
    the cone that ordinary quadrature never resolves; Gauss-Jacobi with that
    exponent as its weight reaches machine precision in a dozen-odd nodes
    (:mod:`shapegen.quad.periods` uses the same nodes). Taking ``(z - C_k)^{a_k} = rho^{a_k} e^{i a_k theta}``
    along the fixed ray angle and unwrapping the remaining factors outwards
    from the cone pins the branch, and with it the grid's absolute
    orientation, which nothing upstream fixes.
    """
    theta = float(np.angle(target - cones[k]))
    r = float(abs(target - cones[k]))
    q = _quantum(a)
    nn = a / q
    other = np.arange(len(cones)) != k

    x, wq = gj_nodes(a[k], 0.0, n)
    order = np.argsort(x)
    z = cones[k] + (r * (1.0 + x[order]) / 2.0) * np.exp(1j * theta)
    F = z[:, None] - cones[None, other]
    ph = np.unwrap((nn[None, other] * np.angle(F)).sum(axis=1))
    G = np.exp((a[None, other] * np.log(np.abs(F))).sum(axis=1) + 1j * q * ph)

    w = (np.exp(1j * (1.0 + a[k]) * theta) * (r / 2.0) ** (1.0 + a[k])
         * np.sum(wq[order] * G))

    # arg W at the endpoint itself, continued from the last quadrature node
    Fe = target - cones[other]
    phe = float((nn[other] * np.angle(Fe)).sum())
    phe += 2.0 * np.pi * round((ph[-1] - phe) / (2.0 * np.pi))
    return complex(w), float(a[k] * theta + q * phe)


def seed_vertex(cones: np.ndarray, idx: np.ndarray, s: float, anchor: int = 0,
                probe_frac: float = 0.25, n_dir: int = 64,
                near_frac: float = 0.10, abs_cap: float = 0.02):
    """A first grid vertex, placed so that cone ``anchor`` falls exactly on a
    lattice point of the grid about to be traced.

    The period solve pins the cones only *relative* to one another (on
    ``((1-i)/2).s.Z[i]``, all 4-fold symmetry centres of the grid). Seeding
    anywhere else puts cone ``anchor`` at a generic offset, where a quarter
    turn of holonomy does not map the grid to itself and nothing closes.

    Method: probe a regular point a little off the cone, integrate
    ``w(probe) - w(C_anchor)`` exactly, and walk off its fractional part.
    Returns ``(zeta, chart, argW)`` of the resulting vertex, as length-1
    arrays.

    Trap: ``near_frac``/``abs_cap`` govern that last walk and must be passed
    through from the caller, not left at :func:`trace_step`'s defaults. An
    error here is a rigid offset of the whole grid; at the tracer defaults it
    put a floor of ``1.3e-7`` on ``merge_residual``.
    """
    cones = np.asarray(cones, dtype=complex)
    a = -np.asarray(idx, dtype=float)
    k = int(anchor)
    gaps = np.abs(cones - cones[k])
    gaps[k] = np.inf
    r = probe_frac * float(gaps.min())

    # a ray that keeps the probe segment as clear of the other cones as it can
    th = np.linspace(0.0, 2.0 * np.pi, n_dir, endpoint=False)
    probes = cones[k] + r * np.exp(1j * th)
    clear = np.abs(probes[:, None] - np.delete(cones, k)[None, :]).min(axis=1)
    p = probes[int(np.argmax(clear))]

    w, arg = _cut_from_cone(k, p, cones, a)
    target = np.round(w.real / s) + 1j * np.round(w.imag / s)
    if target == 0:
        # the nearest lattice point is the anchor cone itself, and a cone is
        # terminal in the trace (its velocity vanishes), so seeding there
        # would expand nothing. Step out to the next lattice point along
        # whichever axis the probe already leans towards.
        target = (1.0 + 0j) if abs(w.real) >= abs(w.imag) else 1j
        target *= np.sign(w.real if abs(w.real) >= abs(w.imag) else w.imag) or 1.0
    delta = s * target - w

    zeta = np.array([p])
    c = np.array([0], dtype=np.int8)
    argv = np.array([arg])
    zeta, c, argv = _switch_chart(zeta, c, argv, np.abs(zeta) > 1.0)

    if abs(delta) > 1e-13 * s:
        zeta, c, argv, _it, ok = trace_step(
            zeta, c, argv, np.array([delta / abs(delta)]), cones, a, abs(delta),
            near_frac=near_frac, abs_cap=abs_cap)
        if not ok[0]:
            raise RuntimeError("seed alignment step did not complete")
    return zeta, c, argv


# --------------------------------------------------------------------------- #
# The trace                                                                   #
# --------------------------------------------------------------------------- #

_DIRS = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j])


@dataclass
class QuadGrid:
    """A closed quad(-dominant) mesh of the sphere, with the diagnostics that
    say whether to believe it.

    ``points``/``frames`` live on the unit sphere;
    :mod:`shapegen.quad.lift` turns the points into positions in R^3.
    ``faces`` is a list of vertex loops -- all of length 4 except where a
    cone's developed image is a cell centre rather than a lattice point (see
    the module docstring).

    Diagnostics: ``euler`` must be 2 and ``coverage`` 1.0 for the mesh to be
    a closed single cover of the sphere. ``merge_residual`` is the trace's
    accuracy -- the worst disagreement between independent arrivals at one
    vertex, as a fraction of the local cell. ``frame_residual`` checks every
    re-arrival differed from the stored frame by a power of ``i`` (a wrong
    merge would not); it is skipped, and ``frames`` is meaningless, at cone
    vertices, where grid directions are ``2pi/(4 -+ 1)`` apart by definition.
    Face connectivity uses the traced tangents, not ``frames``.
    """
    points: np.ndarray            # (n,3) unit vectors
    frames: np.ndarray            # (n,3) unit tangent = developed direction +1
    edges: np.ndarray             # (m,2)
    faces: list                   # list of vertex loops
    cone_vertices: np.ndarray     # vertices sitting on a cone
    cone_of: np.ndarray           # which cone, or -1, per vertex
    cell: float
    euler: int
    coverage: float               # covered solid angle / 4pi
    merge_residual: float
    frame_residual: float
    refused: int
    converged: bool

    @property
    def valences(self) -> np.ndarray:
        v = np.zeros(len(self.points), dtype=int)
        np.add.at(v, self.edges.ravel(), 1)
        return v

    @property
    def face_degrees(self) -> np.ndarray:
        return np.array([len(f) for f in self.faces], dtype=int)


def _tangent_angles(X: np.ndarray, ref: np.ndarray, vecs: np.ndarray):
    """Angles of ``vecs`` about the outward normal ``X``, from the reference
    tangent ``ref``. ``(ref, X x ref, X)`` is right-handed, so the angle
    increases counter-clockwise seen from outside the sphere -- the same
    sense at every vertex, which is all the rotation system needs."""
    e2 = np.cross(X, ref)
    return np.arctan2(np.atleast_2d(vecs) @ e2, np.atleast_2d(vecs) @ ref)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


class _Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def join(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def quad_grid(cones, idx, s: float, anchor: int = 0, merge_frac: float = 0.25,
              max_vertices: int = 200_000, near_frac: float = 0.10,
              abs_cap: float = 0.02) -> QuadGrid:
    """Trace the whole grid and assemble it into a closed mesh.

    ``cones``, ``idx`` and ``s`` are
    :func:`shapegen.quad.periods.solve_lattice`'s output: chart positions,
    indices, cell size. The trace is a wavefront BFS -- every frontier vertex
    steps in all four developed directions at once, arrivals are matched
    against the vertices already found and against each other, and the
    unmatched ones become the next frontier. It stops when nothing new is
    reached, which on a correctly solved configuration is when the grid has
    closed over the whole sphere.

    ``merge_frac`` (fraction of the step chord) is not a tuned tolerance:
    distinct grid vertices are a full cell apart while independent arrivals
    at one vertex agree far more closely (``merge_residual`` reports the
    worst), so a wide range of values gives the identical mesh.

    Vertices that land on a cone are snapped to it (the cone position is
    exact; RK4 against the ``t^(4/3)`` endpoint is not) and not expanded --
    the BFS still reaches their neighbours through the surrounding faces.

    ``near_frac`` and ``abs_cap`` set accuracy, not topology: the integrator
    is 4th order in both, and ``euler``, valences and face degrees are
    unchanged across every setting tried. At the defaults ``merge_residual``
    on the cube fixture is ~1e-8 of a cell; halving both costs ~2x time for
    ~16x accuracy. Stops early with ``converged=False`` at ``max_vertices``.
    """
    cones = np.asarray(cones, dtype=complex)
    idx = np.asarray(idx, dtype=float)
    a = -idx
    s = float(s)
    cone_X = from_chart(cones, 0)

    zeta, c, arg = seed_vertex(cones, idx, s, anchor=anchor)
    V_zeta = [complex(zeta[0])]
    V_c = [np.int8(c[0])]
    V_arg = [float(arg[0])]
    V_X = [from_chart(zeta, c)[0]]
    V_fr = [_unit(chart_tangent(zeta, c, np.exp(-1j * arg)))[0]]
    V_cone = [-1]

    edges: list[tuple[int, int]] = []
    dart_tan: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()

    merge_res = 0.0
    frame_res = 0.0
    refused = 0
    converged = True
    frontier = np.array([0])

    while len(frontier):
        src = np.repeat(frontier, 4)
        z0 = np.array([V_zeta[i] for i in src])
        c0 = np.array([V_c[i] for i in src], dtype=np.int8)
        a0 = np.array([V_arg[i] for i in src])
        # The four developed directions are the same complex numbers at every
        # vertex: each vertex's own carried arg W already absorbs whatever
        # power of i the holonomy handed it, and {1,i,-1,-i} is closed under
        # that. There is no frame to transport by hand.
        d = np.tile(_DIRS, len(frontier))

        z1, c1, a1, _it, done = trace_step(z0, c0, a0, d, cones, a, s,
                                           near_frac=near_frac, abs_cap=abs_cap)
        X0 = from_chart(z0, c0)
        X1 = from_chart(z1, c1)
        chord = np.linalg.norm(X1 - X0, axis=1)
        ok = done & np.isfinite(chord) & (chord > 1e-14)
        refused += int((~ok).sum())

        t0 = _unit(chart_tangent(z0, c0, d * np.exp(-1j * a0)))
        t1 = _unit(chart_tangent(z1, c1, d * np.exp(-1j * a1)))

        # snap arrivals that landed on a cone: the cone position is exact,
        # and the RK4 endpoint there is the one that is not
        dc = np.linalg.norm(X1[:, None, :] - cone_X[None, :, :], axis=2)
        kc = np.argmin(dc, axis=1)
        on_cone = ok & (dc[np.arange(len(kc)), kc] < merge_frac * chord)
        X1[on_cone] = cone_X[kc[on_cone]]

        live = np.nonzero(ok)[0]
        tol = merge_frac * chord
        tgt = np.full(len(ok), -1, dtype=int)

        # (a) match against the vertices already found
        tree = cKDTree(np.asarray(V_X))
        dist, j = tree.query(X1[live])
        hit = dist < tol[live]
        tgt[live[hit]] = j[hit]
        if hit.any():
            merge_res = max(merge_res, float((dist[hit] / tol[live][hit]).max()
                                             * merge_frac))

        # (b) cluster the rest among themselves -- in one wave, up to four
        # frontier vertices can reach the same new vertex at once
        rest = live[~hit]
        fresh: list[int] = []
        if len(rest):
            uf = _Union(len(rest))
            kt = cKDTree(X1[rest])
            kk = min(5, len(rest))
            dd, jj = kt.query(X1[rest], k=kk)
            dd, jj = np.atleast_2d(dd), np.atleast_2d(jj)
            for p in range(len(rest)):
                for q in range(1, kk):
                    o = int(jj[p, q])
                    if dd[p, q] < min(tol[rest[p]], tol[rest[o]]):
                        uf.join(p, o)
                        merge_res = max(merge_res,
                                        float(dd[p, q]) / float(tol[rest[p]])
                                        * merge_frac)
            reps: dict[int, int] = {}
            for p in range(len(rest)):
                root = uf.find(p)
                if root not in reps:
                    if len(V_X) >= max_vertices:
                        converged = False
                        break
                    t = rest[root]
                    V_zeta.append(complex(z1[t]))
                    V_c.append(np.int8(c1[t]))
                    V_arg.append(float(a1[t]))
                    V_X.append(X1[t])
                    V_fr.append(_unit(chart_tangent(
                        z1[t:t + 1], c1[t:t + 1],
                        np.exp(-1j * a1[t:t + 1])))[0])
                    V_cone.append(int(kc[t]) if on_cone[t] else -1)
                    reps[root] = len(V_X) - 1
                    if not on_cone[t]:
                        fresh.append(len(V_X) - 1)
                tgt[rest[p]] = reps[root]

        # (c) every matched arrival must differ from the stored frame by a
        # power of i -- a wrong merge would not, so this is a free check
        for t in live:
            v = int(tgt[t])
            if v < 0:
                continue
            u = int(src[t])
            if V_cone[v] < 0:
                # At a regular vertex the four grid directions are 90 deg
                # apart, so an arrival that is genuinely the same vertex must
                # differ from the stored frame by a power of i and a wrong
                # merge will not. At a cone they are 2pi/(4+-1) apart -- the
                # whole point of a cone -- so the same test there would only
                # measure the cone's index.
                ang = float(_tangent_angles(V_X[v], V_fr[v], -t1[t])[0])
                frame_res = max(frame_res,
                                abs(ang - (np.pi / 2) * round(ang / (np.pi / 2))))
            key = (min(u, v), max(u, v))
            if u != v and key not in seen:
                seen.add(key)
                edges.append((u, v))
                dart_tan.append((t0[t], -t1[t]))
        if not converged:
            break
        # every vertex is created exactly once, so the fresh non-cone ones are
        # precisely those never expanded -- no visited set is needed
        frontier = np.array(sorted(fresh), dtype=int)

    points = np.asarray(V_X)
    frames = np.asarray(V_fr)
    E = np.asarray(edges, dtype=int).reshape(-1, 2)
    faces = faces_from_rotation(points, E, dart_tan)
    cone_of = np.asarray(V_cone, dtype=int)

    euler = len(points) - len(E) + len(faces)
    cov = sum(spherical_area(points[f]) for f in faces) / (4.0 * np.pi)
    return QuadGrid(points=points, frames=frames, edges=E, faces=faces,
                    cone_vertices=np.nonzero(cone_of >= 0)[0], cone_of=cone_of,
                    cell=s, euler=euler, coverage=float(cov),
                    merge_residual=float(merge_res),
                    frame_residual=float(frame_res), refused=int(refused),
                    converged=converged)


# --------------------------------------------------------------------------- #
# Rotation system -> faces                                                    #
# --------------------------------------------------------------------------- #

def faces_from_rotation(points: np.ndarray, edges: np.ndarray, dart_tan):
    """Faces as the orbits of ``phi = sigma . alpha`` on darts.

    ``alpha`` flips a dart and ``sigma`` takes the next dart
    counter-clockwise around its own vertex, so ``phi`` turns one corner and
    its orbits are exactly the faces of the embedded graph. Every dart lies
    in precisely one orbit, so the result is a closed surface however good or
    bad the geometry is, and ``V - E + F`` measures the trace.

    The counter-clockwise order comes from the traced tangents (``dart_tan``
    holds each edge's outgoing tangent at both endpoints) about the outward
    normal, which at a unit-sphere point is the point itself.
    """
    n = len(points)
    if len(edges) == 0:
        return []
    nd = 2 * len(edges)
    src = np.empty(nd, dtype=int)
    tan = np.empty((nd, 3))
    for e, (u, v) in enumerate(edges):
        src[2 * e], src[2 * e + 1] = u, v
        tan[2 * e], tan[2 * e + 1] = dart_tan[e][0], dart_tan[e][1]

    buckets: list[list[int]] = [[] for _ in range(n)]
    for dd in range(nd):
        buckets[src[dd]].append(dd)
    nxt = np.arange(nd)
    for v in range(n):
        ds = buckets[v]
        if not ds:
            continue
        ang = _tangent_angles(points[v], _unit(tan[ds[0]]), tan[ds])
        ds = [ds[i] for i in np.argsort(ang)]
        for i, dd in enumerate(ds):
            nxt[dd] = ds[(i + 1) % len(ds)]

    seen = np.zeros(nd, dtype=bool)
    faces = []
    for d0 in range(nd):
        if seen[d0]:
            continue
        loop = []
        dd = d0
        while not seen[dd]:
            seen[dd] = True
            loop.append(int(src[dd]))
            dd = int(nxt[dd ^ 1])
            if len(loop) > nd:
                break
        # phi turns *clockwise* about the outward normal, so the orbit comes
        # out inward-oriented; reverse it so every face winds counter-clockwise
        # seen from outside and the solid angles in `spherical_area` add up to
        # +4pi rather than -4pi
        faces.append(loop[::-1])
    return faces


# --------------------------------------------------------------------------- #
# Quality                                                                     #
# --------------------------------------------------------------------------- #

def spherical_area(P: np.ndarray) -> float:
    """Solid angle of the spherical polygon through the unit vectors ``P``,
    by fanning from its first vertex and summing signed spherical excesses.
    Over a closed mesh the total is ``4*pi`` exactly when the mesh covers the
    sphere once, which is what ``coverage`` reports."""
    tot = 0.0
    for i in range(1, len(P) - 1):
        A, B, C = P[0], P[i], P[i + 1]
        num = float(A @ np.cross(B, C))
        den = 1.0 + float(A @ B) + float(B @ C) + float(C @ A)
        tot += 2.0 * np.arctan2(num, den)
    return tot


def quad_areas(grid: QuadGrid) -> np.ndarray:
    """Solid angle of every face."""
    return np.array([spherical_area(grid.points[f]) for f in grid.faces])


def developed_cell_area(grid: QuadGrid, cones, idx) -> np.ndarray:
    """Each face's solid angle divided by what the conformal factor predicts
    for one ``s x s`` developed cell at its centre.

    In the developed plane every cell is exactly ``s x s`` -- that is what
    the trace constructs, so measuring it proves nothing. The independent
    statement is that the map is *conformal*: pulled back to the sphere, a
    cell of developed area ``s^2`` must occupy solid angle
    ``s^2 . (2/(1+|z|^2))^2 / |W(z)|^2``. This ratio is 1 where the trace and
    the conformal factor agree, so it measures the trace's accuracy against
    the continuum rather than against itself (median 1.000000 on the cube
    fixture). Its *spread* is the inherent area variation of a conformal quad
    mesh (63x interdecile on the P0 corpus).
    """
    cones = np.asarray(cones, dtype=complex)
    a = -np.asarray(idx, dtype=float)
    A = quad_areas(grid)
    out = np.empty(len(A))
    for i, f in enumerate(grid.faces):
        X = _unit(grid.points[f].mean(axis=0)[None, :])
        c = chart_of(X)
        z = to_chart(X, c)
        lm = form_log_abs(z, c, cones, a)[0]
        conf = (2.0 / (1.0 + abs(z[0]) ** 2)) ** 2 * np.exp(-2.0 * lm)
        n = len(f)
        # a cell centred on a cone occupies only n/4 of a full square
        out[i] = abs(A[i]) / (grid.cell ** 2 * conf * n / 4.0)
    return out


def developed_area(cones, idx, n_samples: int = 200_000) -> float:
    """Total area of the flat structure, ``integral |W|^2`` over the sphere.

    The trace lays down cells of developed area ``s^2``, so this divided by
    ``s^2`` is the face count -- known *before* tracing anything, which is
    what makes :func:`cell_for_faces` possible.

    Each point is evaluated in its own chart, where
    ``dA_flat / dA_sphere = |W|^2 (1+|zeta|^2)^2 / 4``; that ratio is
    chart-independent because both area elements carry the same Jacobian.
    Sampling is a deterministic Fibonacci lattice, not random: the integrand
    has ``r^(-1/2)`` spikes at the cones, which are integrable and of finite
    variance, but a seed-dependent answer here would make the mesh size
    seed-dependent too.
    """
    cones = np.asarray(cones, dtype=complex)
    a = -np.asarray(idx, dtype=float)
    k = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * k / n_samples
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)

    c = chart_of(X)
    zc = to_chart(X, c)
    lm = form_log_abs(zc, c, cones, a)
    dens = np.exp(2.0 * lm) * (1.0 + np.abs(zc) ** 2) ** 2 / 4.0
    return float(4.0 * np.pi * dens.mean())


def cell_for_faces(cones, idx, n_faces: int, n_samples: int = 200_000) -> float:
    """The cell size whose grid has about ``n_faces`` faces --
    ``sqrt(developed_area / n_faces)``.

    **Trap: the result is a request, not a usable cell size -- do not pass it
    straight to :func:`quad_grid`.** Only cell sizes the period solve accepts
    make the grid close (on the cube, only ``edge/m`` for integer ``m``; 200
    faces asks for ``edge/5.77``, which traces to Euler characteristic
    ``-5430``). Pass it to :func:`~shapegen.quad.periods.solve_lattice`,
    which moves the cones so that it closes, or reports that it cannot.

    The prediction is good to a few percent (cone faces cover 3/4 or 5/4 of
    a cell; the integrand has ``r^(-1/2)`` spikes); with ``--faces 20000``
    in the demo, results land within 0.4% of target.
    """
    return float(np.sqrt(developed_area(cones, idx, n_samples) / max(n_faces, 1)))


def quad_isotropy(points: np.ndarray, faces) -> np.ndarray:
    """Per-face longest/shortest edge ratio.

    The developing map is conformal, so a cell that is a perfect square in
    the developed plane comes back a near-square *on the sphere* too, up to
    the second-order variation of the conformal factor across one cell --
    an accuracy measure, unlike developed edge lengths (``s`` by
    construction)."""
    out = []
    for f in faces:
        P = points[f]
        lens = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
        out.append(float(lens.max() / max(lens.min(), 1e-300)))
    return np.array(out)
