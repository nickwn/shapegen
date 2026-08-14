"""Conformal Willmore flow on the demo bottles (Crane–Pinkall–Schröder).

Flows a genus-0 mesh toward a round sphere by repeated spin transformation,
following the reference implementation in ``flo`` (Crane's "Robust fairing via
conformal curvature flow"). Each step is:

    H     = signed mean curvature of the CURRENT mesh
    proj  = -H with {1, n_x, n_y, n_z} projected out in the mass inner product
    rho   = -H + tau * proj
    V     = integrate( spin_transform(V, F, rho) )

Two things are worth understanding before reading the numbers.

*The base surface is the mesh itself, not a sphere.* That is the whole structural
difference from :mod:`bottlegen.geom.encode`, which spin-transforms a fixed base
sphere *into* the target. Here there is no correspondence and no parameterisation
-- the mesh is deformed in place -- which is why the flow needs no cMCF.

*The constraints are what stop it collapsing.* ``rho = -H`` alone drives mean
curvature toward zero, which for a closed surface means shrinking to nothing.
Projecting out the constant and the three normal components removes the scale and
translation modes, leaving a flow that rounds the shape instead. Area is
renormalised each step as well, since Willmore energy is scale-free and the
spin transform does not preserve scale.

**Willmore energy is not a usable convergence test here.** ``W/4pi -> 1`` only in
the continuous limit: an *exact* discrete sphere reads 0.98240 / 0.99556 / 0.99889
at subdiv 2 / 3 / 4, and a distorted mesh can read *lower* than that. This script
measures the least-squares sphere-fit residual instead, which is ~1e-16 on an
exact discrete sphere at every resolution.

Usage:
    python -m scripts.willmore_flow --n 4 --subdiv 3 --tau 0.3
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bottlegen.data.dataset import demo_bottle_meshes
from bottlegen.eval import metrics
from bottlegen.geom import dirac
from bottlegen.geom.curvature import mean_curvature
from bottlegen.geom.operators import cotan_laplacian, vertex_areas

FOURPI = 4.0 * np.pi


def sphere_fit_residual(V: np.ndarray) -> float:
    """Relative max deviation from the least-squares sphere through ``V``.

    Fitting the centre rather than assuming it matters: the flow is Möbius
    invariant, so vertices drift around the sphere even once the *surface* is
    round, and a radius-from-centroid measure would read that drift as
    non-sphericity. This separates "is the surface a sphere" from "are the
    vertices evenly spread on it".

    ``|V|² − 2c·V − k = 0`` is linear in ``(c, k)`` with ``k = r² − |c|²``.
    """
    A = np.column_stack([2.0 * V, np.ones(len(V))])
    sol, *_ = np.linalg.lstsq(A, (V ** 2).sum(1), rcond=None)
    c = sol[:3]
    r = np.sqrt(max(sol[3] + (c ** 2).sum(), 1e-30))
    return float(np.abs(np.linalg.norm(V - c, axis=1) - r).max() / r)


def _constraint_basis(V, n, m):
    """``{1, n_x, n_y, n_z}``, M-orthonormalised by Gram–Schmidt."""
    ip = lambda a, b: float((a * b * m).sum())
    U = []
    for j in range(4):
        v = (np.ones(len(V)) if j == 0 else n[:, j - 1]).copy()
        for u in U:
            v -= ip(v, u) * u
        U.append(v / np.sqrt(max(ip(v, v), 1e-300)))
    return U, ip


def willmore_step(V: np.ndarray, F: np.ndarray, tau: float,
                  mode: str = "flo") -> np.ndarray:
    """One forward-Euler step of conformal Willmore flow. Returns new positions.

    ``mode="flo"`` reproduces the reference: ``rho = -H + tau*proj``. The
    unprojected ``-H`` is what makes it reach a sphere in 3-4 steps.

    ``mode="gradual"`` drops it (``rho = tau*proj``), which is far slower to round
    but takes small, even steps. **Use this whenever the intermediate meshes
    matter rather than only the endpoint.** Measured on a bottle at subdiv 3, the
    quasi-conformal distortion of the resulting sphere against the original is
    0.017 for ``gradual`` versus 0.185 for ``flo`` — the fast mode reaches a
    round surface but wrecks the correspondence getting there, which makes its
    endpoint useless as a parameterisation.
    """
    L, M = cotan_laplacian(V, F)
    m = M.diagonal()
    H, _, n = mean_curvature(V, F, L, M)

    U, ip = _constraint_basis(V, n, m)
    proj = -H
    for u in U:
        proj -= ip(proj, u) * u

    drive = (-H + tau * proj) if mode == "flo" else (tau * proj)
    lam, _ = dirac.spin_transform(V, F, drive)
    Vn = dirac.integrate_edges(V, F, lam)
    return Vn / np.sqrt(vertex_areas(Vn, F).sum() / FOURPI)      # unit-area sphere


def flow_trajectory(V, F, tau=0.5, steps=30, mode="gradual"):
    """Run the flow and return every intermediate mesh, ``[V_0, ..., V_steps]``.

    ``V_0`` is the input, ``V_steps`` the roundest. Used to build reverse-process
    training data, where the whole path matters and not just the endpoint.
    """
    traj = [np.asarray(V, dtype=float).copy()]
    for _ in range(steps):
        traj.append(willmore_step(traj[-1], F, tau, mode=mode))
    return traj


def willmore_flow(V, F, tau=0.3, max_iter=40, tol=5e-3, patience=4, verbose=False):
    """Flow to a sphere. Returns ``(V_best, info)``.

    Stops on whichever comes first: the residual falls below ``tol``, or it fails
    to improve on its best for ``patience`` consecutive steps. The patience rule
    is not optional — the flow reaches its floor in a few iterations and then
    *degrades* (Möbius drift plus the per-step discretisation error of the edge
    rule accumulating), so a fixed iteration count would return a worse mesh than
    the flow actually found.
    """
    hist = [sphere_fit_residual(V)]
    best_V, best_r, best_it, stale = V.copy(), hist[0], 0, 0
    for it in range(1, max_iter + 1):
        V = willmore_step(V, F, tau)
        r = sphere_fit_residual(V)
        hist.append(r)
        if verbose:
            print(f"    iter {it:>3}  residual {r:.3e}")
        if r < best_r:
            best_V, best_r, best_it, stale = V.copy(), r, it, 0
        else:
            stale += 1
        if best_r < tol or stale >= patience:
            break
    return best_V, {"iters_run": it, "best_iter": best_it, "residual": best_r,
                    "history": hist,
                    # 'tol' means it got as round as asked; 'plateau' means it hit
                    # its own floor first. The floor is ~5e-3 at subdiv 3, so a
                    # tighter tol than that reports failure for a flow that did
                    # exactly what it always does.
                    "stop": "tol" if best_r < tol else "plateau"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="how many bottles")
    ap.add_argument("--subdiv", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.3,
                    help="forward-Euler step; 0.3 converges fastest here, "
                         "flo's default of 0.95 needs ~3x more steps")
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="sphere-fit residual accepted as 'round'. The flow's own "
                         "floor is ~5e-3, so anything tighter cannot be reached")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    meshes = demo_bottle_meshes(n=args.n, subdiv=args.subdiv, seed=args.seed)
    ref = icosphere_reference(args.subdiv)
    print(f"{args.n} bottles, subdiv={args.subdiv}, tau={args.tau}, tol={args.tol}")
    print(f"exact discrete sphere at this resolution: residual {ref['residual']:.1e}, "
          f"W/4pi {ref['w']:.5f}\n")

    print(f"{'#':>2} {'start':>9} {'residual':>9} {'best@':>6} {'ran':>4} "
          f"{'W/4pi':>8} {'minang':>7} {'flip':>6} {'sec':>6}  stop")
    iters, res = [], []
    for i, mesh in enumerate(meshes):
        F = mesh.faces
        t0 = time.perf_counter()
        V, info = willmore_flow(mesh.vertices.copy(), F, tau=args.tau,
                                max_iter=args.max_iter, tol=args.tol,
                                verbose=args.verbose)
        q = metrics.triangle_quality(V, F)
        print(f"{i:>2} {info['history'][0]:>9.2e} {info['residual']:>9.2e} "
              f"{info['best_iter']:>6} {info['iters_run']:>4} "
              f"{metrics.willmore_energy(V, F) / FOURPI:>8.5f} "
              f"{q['min_angle']:>7.1f} {metrics.flip_fraction(V, F):>6.3f} "
              f"{time.perf_counter() - t0:>6.1f}  {info['stop']}")
        iters.append(info["best_iter"])
        res.append(info["residual"])

    print(f"\nreached tol={args.tol}: {sum(r < args.tol for r in res)}/{len(res)}; "
          f"best iteration: median {int(np.median(iters))}, max {max(iters)}")
    print(f"residual: median {np.median(res):.2e}, worst {max(res):.2e} "
          f"(exact discrete sphere: {ref['residual']:.1e})")
    print("NOTE: the flow plateaus ~5e-3 and then degrades -- it fairs a surface, "
          "it does not build a parameterisation. For a base sphere you still want "
          "cMCF (residual ~3e-16) or a projection/Mobius-normalisation step.")


def icosphere_reference(subdiv: int) -> dict:
    """Metrics on an exact discrete sphere — the real target, not W/4pi = 1."""
    from bottlegen.geom.mesh import icosphere
    m = icosphere(subdiv)
    V = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    return {"residual": sphere_fit_residual(V),
            "w": metrics.willmore_energy(V, m.faces) / FOURPI}


if __name__ == "__main__":
    main()
