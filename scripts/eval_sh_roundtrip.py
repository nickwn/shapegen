"""Encode → SH → decode round trip as a function of SH bandwidth.

    mesh --cMCF--> sphere param --encode--> rho_hat (bandwidth L)
                                 --decode--> mesh'      (spin transform + Poisson)

Reports the relative Procrustes error of ``mesh'`` against ``mesh`` for each L,
for two encoders:

  baseline   rho from the lambda route (geom.spin.mesh_to_rho), projected to
             bandwidth L afterwards;
  optimised  rho_hat optimised *at* bandwidth L against decoded shape error,
             through the adjoint (geom.encode.encode_sh).

The mesh has to out-resolve the bandwidth or the comparison is meaningless: an
icosphere with n vertices resolves up to roughly l ~ pi/sqrt(4 pi / n), which is
~44 at subdiv 4 and ~22 at subdiv 3.  L=40 therefore needs subdiv >= 4.

Usage:
    python -m scripts.eval_sh_roundtrip --subdiv 4 --lmax 40
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bottlegen.data.dataset import demo_bottle_meshes
from bottlegen.geom import conformal as cf, encode as enc, spin


def resolvable_degree(n: int) -> float:
    """Rough angular Nyquist degree for ``n`` roughly-uniform samples on S²."""
    return np.pi / np.sqrt(4.0 * np.pi / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=40)
    ap.add_argument("--levels", type=str, default="")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmcf-iters", type=int, default=300)
    ap.add_argument("--out", type=str, default="reports/sh_roundtrip.json")
    args = ap.parse_args()

    levels = ([int(x) for x in args.levels.split(",")] if args.levels else
              [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40])
    levels = [L for L in levels if L <= args.lmax]

    mesh = demo_bottle_meshes(n=1, subdiv=args.subdiv, seed=args.seed)[0]
    n = len(mesh.vertices)
    print(f"mesh: n={n} vertices, {len(mesh.faces)} faces, "
          f"resolvable to l~{resolvable_degree(n):.0f}")
    if max(levels) > resolvable_degree(n):
        print(f"  WARNING: L={max(levels)} exceeds the mesh's angular resolution; "
              f"those rows measure the mesh, not the codec.")

    t0 = time.perf_counter()
    sph = cf.mobius_normalize(mesh, cf.spherical_conformal_map(
        mesh, iters=args.cmcf_iters))
    qc = spin.qc_distortion(sph, mesh.vertices, mesh.faces)
    print(f"cMCF param: {time.perf_counter() - t0:.1f}s, "
          f"quasi-conformal distortion {qc:.4f}")

    t0 = time.perf_counter()
    rho0 = spin.mesh_to_rho(mesh, sph=sph)[0]
    print(f"baseline rho (lambda route): {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    Y = enc.real_sh_basis(sph, max(levels))
    print(f"SH basis {Y.shape}: {time.perf_counter() - t0:.1f}s\n")

    print(f"{'L':>4} {'coeffs':>7} {'baseline':>10} {'optimised':>10} "
          f"{'gain':>7} {'start':>6} {'iters':>6} {'|g|':>9} {'sec':>7}")
    rows, c_prev = [], None
    for L in levels:
        K = enc.n_coeffs(L)
        Yl = Y[:, :K]

        c_base = enc.sh_project(rho0, Yl)
        Xb = enc.decode_sh(c_base, sph, mesh.faces, Y=Y)
        err_base = float(np.sqrt(enc.procrustes(mesh.vertices, Xb)[0]))

        # Two starts, keep whichever *ends* better.  Warm-starting alone is not
        # safe: a band that stalls in a bad basin hands that basin to every band
        # after it (measured: L=4 stalled at 0.0944 and L=6,8,10 inherited it,
        # while a cold start at L=10 reached 0.0065).  The starting objective
        # does not predict this -- the stalled point has the *lower* J -- so the
        # only reliable test is to run both.
        starts = [("cold", c_base)]
        if c_prev is not None:
            warm = np.zeros(K)
            warm[:len(c_prev)] = c_prev
            starts.append(("warm", warm))
        t0 = time.perf_counter()
        best = None
        for name, c_start in starts:
            o = enc.encode_sh(mesh, L, sph=sph, Y=Y, c0=c_start, rho0=rho0,
                              maxiter=args.maxiter)
            o["start"] = name
            if best is None or o["err"] < best["err"]:
                best = o
        out = best
        dt = time.perf_counter() - t0
        c_prev = out["coeffs"] if out["err"] < err_base else c_base

        print(f"{L:>4} {K:>7} {err_base:>10.4f} {out['err']:>10.4f} "
              f"{err_base / max(out['err'], 1e-12):>6.2f}x {out['start']:>6} "
              f"{out['niter']:>6} {out['gnorm']:>9.2e} {dt:>7.1f}")
        rows.append({"L": L, "coeffs": K, "baseline": err_base,
                     "optimised": out["err"], "init": out["err0"],
                     "niter": out["niter"], "nfev": out["nfev"], "sec": dt,
                     "gnorm": out["gnorm"], "status": out["status"],
                     "start": out["start"]})

    meta = {"n_vertices": n, "subdiv": args.subdiv, "seed": args.seed,
            "qc_distortion": qc, "maxiter": args.maxiter,
            "resolvable_degree": resolvable_degree(n), "rows": rows}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
