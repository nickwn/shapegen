"""Encode → SH → decode round trip as a function of SH bandwidth.

    mesh --cMCF--> sphere param --encode--> rho_hat (bandwidth L)
                                 --decode--> mesh'      (spin transform + Poisson)

Reports the relative Procrustes error of ``mesh'`` against ``mesh`` for each L,
for two encoders:

  baseline   rho from the direct encoder (geom.spin.mesh_to_rho_direct),
             projected to bandwidth L *afterwards*;
  optimised  rho_hat optimised *at* bandwidth L against decoded shape error,
             through the adjoint (geom.encode.encode_sh).

The distinction between "project afterwards" and "optimise at bandwidth" is the
whole experiment.  The decode is nonlinear, so

    argmin over the band-limited subspace  !=  P_L( argmin over all rho )

and truncating after the fact throws away accuracy in an uncontrolled way.  The
baseline curve flatlining while the optimised one keeps descending is that
inequality made visible.

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
    """Rough angular Nyquist degree for ``n`` roughly-uniform samples on S².

    ``n`` samples spread over 4*pi steradians sit about ``sqrt(4 pi / n)`` radians
    apart, and a degree-l harmonic has features of angular size ``pi / l``.  Set
    them equal.  This is a sanity bound, not a theorem -- an icosphere is not a
    uniform sampling -- but it is the right order and it is what decides whether
    a row of the table is measuring the codec or the mesh.
    """
    return np.pi / np.sqrt(4.0 * np.pi / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdiv", type=int, default=4,
                    help="icosphere subdivision of the test bottle; 4 = 2562 "
                         "vertices, the smallest that honestly carries L=40")
    ap.add_argument("--lmax", type=int, default=40)
    ap.add_argument("--levels", type=str, default="",
                    help="comma-separated bandwidths; default is a log-ish ladder")
    ap.add_argument("--maxiter", type=int, default=60,
                    help="L-BFGS iteration cap PER START (there are up to two)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmcf-iters", type=int, default=300)
    ap.add_argument("--out", type=str, default="reports/sh_roundtrip.json")
    args = ap.parse_args()

    # Denser at low L, where the curve actually moves; sparse above L=20, where
    # successive bands differ little and each row costs minutes.
    levels = ([int(x) for x in args.levels.split(",")] if args.levels else
              [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40])
    levels = [L for L in levels if L <= args.lmax]

    mesh = demo_bottle_meshes(n=1, subdiv=args.subdiv, seed=args.seed)[0]
    n = len(mesh.vertices)
    print(f"mesh: n={n} vertices, {len(mesh.faces)} faces, "
          f"resolvable to l~{resolvable_degree(n):.0f}")
    # Warn rather than refuse: over-resolved rows are still informative as an
    # upper bound, they just stop being a statement about the representation.
    if max(levels) > resolvable_degree(n):
        print(f"  WARNING: L={max(levels)} exceeds the mesh's angular resolution; "
              f"those rows measure the mesh, not the codec.")

    # ----------------------------------------------------------------- setup
    # Everything below is hoisted out of the L loop.  These three are all
    # bandwidth-independent, and the cMCF flow in particular costs more than a
    # short optimisation run, so recomputing it per band would dominate.
    t0 = time.perf_counter()
    sph = cf.mobius_normalize(mesh, cf.spherical_conformal_map(
        mesh, iters=args.cmcf_iters))
    # The quasi-conformal distortion of the parameterisation is the shear the
    # spin transform cannot represent, so it is a rough floor on the round trip.
    # Printed so a bad round trip can be attributed to the param rather than the
    # encoder; it is also drawn as a reference line on the plot.
    qc = spin.qc_distortion(sph, mesh.vertices, mesh.faces)
    print(f"cMCF param: {time.perf_counter() - t0:.1f}s, "
          f"quasi-conformal distortion {qc:.4f}")

    # The direct encoder serves double duty: it is the *baseline* curve, and
    # (projected to each bandwidth) the cold initialiser for the optimiser.
    t0 = time.perf_counter()
    rho0 = spin.mesh_to_rho_direct(mesh, sph)
    print(f"baseline rho (direct): {time.perf_counter() - t0:.1f}s")

    # Built once at the largest bandwidth and column-sliced for every smaller
    # one.  Legal because real_sh_basis orders columns by ascending degree, so
    # the first (L'+1)^2 columns *are* the bandwidth-L' basis -- asserted in
    # tests/test_encode.py::test_sh_basis_slices_by_bandwidth.  At L=40 this is
    # 1681 scipy calls, several seconds, so doing it per band would be silly.
    t0 = time.perf_counter()
    Y = enc.real_sh_basis(sph, max(levels))
    print(f"SH basis {Y.shape}: {time.perf_counter() - t0:.1f}s\n")

    # ------------------------------------------------------------- the sweep
    # Every band is encoded independently, from the lambda-route projection at
    # that bandwidth.  Bands do NOT seed each other.
    #
    # An earlier version also ran a "warm" start (the previous band's solution,
    # zero-padded) and kept whichever ended better.  That was introduced to
    # contain a real failure -- a warm chain stalled at L=4 and passed the bad
    # basin to L=6,8,10, where a cold start reached 0.0065 against the chain's
    # 0.0944.  Preconditioning the optimiser (encode.band_precond) removed the
    # stall at its source, and with it the reason for the second start: measured
    # across L=2..24, the two starts reach the same optimum to 0.00% wherever
    # the optimiser converges, cost the same in total (800 iterations either
    # way), and running cold alone costs 0.013% in the worst row.  It was pure
    # 2x overhead, so it is gone.
    #
    # Columns:
    #   baseline/optimised  relative Procrustes shape error (lower is better)
    #   gain                baseline/optimised
    #   iters, |g|          READ THESE.  A row at the iteration cap with a
    #                       gradient still well above zero is budget-limited,
    #                       not bandwidth-limited, and its error is an upper
    #                       bound rather than a property of the representation.
    print(f"{'L':>4} {'coeffs':>7} {'baseline':>10} {'optimised':>10} "
          f"{'gain':>7} {'iters':>6} {'|g|':>9} {'sec':>7}")
    rows = []
    for L in levels:
        K = enc.n_coeffs(L)
        Yl = Y[:, :K]

        # --- baseline: encode once in vertex space, THEN truncate ------------
        c_base = enc.sh_project(rho0, Yl)
        Xb = enc.decode_sh(c_base, sph, mesh.faces, Y=Y)
        err_base = float(np.sqrt(enc.procrustes(mesh.vertices, Xb)[0]))

        # --- optimised: same projection, but optimised AT this bandwidth -----
        t0 = time.perf_counter()
        out = enc.encode_sh(mesh, L, sph=sph, Y=Y, c0=c_base, rho0=rho0,
                            maxiter=args.maxiter)
        dt = time.perf_counter() - t0

        print(f"{L:>4} {K:>7} {err_base:>10.4f} {out['err']:>10.4f} "
              f"{err_base / max(out['err'], 1e-12):>6.2f}x {out['niter']:>6} "
              f"{out['gnorm']:>9.2e} {dt:>7.1f}")
        # `status` is the L-BFGS termination reason and is the machine-readable
        # form of the iters/|g| caveat above; plot_sh_roundtrip keys off it.
        rows.append({"L": L, "coeffs": K, "baseline": err_base,
                     "optimised": out["err"], "init": out["err0"],
                     "niter": out["niter"], "nfev": out["nfev"], "sec": dt,
                     "gnorm": out["gnorm"], "status": out["status"]})

    # Metadata travels with the rows so a stored result can be interpreted
    # later without re-deriving the mesh size or the resolution bound.
    meta = {"n_vertices": n, "subdiv": args.subdiv, "seed": args.seed,
            "qc_distortion": qc, "maxiter": args.maxiter,
            "resolvable_degree": resolvable_degree(n), "rows": rows}
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
