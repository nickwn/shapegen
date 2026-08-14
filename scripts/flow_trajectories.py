"""Build reverse-process training data: bottle -> sphere flow, then the rho steps
that undo it.

The generative story this serves: conformal Willmore flow is a *forward* process
that destroys shape (bottle -> sphere) the way noise does in a diffusion model.
If we can express the reverse of each step as a change in rho on a FIXED base
sphere, then a network that predicts ``delta rho_hat`` from ``(rho_hat, t)``
generates bottles by starting at the sphere (rho_hat = 0) and integrating
backwards. This script produces the (state, target) pairs.

Pipeline, per bottle:

  1. flow          V_0 (bottle) -> ... -> V_T (round), recording every step,
                   in ``gradual`` mode so the steps are small and the
                   correspondence survives (flo's fast mode reaches a sphere in
                   3 steps but with 10x the quasi-conformal distortion).
  2. gauge fix     project V_T onto the exact unit sphere, then Mobius-normalise
                   against the ORIGINAL bottle's conformal density. This kills
                   the l=1 Mobius freedom, which the flow otherwise drifts along
                   freely -- Willmore energy is Mobius invariant, so nothing in
                   the flow pins it.
  3. encode        for every t, rho_t on the SAME base sphere S, band-limited to
                   L. One fixed sphere and one fixed SH basis for the whole
                   trajectory is what makes the latent space consistent; a
                   per-step parameterisation would not be learnable.
  4. refine        walk t backwards, optimising rho_hat_t through the adjoint so
                   ``decode(S, rho_hat_t)`` matches ``V_t``, warm-started from
                   ``rho_hat_{t+1} + (local band-limited difference)``.
  5. targets       ``d_t = rho_hat_{t-1} - rho_hat_t`` -- the change that undoes
                   one flow step. This is what a model learns to predict.

The trajectory is terminated at ``S`` itself, so ``rho_hat_T == 0`` exactly
(``decode(S, 0) == S`` is the machine-precision identity gate). It then runs from
a zero latent at the sphere out to the bottle -- the same shape as a diffusion
trajectory, with t playing the role of noise level. Note the final step
``V_{T-1} -> S`` absorbs the Mobius snap and so is larger than a flow step; its
``|delta|`` is reported separately and it can be dropped if that outlier hurts
training.

Output is an .npz per bottle with S, F, the coefficient path, the deltas, and
the per-step reconstruction error.

Usage:
    python -m scripts.flow_trajectories --n 2 --subdiv 3 --L 12 --steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from bottlegen.data.dataset import demo_bottle_meshes
from bottlegen.geom import conformal as cf, encode as enc, spin
from bottlegen.geom.mesh import Mesh
from scripts.willmore_flow import flow_trajectory, sphere_fit_residual


def gauge_fixed_sphere(traj, mesh) -> np.ndarray:
    """Endpoint of the flow, projected to the unit sphere and Mobius-normalised.

    Two separate fixes, both needed:

    * the flow plateaus ~5e-3 short of a sphere and cannot close that gap on its
      own (it fairs a surface, it has no target to project onto), so we project;
    * Willmore energy is Mobius invariant, so the flow wanders through the
      3-parameter family of ball automorphisms without any energy penalty. Left
      alone, two bottles would land in arbitrarily different gauges and their
      latents would not be comparable. ``mobius_normalize`` weights the conformal
      barycenter by the ORIGINAL bottle's areas, which makes the fixed point
      canonical for that shape.
    """
    S = traj[-1] - traj[-1].mean(0)
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    return cf.mobius_normalize(mesh, S)


def local_coeffs(traj, S, F, Yl):
    """Band-limited lambda-route rho for every mesh in the trajectory.

    Cheap and closed-form (no optimisation). Used for the *initialiser* and for
    the difference that seeds each refinement -- not as the final answer, since
    the local encoder minimises Dirac consistency rather than shape error and
    sits ~2x off the achievable optimum.
    """
    out = []
    for V in traj:
        rho = spin.mesh_to_rho(Mesh(V, F), sph=S)[0]
        out.append(enc.sh_project(rho, Yl))
    return out


def refine_backwards(traj, S, F, Y, L, maxiter, verbose=False):
    """Optimise rho_hat_t for every t, walking from the sphere back to the bottle.

    Backwards because that is the direction the steps chain: each refinement is
    warm-started from the already-refined *next* step plus the band-limited local
    difference, which is the user-facing "change in rho" and an excellent guess
    (consecutive flow steps are small by construction). Forward order would have
    nothing to warm-start from at the bottle end, where the problem is hardest.
    """
    Yl = Y[:, :enc.n_coeffs(L)]
    loc = local_coeffs(traj, S, F, Yl)
    T = len(traj) - 1

    opt = [None] * (T + 1)
    errs = [None] * (T + 1)
    iters = [None] * (T + 1)

    for t in range(T, -1, -1):
        if t == T:
            # The final state IS S, so rho_hat_T is 0 up to round-off and the
            # local encoder already lands there.
            c0 = loc[t]
        else:
            # rho_hat_{t+1} + (local difference) -- the "band-limited rho
            # difference" start. Uses the *refined* next state plus a *local*
            # delta, so errors do not accumulate down the chain.
            c0 = opt[t + 1] + (loc[t] - loc[t + 1])
        o = enc.encode_sh(Mesh(traj[t], F), L, sph=S, Y=Y, c0=c0,
                          rho0=None, maxiter=maxiter)
        opt[t], errs[t], iters[t] = o["coeffs"], o["err"], o["niter"]
        if verbose:
            print(f"    t={t:>3}  err={o['err']:.5f}  iters={o['niter']:>4}",
                  flush=True)

    opt = np.array(opt)
    # d_t undoes the flow step that produced V_t: applied at state t it lands on
    # t-1, i.e. one step closer to the bottle. d[0] is unused (no step before 0).
    deltas = np.zeros_like(opt)
    deltas[1:] = opt[:-1] - opt[1:]

    # The generative use integrates the deltas from the sphere outwards, so the
    # path must telescope exactly: rho_0 == rho_T + sum(d). Cheap to assert, and
    # it is the one place an off-by-one in the backward walk would hide.
    assert np.allclose(opt[-1] + deltas[1:].sum(0), opt[0], atol=1e-10), \
        "delta path does not telescope back to the bottle state"
    return opt, deltas, np.array(errs), np.array(iters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="how many bottles")
    ap.add_argument("--subdiv", type=int, default=3)
    ap.add_argument("--L", type=int, default=12, help="SH bandwidth of the latent")
    ap.add_argument("--steps", type=int, default=20, help="flow steps per bottle")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--maxiter", type=int, default=200,
                    help="adjoint-optimiser cap per step. Do not cut this to save "
                         "time: under-converged states inject optimiser noise "
                         "straight into the deltas, which ARE the training "
                         "targets (measured: median |delta| 1.15 at maxiter=60 "
                         "vs 0.46 at 250, a 2.5x inflation of pure noise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/trajectories")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    meshes = demo_bottle_meshes(n=args.n, subdiv=args.subdiv, seed=args.seed)
    K = enc.n_coeffs(args.L)
    print(f"{args.n} bottles, subdiv={args.subdiv}, L={args.L} ({K} coeffs), "
          f"{args.steps} flow steps, tau={args.tau}\n")

    for i, mesh in enumerate(meshes):
        F = mesh.faces
        t0 = time.perf_counter()

        traj = flow_trajectory(mesh.vertices, F, tau=args.tau,
                               steps=args.steps, mode="gradual")
        S = gauge_fixed_sphere(traj, mesh)
        # Terminate the trajectory at S itself. Without this the last state is
        # the flow endpoint V_T, which is round but sits in a different Mobius
        # gauge than S, so rho_T encodes that reparameterisation and is NOT
        # small (measured 3.58 against 6.06 at the bottle end). Generation wants
        # a canonical starting latent, and decode(S, 0) == S exactly, so making
        # S the final state puts the trajectory's origin at rho = 0.
        traj = traj + [S]
        Y = enc.real_sh_basis(S, args.L)
        qc = spin.qc_distortion(S, mesh.vertices, F)
        t_flow = time.perf_counter() - t0

        opt, deltas, errs, iters = refine_backwards(
            traj, S, F, Y, args.L, args.maxiter, verbose=args.verbose)

        path = outdir / f"bottle_{i:03d}.npz"
        np.savez_compressed(
            path, sphere=S, faces=F, coeffs=opt, deltas=deltas,
            errors=errs, iters=iters,
            trajectory=np.array(traj), L=args.L, tau=args.tau,
            qc_distortion=qc, subdiv=args.subdiv, seed=args.seed)

        dn_all = np.linalg.norm(deltas[1:], axis=1)
        dn = dn_all[:-1]           # flow steps only
        dn_snap = dn_all[-1]       # the V_{T-1} -> S Mobius snap
        print(f"bottle {i}: flow {t_flow:.1f}s + refine "
              f"{time.perf_counter() - t0 - t_flow:.0f}s   "
              f"sphere resid={sphere_fit_residual(traj[-1]):.1e} qc={qc:.4f}")
        print(f"  reconstruction err  sphere-end {errs[-1]:.4f} -> "
              f"bottle-end {errs[0]:.4f}   (worst {errs.max():.4f}, "
              f"median {np.median(errs):.4f})")
        print(f"  latent norm         {np.linalg.norm(opt[-1]):.3f} -> "
              f"{np.linalg.norm(opt[0]):.3f}")
        print(f"  |delta| flow steps  min {dn.min():.4f}  median "
              f"{np.median(dn):.4f}  max {dn.max():.4f}")
        print(f"  |delta| mobius snap {dn_snap:.4f}  "
              f"({dn_snap / np.median(dn):.1f}x the median flow step)")
        print(f"  optimiser iters     median {int(np.median(iters))}, "
              f"max {iters.max()}    -> {path}\n")


if __name__ == "__main__":
    main()
