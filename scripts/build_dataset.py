"""Encode a shape dataset into rho SH coefficients, ready for generative training.

    meshes ──encode against ONE template sphere──▸ coeffs (N, (L+1)^2)

The shared template is the whole point. rho_hat is defined *relative to* a base
sphere -- ``rho = H e^u - 1`` and ``e^u`` belongs to the map, not the surface --
so codes encoded against per-shape sphere maps (``codec.base_sphere``) do not
share a meaning, and at generation time there is no mesh from which to recover a
sphere anyway. One fixed template sphere (``codec.template_sphere``) for
encoding, training and sampling alike; the demo meshes share its connectivity.

After encoding, the script runs the check that actually matters before training:
fit a Gaussian to the codes, sample from it, and decode. Those samples are not in
the dataset, which is exactly the regime a generative model operates in -- if they
fold, the latent is not safe to sample from and no amount of model quality fixes
it.

Usage:
    python -m scripts.build_dataset --n 16 --L 12 --subdiv 3 --out reports/dataset.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from shapegen import codec
from shapegen.data.dataset import demo_bottle_meshes
from shapegen.eval.metrics import flip_fraction


def validate_samples(coeffs, sphere, faces, n=12, seed=0, radii=None):
    """How far from the data can you sample before the decode stops being valid?

    Sampling is restricted to the span the data actually occupies. With N codes in
    K dimensions the data spans at most N-1 directions, and a diagonal Gaussian
    over all K would put variance in directions nothing constrained -- producing
    folds that say more about the probe than the latent. PCA to the real span,
    then walk outward.

    The output is a *profile*, not a pass/fail: the decode has no obligation to
    return a valid surface for arbitrary rho_hat, so what matters is where the
    safe region ends, and whether a trained model will stay inside it.
    """
    N, K = coeffs.shape
    mu = coeffs.mean(0)
    U, S, Vt = np.linalg.svd(coeffs - mu, full_matrices=False)
    r = min(N - 1, K)
    rng = np.random.default_rng(seed)

    rows = []
    for rad in (radii or (0.5, 1.0, 1.5, 2.0, 3.0)):
        flips = []
        for _ in range(n):
            z = rng.standard_normal(r) * (S[:r] / np.sqrt(N)) * rad
            X = codec.decode(mu + z @ Vt[:r], sphere, faces)
            flips.append(flip_fraction(X, faces) if np.isfinite(X).all() else 1.0)
        flips = np.array(flips)
        rows.append({"radius": rad, "flip_mean": float(flips.mean()),
                     "flip_max": float(flips.max()),
                     "valid_frac": float((flips <= 0.02).mean())})
    return {"span": r, "singular_values": S[:min(5, len(S))].tolist(), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="number of shapes")
    ap.add_argument("--subdiv", type=int, default=3)
    ap.add_argument("--L", type=int, default=12, help="SH bandwidth of the latent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("quotient",), default="quotient",
                    help="encoder; the dataset shares one template sphere, which "
                         "only quotient encodes against")
    ap.add_argument("--out", type=str, default="reports/dataset.npz")
    args = ap.parse_args()

    meshes = demo_bottle_meshes(n=args.n, subdiv=args.subdiv, seed=args.seed)
    template = codec.template_sphere(args.subdiv)
    faces = meshes[0].faces
    K = codec.n_coeffs(args.L)
    print(f"{len(meshes)} shapes, {len(meshes[0].vertices)} verts, "
          f"L={args.L} -> {K} coefficients, method={args.method}\n")

    t0 = time.perf_counter()
    coeffs, results = codec.encode_dataset(meshes, args.L, method=args.method,
                                           sphere=template, n_jobs=-1, verbose=True)
    dt = time.perf_counter() - t0

    errs = np.array([r.err for r in results])
    print(f"\nencoded {args.n} shapes in {dt:.0f}s ({dt/args.n:.0f}s each)")
    print(f"  reconstruction err  mean {errs.mean():.4f}  max {errs.max():.4f}")
    print(f"  code norms          mean {np.linalg.norm(coeffs, axis=1).mean():.3f}"
          f"  spread {np.linalg.norm(coeffs, axis=1).std():.3f}")

    # Each draw costs a full decode, which scales with the vertex count, so size
    # the probe to the mesh: 60 decodes is nothing on a small icosphere and
    # minutes on a face.
    big = len(meshes[0].vertices) > 5000
    v = validate_samples(coeffs, template, faces,
                         n=4 if big else 12,
                         radii=(0.5, 1.0, 2.0) if big else None)
    print(f"\nsampling profile ({args.n} codes span {v['span']} dims; "
          f"12 draws per radius):")
    print(f"  {'radius':>7} {'flip mean':>10} {'flip max':>9} {'valid':>7}")
    for row in v["rows"]:
        print(f"  {row['radius']:>7.1f} {row['flip_mean']:>10.4f} "
              f"{row['flip_max']:>9.4f} {row['valid_frac']:>7.0%}")
    safe = [r["radius"] for r in v["rows"] if r["valid_frac"] == 1.0]
    print(f"  -> fully valid out to {max(safe):.1f} sigma"
          if safe else "  -> WARNING: folds even at the innermost radius")
    print("     A trained model must stay inside this: validity-filter samples.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, coeffs=coeffs, template=template, faces=faces,
                        errors=errs, L=args.L, subdiv=args.subdiv, seed=args.seed)
    print(f"\nwrote {args.out}  (coeffs {coeffs.shape}, template, faces)")


if __name__ == "__main__":
    main()
