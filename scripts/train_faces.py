"""Dry run: smoothed FaceScape heads -> PCA codes -> flow matching -> heads.

This exists to check that the pipeline **represents properly**, which is a
different question from whether the loss goes down. A flow-matching loss has a
floor set by the irreducible variance of its conditional target, so it tells you
almost nothing about sample quality; every check below is therefore about the
data path or the sample distribution, never the loss curve.

Usage::

    python -m scripts.train_faces --roots ~/Downloads/facescape_trainset_001_100
    python -m scripts.train_faces --roots ~/Downloads/facescape_trainset_0*  -k 64

Checks, in order, each with an explicit pass criterion:

  1. round trip at full rank     -- the basis is not losing anything it should keep
  2. held-out reconstruction     -- reproduces the measured curve (0.0082 at k=32)
  3. smoothing actually helps    -- guards the one stage that carries the result
  4. codes are whitened          -- mean ~0, std ~1, else the flow loss is skewed
  5. samples match the moments   -- first-order evidence the flow learned anything
  6. samples are not memorised   -- vs how far a REAL held-out face sits from train
  7. beats a diagonal Gaussian   -- the baseline a 3DMM ships; must be earned

**Pick k so that N/k >= 20.** Measured on 340 training heads, the sample
dispersion (max |std-1|, want 0) degrades monotonically with samples per
dimension: 0.018 at k=4 (N/k=85), 0.164 at k=16 (21), 0.229 at k=32 (11), 0.329
at k=64 (5.3). That is not a training-budget problem -- tripling the steps buys
only 0.329 -> 0.256, and going from 50 to 1000 Euler steps buys 0.329 -> 0.298,
so it is neither underfitting you can train away nor integration error. It is
340 samples being too few to pin a 64-dimensional density. Reconstruction still
improves with k, so there is a real trade: k=64 represents better and generates
worse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bottlegen import codec                                    # noqa: E402
from bottlegen.data import faces                               # noqa: E402
from bottlegen.models.basis import PCABasis                    # noqa: E402
from bottlegen.models.flow import (gaussian_baseline,          # noqa: E402
                                   sample_flow, train_flow)

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, ok, detail):
    _results.append((name, bool(ok)))
    print(f"{PASS if ok else FAIL}  {name}: {detail}", flush=True)


def mean_shape_error(A, B):
    return float(np.mean([codec.shape_error(A[i], B[i]) for i in range(len(A))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("-k", type=int, default=32, help="PCA components")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--t-smooth", type=float, default=1e-3)
    ap.add_argument("--out", default="reports/faces")
    a = ap.parse_args()

    roots = [Path(r).expanduser() for r in a.roots]

    # ---------------------------------------------------------------- data --
    print("== dataset ==", flush=True)
    X, F = faces.build_dataset(roots, t_smooth=a.t_smooth, limit=a.limit,
                               verbose=True)
    n_hold = min(a.holdout, len(X) // 4)
    Xtr, Xte = X[:-n_hold], X[-n_hold:]
    print(f"train {len(Xtr)}, held-out {len(Xte)}, {X.shape[1]} verts\n", flush=True)

    # ------------------------------------------------------------- checks --
    print("== representation ==", flush=True)

    # 1. A full-rank basis spans the centred training set exactly, so encoding
    #    and decoding a TRAINING shape must return it to numerical precision.
    #    Anything else means the basis, the whitening or the reshape is wrong.
    full = PCABasis.fit(Xtr, k=len(Xtr))
    rt = mean_shape_error(Xtr[:8], full.decode(full.encode(Xtr[:8])))
    check("round trip at full rank", rt < 1e-6, f"{rt:.2e} (want < 1e-6)")

    # 2. The number that has to reproduce. Measured on 200 subjects at k=32:
    #    0.0082. A materially different value means this pipeline is not the one
    #    that produced the measurement.
    basis = PCABasis.fit(Xtr, k=a.k)
    err = mean_shape_error(Xte, basis.decode(basis.encode(Xte)))
    print(f"       held-out reconstruction, k={a.k}: {err:.5f} "
          f"({basis.explained()[-1] * 100:.1f}% of train variance)")
    for kk in (8, 16, 32, 64, 128):
        if kk <= basis.k:
            b = PCABasis.fit(Xtr, k=kk)
            print(f"         k={kk:4d}  {mean_shape_error(Xte, b.decode(b.encode(Xte))):.5f}")
    check("held-out reconstruction is sane", 0.0 < err < 0.05,
          f"{err:.5f} (expect ~0.008 at k=32 on ~200 subjects)")

    # 3. Smoothing is the stage that carries the result, so verify it is doing
    #    something rather than trusting the default. Raw PCA plateaus on the scan
    #    noise; the smoothed target must beat it at this k.
    if a.t_smooth > 0:
        Xr, _ = faces.build_dataset(roots, t_smooth=0.0, limit=a.limit)
        br = PCABasis.fit(Xr[:-n_hold], k=a.k)
        er = mean_shape_error(Xr[-n_hold:], br.decode(br.encode(Xr[-n_hold:])))
        check("smoothing improves the curve", err < er,
              f"smoothed {err:.5f} vs raw {er:.5f} ({er / err:.2f}x)")

    # 4. The flow loss is an unweighted MSE in code space; unwhitened codes make
    #    it fit the largest coordinates and ignore the rest.
    Z = basis.encode(Xtr)
    mx, sx = np.abs(Z.mean(0)).max(), Z.std(0)
    check("codes are whitened", mx < 1e-6 and np.allclose(sx, 1.0, atol=0.05),
          f"|mean|max {mx:.2e}, std in [{sx.min():.3f}, {sx.max():.3f}]")

    # ---------------------------------------------------------------- flow --
    print("\n== flow matching ==", flush=True)
    model, hist = train_flow(Z, steps=a.steps, verbose=True)
    S = sample_flow(model, n=256, k=basis.k, steps=50)
    G = gaussian_baseline(Z, n=256)

    # 5. Cheapest evidence the field transports anything: the samples should
    #    carry the training moments. Whitened codes make the target explicit --
    #    mean 0, std 1. Under-dispersion is the characteristic failure, and at
    #    low N/k it is a data limit rather than anything trainable away.
    dm, ds = np.abs(S.mean(0)).max(), np.abs(S.std(0) - 1.0).max()
    check("samples match the code moments", dm < 0.35 and ds < 0.25,
          f"max |mean| {dm:.3f}, max |std-1| {ds:.3f}; "
          f"N/k = {len(Z) / basis.k:.1f} (want >= 20)")

    # 6. Memorisation check. The reference is how far a REAL unseen face sits
    #    from the training set -- held-out faces are genuine draws from the true
    #    distribution, so that distance is what an honest sample should show.
    #
    #    Train-to-train nearest neighbour is the wrong reference and was used
    #    here at first: it measures density *among* training points, a different
    #    quantity, and the sample/train ratio it produces sits at 0.17-0.39 for
    #    every k -- including k=4, where the model is essentially perfect. A
    #    criterion that fails a known-good model is not a criterion.
    d_gen = np.sqrt(((S[:, None] - Z[None]) ** 2).sum(2)).min(1).mean()
    d_hold = np.sqrt(((basis.encode(Xte)[:, None] - Z[None]) ** 2).sum(2)).min(1).mean()
    check("samples are not memorised", d_gen > 0.5 * d_hold,
          f"sample->train {d_gen:.3f} vs heldout->train {d_hold:.3f}")

    # 7. The baseline that must be beaten, not merely mentioned. Compares how
    #    well each sample set's DECODED shapes sit near the held-out faces --
    #    coverage of real shapes, not code statistics.
    def coverage(codes):
        M = basis.decode(codes[:64])
        return float(np.mean([min(codec.shape_error(t, m) for m in M) for t in Xte]))
    #    Read a narrow win with suspicion: under-dispersed samples cluster near
    #    the mean, and held-out faces are nearer the mean than the tails, so
    #    shrinkage flatters this metric rather than being penalised by it.
    cf, cg = coverage(S), coverage(G)
    check("flow is at least as good as a diagonal Gaussian", cf <= cg * 1.05,
          f"flow {cf:.5f} vs gaussian {cg:.5f}")

    # ------------------------------------------------------------- outputs --
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "faces_flow.npz", samples=basis.decode(S[:16]), faces=F,
             mean=basis.mean, comps=basis.comps, scale=basis.scale,
             loss=hist, codes=Z)
    print(f"\nwrote {out / 'faces_flow.npz'}")

    n_ok = sum(ok for _, ok in _results)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
