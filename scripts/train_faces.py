"""Dry run: smoothed FaceScape heads -> codes -> flow matching -> heads.

Runs against either representation. ``--basis rho`` (the default) is the one
``SH_FLOW_PLAN.md`` proposes: the model's output *is* a band-limited SH
coefficient vector of ρ, and decoding is the spin transform. ``--basis pca`` is
the position baseline, which on registered faces is measurably better, and it is
kept reachable so that comparison stays checked rather than remembered.

This exists to verify the pipeline **represents properly**, which is a different
question from whether the loss goes down. A flow-matching loss floor is set by
the irreducible variance of its conditional target, so it says almost nothing
about sample quality; every check below is about the data path or the sample
distribution, never the loss curve.

Usage::

    python -m scripts.train_faces --roots ~/Downloads/facescape_trainset_001_100
    python -m scripts.train_faces --roots ~/Downloads/facescape_* --basis pca -k 64

**Size the code so that N / d_eff >= 20.** Measured on 340 training heads, the
sample dispersion (``max |std-1|``, want 0) degrades monotonically with samples
per dimension: 0.018 at 85, 0.164 at 21, 0.229 at 11, 0.329 at 5.3. That is not a
training-budget problem — tripling the steps buys 0.329 -> 0.256 and 50 -> 1000
Euler steps buys 0.329 -> 0.298. It is too few samples to pin the density.

For ρ̂ the nominal ``(L+1)²`` is **not** that dimension: heads span a
low-dimensional identity manifold inside it, so the target is near-singular and
flow matching onto it is ill-posed rather than merely data-starved. Check 0
reports the participation ratio, which is the number that actually decides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bottlegen import codec                                    # noqa: E402
from bottlegen.data import faces                               # noqa: E402
from bottlegen.geom import dirac                               # noqa: E402
from bottlegen.geom.mesh import Mesh                           # noqa: E402
from bottlegen.models.basis import PCABasis, RhoBasis          # noqa: E402
from bottlegen.models.flow import (gaussian_baseline,          # noqa: E402
                                   sample_flow, train_flow)

_results = []


def check(name, ok, detail):
    _results.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def mean_shape_error(A, B):
    return float(np.mean([codec.shape_error(A[i], B[i]) for i in range(len(A))]))


def build_basis(kind, X, F, k, L, sphere=None, fields=None):
    """One switch, so ``--basis`` is a real choice rather than an edit.

    For ρ the sphere is fitted once and reused across the whole sweep: ρ̂ is
    defined *relative to* a base sphere, and at sample time there is no mesh to
    recover one from, so it must be fixed in advance and shared by every code.
    Refitting it per bandwidth would silently compare codes of different meanings.

    ``fields`` are cached per-vertex ρ. They do not depend on L, so passing them
    turns the bandwidth sweep from ``N x |L|`` edge-least-squares solves into
    ``N`` — the difference between minutes and hours on 26k-vertex heads.
    """
    if kind == "pca":
        return PCABasis.fit(X, k=k), f"PCA k={min(k, len(X) - 1)}", None
    if sphere is None:
        sphere = codec.base_sphere(Mesh(X.mean(0), F))
    return (RhoBasis.fit(X, F, L, sphere, method="direct", fields=fields),
            f"rho-hat L={L} ({codec.n_coeffs(L)} coeffs)", sphere)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--basis", choices=("rho", "pca"), default="rho")
    ap.add_argument("-k", type=int, default=32, help="PCA components")
    ap.add_argument("-L", type=int, default=16, help="rho-hat bandwidth")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--t-smooth", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="SH_FLOW_PLAN 4(b): variance floor on the target")
    ap.add_argument("--t-end", type=float, default=1.0,
                    help="SH_FLOW_PLAN 4(c): stop the ODE short of t=1")
    ap.add_argument("--cache", default=None,
                    help="npz from scripts/encode_faces.py; skips load+encode")
    ap.add_argument("--out", default="reports/faces")
    a = ap.parse_args()

    roots = [Path(r).expanduser() for r in a.roots]
    rho = a.basis == "rho"

    # ---------------------------------------------------------------- data --
    # cap and align are decided by the representation, not by taste: rho needs a
    # closed surface, and cannot see rotation at all (measured 4e-15), so
    # aligning for it would be pure cost.
    print(f"== dataset (basis={a.basis}, cap={rho}, align={not rho}) ==", flush=True)
    sph = fields = None
    if a.cache:
        d = np.load(a.cache)
        X, F, sph, fields = d["X"], d["faces"], d["sphere"], d["fields"]
        print(f"cache {a.cache}: {len(X)} heads, sphere and rho fields included",
              flush=True)
    else:
        out = faces.build_dataset(roots, t_smooth=a.t_smooth, limit=a.limit,
                                  cap=rho, align=not rho,
                                  base="icosphere" if rho else "native",
                                  verbose=True)
        X, F, sph = out if rho else (*out, None)
    n_hold = min(a.holdout, len(X) // 4)
    Xtr, Xte = X[:-n_hold], X[-n_hold:]
    f_tr = f_te = None
    if fields is not None:
        f_tr, f_te = fields[:-n_hold], fields[-n_hold:]
    print(f"train {len(Xtr)}, held-out {len(Xte)}, {X.shape[1]} verts\n", flush=True)

    print("== representation ==", flush=True)
    if rho and fields is None:
        # Fields BEFORE the first fit, not after: they do not depend on L, so
        # computing them up front means the whole bandwidth sweep costs one pass
        # over the data instead of one per bandwidth.
        if sph is None:
            sph = codec.base_sphere(Mesh(Xtr.mean(0), F))
        fields = RhoBasis(sph, F, L=1).rho_fields(X)
        f_tr, f_te = fields[:-n_hold], fields[-n_hold:]
        print(f"       rho fields {fields.shape} computed once for the sweep",
              flush=True)
    basis, label, sph = build_basis(a.basis, Xtr, F, a.k, a.L, sphere=sph,
                                    fields=f_tr)
    kdim = basis.k
    print(f"       {label}", flush=True)

    # 1. Basis-appropriate sanity. PCA at full rank spans the centred training
    #    set exactly; rho is not a linear basis, so the contract there is that it
    #    reproduces what the codec itself reports for the same encode.
    if rho:
        e_b = mean_shape_error(Xtr[:6],
                               basis.decode(basis.encode(Xtr[:6], fields=f_tr[:6])))
        e_c = float(np.mean([codec.encode(Mesh(x, F), a.L, sphere=basis.sphere,
                                          method="direct").err for x in Xtr[:6]]))
        check("basis agrees with the codec", abs(e_b - e_c) < 1e-6 * max(e_c, 1.0),
              f"basis {e_b:.5f} vs codec {e_c:.5f}")
    else:
        full = PCABasis.fit(Xtr, k=len(Xtr))
        rt = mean_shape_error(Xtr[:6], full.decode(full.encode(Xtr[:6])))
        check("round trip at full rank", rt < 1e-6, f"{rt:.2e} (want < 1e-6)")

    # 2. Held-out reconstruction: the ceiling the generative model inherits and
    #    cannot beat, whatever the flow does afterwards.
    # A rho decode is an eigensolve on a 4n x 4n matrix -- 5.6 s per 26k-vertex
    # head -- so the reconstruction checks run on a subset. PCA decodes are a
    # matrix product and can use everything.
    nd = 12 if rho else len(Xte)
    enc_te = ((lambda b: b.encode(Xte[:nd], fields=f_te[:nd])) if rho
              else (lambda b: b.encode(Xte)))
    err = mean_shape_error(Xte[:nd], basis.decode(enc_te(basis)))
    print(f"       held-out reconstruction: {err:.5f}"
          f"{f'  (over {nd} of {len(Xte)} held-out)' if rho else ''}")
    for s in ((4, 8, 12, 16) if rho else (8, 16, 32, 64)):
        if not rho and s > len(Xtr) - 1:
            continue
        b, _, _ = build_basis(a.basis, Xtr, F, s, s, sphere=sph, fields=f_tr)
        print(f"         {'L' if rho else 'k'}={s:4d}  "
              f"{mean_shape_error(Xte[:nd], b.decode(enc_te(b))):.5f}", flush=True)
    check("held-out reconstruction is sane", 0.0 < err < 0.3, f"{err:.5f}")

    # 3. Smoothing is the stage that carries the result -- raw PCA plateaus on
    #    the scan noise -- so verify it rather than trusting the default.
    if a.t_smooth > 0:
        ro = faces.build_dataset(roots, t_smooth=0.0, limit=a.limit, cap=rho,
                                 align=not rho,
                                 base="icosphere" if rho else "native")
        Xr = ro[0]
        fr = RhoBasis(sph, F, L=1).rho_fields(Xr) if rho else None
        br, _, _ = build_basis(a.basis, Xr[:-n_hold], F, a.k, a.L, sphere=sph,
                               fields=fr[:-n_hold] if rho else None)
        er = mean_shape_error(
            Xr[-n_hold:][:nd],
            br.decode(br.encode(Xr[-n_hold:][:nd],
                                fields=fr[-n_hold:][:nd] if rho else None)))
        check("smoothing improves the curve", err < er,
              f"smoothed {err:.5f} vs raw {er:.5f} ({er / err:.2f}x)")

    # 4. The flow loss is an unweighted MSE in code space; unwhitened codes make
    #    it fit the largest coordinates and ignore the rest. rho is whitened
    #    band-wise, so its per-coefficient std is near 1 rather than exactly 1.
    Z = basis.encode(Xtr, fields=f_tr) if rho else basis.encode(Xtr)
    mx, sx = np.abs(Z.mean(0)).max(), Z.std(0)
    if rho:
        # RhoBasis whitens BAND-wise, pooling the 2l+1 coefficients of each band,
        # because per-coefficient std is noisy when N is not >> K. So unit std per
        # *coefficient* is the wrong thing to assert -- within a band the spread
        # is real signal (measured 0.04 to 1.25). The invariant is that each band
        # pools to 1.
        bi = codec.band_index(basis.L)
        per_band = np.array([Z[:, bi == l].std() for l in range(basis.L + 1)])
        ok = mx < 1e-6 and np.abs(per_band - 1.0).max() < 0.05
        detail = (f"|mean|max {mx:.2e}, per-band std in "
                  f"[{per_band.min():.3f}, {per_band.max():.3f}], "
                  f"per-coefficient [{sx.min():.3f}, {sx.max():.3f}]")
    else:
        ok = mx < 1e-6 and sx.max() < 1.05 and sx.min() > 0.95
        detail = f"|mean|max {mx:.2e}, std in [{sx.min():.3f}, {sx.max():.3f}]"
    check("codes are whitened", ok, detail)

    # 0. THE contingent check, reported here because it needs the codes. The
    #    nominal code size is NOT the dimension the flow must learn: the heads
    #    span a low-dimensional identity manifold inside it, so the target is
    #    near-singular. The participation ratio is the honest number, and
    #    N / d_eff is what check 5 should be read against.
    sv = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    v = sv ** 2 / (sv ** 2).sum()
    d_eff = float(1.0 / (v ** 2).sum())
    d99 = int(np.searchsorted(np.cumsum(v), 0.99) + 1)
    print(f"       effective dimension: participation ratio {d_eff:.1f}, "
          f"{d99} directions hold 99% of {kdim} nominal; "
          f"N/d_eff = {len(Z) / d_eff:.1f}", flush=True)

    # 8. rho-specific: the gauge everything assumed was broken.
    if rho:
        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        z0, z1 = basis.encode(Xte[:3]), basis.encode(Xte[:3] @ Q)
        rel = np.linalg.norm(z0 - z1) / max(np.linalg.norm(z0), 1e-30)
        check("rho-hat is rotation-invariant", rel < 1e-6,
              f"{rel:.2e} under a random rotation")

    # ---------------------------------------------------------------- flow --
    print(f"\n== flow matching (sigma={a.sigma}, t_end={a.t_end}) ==", flush=True)
    model, hist = train_flow(Z, steps=a.steps, sigma=a.sigma, verbose=True)
    S = sample_flow(model, n=256, k=kdim, steps=50, t_end=a.t_end)
    G = gaussian_baseline(Z, n=256)

    # 5. Under-dispersion is the characteristic failure; read it against
    #    N / d_eff, not against the nominal code size.
    dm, ds = np.abs(S.mean(0)).max(), np.abs(S.std(0) - Z.std(0)).max()
    check("samples match the code moments", dm < 0.35 and ds < 0.25,
          f"max |mean| {dm:.3f}, max |std gap| {ds:.3f}; "
          f"N/d_eff = {len(Z) / d_eff:.1f} (want >= 20)")

    # 6. Memorisation, as a TAIL statistic rather than a mean.
    #
    #    Comparing mean nearest-neighbour distances does not isolate memorisation:
    #    under-dispersed samples cluster toward the mean, where training points are
    #    denser, so they score "closer to train" for a reason that has nothing to
    #    do with copying. Measured, checks 5 and 6 fail and pass together, which is
    #    the giveaway that they are one measurement wearing two hats.
    #
    #    Memorisation is a tail event: a copied sample sits far closer to its
    #    nearest training point than any genuine draw does. So calibrate on real
    #    unseen heads and count how many generated samples fall below where almost
    #    none of them do.
    d_gen = np.sqrt(((S[:, None] - Z[None]) ** 2).sum(2)).min(1)
    d_hold = np.sqrt(((enc_te(basis)[:, None] - Z[None]) ** 2).sum(2)).min(1)
    thr = float(np.quantile(d_hold, 0.05))
    frac = float((d_gen < thr).mean())
    check("samples are not memorised", frac < 0.20,
          f"{frac * 100:.0f}% of samples are nearer the training set than 95% of "
          f"real held-out heads are (threshold {thr:.2f}); "
          f"means {d_gen.mean():.2f} vs {d_hold.mean():.2f}")

    # 7. Read a narrow win with suspicion: under-dispersed samples cluster near
    #    the mean, and held-out heads are nearer the mean than the tails, so
    #    shrinkage flatters this metric rather than being penalised by it.
    def coverage(codes):
        M = basis.decode(codes[:8 if rho else 32])
        return float(np.mean([min(codec.shape_error(t, m) for m in M)
                              for t in Xte[:nd]]))
    cf, cg = coverage(S), coverage(G)
    check("flow is at least as good as a diagonal Gaussian", cf <= cg * 1.05,
          f"flow {cf:.5f} vs gaussian {cg:.5f}")

    # 9/10. rho-specific. gamma falls out of the decode for free and is a genuine
    #       off-manifold score: no other representation here hands you one.
    if rho:
        gam = lambda C: np.array([dirac.spin_transform(basis.sphere, F,
                                                       basis.rho_field(c))[1]
                                  for c in C[:6]])
        g_tr, g_gen = gam(Z), gam(S)
        print(f"       gamma: train {g_tr.mean():.3e}, generated {g_gen.mean():.3e}")
        check("generated codes are as realisable as the data",
              g_gen.mean() < 5.0 * g_tr.mean(),
              f"{g_gen.mean():.3e} vs {g_tr.mean():.3e} (want < 5x)")
        M = basis.decode(S[:6])
        check("decoded samples are valid meshes",
              np.isfinite(M).all() and all(Mesh(m, F).genus == 0 for m in M),
              f"{len(M)} samples, all finite and genus 0")

    # ------------------------------------------------------------- outputs --
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"faces_flow_{a.basis}.npz", samples=basis.decode(S[:16]),
             faces=F, loss=hist, codes=Z, gen_codes=S)
    print(f"\nwrote {out / f'faces_flow_{a.basis}.npz'}")

    n_ok = sum(ok for _, ok in _results)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
