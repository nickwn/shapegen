"""One-time encode of a FaceScape set to cached ρ fields, for the SH flow model.

Encoding is the expensive stage and it never changes between training runs, so it
belongs in a cache rather than in the training loop. What gets cached is the
per-vertex **ρ field**, not the SH coefficients:

    ρ_direct does not depend on L. Only the projection does.

So one cache serves every bandwidth. The costly part is ``u_edge_lsq`` — an LSQR
over every edge of a 26 321-vertex head — plus the single cMCF that builds the
shared template sphere. Sweeping L from a cached field is then just a matrix
product per mesh.

The sphere is cached alongside, and that is not optional: ρ̂ is defined *relative
to* a base sphere, so a cache read back against a different sphere silently means
something else. It is stored in the same file and checked on load.

Usage::

    python -m scripts.encode_faces --roots ~/Downloads/facescape_trainset_0* \\
        --out reports/faces/rho_neutral.npz
    python -m scripts.encode_faces --roots ~/Downloads/facescape_* \\
        --expressions 1_neutral 2_smile 16_grin --out reports/faces/rho_multi.npz

Then::

    d = np.load(path)
    basis = RhoBasis.fit(d["X"], d["faces"], L, d["sphere"], fields=d["fields"])

**Choosing ``--remesh`` and L.**

Scored against the **original** FaceScape head, topology-free (chamfer), 6 heads,
``method="residual"`` at L=40 — see the warning below for why the reference
matters. Reported twice: over the whole head, and over the **face region** alone
(a 0.7-radius ball about the nose tip, 40% of vertices). Both are needed, because
most of a FaceScape head is bald cranium and neck and a whole-head mean is
dominated by them:

    remesh  smooth |  whole: target   decode |   face: target   decode
       2.5   1e-03 |        0.02327  0.02833 |        0.01338  0.01936
       0.8   1e-03 |        0.01731  0.02598 |        0.01119  0.01927   <- default
       0.8       0 |        0.01521  0.02716 |        0.00866  0.02325
       0.4   1e-03 |        0.01691  0.02614 |        0.01103  0.01952

* **0.8% is the right target, and 2.5% was costing real detail.** At 2.5% the
  remeshed mean head is only **1807 vertices** — 10 mm edges on a 404 mm bbox
  diagonal, wider than a nostril — so the composed transfer is piecewise-linear
  over ~3600 triangles no matter how fine the icosphere is. The target sits
  0.02327 from the original head against 0.01731 at 0.8%.
* **But almost none of that reaches the face.** Going 2.5% -> 0.8% improves the
  whole-head decode 8% and the face decode **0%** (0.01936 -> 0.01927). The gain
  is on the cranium and neck, where the coarse remesh was faceting an otherwise
  smooth surface. In the face the encoder, not the target, is the binding
  constraint — the error there is Gibbs ringing at the lip seam, eyelid creases
  and nostril, which no amount of target detail addresses.
* **0.4% buys nothing over 0.8%** and costs 4x the vertices.
* **Smoothing must stay on, but 1e-3 was ten times too much.** Swept at
  ``remesh=0.8``, L=32, ``residual``, 6 heads:

    t_smooth |  whole: target   decode  |  face: target   decode
       0     |       0.01521  0.02905   |      0.00866  0.02452
       1e-4  |       0.01582  0.02611   |      0.00923  0.01727   <- default
       3e-4  |       0.01618  0.02601   |      0.00973  0.01785
       1e-3  |       0.01731  0.02640   |      0.01119  0.01918
       3e-3  |       0.02010  0.02731   |      0.01472  0.02065
       1e-2  |       0.02563  0.02985   |      0.02121  0.02391

  Removing it entirely costs **29.6% on the face** and 10.1% whole-head. Going
  from the old 1e-3 to 1e-4 is worth **10.0% on the face** and 1.1% whole-head.
  Both metrics put the optimum in [1e-4, 3e-4]; 1e-4 wins the face by 3.3% and
  loses whole-head by 0.4%, so the face decides it.

  The interior optimum is a trade between two errors. The **target** degrades
  monotonically with t (0.00866 -> 0.02121 on the face) and is charged for it in
  full, since everything is scored against the raw scan. The **decode** improves
  up to 1e-4 because H is a second derivative and amplifies per-subject scan
  noise by ~l^2, which the encoder then spends bands chasing. Past 1e-4 the lost
  detail dominates.

  A crisper-looking decode that measures worse is high-frequency error, not
  recovered detail — ``--t-smooth 0`` is the clearest case: best target of all,
  worst decode of all.

For L, see :func:`bottlegen.codec.encode` — **L=32 is the operating point**.
L=40 costs 54% more coefficients for 1.6% on the whole head and is marginally
worse on the face. The old "gamma under 1e-2" rule applied to ``direct`` only;
``residual`` is still improving at gamma 4.8e-2.

.. warning::

   **Score against a reference that does not move.** An earlier version of this
   docstring claimed remeshing was worth 25-32% and that coarser was better, with
   a paired t-test to match. All of it was an artifact: those numbers compared the
   decode to the **resampled** target, and a coarser remesh degrades that target
   (measured against the original head: 0.0147 at 0.8%, 0.0196 at 2.5%, 0.0245 at
   5.0%). Coarser settings were being graded on a softer reference.

   This same mistake was made three times in one session — once for the icosphere
   resampling, twice here — because the reference and the configuration were
   changed together and the difference attributed to the configuration. Any
   comparison that alters the mesh must be scored against the original input with
   a topology-free metric, never against its own resampled target.

   A fourth trap, of the same family: a **whole-head** mean is dominated by the
   cranium and neck, so it can move a lot while the face does not move at all.
   Report :func:`~bottlegen.eval.metrics.chamfer_region` about
   :func:`~bottlegen.eval.metrics.nose_tip` alongside it. Getting the tip on the
   wrong end of the depth axis scores the occiput and reports gains that are not
   about the face — that happened too.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bottlegen import codec                                    # noqa: E402
from bottlegen.data import faces                               # noqa: E402
from bottlegen.geom.mesh import Mesh                           # noqa: E402
from bottlegen.geom import spin                               # noqa: E402
from bottlegen.models.basis import RhoBasis                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--expressions", nargs="+", default=["1_neutral"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--t-smooth", type=float, default=1e-4)
    ap.add_argument("--base", choices=("icosphere", "native"), default="icosphere")
    ap.add_argument("--subdiv", type=int, default=6)
    ap.add_argument("--remesh", type=float, default=0.8,
                    help="isotropic remesh target %% of bbox diagonal before "
                         "cMCF; 0 disables. 0.8%% measured best (see module doc).")
    ap.add_argument("--out", default="reports/faces/rho_cache.npz")
    a = ap.parse_args()

    roots = [Path(r).expanduser() for r in a.roots]

    # cap=True because rho needs a closed surface; align=False because rho is
    # exactly rotation-invariant on a shared template (measured 4e-15), so
    # aligning would be pure cost.
    t0 = time.perf_counter()
    out = faces.build_dataset(roots, expressions=a.expressions,
                              t_smooth=a.t_smooth, cap=True, align=False,
                              base=a.base, subdiv=a.subdiv,
                              remesh=(a.remesh or None), limit=a.limit,
                              verbose=True)
    if a.base == "icosphere":
        X, F, sphere = out
    else:
        X, F = out
        # ONE sphere for the whole set, from the mean head so it is not tied to
        # whichever subject happened to be first.
        sphere = codec.base_sphere(Mesh(X.mean(0), F))
    print(f"loaded + smoothed + parameterised {len(X)} heads ({X.shape[1]} "
          f"verts) in {time.perf_counter() - t0:.0f}s", flush=True)

    # A base sphere must reproduce rho = 0 against itself. Check it rather than
    # assume it: a cMCF sphere of a head scores 8.54 here against an icosphere's
    # 0.004, because two stages of the codec divide by its triangle areas.
    resid = spin.base_sphere_residual(sphere, F)
    print(f"base-sphere residual {resid:.4f} "
          f"({'OK' if resid < 0.05 else 'POOR -- expect degraded reconstruction'})",
          flush=True)
    if resid > 0.5:
        raise SystemExit(
            f"base-sphere residual {resid:.3f} is too large to encode against; "
            f"pass --base icosphere (or a finer --subdiv)")

    t0 = time.perf_counter()
    fields = RhoBasis(sphere, F, L=1).rho_fields(X)
    dt = time.perf_counter() - t0
    print(f"rho fields {fields.shape} in {dt:.0f}s ({dt / len(X):.2f}s/mesh)",
          flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # float32 for the vertex block: it is by far the biggest array (26 321 x 3
    # per mesh) and everything it feeds is a Procrustes shape error, which is
    # scale- and rotation-normalised and nowhere near float32-limited. The rho
    # fields stay float64 -- they are 1/3 the size and feed an eigensolve.
    np.savez_compressed(out, X=X.astype(np.float32), faces=F, sphere=sphere,
                        fields=fields, expressions=np.array(a.expressions),
                        t_smooth=a.t_smooth)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

    # Report the reconstruction ceiling right here, since it is the number the
    # generative model inherits and cannot beat, and it is nearly free now.
    print(f"\n{'L':>4s} {'coeffs':>7s} {'shape error':>12s}")
    for L in (8, 12, 16, 24, 32):
        b = RhoBasis.fit(X, F, L, sphere, fields=fields)
        R = b.decode(b.encode(X[:16], fields=fields[:16]))
        e = float(np.mean([codec.shape_error(X[i], R[i]) for i in range(len(R))]))
        print(f"{L:4d} {codec.n_coeffs(L):7d} {e:12.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
