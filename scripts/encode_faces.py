"""One-time build of a FaceScape set for the SH flow model: smoothed, capped,
pose-aligned heads on one shared sphere.

What gets cached is the **positions and the sphere**, not codes: ``quotient``
codes are fitted at one bandwidth, and at ~0.3 s per head encoding is cheap next
to loading, smoothing and remeshing, so :class:`~shapegen.models.basis.RhoBasis`
encodes at whatever L it is fitted at.

The sphere is cached alongside, and that is not optional: ρ̂ is defined relative
to a base sphere, so a cache read back against a different sphere silently means
something else. By default it is a uniform icosphere (subdiv 6) resampled through
the sphere map of the remeshed mean head (:func:`shapegen.codec.base_sphere`,
see :func:`shapegen.data.faces.build_dataset`).

Usage::

    python -m scripts.encode_faces --roots ~/Downloads/facescape_trainset_0* \\
        --out reports/faces/rho_neutral.npz
    python -m scripts.encode_faces --roots ~/Downloads/facescape_* \\
        --expressions 1_neutral 2_smile 16_grin --out reports/faces/rho_multi.npz

Then::

    d = np.load(path)
    basis = RhoBasis.fit(d["X"], d["faces"], L, d["sphere"])

**Defaults** (6 heads, chamfer against the ORIGINAL head, whole / face region;
measured with the since-removed residual encoder and mesh decoder):

* ``--remesh 0.8``: remeshing at all is worth ~14%; 2.5% -> 0.8% is worth 8%
  whole-head and 0% on the face (the gain is cranium faceting); 0.4% buys
  nothing at 4x the vertices. In the face the encoder, not the target, binds --
  Gibbs ringing at the lip seam, eyelid creases and nostril.
* ``--t-smooth 1e-4``: removing smoothing costs 29.6% on the face, the old 1e-3
  costs 10%. The optimum trades target detail lost (monotone in t) against scan
  noise H amplifies by ~ℓ², which the encoder spends bands chasing. A
  crisper-looking decode that measures worse is high-frequency error, not
  recovered detail.
* L=32 is the operating point (see :func:`shapegen.codec.encode`).

.. warning::

   **Score against a reference that does not move** -- the original head, with a
   topology-free metric (:func:`~shapegen.eval.metrics.chamfer`), never the
   resampled target this script builds; a coarser remesh softens that target and
   has reported 25-32% gains that were not real. **And score the face
   separately** with :func:`~shapegen.eval.metrics.chamfer_region` about
   :func:`~shapegen.eval.metrics.nose_tip`: a whole-head mean is dominated by
   cranium and neck, and a tip on the wrong end of the depth axis scores the
   occiput.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shapegen import codec                                    # noqa: E402
from shapegen.data import faces                               # noqa: E402
from shapegen.geom.mesh import Mesh                           # noqa: E402
from shapegen.models.basis import RhoBasis                    # noqa: E402


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
                         "the sphere map; 0 disables. 0.8%% measured best (see module doc).")
    ap.add_argument("--out", default="reports/faces/rho_cache.npz")
    a = ap.parse_args()

    roots = [Path(r).expanduser() for r in a.roots]

    # cap=True because rho needs a closed surface; align=True because quotient
    # codes on a shared sphere move ~7% under a random rotation (L=32), against
    # ~31% between subjects, so unaligned pose would read as shape.
    t0 = time.perf_counter()
    out = faces.build_dataset(roots, expressions=a.expressions,
                              t_smooth=a.t_smooth, cap=True, align=True,
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

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # float32 for the vertex block: it is by far the biggest array (26 321 x 3
    # per mesh) and everything it feeds is a Procrustes shape error, which is
    # scale- and rotation-normalised and nowhere near float32-limited.
    np.savez_compressed(out, X=X.astype(np.float32), faces=F, sphere=sphere,
                        expressions=np.array(a.expressions),
                        t_smooth=a.t_smooth)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")

    # The reconstruction ceiling of the codes: the number the generative model
    # inherits and cannot beat (fitted on 16 heads: this is a codec number).
    print(f"\n{'L':>4s} {'coeffs':>7s} {'shape error':>12s}")
    for L in (8, 12, 16, 24, 32):
        b = RhoBasis.fit(X[:16], F, L, sphere)
        R = b.decode(b.encode(X[:16]))
        e = float(np.mean([codec.shape_error(X[i], R[i]) for i in range(len(R))]))
        print(f"{L:4d} {codec.n_coeffs(L):7d} {e:12.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
