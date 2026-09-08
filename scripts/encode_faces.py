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
    ap.add_argument("--t-smooth", type=float, default=1e-3)
    ap.add_argument("--base", choices=("icosphere", "native"), default="icosphere")
    ap.add_argument("--subdiv", type=int, default=6)
    ap.add_argument("--remesh", type=float, default=0.8,
                    help="isotropic remesh target %% of bbox diagonal before "
                         "cMCF; 0 disables. Worth 25-32%% of reconstruction.")
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
