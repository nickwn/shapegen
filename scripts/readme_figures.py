"""Render the README's pipeline figures from one cached head.

Stages, left to right: the input mesh, its conformal sphere map coloured by
the encoded ρ, decodes at increasing bandwidth L, and the quad mesh computed
from the code. A second figure shows heads sampled from the flow model.

**Input.** ``reports/faces/rho_ico6.npz`` (``scripts/encode_faces.py``:
smoothed FaceScape heads resampled onto icosphere 6, closed genus 0) and, for
the samples, ``reports/faces/faces_flow_rho.npz`` (``scripts/train_faces.py
--basis rho``).

Usage::

    python -m scripts.readme_figures                  # writes docs/img/*.png
    python -m scripts.readme_figures --subject 3 --out docs/img

Rendering is a plain orthographic painter's algorithm (back faces culled,
Lambertian shading) on a 2-D ``PolyCollection``: fast, and good enough for a
convex-ish head seen from the front.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")                               # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from shapegen import codec                          # noqa: E402
from shapegen.geom.encode import eval_real_sh      # noqa: E402
from shapegen.geom.mesh import Mesh                 # noqa: E402
from shapegen.quad.pipeline import quad_mesh       # noqa: E402

BASE = np.array([0xc9, 0xcf, 0xd8]) / 255.0         # neutral clay
QUAD = np.array([0xa8, 0xc8, 0xe8]) / 255.0


# --------------------------------------------------------------------------- #
# geometry helpers                                                            #
# --------------------------------------------------------------------------- #

def similarity_align(X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate, scale and translate ``X`` onto ``target`` (same vertex order):
    the decode is defined up to a similarity, so this is display-only."""
    mx, mt = X.mean(0), target.mean(0)
    A, B = X - mx, target - mt
    U, s, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    scale = (s * np.diag(D)).sum() / (A ** 2).sum()
    return scale * A @ R + mt


def view_frame(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rows: screen-right, screen-up, towards-camera."""
    f = forward / np.linalg.norm(forward)
    u = up - (up @ f) * f
    u /= np.linalg.norm(u)
    return np.stack([np.cross(u, f), u, f])


def rotate_y(R: np.ndarray, deg: float) -> np.ndarray:
    """Turn the camera ``deg`` degrees about screen-up (a three-quarter view)."""
    a = np.radians(deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return Ry @ R


def render(ax, P: np.ndarray, faces, R: np.ndarray, colors=None, base=BASE,
           edges: float = 0.0, edgecolor="#2b3440"):
    """Orthographic render of the polygon mesh ``(P, faces)`` in camera frame
    ``R`` (see :func:`view_frame`). ``faces`` is an ``(m, k)`` array or a list
    of variable-length loops. ``colors`` (per face, RGB) replaces ``base``;
    shading multiplies either."""
    Q = (P - P.mean(0)) @ R.T
    Q /= np.abs(Q[:, :2]).max()
    loops = [np.asarray(f) for f in faces]
    polys = [Q[f] for f in loops]
    n = np.array([np.cross(p - np.roll(p, -1, 0), p + np.roll(p, -1, 0)).sum(0)
                  for p in polys])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    # one global orientation check (outward normals), never per face
    cen = np.array([p.mean(0) for p in polys])
    if np.einsum("ij,ij->i", n, cen - Q.mean(0)).sum() < 0:
        n = -n
    keep = n[:, 2] > 0
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = 0.22 + 0.78 * np.clip(n @ light, 0, 1) + 0.12 * n[:, 2] ** 8
    base_c = np.broadcast_to(base, (len(polys), 3)) if colors is None else colors
    col = np.clip(base_c * shade[:, None], 0, 1)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(cen[idx, 2])]              # far to near
    pc = PolyCollection([polys[i][:, :2] for i in idx], facecolors=col[idx],
                        edgecolors=edgecolor if edges else "face",
                        linewidths=edges if edges else 0.15, antialiased=True)
    ax.add_collection(pc)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def head_frame(X: np.ndarray) -> tuple[np.ndarray, int]:
    """The frontal camera and the nose-tip vertex. FaceScape heads (and the
    ico-resampled cache) are +z forward, +y up; the nose is the most forward
    vertex near the vertical midline (an ear can stick out further on a
    turned head, so restrict to |x| small)."""
    C = X - X.mean(0)
    mid = np.abs(C[:, 0]) < 0.1 * np.abs(C[:, 0]).max()
    nose = int(np.flatnonzero(mid)[np.argmax(C[mid, 2])])
    return view_frame(np.array([0.0, 0, 1]), np.array([0.0, 1, 0])), nose


# --------------------------------------------------------------------------- #
# figures                                                                     #
# --------------------------------------------------------------------------- #

def pipeline_figure(X, F, out: Path, Ls=(8, 16, 32), quad_faces=4000,
                    turn=-28.0):
    mesh = Mesh(X, F)
    R, nose = head_frame(X)
    R = rotate_y(R, turn)

    encs = {}
    for L in Ls:
        t = time.perf_counter()
        encs[L] = codec.encode(mesh, L)
        print(f"  encode L={L}: {encs[L]}  {time.perf_counter() - t:.1f} s", flush=True)
    Lmax = max(Ls)
    res = encs[Lmax]
    sph = res.sphere

    t = time.perf_counter()
    q = quad_mesh(res.coeffs, Lmax, target_faces=quad_faces)
    print(f"  quad mesh: ok={q['ok']} faces={len(q.get('faces', []))} "
          f"{time.perf_counter() - t:.1f} s", flush=True)
    if not q["ok"]:
        raise RuntimeError(q["reason"])

    ncols = 2 + len(Ls) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3.2), dpi=150)

    render(axes[0], X, F, R)
    axes[0].set_title(f"input mesh\n{len(X)} vertices", fontsize=9)

    # sphere map coloured by ρ; camera on the nose's image so the face faces us
    rho = eval_real_sh(sph, res.coeffs, Lmax)
    rho_f = rho[F].mean(1)
    lim = np.percentile(np.abs(rho_f), 98)
    cmap = plt.get_cmap("RdBu_r")
    cols = cmap(0.5 + 0.5 * np.clip(rho_f / lim, -1, 1))[:, :3]
    Rs = view_frame(sph[nose], R[1] - (R[1] @ sph[nose]) * sph[nose])
    Rs = rotate_y(Rs, turn)
    render(axes[1], sph, F, Rs, colors=cols)
    axes[1].set_title("conformal sphere map\ncoloured by ρ", fontsize=9)

    for ax, L in zip(axes[2:], Ls):
        Y = similarity_align(codec.decode(encs[L].coeffs, encs[L].sphere), X)
        render(ax, Y, F, R)
        ax.set_title(f"decoded, L={L}\n{(L + 1) ** 2} numbers, err {encs[L].err:.3f}",
                     fontsize=9)

    # the quad mesh's lift is in the decoder's frame: carry the similarity that
    # takes the L=Lmax decode onto the input
    qf = q["faces"]
    Pq = _carry_similarity(codec.decode(res.coeffs, sph), X, q["P3"])
    render(axes[-1], Pq, qf, R, base=QUAD, edges=0.25)
    axes[-1].set_title(f"quad mesh from z\n{len(qf)} faces", fontsize=9)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.02)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def _carry_similarity(src, dst, P):
    """The similarity taking ``src`` onto ``dst`` (corresponding rows), applied to ``P``."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, s, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    scale = (s * np.diag(D)).sum() / (A ** 2).sum()
    return scale * (P - ms) @ R + md


def samples_figure(samples, F, ref, out: Path, n=6, turn=-28.0):
    """``samples`` are decodes (arbitrary similarity): align each to ``ref``,
    a head in the dataset frame on the same connectivity."""
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6), dpi=150)
    R, _ = head_frame(ref)
    R = rotate_y(R, turn)
    for ax, X in zip(axes, samples[:n]):
        render(ax, similarity_align(X, ref), F, R)
    fig.suptitle("sampled from the flow model, decoded (no input mesh)", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.02)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="reports/faces/rho_ico6.npz")
    ap.add_argument("--flow", default="reports/faces/faces_flow_rho.npz")
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--out", default="docs/img")
    ap.add_argument("--quad-faces", type=int, default=4000)
    ap.add_argument("--only", choices=["pipeline", "samples"], default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.cache)
    X = np.asarray(d["X"][a.subject], float)
    if a.only in (None, "pipeline"):
        pipeline_figure(X, d["faces"], out / "pipeline.png",
                        quad_faces=a.quad_faces)
    if a.only in (None, "samples") and Path(a.flow).exists():
        f = np.load(a.flow)
        assert np.array_equal(f["faces"], d["faces"]), "samples not on the cache's mesh"
        samples_figure(f["samples"], f["faces"], X, out / "samples.png")


if __name__ == "__main__":
    main()
