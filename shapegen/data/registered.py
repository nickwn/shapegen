"""Loading template-registered mesh datasets (COMA, DFAUST, SMAL, ...).

These are the ideal input for the codec: every mesh shares one connectivity, is
genus 0 and watertight, so the correspondence problem that dominates real-world
data is already solved. What this module does is load them and *verify* those
properties rather than assume them -- a single stray mesh with different topology
would silently corrupt an entire encoded dataset.

COMA and FLAME are license-gated (registration + a non-commercial agreement at
coma.is.tue.mpg.de / flame.is.tue.mpg.de). Download them yourself; this module
only reads what you already have.

Minimal PLY reader instead of a trimesh dependency: these files are a fixed,
simple format and the alternative is a large dependency for one function.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = ["read_ply", "read_obj", "read_mesh", "fill_holes", "apply_caps",
           "boundary_loops", "load_registered", "check_registered"]

_NP = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
       "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
       "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
       "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4"}


def read_ply(path):
    """``(V (n,3) float64, F (m,3) int64)`` from an ascii or binary-LE PLY.

    Handles the vertex properties these datasets actually carry (xyz plus
    optional normals/colour) by reading the declared layout rather than assuming
    xyz-only, and triangulates any polygon fans in the face list.
    """
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    header = raw[:end].decode("ascii", "replace").splitlines()
    body_at = end + (2 if raw[end:end + 2] == b"\r\n" else 1)

    fmt, elements, cur = None, [], None
    for line in header:
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            cur = {"name": t[1], "count": int(t[2]), "props": []}
            elements.append(cur)
        elif t[0] == "property" and cur is not None:
            if t[1] == "list":
                cur["props"].append(("list", t[2], t[3], t[4]))
            else:
                cur["props"].append(("scalar", t[1], t[2]))
    if fmt is None:
        raise ValueError(f"{path}: no format line in PLY header")

    V = F = None
    if fmt == "ascii":
        toks = raw[body_at:].split()
        k = 0
        for el in elements:
            if el["name"] == "vertex":
                names = [p[2] for p in el["props"]]
                w = len(names)
                block = np.array(toks[k:k + el["count"] * w], dtype=float)
                k += el["count"] * w
                block = block.reshape(el["count"], w)
                V = block[:, [names.index(a) for a in ("x", "y", "z")]]
            else:
                faces = []
                for _ in range(el["count"]):
                    n = int(toks[k]); k += 1
                    idx = [int(x) for x in toks[k:k + n]]; k += n
                    faces += [[idx[0], idx[i], idx[i + 1]] for i in range(1, n - 1)]
                if el["name"] == "face":
                    F = np.array(faces, dtype=np.int64)
    elif fmt == "binary_little_endian":
        off = body_at
        for el in elements:
            if el["name"] == "vertex":
                dt = np.dtype([(p[2], "<" + _NP[p[1]]) for p in el["props"]])
                block = np.frombuffer(raw, dtype=dt, count=el["count"], offset=off)
                off += dt.itemsize * el["count"]
                V = np.stack([block["x"], block["y"], block["z"]], axis=1).astype(float)
            else:
                faces = []
                for _ in range(el["count"]):
                    _, cnt_t, idx_t, _ = el["props"][0]
                    cs = np.dtype(_NP[cnt_t]).itemsize
                    n = int(np.frombuffer(raw, np.dtype("<" + _NP[cnt_t]),
                                          1, off)[0]); off += cs
                    idx = np.frombuffer(raw, np.dtype("<" + _NP[idx_t]), n, off)
                    off += np.dtype(_NP[idx_t]).itemsize * n
                    faces += [[int(idx[0]), int(idx[i]), int(idx[i + 1])]
                              for i in range(1, n - 1)]
                if el["name"] == "face":
                    F = np.array(faces, dtype=np.int64)
    else:
        raise ValueError(f"{path}: unsupported PLY format {fmt!r} "
                         f"(need ascii or binary_little_endian)")
    if V is None or F is None:
        raise ValueError(f"{path}: missing vertex or face element")
    return V, F


def check_registered(meshes):
    """Verify the invariants the codec needs. Returns a dict; raises on breach.

    Checked rather than assumed because a single odd mesh corrupts a whole
    encoded dataset *silently* -- the codes would simply mean different things.
    """
    from ..eval.metrics import genus  # noqa: PLC0415
    F0 = meshes[0].faces
    n0 = len(meshes[0].vertices)
    for i, m in enumerate(meshes):
        if len(m.vertices) != n0:
            raise ValueError(f"mesh {i}: {len(m.vertices)} vertices, expected "
                             f"{n0} — the set is not template-registered")
        if not np.array_equal(m.faces, F0):
            raise ValueError(f"mesh {i}: connectivity differs from mesh 0 — the "
                             f"set is not template-registered")
    loops = boundary_loops(F0)
    if loops:
        raise ValueError(
            f"surface is OPEN: {len(loops)} boundary loop(s) of sizes "
            f"{sorted((len(l) for l in loops), reverse=True)}. Every stage of the "
            f"codec assumes a closed surface — the Dirac eigensolve, the "
            f"cotan-Poisson integration and cMCF are all undefined with boundary. "
            f"Run `fill_holes` first.")
    g = genus(meshes[0].vertices, F0)
    if g != 0:
        raise ValueError(f"genus {g}; the spin-transform codec is genus-0 only")
    return {"n_meshes": len(meshes), "n_vertices": n0, "n_faces": len(F0),
            "genus": g}


def load_registered(root, pattern="**/*.ply", limit=None, stride=1):
    """Load a directory tree of registered meshes into ``[Mesh]``.

    ``stride`` subsamples: COMA-style data is 60 fps video, so consecutive frames
    are near-duplicates and encoding all of them mostly buys redundancy.
    """
    from ..geom.mesh import Mesh  # noqa: PLC0415
    paths = sorted(Path(root).glob(pattern))[::stride]
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")
    if limit is not None:
        paths = paths[:limit]
    meshes = []
    for p in paths:
        V, F = read_ply(p)
        meshes.append(Mesh(V, F))
    check_registered(meshes)
    return meshes


def read_obj(path):
    """``(V, F)`` from a Wavefront OBJ. Polygons are fan-triangulated.

    Ignores everything but ``v`` and ``f`` — normals, UVs and materials are not
    geometry the codec can use, and FaceScape ships them alongside.
    """
    V, F = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                V.append([float(x) for x in t[1:4]])
            elif t[0] == "f":
                idx = [int(q.split("/")[0]) - 1 for q in t[1:]]
                F += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(V, dtype=float), np.array(F, dtype=np.int64)


def read_mesh(path):
    """Dispatch on suffix."""
    p = Path(path)
    if p.suffix.lower() == ".ply":
        return read_ply(p)
    if p.suffix.lower() == ".obj":
        return read_obj(p)
    raise ValueError(f"{p.name}: expected .ply or .obj")


def boundary_loops(F):
    """Ordered boundary loops as vertex-index lists, with their winding.

    A boundary edge belongs to exactly one face, and that face fixes its
    direction. The loops are returned in that *directed* order, which is what
    lets :func:`fill_holes` orient the caps consistently with the rest of the
    surface — get this wrong and the mesh is closed but inside-out in patches,
    which the codec would not notice and the decode would silently mangle.
    """
    from collections import Counter  # noqa: PLC0415
    undirected = Counter()
    directed = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            undirected[(min(a, b), max(a, b))] += 1
            directed[(a, b)] = True
    nxt = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if undirected[(min(a, b), max(a, b))] == 1:
                nxt[a] = b                      # directed as the lone face sees it
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, cur = [], start
        while cur is not None and cur not in seen:
            seen.add(cur)
            loop.append(cur)
            cur = nxt.get(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_holes(V, F, loops=None):
    """Cap every boundary loop with a centroid fan. ``(V2, F2, added_faces)``.

    Returns the added faces separately: for a **registered** dataset the caps are
    a fixed face list, so compute them once on one mesh and reuse. The new vertex
    positions still differ per mesh (each cap centroid follows its own loop), but
    the *combinatorics* are shared, which is what keeps the set registered.

    Cap winding is the reverse of the boundary's directed traversal, which is what
    makes the result consistently oriented rather than merely closed.
    """
    loops = boundary_loops(F) if loops is None else loops
    V2 = list(np.asarray(V, dtype=float))
    added = []
    for loop in loops:
        c = len(V2)
        V2.append(np.asarray(V, dtype=float)[loop].mean(0))
        n = len(loop)
        for i in range(n):
            a, b = loop[i], loop[(i + 1) % n]
            added.append([b, a, c])             # reversed: closes the surface
    added = np.array(added, dtype=np.int64) if added else np.zeros((0, 3), np.int64)
    return np.array(V2), np.vstack([np.asarray(F), added]), added


def apply_caps(V, F, loops, added):
    """Apply a precomputed cap face list to another mesh of the same topology."""
    V2 = list(np.asarray(V, dtype=float))
    for loop in loops:
        V2.append(np.asarray(V, dtype=float)[loop].mean(0))
    return np.array(V2), np.vstack([np.asarray(F), added])
