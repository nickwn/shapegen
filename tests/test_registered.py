"""Tests for the registered-dataset loader.

COMA/FLAME are license-gated, so these exercise the reader and the invariant
checks against synthetic data — which is what actually needs testing: a silent
topology mismatch would corrupt an encoded dataset without raising anything.
"""

import numpy as np
import pytest

from bottlegen.data.dataset import demo_bottle_meshes
from bottlegen.data.registered import check_registered, read_ply
from bottlegen.geom.mesh import Mesh, icosphere


def _write_ply(path, V, F, binary=True):
    if binary:
        head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"element face {len(F)}\n"
                f"property list uchar int vertex_indices\nend_header\n")
        buf = bytearray(head.encode())
        buf += np.asarray(V, "<f4").tobytes()
        for f in F:
            buf += np.uint8(3).tobytes() + np.asarray(f, "<i4").tobytes()
    else:
        head = (f"ply\nformat ascii 1.0\nelement vertex {len(V)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"element face {len(F)}\n"
                f"property list uchar int vertex_indices\nend_header\n")
        buf = bytearray(head.encode())
        buf += "".join(f"{a} {b} {c}\n" for a, b, c in V).encode()
        buf += "".join(f"3 {a} {b} {c}\n" for a, b, c in F).encode()
    path.write_bytes(bytes(buf))


@pytest.mark.parametrize("binary", (True, False))
def test_ply_roundtrip(tmp_path, binary):
    m = icosphere(2)
    p = tmp_path / "m.ply"
    _write_ply(p, m.vertices, m.faces, binary=binary)
    V, F = read_ply(p)
    assert np.allclose(V, m.vertices, atol=1e-6)
    assert np.array_equal(F, m.faces)


def test_check_registered_accepts_a_registered_set():
    info = check_registered(demo_bottle_meshes(n=3, subdiv=2, seed=0))
    assert info["genus"] == 0 and info["n_meshes"] == 3


def test_check_registered_rejects_mismatched_topology():
    """The failure this guards against is silent: differing connectivity means
    the codes mean different things, with nothing to signal it."""
    a = demo_bottle_meshes(n=1, subdiv=2, seed=0)[0]
    b = demo_bottle_meshes(n=1, subdiv=3, seed=0)[0]
    with pytest.raises(ValueError, match="not template-registered"):
        check_registered([a, b])


def test_check_registered_rejects_nonzero_genus():
    nu, nv, R, r = 16, 8, 1.0, 0.4
    u = np.arange(nu) * 2 * np.pi / nu
    v = np.arange(nv) * 2 * np.pi / nv
    U, V_ = np.meshgrid(u, v, indexing="ij")
    P = np.stack([(R + r * np.cos(V_)) * np.cos(U),
                  (R + r * np.cos(V_)) * np.sin(U), r * np.sin(V_)],
                 -1).reshape(-1, 3)
    idx = lambda i, j: (i % nu) * nv + (j % nv)
    F = np.array([f for i in range(nu) for j in range(nv)
                  for f in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                            [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])],
                 dtype=np.int64)
    with pytest.raises(ValueError, match="genus-0 only"):
        check_registered([Mesh(P, F)])
