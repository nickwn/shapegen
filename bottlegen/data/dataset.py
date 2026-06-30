"""``BottleSigma`` torch dataset + free Wigner-D rotation augmentation.

The augmentation core (``wigner_rotate_sigma``, ``random_so3``, real packing) is
pure NumPy and fully testable without torch; the ``BottleSigma`` wrapper imports
torch lazily so the module loads in a torch-free environment (the geometry /
spectral pipeline never needs it).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ..flow import state
from ..spectral import swsh

__all__ = [
    "random_so3", "wigner_rotate_sigma", "list_cache", "load_record",
    "BottleSigma", "demo_bumpy_dataset",
]


def random_so3(rng=None) -> np.ndarray:
    """Uniformly random rotation matrix (Shoemake's quaternion method)."""
    rng = np.random.default_rng() if rng is None else rng
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wigner_rotate_sigma(sigma_hat: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply the SO(3) rotation ``R`` to the spin-0 state σ̂ (Wigner-D action).
    Unitary per degree, so ``‖σ̂_ℓ‖`` is preserved."""
    return swsh.rotate_coeffs(sigma_hat, R)


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #

def list_cache(cache_dir: str):
    """Sorted list of cached ``*.npz`` record paths under ``cache_dir``."""
    return sorted(glob.glob(os.path.join(cache_dir, "*.npz")))


def load_record(path: str) -> dict:
    """Load one cached record (``sigma_hat`` plus metadata)."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def demo_bumpy_dataset(n=16, L=12, subdiv=3, seed=0):
    """Synthesize a small in-class σ̂ dataset (no ShapeNet needed) by encoding
    randomly deformed genus-0 'bottle-ish' meshes — a surface of revolution with
    a neck plus low-order bumps, some strong enough to be non-star. For demos /
    the Session 7 sample script when ``cache/`` is empty.
    """
    from ..geom import conformal as cf
    from ..geom.mesh import Mesh, icosphere
    rng = np.random.default_rng(seed)
    base = icosphere(subdiv)
    V0 = base.vertices
    z = V0[:, 2]
    out = []
    for _ in range(n):
        # bottle profile: wide body, pinched neck near the top
        neck = 1.0 - 0.45 * np.exp(-((z - 0.7) ** 2) / 0.05)
        body = 1.0 + 0.18 * (z ** 2 - 0.3)
        r = neck * body
        # random low-order bumps (can push concave -> non-star)
        r = r + 0.12 * rng.standard_normal() * (V0[:, 0] ** 2 - V0[:, 1] ** 2)
        r = r + 0.10 * rng.standard_normal() * V0[:, 0] * V0[:, 2]
        m = Mesh(V0 * r[:, None], base.faces)
        out.append(cf.mesh_to_sigma(m, L=L))
    return out


# --------------------------------------------------------------------------- #
# Torch dataset (lazy torch import)
# --------------------------------------------------------------------------- #

def _torch():
    import torch  # noqa: PLC0415 — lazy so the module loads without torch
    return torch


class BottleSigma:
    """Map-style dataset over cached bottle σ̂ with optional rotation aug.

    A plain ``__len__``/``__getitem__`` object — torch's ``DataLoader`` consumes
    it directly (no ``Dataset`` subclassing needed). Accepts either a
    ``cache_dir`` of ``*.npz`` records or an in-memory ``items`` list of σ̂
    arrays (handy for tests). ``__getitem__`` returns a real-packed float32
    tensor (the degree-tokenized layout is available via
    :func:`bottlegen.flow.state.perdeg_pack`).
    """

    def __init__(self, cache_dir=None, items=None, L=16, rotate=True, seed=0):
        if (cache_dir is None) == (items is None):
            raise ValueError("provide exactly one of cache_dir / items")
        self.cache_dir = cache_dir
        self.paths = list_cache(cache_dir) if cache_dir else None
        self._items = None if items is None else [np.asarray(s) for s in items]
        self.L = L
        self.rotate = rotate
        self._rng = np.random.default_rng(seed)
        self.stats = state.perdeg_std(np.stack(self._all_sigma())) \
            if self.__len__() else None

    def _all_sigma(self):
        if self._items is not None:
            return self._items
        return [load_record(p)["sigma_hat"] for p in self.paths]

    def __len__(self):
        return len(self._items) if self._items is not None else len(self.paths)

    def _get_sigma(self, i):
        if self._items is not None:
            return self._items[i]
        return load_record(self.paths[i])["sigma_hat"]

    def __getitem__(self, i):
        s = state.gauge_fix(np.asarray(self._get_sigma(i), dtype=complex))
        if self.rotate:
            s = wigner_rotate_sigma(s, random_so3(self._rng))
        vec = state.real_pack(s).astype(np.float32)
        torch = _torch()
        return torch.from_numpy(vec)
