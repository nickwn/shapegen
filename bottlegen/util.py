"""Device + precision split (see CONVENTIONS.md).

MPS has no float64, so all of ``geom/``, ``flow/willmore.py``, the conformal
param and the Dirac solve run in **NumPy/SciPy float64 on CPU**; the networks run
**float32 on MPS** (or CPU as a fallback).
"""

from __future__ import annotations

GEOM_DTYPE = "float64"   # numpy, CPU


def device(prefer_mps: bool = True):
    """Return the network device: MPS if available and preferred, else CPU."""
    import torch
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def net_dtype():
    import torch
    return torch.float32
