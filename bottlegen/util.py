"""Device + precision split (see CONVENTIONS.md).

MPS has no float64, so all of ``geom/`` — the conformal parameterisation and the
Dirac solve included — runs in **NumPy/SciPy float64 on CPU**. The float32/MPS
half is a placeholder: there are no networks in this package any more.
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
