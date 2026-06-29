"""Session 5 acceptance tests: state packing + gauge fix.

State round trips (flat and per-degree) and gauge-fix idempotency.
"""

import numpy as np
import pytest

from bottlegen.flow import state
from bottlegen.spectral import swsh

L = 12


def _rand_sigma(seed=0):
    rng = np.random.default_rng(seed)
    C = swsh.empty_coeffs(L)
    for l in range(2, L + 1):
        for m in range(-l, l + 1):
            C[l, l + m] = rng.standard_normal() + 1j * rng.standard_normal()
    return C


def test_real_pack_round_trip():
    C = _rand_sigma(1)
    v = state.real_pack(C)
    assert v.shape[0] == state.flat_dim(L)
    C2 = state.real_unpack(v, L)
    assert np.max(np.abs(C2 - C)) < 1e-12


def test_perdeg_pack_round_trip():
    C = _rand_sigma(2)
    pd = state.perdeg_pack(C)
    assert [len(x) for x in pd] == [2 * (2 * l + 1) for l in range(2, L + 1)]
    C2 = state.perdeg_unpack(pd, L)
    assert np.max(np.abs(C2 - C)) < 1e-12


def test_gauge_fix_zeros_low_l_and_idempotent():
    C = _rand_sigma(3)
    C[0, 0] = 5.0
    C[1, 1] = 2.0 - 1j
    g = state.gauge_fix(C)
    assert np.allclose(g[0], 0) and np.allclose(g[1], 0)
    assert np.max(np.abs(state.gauge_fix(g) - g)) == 0.0   # idempotent


def test_perdeg_std_positive_and_shaped():
    stack = np.stack([_rand_sigma(i) for i in range(5)])
    std = state.perdeg_std(stack)
    assert std.shape == (L + 1,)
    assert (std[2:] > 0).all() and std[0] == 0 and std[1] == 0


def test_sobolev_weights_increasing():
    w = state.sobolev_weights(L, beta=1.0)
    assert (np.diff(w[2:]) > 0).all()
    assert w[0] == 1.0
