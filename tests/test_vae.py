"""Session 9 acceptance tests: DiffusionNet set-latent VAE.

Maps to Appendix B row 12:
  12. VAE recon + smooth traversal                  small / visual

Plus: DiffusionNet operator sanity, KL behaves, permutation-invariant latents
(vertex shuffle), and smooth latent traversal. CPU + small (sub-2) meshes; the
VAE is trained once per module (a shared fixture) for speed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bottlegen.flow import state                            # noqa: E402
from bottlegen.geom import conformal as cf                  # noqa: E402
from bottlegen.geom.diffusion_ops import (lb_eigenbasis,    # noqa: E402
                                          mesh_operators,
                                          tangent_gradient_operators)
from bottlegen.geom.mesh import Mesh, icosphere             # noqa: E402
from bottlegen.models import training                       # noqa: E402
from bottlegen.models.diffusionnet import ops_to_torch      # noqa: E402
from bottlegen.models.pack import PackSpec                  # noqa: E402
from bottlegen.models.set_vae import SetVAE                 # noqa: E402

L = 8
VAE_CFG = dict(in_ch=6, C=64, M=12, zdim=12, n_blocks=3, dec_d=96, heads=4,
               dec_depth=2)
N = 3


def _shape(seed, sub=2):
    """A strongly, distinctly deformed genus-0 mesh (random low-order radial)."""
    rng = np.random.default_rng(seed)
    s = icosphere(sub)
    V = s.vertices.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    a = rng.uniform(-0.4, 0.4, 6)
    r = (1 + a[0] * (z ** 2 - 0.3) + a[1] * (x ** 2 - y ** 2) + a[2] * x * y
         + a[3] * y * z + a[4] * x * z + a[5] * z)
    return Mesh(V * np.clip(r, 0.3, None)[:, None], s.faces)


@pytest.fixture(scope="module")
def trained():
    meshes = [_shape(i) for i in range(N)]
    sigmas = [cf.mesh_to_sigma(m, L=L) for m in meshes]
    model, ops, hist = training.fit_vae(
        meshes, sigmas, L, k=16, vae_cfg=VAE_CFG, steps=900, batch=N,
        lr=2e-3, beta=1e-5, beta_warmup=150, whiten=False, seed=0)
    model.eval()
    return {"model": model, "ops": ops, "sigmas": sigmas, "hist": hist}


# --------------------------------------------------------------------------- #
# Operator sanity                                                             #
# --------------------------------------------------------------------------- #

def test_lb_eigenbasis_constant_mode():
    m = icosphere(3)
    evals, evecs = lb_eigenbasis(m.vertices, m.faces, k=16)
    assert evals[0] < 1e-6 and (np.diff(evals) > -1e-9).all()
    v0 = evecs[:, 0]
    assert np.std(v0) / (abs(np.mean(v0)) + 1e-9) < 1e-3   # constant 0-mode


def test_tangent_gradient_kills_constants():
    m = icosphere(3)
    gX, gY = tangent_gradient_operators(m.vertices, m.faces)
    one = np.ones(len(m.vertices))
    assert np.abs(gX @ one).max() < 1e-8
    assert np.abs(gY @ one).max() < 1e-8


def test_vae_forward_shapes():
    m = _shape(0)
    ops = ops_to_torch(mesh_operators(m.vertices, m.faces, k=16))
    vae = SetVAE(L=L, **VAE_CFG)
    pred, mu, lv = vae([ops])
    assert pred.shape == (1, PackSpec(L).dim)
    assert mu.shape == (1, VAE_CFG["M"], VAE_CFG["zdim"])
    assert vae.num_params() > 0


# --------------------------------------------------------------------------- #
# Permutation-invariant latents (vertex shuffle)                              #
# --------------------------------------------------------------------------- #

def test_permutation_invariant_latents(trained):
    model = trained["model"]
    m = _shape(1)
    ops = ops_to_torch(mesh_operators(m.vertices, m.faces, k=16))
    perm = torch.randperm(ops["feats"].shape[0])
    pops = {
        "evals": ops["evals"],
        "evecs": ops["evecs"][perm],
        "gradX": ops["gradX"][perm][:, perm],
        "gradY": ops["gradY"][perm][:, perm],
        "feats": ops["feats"][perm],
    }
    with torch.no_grad():
        mu0, _ = model.encode_one(ops)
        mu1, _ = model.encode_one(pops)
    assert torch.allclose(mu0, mu1, atol=1e-5)


# --------------------------------------------------------------------------- #
# Test 12: autoencode -> distinct, faithful per-mesh σ̂                        #
# --------------------------------------------------------------------------- #

def test_vae_autoencode(trained):
    model, ops, sigmas, hist = (trained["model"], trained["ops"],
                                trained["sigmas"], trained["hist"])
    recon0 = np.mean([h[1] for h in hist[:10]])
    reconf = np.mean([h[1] for h in hist[-20:]])
    print(f"[vae overfit] recon {recon0:.3e} -> {reconf:.3e}")
    assert reconf < 0.05 * recon0

    with torch.no_grad():
        preds = [model.decode(model.encode([ops[i]])[0])[0].cpu().numpy()
                 for i in range(N)]
    tgts = [state.real_pack(state.gauge_fix(sigmas[i])) for i in range(N)]
    C = np.array([[np.corrcoef(preds[i], tgts[j])[0, 1] for j in range(N)]
                  for i in range(N)])
    print(f"[vae recon] self-corr {np.round(np.diag(C), 3)} "
          f"off-diag {(C.sum() - np.trace(C)) / (N * N - N):.3f}")
    assert (np.diag(C) > 0.9).all()                        # faithful per mesh
    assert np.diag(C).mean() > (C.sum() - np.trace(C)) / (N * N - N) + 0.5


def test_kl_sane(trained):
    kl = np.mean([h[2] for h in trained["hist"][-10:]])
    assert np.isfinite(kl) and kl >= 0


def test_latent_traversal_smooth(trained):
    model, ops = trained["model"], trained["ops"]
    with torch.no_grad():
        mu_a, _ = model.encode([ops[0]])
        mu_b, _ = model.encode([ops[1]])
        sigs = [model.decode((1 - t) * mu_a + t * mu_b)[0].cpu().numpy()
                for t in np.linspace(0, 1, 6)]
    steps = [np.linalg.norm(sigs[i + 1] - sigs[i]) for i in range(len(sigs) - 1)]
    total = np.linalg.norm(sigs[-1] - sigs[0])
    print(f"[traversal] total={total:.3f} steps={[round(s, 3) for s in steps]}")
    assert total > 0.05                                    # latents drive output
    assert max(steps) < 3 * np.mean(steps)                 # smooth, no jumps
    assert sum(steps) < 1.5 * total                        # ~direct path
