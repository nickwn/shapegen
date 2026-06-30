#!/usr/bin/env python
"""VAE reconstruction + latent-traversal contact sheets (Session 9 DoD).

    python scripts/sample_vae.py                   # trains a demo VAE inline
    python scripts/sample_vae.py train.ckpt=runs/sota/vae.pt   # load a checkpoint

For the DiffusionNet VAE the per-mesh operators are needed, so this trains a
small VAE on the demo bottles inline (or loads a checkpoint + retrains briefly
if operators are absent), then writes:
  - `recon.png`    — input vs VAE-reconstructed bottles
  - `traversal.png`— a row of decodes interpolating between two latents
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen import util
from bottlegen.data import dataset
from bottlegen.decode.reconstruct import decode
from bottlegen.eval import viz
from bottlegen.geom import conformal as cf
from bottlegen.geom.mesh import icosphere
from bottlegen.models import training


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/sota.yaml"),
                          OmegaConf.from_cli())
    device = util.device()
    out = cfg.get("sample", {}).get("out_dir", "reports/sota_vae")
    os.makedirs(out, exist_ok=True)

    n = int(cfg.train.demo_n)
    meshes = dataset.demo_bottle_meshes(n=n, subdiv=3, seed=cfg.train.seed)
    sigmas = [cf.mesh_to_sigma(m, L=cfg.L) for m in meshes]
    print(f"[vae] training demo VAE on {n} bottles ...")
    model, ops, hist = training.fit_vae(
        meshes, sigmas, cfg.L, k=cfg.lb_k,
        vae_cfg=OmegaConf.to_container(cfg.vae), steps=cfg.train.steps,
        batch=min(cfg.train.batch, n), lr=cfg.train.lr, beta=cfg.train.beta,
        beta_warmup=cfg.train.beta_warmup, whiten=cfg.train.whiten,
        seed=cfg.train.seed, device=device, log_every=cfg.train.log_every)
    model.eval()
    print(f"[vae] final recon {np.mean([h[1] for h in hist[-50:]]):.3e}")

    base = icosphere(cfg.base_subdiv)

    def to_mesh(sigma_packed):
        from bottlegen.models.pack import packed_to_sigma
        sig = packed_to_sigma(sigma_packed, cfg.L)
        return decode(base.vertices, base.faces, sig, w_cap=120)

    # reconstruction: input mesh vs decode(VAE(mesh))
    k = min(6, n)
    rec_meshes, titles = [], []
    with torch.no_grad():
        for i in range(k):
            mu, _ = model.encode([ops[i]])
            pred = model.decode(mu)[0].cpu().numpy()
            V, F = to_mesh(pred)
            rec_meshes += [(meshes[i].vertices, meshes[i].faces), (V, F)]
            titles += [f"input {i}", f"VAE recon {i}"]
    viz.contact_sheet(rec_meshes, titles, f"{out}/recon.png", cols=4)

    # latent traversal between two bottles
    with torch.no_grad():
        mu_a, _ = model.encode([ops[0]])
        mu_b, _ = model.encode([ops[min(1, n - 1)]])
        trav, ttl = [], []
        for j, t in enumerate(np.linspace(0, 1, 6)):
            pred = model.decode((1 - t) * mu_a + t * mu_b)[0].cpu().numpy()
            trav.append(to_mesh(pred))
            ttl.append(f"t={t:.1f}")
    viz.contact_sheet(trav, ttl, f"{out}/traversal.png", cols=6)
    print(f"[done] recon + traversal contact sheets -> {out}")


if __name__ == "__main__":
    main()
