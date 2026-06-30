#!/usr/bin/env python
"""Train the DiffusionNet set-latent VAE (Session 9).

    python train/train_vae.py                      # configs/sota.yaml
    python train/train_vae.py train.synthetic=true # demo without ShapeNet

The VAE autoencodes meshes → σ̂ through an M-token latent set. Reads cached
(mesh, σ̂) records from ``cfg.cache_dir`` or falls back to the synthetic demo
bottles. Saves the checkpoint to ``train.ckpt``; freeze it for the rectified
flow (Session 10).
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen import util
from bottlegen.data import dataset
from bottlegen.geom import conformal as cf
from bottlegen.geom.mesh import Mesh
from bottlegen.models import training


def load_meshes_and_sigmas(cfg):
    paths = dataset.list_cache(cfg.cache_dir) if cfg.cache_dir else []
    if paths and not cfg.train.get("synthetic", False):
        print(f"[data] {len(paths)} cached bottles from {cfg.cache_dir}")
        meshes, sigmas = [], []
        for p in paths:
            rec = dataset.load_record(p)
            meshes.append(Mesh(rec["mesh_V"], rec["mesh_F"]))
            sigmas.append(rec["sigma_hat"])
        return meshes, sigmas
    print("[data] no cache -> synthetic demo bottles")
    meshes = dataset.demo_bottle_meshes(n=cfg.train.demo_n, subdiv=3,
                                        seed=cfg.train.seed)
    sigmas = [cf.mesh_to_sigma(m, L=cfg.L) for m in meshes]
    return meshes, sigmas


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/sota.yaml"),
                          OmegaConf.from_cli())
    device = util.device()
    print(f"[device] {device}")
    meshes, sigmas = load_meshes_and_sigmas(cfg)

    model, ops, hist = training.fit_vae(
        meshes, sigmas, cfg.L, k=cfg.lb_k,
        vae_cfg=OmegaConf.to_container(cfg.vae),
        steps=cfg.train.steps, batch=cfg.train.batch, lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay, beta=cfg.train.beta,
        beta_warmup=cfg.train.beta_warmup, sobolev_beta=cfg.train.sobolev_beta,
        whiten=cfg.train.whiten, seed=cfg.train.seed, device=device,
        log_every=cfg.train.log_every)

    ckpt = cfg.train.ckpt
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg),
                "recon_final": float(np.mean([h[1] for h in hist[-50:]]))}, ckpt)
    print(f"[done] recon {np.mean([h[1] for h in hist[-50:]]):.3e} -> {ckpt}")


if __name__ == "__main__":
    main()
