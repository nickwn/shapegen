#!/usr/bin/env python
"""Train the rectified flow over the frozen VAE latents (Session 10).

    python train/train_rf.py                       # vanilla, configs/sota.yaml
    python train/train_rf.py rf.mode=physics
    python train/train_rf.py train.synthetic=true  # demo without ShapeNet

Loads (or trains inline) the SetVAE, freezes it, extracts the data latents, and
trains a LatentDiT rectified flow (vanilla ``z1−z0`` target, or the physics-
informed decoder-VJP reverse-Willmore velocity). Saves the VAE + flow together.
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
from bottlegen.geom.diffusion_ops import mesh_operators
from bottlegen.models import training
from bottlegen.models.diffusionnet import ops_to_torch
from bottlegen.models.pack import kappa_packed
from bottlegen.models.set_vae import SetVAE


def get_vae_and_ops(cfg, meshes, sigmas, device):
    vae = SetVAE(L=cfg.L, **OmegaConf.to_container(cfg.vae)).to(device)
    if os.path.exists(cfg.rf.vae_ckpt) and not cfg.train.get("synthetic", False):
        vae.load_state_dict(torch.load(cfg.rf.vae_ckpt, map_location=device,
                                       weights_only=False)["model"])
        ops = [ops_to_torch(mesh_operators(m.vertices, m.faces, cfg.lb_k), device)
               for m in meshes]
        print(f"[vae] loaded {cfg.rf.vae_ckpt}")
    else:
        print("[vae] training inline on demo bottles ...")
        vae, ops, _ = training.fit_vae(
            meshes, sigmas, cfg.L, k=cfg.lb_k,
            vae_cfg=OmegaConf.to_container(cfg.vae), steps=cfg.train.steps,
            batch=min(cfg.train.batch, len(meshes)), lr=cfg.train.lr,
            beta=cfg.train.beta, beta_warmup=cfg.train.beta_warmup,
            whiten=cfg.train.whiten, seed=cfg.train.seed, device=device,
            log_every=cfg.train.log_every)
    return vae, ops


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/sota.yaml"),
                          OmegaConf.from_cli())
    device = util.device()
    print(f"[device] {device}  mode={cfg.rf.mode}")

    meshes = dataset.demo_bottle_meshes(n=cfg.train.demo_n, subdiv=3,
                                        seed=cfg.train.seed)
    sigmas = [cf.mesh_to_sigma(m, L=cfg.L) for m in meshes]
    vae, ops = get_vae_and_ops(cfg, meshes, sigmas, device)
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()

    z1 = training.vae_latents(vae, ops, device)
    print(f"[rf] {z1.shape[0]} latents, shape {tuple(z1.shape[1:])}")
    kap = torch.as_tensor(kappa_packed(cfg.L), dtype=torch.float32,
                          device=device) if cfg.rf.mode == "physics" else None

    rf, ema, hist = training.fit_rect_flow(
        z1, mode=cfg.rf.mode, decoder=(vae.dec if cfg.rf.mode == "physics"
                                       else None), kappa_packed=kap,
        rf_cfg=dict(d=cfg.rf.d, depth=cfg.rf.depth, heads=cfg.rf.heads,
                    t_embed=cfg.rf.t_embed),
        steps=cfg.rf.steps, batch=cfg.rf.batch, lr=cfg.rf.lr,
        ema_decay=cfg.rf.ema_decay, seed=cfg.rf.seed, device=device,
        log_every=cfg.rf.log_every)

    ckpt = cfg.rf.ckpt
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"rf": ema.shadow, "vae": vae.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "loss_final": float(np.mean(hist[-50:]))}, ckpt)
    print(f"[done] {cfg.rf.mode} RF loss {np.mean(hist[-50:]):.3e} -> {ckpt}")


if __name__ == "__main__":
    main()
