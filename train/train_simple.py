#!/usr/bin/env python
"""Train the Simple SWSH cold-diffusion DiT (Session 7).

    python train/train_simple.py                       # configs/simple.yaml
    python train/train_simple.py L=24 train.steps=40000
    python train/train_simple.py train.synthetic=true  # demo without ShapeNet

Reads cached σ̂ from ``cfg.cache_dir`` (built by the data stage). If that is
empty (or ``train.synthetic=true``) it falls back to a synthetic demo dataset so
the pipeline is runnable end-to-end. Saves the EMA checkpoint to ``train.ckpt``.
"""

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen import util
from bottlegen.data import dataset
from bottlegen.flow import schedule
from bottlegen.models import training


def load_dataset(cfg):
    paths = dataset.list_cache(cfg.cache_dir) if cfg.cache_dir else []
    if paths and not cfg.train.get("synthetic", False):
        print(f"[data] {len(paths)} cached bottles from {cfg.cache_dir}")
        return [dataset.load_record(p)["sigma_hat"] for p in paths]
    print("[data] no cache found -> synthetic demo dataset")
    return dataset.demo_bumpy_dataset(n=cfg.train.get("demo_n", 24),
                                      L=cfg.L, subdiv=3, seed=cfg.train.seed)


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/simple.yaml"),
                          OmegaConf.from_cli())
    device = util.device()
    print(f"[device] {device}")
    sigma_stack = load_dataset(cfg)
    t_max = schedule.tmax_for_band(cfg.L, cfg.flow.top_band_damp)

    model, ema, hist = training.fit_simple(
        sigma_stack, cfg.L, t_max,
        model_cfg=OmegaConf.to_container(cfg.model),
        steps=cfg.train.steps, batch=cfg.train.batch, lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay, ema_decay=cfg.train.ema_decay,
        sobolev_beta=cfg.train.sobolev_beta, whiten=cfg.train.whiten,
        rotate_aug=cfg.train.rotate_aug, warmup=cfg.train.warmup,
        seed=cfg.train.seed, device=device, log_every=cfg.train.log_every)

    ckpt = cfg.train.ckpt
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    torch.save({"ema": ema.shadow, "model": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg), "t_max": t_max,
                "loss_final": float(np.mean(hist[-50:]))}, ckpt)
    print(f"[done] final loss {np.mean(hist[-50:]):.3e}  ->  {ckpt}")


if __name__ == "__main__":
    main()
