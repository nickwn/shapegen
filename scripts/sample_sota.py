#!/usr/bin/env python
"""Generate bottles with the SOTA pipeline: rectified flow → VAE decoder → σ̂ →
robust Dirac decode → mesh (Session 10 DoD).

    python scripts/sample_sota.py rf.ckpt=runs/sota/rf.pt sample.n=16

Writes OBJ files, a metrics JSON (Willmore, genus, non-star), and a contact
sheet to ``sample.out_dir``.
"""

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen import util
from bottlegen.decode.reconstruct import decode
from bottlegen.eval import metrics, viz
from bottlegen.geom.mesh import icosphere
from bottlegen.models.pack import packed_to_sigma
from bottlegen.models.rect_flow import LatentDiT, heun_sample
from bottlegen.models.set_vae import SetVAE


def save_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in F:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/sota.yaml"),
                          OmegaConf.from_cli())
    device = util.device()
    ck = torch.load(cfg.rf.ckpt, map_location=device, weights_only=False)
    c = OmegaConf.create(ck["cfg"])
    L, M, zdim = c.L, c.vae.M, c.vae.zdim

    vae = SetVAE(L=L, **OmegaConf.to_container(c.vae)).to(device)
    vae.load_state_dict(ck["vae"])
    vae.eval()
    rf = LatentDiT(M, zdim, d=c.rf.d, depth=c.rf.depth, heads=c.rf.heads,
                   t_embed=c.rf.t_embed).to(device)
    rf.load_state_dict(ck["rf"])
    rf.eval()

    torch.manual_seed(cfg.sample.get("seed", 0))
    z0 = torch.randn(cfg.sample.n, M, zdim, device=device)
    z1 = heun_sample(rf, z0, n_steps=cfg.sample.steps)
    with torch.no_grad():
        sigmas = vae.dec(z1).cpu().numpy()

    base = icosphere(c.base_subdiv)
    out = cfg.sample.out_dir
    os.makedirs(out, exist_ok=True)
    meshes, mets, titles, hot = [], [], [], []
    for i, vec in enumerate(sigmas):
        sig = packed_to_sigma(vec, L)
        V, F, info = decode(base.vertices, base.faces, sig, w_cap=120,
                            return_info=True)
        save_obj(os.path.join(out, f"bottle_{i:02d}.obj"), V, F)
        ns = metrics.nonstar_fraction(V, F)
        rec = {"W": metrics.willmore_energy(V, F), "genus": metrics.genus(V, F),
               "nonstar": ns, "amplitude": info["amplitude"]}
        meshes.append((V, F)); mets.append(rec); hot.append(ns > 0.02)
        titles.append(f"W={rec['W']:.1f} s={info['amplitude']}")
        print(f"  sample {i:02d}: W={rec['W']:.2f} genus={rec['genus']} "
              f"nonstar={ns:.3f}")

    viz.contact_sheet(meshes, titles, os.path.join(out, "contact_sheet.png"),
                      highlight=hot)
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump({"mode": c.rf.mode, "samples": mets}, fh, indent=2)
    print(f"[done] {len(meshes)} SOTA bottles ({c.rf.mode}) -> {out}")


if __name__ == "__main__":
    main()
