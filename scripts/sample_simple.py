#!/usr/bin/env python
"""Sample bottles from the trained Simple model and decode them.

    python scripts/sample_simple.py                    # uses train.ckpt
    python scripts/sample_simple.py sample.n=16 sample.noise=1e-4

Cold-diffusion restoration from the terminal prior -> σ̂ -> robust Dirac decode
(Session 8 trust region) -> mesh. Writes OBJ files, a metrics JSON (Willmore,
genus, non-star fraction, decode amplitude) and a contact-sheet PNG (non-star
samples highlighted) to ``sample.out_dir``.
"""

import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen import util
from bottlegen.data import dataset
from bottlegen.decode.reconstruct import decode
from bottlegen.eval import metrics, viz
from bottlegen.flow import state
from bottlegen.geom.mesh import icosphere
from bottlegen.models import cold_diffusion as cd
from bottlegen.models import pack
from bottlegen.models.training import make_model


def save_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in F:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def build_prior(cfg, L, t_max):
    paths = dataset.list_cache(cfg.cache_dir) if cfg.cache_dir else []
    if paths and not cfg.train.get("synthetic", False):
        data = [dataset.load_record(p)["sigma_hat"] for p in paths]
    else:
        data = dataset.demo_bumpy_dataset(n=cfg.train.get("demo_n", 24),
                                          L=L, subdiv=3, seed=cfg.train.seed)
    packed = np.stack([state.real_pack(state.gauge_fix(np.asarray(d, complex)))
                       for d in data])
    return cd.TerminalPriorPacked(pack.PackSpec(L).dim).fit(
        packed, t_max, pack.kappa_packed(L))


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/simple.yaml"),
                          OmegaConf.from_cli())
    device = util.device()
    ckpt = torch.load(cfg.train.ckpt, map_location=device, weights_only=False)
    L = ckpt["cfg"]["L"]
    t_max = ckpt["t_max"]

    model = make_model(L, ckpt["cfg"]["model"], device)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    cdiff = cd.ColdDiffusion(L, t_max, device=device)
    prior = build_prior(cfg, L, t_max)
    states = cdiff.sample(model, prior, n=cfg.sample.n, T=cfg.sample.steps,
                          rng=np.random.default_rng(cfg.train.seed),
                          noise=cfg.sample.noise)

    base = icosphere(cfg.base_subdiv)
    out_dir = cfg.sample.out_dir
    os.makedirs(out_dir, exist_ok=True)
    meshes, mets, titles, hot = [], [], [], []
    for i, vec in enumerate(states):
        sigma = pack.packed_to_sigma(vec, L)
        V, F, info = decode(base.vertices, base.faces, sigma, w_cap=120,
                            return_info=True)
        save_obj(os.path.join(out_dir, f"bottle_{i:02d}.obj"), V, F)
        ns = metrics.nonstar_fraction(V, F)
        rec = {"W": metrics.willmore_energy(V, F), "genus": metrics.genus(V, F),
               "nonstar": ns, "amplitude": info["amplitude"]}
        meshes.append((V, F)); mets.append(rec)
        titles.append(f"W={rec['W']:.1f} s={info['amplitude']}")
        hot.append(ns > 0.02)
        print(f"  sample {i:02d}: W={rec['W']:.2f} genus={rec['genus']} "
              f"nonstar={ns:.3f} amp={info['amplitude']}")

    viz.contact_sheet(meshes, titles, os.path.join(out_dir, "contact_sheet.png"),
                      highlight=hot)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"t_max": t_max, "samples": mets}, fh, indent=2)
    print(f"[done] {len(meshes)} bottles -> {out_dir} "
          f"(mean non-star {np.mean([m['nonstar'] for m in mets]):.3f})")


if __name__ == "__main__":
    main()
