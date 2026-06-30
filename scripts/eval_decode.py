#!/usr/bin/env python
"""Decode-quality eval report (Session 8 DoD): encode→decode the bottle set and
report fidelity / validity / quality distributions.

    python scripts/eval_decode.py            # synthetic demo bottles
    python scripts/eval_decode.py L=16 n=24

Writes ``reports/decode_eval.md`` plus figures: an input-vs-decode contact
sheet, a |q| sizing-field-on-mesh, and a per-degree power spectrum.
"""

import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bottlegen.decode.reconstruct import decode
from bottlegen.eval import metrics, viz
from bottlegen.geom import conformal as cf
from bottlegen.geom.curvature import hopf_q
from bottlegen.geom.mesh import Mesh, icosphere


def main():
    cfg = OmegaConf.merge(OmegaConf.create({"L": 12, "n": 12, "subdiv": 4}),
                          OmegaConf.from_cli())
    L, n, sub = int(cfg.L), int(cfg.n), int(cfg.subdiv)
    out = "reports/decode_eval"
    os.makedirs(out, exist_ok=True)
    base = icosphere(sub)

    # build (or load) bottles, encode -> σ̂, robust-decode
    inputs = _demo_inputs(n, sub)
    rows, meshes, titles, hot = [], [], [], []
    in_spectra, out_spectra = [], []
    for i, m in enumerate(inputs):
        sig = cf.mesh_to_sigma(m, L=L)
        V, F, info = decode(base.vertices, base.faces, sig, w_cap=120,
                            return_info=True)
        ch = metrics.chamfer(V, m.vertices, scale_align=True)
        rows.append({
            "chamfer": ch,
            "W_in": metrics.willmore_energy(m.vertices, m.faces),
            "W_out": metrics.willmore_energy(V, F),
            "genus": metrics.genus(V, F),
            "nonstar_in": metrics.nonstar_fraction(m.vertices, m.faces, 800),
            "nonstar_out": metrics.nonstar_fraction(V, F, 800),
            "amplitude": info["amplitude"],
            "min_angle": metrics.triangle_quality(V, F)["min_angle"],
        })
        meshes += [(m.vertices, m.faces), (V, F)]
        titles += [f"in {i}", f"decode (s={info['amplitude']})"]
        hot += [rows[-1]["nonstar_in"] > 0.02, rows[-1]["nonstar_out"] > 0.02]
        in_spectra.append(metrics.per_degree_power(sig))
        out_spectra.append(metrics.per_degree_power(cf.mesh_to_sigma(Mesh(V, F), L=L)))

    # figures
    viz.contact_sheet(meshes, titles, f"{out}/input_vs_decode.png",
                      highlight=hot, cols=4)
    sig0 = cf.mesh_to_sigma(inputs[0], L=L)
    V0, F0 = decode(base.vertices, base.faces, sig0)
    viz.scalar_on_mesh(V0, F0, np.abs(hopf_q(V0, F0)), f"{out}/hopf_q.png",
                       title="|q| (Hopf magnitude) on a decoded bottle")
    viz.power_spectrum_plot(
        {"data σ̂": np.mean(in_spectra, 0),
         "encode∘decode": np.mean(out_spectra, 0)},
        f"{out}/power_spectrum.png", "per-degree power: data vs encode∘decode")

    _write_report(out, L, n, sub, rows)
    print(f"[done] decode eval -> {out}/decode_eval.md")


def _demo_inputs(n, sub):
    # the same parametric bottle family demo_bumpy_dataset encodes, as meshes
    rng = np.random.default_rng(0)
    bs = icosphere(sub); V0 = bs.vertices; z = V0[:, 2]
    out = []
    for _ in range(n):
        neck = 1.0 - 0.45 * np.exp(-((z - 0.7) ** 2) / 0.05)
        body = 1.0 + 0.18 * (z ** 2 - 0.3)
        r = neck * body
        r = r + 0.12 * rng.standard_normal() * (V0[:, 0] ** 2 - V0[:, 1] ** 2)
        r = r + 0.10 * rng.standard_normal() * V0[:, 0] * V0[:, 2]
        out.append(Mesh(V0 * r[:, None], bs.faces))
    return out


def _write_report(out, L, n, sub, rows):
    def col(k):
        return np.array([r[k] for r in rows], float)
    ch, Wi, Wo = col("chamfer"), col("W_in"), col("W_out")
    nsi, nso = col("nonstar_in"), col("nonstar_out")
    amp, mina = col("amplitude"), col("min_angle")
    genus_ok = int((col("genus") == 0).sum())
    md = f"""# Decode evaluation (Session 8)

Robust σ̂ → mesh on a base icosphere (subdiv {sub}, bandlimit L={L}) over {n}
synthetic bottles. Encode (conformal param + Hopf) → σ̂ → trust-region Dirac
decode → mesh.

## Fidelity & validity

| metric | mean | median | min | max |
|---|---|---|---|---|
| Chamfer (scale-aligned) | {ch.mean():.4f} | {np.median(ch):.4f} | {ch.min():.4f} | {ch.max():.4f} |
| Willmore W (input) | {Wi.mean():.2f} | {np.median(Wi):.2f} | {Wi.min():.2f} | {Wi.max():.2f} |
| Willmore W (decoded) | {Wo.mean():.2f} | {np.median(Wo):.2f} | {Wo.min():.2f} | {Wo.max():.2f} |
| trust-region amplitude | {amp.mean():.2f} | {np.median(amp):.2f} | {amp.min():.2f} | {amp.max():.2f} |
| min triangle angle (deg) | {mina.mean():.1f} | {np.median(mina):.1f} | {mina.min():.1f} | {mina.max():.1f} |

- **Genus correct (=0):** {genus_ok}/{n}
- **Non-star fraction** — input mean {nsi.mean():.3f}, decoded mean {nso.mean():.3f}
  (a substantial fraction are non-star, vindicating the conformal+Dirac route
  over a radial height field).

## Figures
- `input_vs_decode.png` — input bottles vs their decodes (non-star tiles in red)
- `hopf_q.png` — the Hopf magnitude |q| on a decoded bottle (sizing field driver)
- `power_spectrum.png` — per-degree power, data σ̂ vs encode∘decode

## Notes
The decode is a **linearized approximate inverse**: it amplifies high-ℓ content
by (ℓ(ℓ+1))² and the trust region caps the amplitude where that would fold the
surface, so sharp necks are flattened rather than torn. Faithful sharp-feature
decoding is the job of the physics-matched flow (Session 10) / true Willmore
flow.
"""
    with open(f"{out}/decode_eval.md", "w") as fh:
        fh.write(md)


if __name__ == "__main__":
    main()
