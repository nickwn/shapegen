"""Render the encode→SH→decode sweep from ``reports/sh_roundtrip.json``.

Log-y error against bandwidth for both encoders, with rows that hit the
iteration cap marked — those are iteration-limited, not bandwidth-limited, and
reading them as converged overstates what the bandwidth costs.

Usage:
    python -m scripts.plot_sh_roundtrip [--json reports/sh_roundtrip.json]
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="reports/sh_roundtrip.json")
    ap.add_argument("--out", default="reports/sh_roundtrip.png")
    args = ap.parse_args()

    d = json.load(open(args.json))
    rows = d["rows"]
    L = [r["L"] for r in rows]
    base = [r["baseline"] for r in rows]
    opt = [r["optimised"] for r in rows]
    capped = [r for r in rows if "LIMIT" in r.get("status", "")]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(L, base, "o-", label="baseline (λ route, projected)", color="#b0562a")
    ax.plot(L, opt, "s-", label="optimised (adjoint, at bandwidth)", color="#2a6fb0")
    if capped:
        ax.plot([r["L"] for r in capped], [r["optimised"] for r in capped],
                "s", mfc="none", mec="#2a6fb0", ms=13, mew=1.6,
                label="hit iteration cap (not converged)")
    ax.axhline(d["qc_distortion"], ls=":", color="grey", lw=1)
    ax.annotate(f"cMCF quasi-conformal distortion  {d['qc_distortion']:.4f}",
                (L[0], d["qc_distortion"]), textcoords="offset points",
                xytext=(4, 5), fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("SH bandwidth L")
    ax.set_ylabel("relative Procrustes shape error")
    ax.set_title(f"encode → SH → decode round trip  "
                 f"(bottle, n={d['n_vertices']}, resolvable to ℓ≈"
                 f"{d['resolvable_degree']:.0f})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
