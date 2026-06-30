"""Rendering helpers (matplotlib, Agg backend) for eval figures."""

from __future__ import annotations

import numpy as np

__all__ = ["contact_sheet", "scalar_on_mesh", "power_spectrum_plot"]


def _ax_mesh(ax, V, F, color="#6699cc", facecolor=None):
    if facecolor is not None:
        import matplotlib.tri as mtri  # noqa: F401
        tri = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                              linewidth=0, antialiased=True)
        tri.set_array(facecolor)
    else:
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color=color,
                        edgecolor="none")
    ax.set_axis_off()
    m = np.abs(V).max()
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.view_init(elev=12, azim=20)


def contact_sheet(meshes, titles, path, highlight=None, cols=None):
    """Grid of decoded meshes with per-tile titles. ``highlight`` is a list of
    bool (red title) e.g. for non-star samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(meshes)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    for i, (V, F) in enumerate(meshes):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _ax_mesh(ax, V, F)
        hot = highlight is not None and highlight[i]
        ax.set_title(titles[i], fontsize=8, color="crimson" if hot else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def scalar_on_mesh(V, F, scalar, path, title="", cmap="viridis", clip=(2, 98)):
    """Render a per-vertex scalar field on a mesh (e.g. |q| or the sizing
    field). Colors are clipped to the ``clip`` percentile range so outlier
    spikes (e.g. creases) don't flatten the colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    face_c = scalar[F].mean(1)
    vmin, vmax = np.percentile(face_c, clip)
    tri = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, cmap=cmap,
                          linewidth=0, antialiased=True)
    tri.set_array(face_c)
    tri.set_clim(vmin, vmax)
    ax.set_axis_off()
    m = np.abs(V).max()
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.view_init(elev=12, azim=20)
    ax.set_title(title, fontsize=10)
    fig.colorbar(tri, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def power_spectrum_plot(spectra: dict, path, title="per-degree power"):
    """Overlay per-degree power spectra (dict label -> array over ℓ)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, p in spectra.items():
        ax.semilogy(np.arange(len(p)), np.maximum(p, 1e-12), marker="o",
                    ms=3, label=label)
    ax.set_xlabel("degree ℓ"); ax.set_ylabel("power")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
