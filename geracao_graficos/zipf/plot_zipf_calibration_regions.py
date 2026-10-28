"""
plot_zipf_calibration_regions.py
--------------------------------
Generates the scientific figures of the dynamic Zipf calibration of the City Clustering
Algorithm (CCA) for each of the 6 demographic regions and a 6-region comparative panel:
  (a) rank-size OLS exponent alpha(D, L) over the full (D, L) sweep,
  (b) coefficient of determination R^2(D, L) with the R^2 >= 0.98 admissible region,
  (c) rank-size distribution S(r) of the selected CCA clusters.
The selected pair (D*, L*) minimizes |alpha - 1| subject to R^2 >= 0.98 and L >= L_min.
Strictly follows `AGENTS.md` and `diretrizes/` (English text, 'plasma', vertical colorbars,
300 DPI PNG + vector PDF in `geracao_graficos/zipf/`).
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for p in (REPO_ROOT, THIS_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diretrizes.templates.scientific_plot_utils import (
    set_scientific_style,
    add_panel_label,
    add_vertical_colorbar,
    save_scientific_figure,
)
from regioes_plot import REGIOES_PLOT

RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
R2_MIN = 0.98


def _carregar(tid):
    z = np.load(RESULTS_DIR / f"zipf_matriz_{tid}.npz")
    with open(RESULTS_DIR / f"cca_calibracao_{tid}.json", "r", encoding="utf-8") as f:
        cal = json.load(f)
    return z, cal


def _extent(D_grade, L_grade):
    dx = D_grade[1] - D_grade[0]
    dy = L_grade[1] - L_grade[0]
    return (D_grade[0] - dx / 2, D_grade[-1] + dx / 2, L_grade[0] - dy / 2, L_grade[-1] + dy / 2)


def _mascarar_sub_pixel(ax, z, cal, matrizes):
    """Hatches the inadmissible band L < L_min (sub-pixel linking distance) and hides its values."""
    L_grade = z["l"]
    dy = L_grade[1] - L_grade[0]
    L_min = float(cal["L_MIN_metros"])
    inval = L_grade < L_min
    ax.axhspan(L_grade[0] - dy / 2, L_grade[inval].max() + dy / 2 if inval.any() else L_grade[0] - dy / 2,
               facecolor="#d0d0d0", edgecolor="#777777", hatch="///", lw=0.0, zorder=3)
    return [np.where(inval[:, None], np.nan, m) for m in matrizes]


def _painel_alpha(fig, ax, z, cal, compacto=False):
    D_grade, L_grade = z["D"], z["l"]
    alpha, r2 = _mascarar_sub_pixel(ax, z, cal, [z["alpha_ols100"], z["r2_ols100"]])
    im = ax.imshow(alpha, origin="lower", extent=_extent(D_grade, L_grade), aspect="auto",
                   cmap="plasma", norm=Normalize(vmin=0.6, vmax=1.6), interpolation="nearest")
    add_vertical_colorbar(fig, im, ax, label="Rank-size exponent $\\alpha$ ($N=100$)", shrink=0.84 if compacto else 1.0)
    if np.isfinite(alpha).sum() > 4 and np.nanmin(alpha) < 1.0 < np.nanmax(alpha):
        ax.contour(D_grade, L_grade, alpha, levels=[1.0], colors="white", linewidths=1.2)
    if np.isfinite(r2).sum() > 4 and np.nanmin(r2) < R2_MIN < np.nanmax(r2):
        ax.contour(D_grade, L_grade, r2, levels=[R2_MIN], colors="#00f5d4", linewidths=1.0, linestyles="--")
    ax.plot(float(cal["D_star_hab_km2"]), float(cal["l_metros"]), marker="*", ms=11 if not compacto else 9,
            mfc="#00f5d4", mec="#111111", mew=0.9, ls="none")
    ax.set_xlabel("Density threshold $D$ (people / km$^2$)" if not compacto else "$D$ (people / km$^2$)")
    ax.set_ylabel("Distance threshold $L$ (m)" if not compacto else "$L$ (m)")


def generate_zipf_calibration_figures():
    set_scientific_style()

    for reg in REGIOES_PLOT:
        tid = reg["id"]
        if not (RESULTS_DIR / f"zipf_matriz_{tid}.npz").exists():
            continue
        z, cal = _carregar(tid)
        D_grade, L_grade = z["D"], z["l"]
        pop_cidades = np.sort(z["pop_cidades"][z["pop_cidades"] > 0])[::-1]

        fig, (ax_a, ax_r2, ax_rank) = plt.subplots(1, 3, figsize=(13.2, 3.9))
        fig.subplots_adjust(wspace=0.55, left=0.06, right=0.96, top=0.84, bottom=0.16)

        _painel_alpha(fig, ax_a, z, cal)
        ax_a.set_title(f"{reg['label']}: Zipf exponent $\\alpha(D, L)$", pad=8)
        add_panel_label(ax_a, "a", x=-0.18, y=1.05)

        (r2,) = _mascarar_sub_pixel(ax_r2, z, cal, [z["r2_ols100"]])
        im2 = ax_r2.imshow(r2, origin="lower", extent=_extent(D_grade, L_grade), aspect="auto",
                           cmap="plasma", norm=Normalize(vmin=0.90, vmax=1.0), interpolation="nearest")
        add_vertical_colorbar(fig, im2, ax_r2, label="Coefficient of determination $R^2$")
        if np.isfinite(r2).sum() > 4 and np.nanmin(r2) < R2_MIN < np.nanmax(r2):
            ax_r2.contour(D_grade, L_grade, r2, levels=[R2_MIN], colors="#00f5d4", linewidths=1.0, linestyles="--")
        ax_r2.plot(float(cal["D_star_hab_km2"]), float(cal["l_metros"]), marker="*", ms=11,
                   mfc="#00f5d4", mec="#111111", mew=0.9, ls="none")
        ax_r2.set_xlabel("Density threshold $D$ (people / km$^2$)")
        ax_r2.set_ylabel("Distance threshold $L$ (m)")
        ax_r2.set_title("Rank-size fit quality $R^2(D, L)$", pad=8)
        add_panel_label(ax_r2, "b", x=-0.18, y=1.05)

        ranks = np.arange(1, len(pop_cidades) + 1, dtype=np.float64)
        idx_sub = np.unique(np.geomspace(1, len(pop_cidades), num=min(350, len(pop_cidades)), dtype=int) - 1)
        sc = ax_rank.scatter(ranks[idx_sub], pop_cidades[idx_sub], c=pop_cidades[idx_sub], cmap="plasma",
                             norm=LogNorm(vmin=max(1.0, pop_cidades.min()), vmax=pop_cidades.max()),
                             s=14, alpha=0.9, edgecolors="none", zorder=4)
        add_vertical_colorbar(fig, sc, ax_rank, label="Cluster population $S_r$ (people)")
        r_ref = np.logspace(0, np.log10(min(100, len(pop_cidades))), 50)
        ax_rank.plot(r_ref, pop_cidades[0] * r_ref ** (-1.0), color="#111111", ls="--", lw=1.1,
                     label="Zipf law $S_r \\propto r^{-1}$", zorder=5)
        ax_rank.plot(r_ref, pop_cidades[0] * r_ref ** (-float(cal["alpha_ols"])), color="#d90429", lw=1.1,
                     label=f"OLS fit $\\alpha={cal['alpha_ols']:.2f}$, $R^2={cal['r2_ols']:.3f}$", zorder=6)
        ax_rank.set_xscale("log")
        ax_rank.set_yscale("log")
        ax_rank.set_xlabel("Cluster rank $r$")
        ax_rank.set_ylabel("Cluster population $S_r$ (people)")
        ax_rank.set_title(
            f"$D^*={int(cal['D_star_hab_km2'])}$ people/km$^2$, $L^*={int(cal['l_metros'])}$ m ($N={len(pop_cidades):,}$)",
            pad=8,
        )
        ax_rank.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
        ax_rank.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=6.0)
        add_panel_label(ax_rank, "c", x=-0.18, y=1.05)

        save_scientific_figure(fig, str(THIS_DIR / f"zipf_calibration_{tid}"), dpi=300)
        plt.close(fig)

    fig6, axes6 = plt.subplots(2, 3, figsize=(11.6, 6.6))
    fig6.subplots_adjust(wspace=0.50, hspace=0.50, left=0.07, right=0.95, top=0.90, bottom=0.09)
    for idx_t, (ax_r, reg) in enumerate(zip(axes6.ravel(), REGIOES_PLOT)):
        z, cal = _carregar(reg["id"])
        _painel_alpha(fig6, ax_r, z, cal, compacto=True)
        ax_r.set_title(
            f"{reg['label']}\n$D^*={int(cal['D_star_hab_km2'])}$, $L^*={int(cal['l_metros'])}$ m, "
            f"$\\alpha={cal['alpha_ols']:.2f}$, $R^2={cal['r2_ols']:.3f}$",
            fontsize=7.4, pad=5,
        )
        add_panel_label(ax_r, chr(ord("a") + idx_t), x=-0.20, y=1.04)
    save_scientific_figure(fig6, str(THIS_DIR / "zipf_calibration_6regions"), dpi=300)
    plt.close(fig6)


if __name__ == "__main__":
    generate_zipf_calibration_figures()
