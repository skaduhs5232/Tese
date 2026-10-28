"""
plot_noaa_global_sampling.py
----------------------------
Generates the scientific figures for the NOAA NCEI Marine Microplastics Hub database
(GeoJSON export, 29,076 records; 12,088 coastal records after excluding 'Ocean water':
Nurdle Patrol, Beach and Ocean sediment) and the 6 regional sampling maps,
strictly following `AGENTS.md` and `diretrizes/`:
  - 100% English text
  - Colormap: 'plasma'
  - World map locator inset in the bottom-left corner (`lower left`) in neutral gray (#d0d0d0)
  - Vertical colorbar on the right (`orientation='vertical'`, `location='right'`)
  - Exported in both 300 DPI PNG and vector PDF in `geracao_graficos/noaa/`
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from shapely.geometry import shape

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for p in (REPO_ROOT, THIS_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diretrizes.templates.scientific_plot_utils import (
    WORLD_GEOJSON,
    set_scientific_style,
    add_panel_label,
    add_scale_bar,
    add_vertical_colorbar,
    add_world_inset,
    format_geo_axes,
    save_scientific_figure,
)
from notebooks.executar_pipeline_global import carregar_noaa_limpo
from regioes_plot import REGIOES_PLOT


def _draw_vector_land(ax, land_color="#ebebeb", border_color="#777777", lw=0.5):
    if os.path.exists(WORLD_GEOJSON):
        with open(WORLD_GEOJSON, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        for feature in world_data["features"]:
            geom = shape(feature["geometry"])
            geoms = getattr(geom, "geoms", [geom])
            for poly in geoms:
                xs, ys = poly.exterior.xy
                ax.fill(xs, ys, fc=land_color, ec=border_color, lw=lw, zorder=2)


def generate_noaa_sampling_figure():
    set_scientific_style()

    noaa_csv = REPO_ROOT / "dados" / "noaa_microplastics.csv"
    df = carregar_noaa_limpo(noaa_csv, apenas_costeiro=True)

    zoom_box = (-112.0, 6.0, -64.0, 46.0)
    sub_zoom = df[
        (df["lon"] >= zoom_box[0])
        & (df["lon"] <= zoom_box[2])
        & (df["lat"] >= zoom_box[1])
        & (df["lat"] <= zoom_box[3])
        & (df["medida"] > 0)
    ].sort_values("medida", ascending=True)

    fig = plt.figure(figsize=(9.0, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46)

    # Panel (a): Zoomed Coastal & Beach Microplastic Concentration Map + Bottom-Left Inset + Vertical Right Colorbar
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#eef3f8")
    _draw_vector_land(ax_map, land_color="#ececec", border_color="#666666", lw=0.55)

    sc = ax_map.scatter(
        sub_zoom["lon"].to_numpy(),
        sub_zoom["lat"].to_numpy(),
        c=np.clip(sub_zoom["medida"].to_numpy(), 1e-2, 1e4),
        cmap="plasma",
        norm=LogNorm(vmin=1e-2, vmax=1e4),
        s=10.0,
        alpha=0.88,
        edgecolors="none",
        zorder=5,
        rasterized=True,
    )

    format_geo_axes(
        ax_map,
        box=zoom_box,
        xticks=(-110, -100, -90, -80, -70),
        yticks=(10, 20, 30, 40),
    )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("On-Land & Coastal Sampling (NOAA)", pad=8)
    add_panel_label(ax_map, "a", x=-0.16, y=1.05)
    add_scale_bar(ax_map, km=500, pos=(0.68, 0.06))
    add_vertical_colorbar(fig, sc, ax_map, label="Coastal microplastic abundance $M$")

    add_world_inset(
        fig,
        ax_map,
        zoom_box=zoom_box,
        use_axes_coords=True,
        corner="auto",
    )

    # Panel (b): Empirical CCDF Across Coastal/Shoreline Physical Measurement Regimes
    ax_dist = fig.add_subplot(gs[0, 1])
    cmap_plasma = plt.get_cmap("plasma")
    regimes = [
        ("pieces/10 mins", "Beach Nurdle Patrol (pieces / 10 min)", cmap_plasma(0.15)),
        ("pieces/m3", "Beach Survey (pieces / m$^3$)", cmap_plasma(0.55)),
        ("pieces kg-1 d.w.", "Coastal Sediment (pieces / kg d.w.)", cmap_plasma(0.85)),
    ]

    for un_raw, label_en, col in regimes:
        vals = np.sort(df.loc[(df["unidade"] == un_raw) & (df["medida"] > 0), "medida"].to_numpy())
        n_total_un = int((df["unidade"] == un_raw).sum())
        if vals.size > 2:
            ccdf = 1.0 - np.arange(vals.size, dtype=np.float64) / vals.size
            idx_plot = np.unique(np.geomspace(1, vals.size, num=min(250, vals.size), dtype=int) - 1)
            ax_dist.plot(
                vals[idx_plot],
                ccdf[idx_plot],
                color=col,
                lw=1.5,
                label=f"{label_en}\n($N={n_total_un:,}$, positive $n={vals.size:,}$)",
                zorder=4,
            )

    ax_dist.set_xscale("log")
    ax_dist.set_yscale("log")
    ax_dist.set_xlim(1e-3, 1e6)
    ax_dist.set_ylim(2e-4, 1.3)
    ax_dist.set_xlabel("Microplastic Measurement $m$ (unit-specific)")
    ax_dist.set_ylabel("Exceedance Probability $P(M \\geq m)$")
    ax_dist.set_title("Global Distribution by Sampling Regime", pad=8)
    ax_dist.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
    ax_dist.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=5.8)
    add_panel_label(ax_dist, "b", x=-0.16, y=1.04)

    out_base = str(THIS_DIR / "noaa_global_microplastic_regimes")
    save_scientific_figure(fig, out_base, dpi=300)
    plt.close(fig)

    # Generate 6-region NOAA coastal/marine sampling overview (each with bottom-left world inset & vertical right colorbar)
    fig6, axes6 = plt.subplots(2, 3, figsize=(11.4, 6.8))
    fig6.subplots_adjust(wspace=0.44, hspace=0.38)
    for idx_t, (ax_r, reg) in enumerate(zip(axes6.ravel(), REGIOES_PLOT)):
        t_box = reg["box"]
        sub_t = df[
            (df["lon"] >= t_box[0])
            & (df["lon"] <= t_box[2])
            & (df["lat"] >= t_box[1])
            & (df["lat"] <= t_box[3])
            & (df["medida"] > 0)
        ].sort_values("medida", ascending=True)

        ax_r.set_facecolor("#eef3f8")
        _draw_vector_land(ax_r, land_color="#ececec", border_color="#666666", lw=0.45)
        sc_r = ax_r.scatter(
            sub_t["lon"].to_numpy(),
            sub_t["lat"].to_numpy(),
            c=np.clip(sub_t["medida"].to_numpy(), 1e-3, 1e4),
            cmap="plasma",
            norm=LogNorm(vmin=1e-3, vmax=1e4),
            s=8.0,
            alpha=0.85,
            edgecolors="none",
            zorder=5,
            rasterized=True,
        )
        format_geo_axes(ax_r, box=t_box)
        ax_r.set_title(f"{reg['label']} ($n={len(sub_t):,}$)", fontsize=8.0, pad=6)
        add_panel_label(ax_r, chr(ord("a") + idx_t), x=-0.16, y=1.06)
        add_vertical_colorbar(fig6, sc_r, ax_r, label="Abundance $M$", shrink=0.82)
        add_world_inset(
            fig6,
            ax_r,
            zoom_box=t_box,
            use_axes_coords=True,
            corner="auto",
        )

    save_scientific_figure(fig6, str(THIS_DIR / "noaa_sampling_6regions"), dpi=300)
    plt.close(fig6)


if __name__ == "__main__":
    generate_noaa_sampling_figure()
