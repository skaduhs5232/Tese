"""
plot_global_islands_clusters.py
-------------------------------
Generates the scientific figures for the GEBCO 2026 site-percolation height clusters (h > 0 m)
the 2 contiguous hydrological blocks used by the IPBA (Americas; Afro-Eurasia + Oceania)
and the 6 regional elevation maps, strictly following `AGENTS.md` and `diretrizes/`:
  - 100% English labels, titles, and annotations
  - Colormap: 'plasma'
  - World map locator inset in the bottom-left corner (`lower left`) in neutral gray (#d0d0d0)
  - Vertical colorbar on the right (`orientation='vertical'`, `location='right'`)
  - Exported in both 300 DPI PNG and vector PDF in `geracao_graficos/gebco/`
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for p in (REPO_ROOT, THIS_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diretrizes.templates.scientific_plot_utils import (
    set_scientific_style,
    add_panel_label,
    add_scale_bar,
    add_vertical_colorbar,
    add_world_inset,
    format_geo_axes,
    save_scientific_figure,
)
from notebooks.gebco_cluster_dvider.geotiff.geotiff import ReadGeoTiff
from regioes_plot import REGIOES_PLOT


def generate_gebco_clusters_figure():
    set_scientific_style()

    catalogo_path = (
        REPO_ROOT / "notebooks" / "gebco_cluster_dvider" / "datasets" / "processed" / "catalogo_territorios_ilhas.json"
    )
    gebco_path = (
        REPO_ROOT / "notebooks" / "gebco_cluster_dvider" / "datasets" / "processed" / "gebco_2026_sub_ice_geotiff.tif"
    )

    with open(catalogo_path, "r", encoding="utf-8") as f:
        catalogo = json.load(f)

    info, arr = ReadGeoTiff(gebco_path)
    gt = info["GeoTransform"]
    nrows, ncols = arr.shape
    lons = gt[0] + (np.arange(ncols) + 0.5) * gt[1]
    lats = gt[3] + (np.arange(nrows) + 0.5) * gt[5]

    zoom_box = (-125.0, -5.0, -55.0, 52.0)
    col_mask = (lons >= zoom_box[0]) & (lons <= zoom_box[2])
    row_mask = (lats >= zoom_box[1]) & (lats <= zoom_box[3])

    arr_zoom = arr[np.ix_(row_mask, col_mask)].astype(np.float64)
    elev_land = np.where(arr_zoom > 0, np.maximum(1.0, arr_zoom), np.nan)

    fig = plt.figure(figsize=(9.0, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46)

    # Panel (a): Zoomed Elevation Cluster Map + Bottom-Left Grayscale World Inset + Vertical Right Colorbar
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#eef3f8")

    im = ax_map.imshow(
        elev_land,
        extent=(zoom_box[0], zoom_box[2], zoom_box[1], zoom_box[3]),
        origin="upper",
        cmap="plasma",
        norm=LogNorm(vmin=1.0, vmax=4500.0),
        interpolation="nearest",
        zorder=2,
    )

    format_geo_axes(
        ax_map,
        box=zoom_box,
        xticks=(-120, -100, -80, -60),
        yticks=(0, 15, 30, 45),
    )
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Percolated Land Clusters ($h_i > 0$ m)", pad=8)
    add_panel_label(ax_map, "a", x=-0.14, y=1.04)
    add_scale_bar(ax_map, km=1000, pos=(0.68, 0.06))
    add_vertical_colorbar(fig, im, ax_map, label="Elevation above sea level (m)")

    add_world_inset(
        fig,
        ax_map,
        zoom_box=zoom_box,
        use_axes_coords=True,
        corner="auto",
    )

    # Panel (b): Spherical Area Rank-Size Distribution of Top Percolated Landmasses + Vertical Right Colorbar
    ax_rank = fig.add_subplot(gs[0, 1])
    top_ilhas = catalogo["top_ilhas_percoladas"]
    ranks = np.array([item["rank"] for item in top_ilhas], dtype=np.float64)
    areas_km2 = np.array([item["area_esferica_km2"] for item in top_ilhas], dtype=np.float64)

    ax_rank.plot(ranks, areas_km2, color="#444444", lw=1.0, ls="--", alpha=0.7, zorder=2)
    sc_rank = ax_rank.scatter(
        ranks,
        areas_km2,
        c=areas_km2,
        cmap="plasma",
        norm=LogNorm(vmin=areas_km2.min(), vmax=areas_km2.max()),
        s=36,
        edgecolors="#222222",
        linewidths=0.6,
        zorder=4,
    )
    add_vertical_colorbar(fig, sc_rank, ax_rank, label="Spherical cluster area $A_r$ (km$^2$)")

    ax_rank.set_xscale("log")
    ax_rank.set_yscale("log")
    ax_rank.set_xlim(0.75, 42.0)
    ax_rank.set_ylim(7e4, 1.6e8)
    ax_rank.set_xlabel("Percolation Cluster Rank $r$")
    ax_rank.set_ylabel("Spherical Cluster Area $A_r$ (km$^2$)")
    ax_rank.set_title("Spherical Area of Global Land Clusters", pad=8)
    ax_rank.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
    add_panel_label(ax_rank, "b", x=-0.16, y=1.04)

    island_labels = [
        (0, "Afro-Eurasia", (1.28, 0.90)),
        (1, "Americas", (1.30, 0.85)),
        (2, "Antarctica", (1.30, 0.85)),
        (3, "Australia", (1.30, 0.85)),
        (4, "Greenland", (1.30, 0.85)),
        (5, "New Guinea", (1.30, 1.10)),
        (7, "Madagascar", (1.32, 0.95)),
    ]
    for idx_i, name_en, (fx, fy) in island_labels:
        if idx_i < len(ranks):
            r_val = ranks[idx_i]
            a_val = areas_km2[idx_i]
            ax_rank.annotate(
                name_en,
                xy=(r_val, a_val),
                xytext=(r_val * fx, a_val * fy),
                fontsize=6.2,
                color="#222222",
                arrowprops=dict(arrowstyle="-", color="#666666", lw=0.5, shrinkA=1, shrinkB=2),
                zorder=6,
            )

    total_area_mkm2 = catalogo["area_emersa_global_km2"] / 1e6
    n_clusters = catalogo["total_clusters_globais"]
    summary_txt = (
        f"Spherical Area (Eqs. 3–4)\n"
        f"Clusters ($h > 0$ m): {n_clusters:,}\n"
        f"Total: {total_area_mkm2:.2f}$\\times 10^6$ km$^2$"
    )
    ax_rank.text(
        0.03,
        0.04,
        summary_txt,
        transform=ax_rank.transAxes,
        fontsize=6.0,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92),
        zorder=8,
    )

    out_base = str(THIS_DIR / "gebco_global_islands_clusters")
    save_scientific_figure(fig, out_base, dpi=300)
    plt.close(fig)

    blocos = []
    for bloco in catalogo["blocos_hidrologicos"]:
        b_info, b_dem = ReadGeoTiff(REPO_ROOT / bloco["arquivo_tif"])
        b_gt = b_info["GeoTransform"]
        b_h, b_w = b_dem.shape
        b_box = (b_gt[0], b_gt[3] + b_h * b_gt[5], b_gt[0] + b_w * b_gt[1], b_gt[3])
        blocos.append((bloco, b_dem, b_gt, b_box))

    # Hydrological blocks: IPBA runs on 2 contiguous landmass blocks so that no basin is cut by a box edge
    larguras = [b[3][2] - b[3][0] for b in blocos]
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": larguras})
    fig2.subplots_adjust(wspace=0.22)
    for idx_b, (ax_b, (bloco, b_dem, b_gt, b_box)) in enumerate(zip(axes2, blocos)):
        elev_b = np.where(b_dem > 0, np.maximum(1.0, b_dem.astype(np.float64)), np.nan)
        ax_b.set_facecolor("#eef3f8")
        im_b = ax_b.imshow(elev_b, extent=(b_box[0], b_box[2], b_box[1], b_box[3]), origin="upper", cmap="plasma",
                           norm=LogNorm(vmin=1.0, vmax=4500.0), interpolation="nearest", zorder=2)
        format_geo_axes(ax_b, box=b_box)
        ax_b.set_title(
            f"{bloco['nome_en']}\n({bloco['n_massas_terra']:,} landmasses, "
            f"{bloco['area_esferica_km2'] / 1e6:.2f} M km$^2$)",
            fontsize=8.0, pad=6,
        )
        add_panel_label(ax_b, chr(ord("a") + idx_b), x=-0.10, y=1.05)
        add_vertical_colorbar(fig2, im_b, ax_b, label="Elevation (m)", shrink=0.80)
        add_world_inset(fig2, ax_b, zoom_box=b_box, use_axes_coords=True, corner="auto")
    save_scientific_figure(fig2, str(THIS_DIR / "gebco_hydrological_blocks"), dpi=300)
    plt.close(fig2)

    bloco_por_regiao = {
        "americas_norte_central_caribe": 0, "america_do_sul": 0, "europa_mediterraneo": 1,
        "africa_madagascar": 1, "asia_continental_arquipelagos": 1, "oceania_australia": 1,
    }
    fig6, axes6 = plt.subplots(2, 3, figsize=(11.4, 6.8))
    fig6.subplots_adjust(wspace=0.44, hspace=0.38)
    for idx_t, (ax_r, reg) in enumerate(zip(axes6.ravel(), REGIOES_PLOT)):
        _, b_dem, b_gt, _ = blocos[bloco_por_regiao[reg["id"]]]
        lons = b_gt[0] + (np.arange(b_dem.shape[1]) + 0.5) * b_gt[1]
        lats = b_gt[3] + (np.arange(b_dem.shape[0]) + 0.5) * b_gt[5]
        box = reg["box"]
        cm = (lons >= box[0]) & (lons <= box[2])
        rm = (lats >= box[1]) & (lats <= box[3])
        sub = b_dem[np.ix_(rm, cm)]
        ext = (lons[cm][0], lons[cm][-1], lats[rm][-1], lats[rm][0])
        elev_t = np.where(sub > 0, np.maximum(1.0, sub.astype(np.float64)), np.nan)

        ax_r.set_facecolor("#eef3f8")
        im_r = ax_r.imshow(elev_t, extent=ext, origin="upper", cmap="plasma",
                           norm=LogNorm(vmin=1.0, vmax=4500.0), interpolation="nearest", zorder=2)
        format_geo_axes(ax_r, box=box)
        ax_r.set_title(reg["label"], fontsize=8.0, pad=6)
        add_panel_label(ax_r, chr(ord("a") + idx_t), x=-0.16, y=1.06)
        add_vertical_colorbar(fig6, im_r, ax_r, label="Elevation (m)", shrink=0.82)
        add_world_inset(fig6, ax_r, zoom_box=box, use_axes_coords=True, corner="auto")

    save_scientific_figure(fig6, str(THIS_DIR / "gebco_elevation_6regions"), dpi=300)
    plt.close(fig6)


if __name__ == "__main__":
    generate_gebco_clusters_figure()
