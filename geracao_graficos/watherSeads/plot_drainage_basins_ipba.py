"""
plot_drainage_basins_ipba.py
----------------------------
Generates the scientific figures for the Invasion Percolation-Based Algorithm (IPBA)
drainage basins computed on 2 contiguous hydrological blocks, with endorheic basins
(no ocean outlet, excluded from the marine-litter analysis) shaded in hatched gray,
and the universal spherical basin-area scaling (Oliveira et al., 2019).
Strictly follows `AGENTS.md` and `diretrizes/`:
  - 100% English text
  - Colormap: 'plasma'
  - Gray world locator inset with red zoom box, placed in a free ocean corner
  - Vertical colorbar on the right
  - Exported in both 300 DPI PNG and vector PDF in `geracao_graficos/watherSeads/`
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Patch

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
from calculos.geometria_esferica import pixel_area_by_latitude, somar_areas_por_bacia
from regioes_plot import REGIOES_PLOT

RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
COR_ENDO = ListedColormap(["#b5b5b5"])


def _compute_basin_area_map(basin_grid: np.ndarray, gt: tuple):
    nrows, ncols = basin_grid.shape
    lats = gt[3] + (np.arange(nrows) + 0.5) * gt[5]
    row_areas = pixel_area_by_latitude(lats, abs(gt[1]), abs(gt[5]))
    b_ids, _, b_areas_km2 = somar_areas_por_bacia(np.where(basin_grid > 0, basin_grid, 0), row_areas)
    max_id = int(b_ids.max()) + 1 if b_ids.size else 1
    area_lut = np.full(max_id, np.nan, dtype=np.float64)
    area_lut[b_ids] = b_areas_km2
    return np.where(basin_grid > 0, area_lut[np.clip(basin_grid, 0, max_id - 1)], np.nan)


def _recortar(grid, gt, box):
    nrows, ncols = grid.shape
    lons = gt[0] + (np.arange(ncols) + 0.5) * gt[1]
    lats = gt[3] + (np.arange(nrows) + 0.5) * gt[5]
    cm = (lons >= box[0]) & (lons <= box[2])
    rm = (lats >= box[1]) & (lats <= box[3])
    sub = grid[np.ix_(rm, cm)]
    ext = (lons[cm][0] - gt[1] / 2, lons[cm][-1] + gt[1] / 2, lats[rm][-1] + gt[5] / 2, lats[rm][0] - gt[5] / 2)
    return sub, ext


def _desenhar_bacias(ax, grid, area_map, ext, passo_hachura=3):
    im = ax.imshow(area_map, extent=ext, origin="upper", cmap="plasma",
                   norm=LogNorm(vmin=10.0, vmax=7.0e6), interpolation="nearest", zorder=2)
    endo = grid < 0
    if endo.any():
        ax.imshow(np.where(endo, 1.0, np.nan), extent=ext, origin="upper", cmap=COR_ENDO,
                  interpolation="nearest", zorder=2)
        m = endo[::passo_hachura, ::passo_hachura].astype(float)
        xs = np.linspace(ext[0], ext[1], m.shape[1])
        ys = np.linspace(ext[3], ext[2], m.shape[0])
        ax.contourf(xs, ys, m, levels=[0.5, 1.5], colors="none", hatches=["/////"], zorder=3)
    big = np.where(grid > 0, area_map, 0.0) >= 25000.0
    divide = (grid != 0) & (big | endo) & ((grid != np.roll(grid, 1, axis=0)) | (grid != np.roll(grid, 1, axis=1)))
    ax.imshow(np.where(divide, 1.0, np.nan), extent=ext, origin="upper", cmap="Greys",
              vmin=0.0, vmax=1.2, alpha=0.45, interpolation="nearest", zorder=4)
    return im


def generate_ipba_basins_figure():
    set_scientific_style()
    df_bacias = pd.read_csv(RESULTS_DIR / "bacias_global_consolidado.csv")
    legenda_endo = Patch(facecolor="#b5b5b5", edgecolor="#555555", hatch="/////", label="Endorheic (no ocean outlet)")

    info, grid_full = ReadGeoTiff(RESULTS_DIR / "bacias_bloco_americas_ipba.tif")
    gt = info["GeoTransform"]
    zoom_box = (-125.0, 8.0, -60.0, 52.0)
    grid_zoom, ext = _recortar(grid_full, gt, zoom_box)
    gt_zoom = (ext[0], gt[1], 0.0, ext[3], 0.0, gt[5])
    area_map = _compute_basin_area_map(grid_zoom, gt_zoom)

    fig = plt.figure(figsize=(9.0, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#eef3f8")
    im = _desenhar_bacias(ax_map, grid_zoom, area_map, ext)
    format_geo_axes(ax_map, box=zoom_box, xticks=(-120, -105, -90, -75, -60), yticks=(10, 20, 30, 40, 50))
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("IPBA Drainage Basins by Spherical Area", pad=8)
    add_panel_label(ax_map, "a", x=-0.14, y=1.04)
    add_scale_bar(ax_map, km=1000, pos=(0.68, 0.06))
    add_vertical_colorbar(fig, im, ax_map, label="Drainage basin area $A$ (km$^2$)")
    add_world_inset(fig, ax_map, zoom_box=zoom_box, use_axes_coords=True, corner="auto")
    ax_map.legend(handles=[legenda_endo], loc="upper right", frameon=True, facecolor="#ffffff",
                  edgecolor="#cccccc", fontsize=5.8)

    ax_ccdf = fig.add_subplot(gs[0, 1])
    exo = df_bacias[~df_bacias["is_endorheic"]]
    cores = plt.get_cmap("plasma")(np.linspace(0.08, 0.90, len(REGIOES_PLOT)))
    for reg, col in zip(REGIOES_PLOT, cores):
        a = np.sort(exo.loc[exo["territorio_id"] == reg["id"], "area_km2"].to_numpy())[::-1]
        if a.size > 5:
            ax_ccdf.plot(a, np.arange(1, a.size + 1) / a.size, lw=1.2, color=col,
                         label=f"{reg['label']} ($n={a.size}$)", zorder=4)
    x_ref = np.logspace(3.6, 6.8, 50)
    ax_ccdf.plot(x_ref, 0.85 * (x_ref / x_ref[0]) ** (-0.90), color="#111111", ls="--", lw=1.2,
                 label="Universal law $\\tau \\approx 1.90$ (Oliveira et al.)", zorder=6)
    ax_ccdf.set_xscale("log")
    ax_ccdf.set_yscale("log")
    ax_ccdf.set_xlim(3.0e3, 1.0e7)
    ax_ccdf.set_ylim(1e-3, 1.4)
    ax_ccdf.set_xlabel("Spherical Basin Area $A_0$ (km$^2$)")
    ax_ccdf.set_ylabel("Cumulative Probability $P(A \\geq A_0)$")
    ax_ccdf.set_title("Exorheic Major Basins ($P = 0.90$)", pad=8)
    ax_ccdf.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
    ax_ccdf.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=5.6)
    add_panel_label(ax_ccdf, "b", x=-0.16, y=1.04)

    save_scientific_figure(fig, str(THIS_DIR / "ipba_drainage_basins_universal_scaling"), dpi=300)
    plt.close(fig)

    fig6, axes6 = plt.subplots(2, 3, figsize=(11.6, 7.0))
    fig6.subplots_adjust(wspace=0.46, hspace=0.40)
    for idx_t, (ax_r, reg) in enumerate(zip(axes6.ravel(), REGIOES_PLOT)):
        t_info, t_grid = ReadGeoTiff(RESULTS_DIR / f"bacias_{reg['id']}_ipba.tif")
        t_gt = t_info["GeoTransform"]
        sub, ext_r = _recortar(t_grid, t_gt, reg["box"])
        gt_r = (ext_r[0], t_gt[1], 0.0, ext_r[3], 0.0, t_gt[5])
        area_r = _compute_basin_area_map(sub, gt_r)
        ax_r.set_facecolor("#eef3f8")
        im_r = _desenhar_bacias(ax_r, sub, area_r, ext_r, passo_hachura=4)
        format_geo_axes(ax_r, box=reg["box"])
        ax_r.set_title(f"{reg['label']}: IPBA basins", fontsize=8.0, pad=6)
        add_panel_label(ax_r, chr(ord("a") + idx_t), x=-0.16, y=1.06)
        add_vertical_colorbar(fig6, im_r, ax_r, label="Basin area $A$ (km$^2$)", shrink=0.82)
        add_world_inset(fig6, ax_r, zoom_box=reg["box"], use_axes_coords=True, corner="auto")
    fig6.legend(handles=[legenda_endo], loc="lower center", ncol=1, frameon=True, facecolor="#ffffff",
                edgecolor="#cccccc", fontsize=7.0, bbox_to_anchor=(0.5, 0.0))
    save_scientific_figure(fig6, str(THIS_DIR / "ipba_drainage_basins_6regions"), dpi=300)
    plt.close(fig6)


if __name__ == "__main__":
    generate_ipba_basins_figure()
