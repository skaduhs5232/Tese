"""
generate_cca_ipba_gifs.py
-------------------------
Generates two scientific animated GIFs in `geracao_graficos/gifs/` illustrating the
step-by-step construction of the two core algorithms from the advisor's repositories:
  1. `cca_urban_clustering_evolution.gif` (`notebooks/CCA`):
     Shows the City Clustering Algorithm (CCA) activating and merging urban clusters
     across the density threshold cascade D (from 2,000 people/km^2 down to the dynamically
     calibrated D* of North & Central America, at the calibrated distance L*)
     and the emergence of Zipf's rank-size power law (S_r ~ r^-1).
  2. `ipba_watershed_percolation_evolution.gif` (`notebooks/IPBA`):
     Shows the Invasion Percolation-Based Algorithm (IPBA) progressively invading the
     continental topography from the coastal sinks S_k (sea level 0 m up to the continental
     divides > 3,500 m) and the emergence of the universal drainage basin area law (Oliveira et al., 2019).

Strictly follows all rules in `AGENTS.md` and `diretrizes/`:
  - 100% English text inside all frames
  - Colormap: 'plasma'
  - Grayscale world map locator inset in the bottom-left corner (`lower left`)
  - Vertical colorbar on the right (`orientation='vertical'`, `location='right'`)
"""

import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
CCA_DIR = REPO_ROOT / "notebooks" / "CCA"
IPBA_DIR = REPO_ROOT / "notebooks" / "IPBA"

for p in (REPO_ROOT, CCA_DIR, IPBA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cca.city_clustering import cca_grid as cca_grid_orientador
from diretrizes.templates.scientific_plot_utils import (
    set_scientific_style,
    add_panel_label,
    add_scale_bar,
    add_vertical_colorbar,
    add_world_inset,
    format_geo_axes,
)
from notebooks.gebco_cluster_dvider.geotiff.geotiff import ReadGeoTiff
from notebooks.executar_pipeline_global import ler_janela_gpw, codigo_regiao
from calculos import (
    pixel_area_by_latitude,
    somar_areas_por_bacia,
    calcular_densidade_gpw,
)
from calculos.alometria_estatistica import alpha_ols_rank_tamanho


def _fig_to_pil(fig, dpi: int = 160) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return im


def generate_cca_gif():
    set_scientific_style()
    THIS_DIR.mkdir(parents=True, exist_ok=True)

    gpw_path = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
    zoom_box = (-125.0, 7.0, -65.0, 50.0)
    with open(REPO_ROOT / "notebooks" / "results" / "cca_calibracao_americas_norte_central_caribe.json", "r", encoding="utf-8") as f:
        cal = json.load(f)
    d_opt = int(cal["D_star_hab_km2"])
    l_km = float(cal["l_metros"]) / 1000.0

    # Same 1 arc-min aggregation as the pipeline calibration
    pop_gpw, lats, lons, gt = ler_janela_gpw(gpw_path, zoom_box, fator_agregacao=2)
    lon2d, lat2d = np.meshgrid(lons, lats)
    pop_gpw[codigo_regiao(lat2d, lon2d) != 0] = 0.0

    dlon, dlat = abs(gt[1]), abs(gt[5])
    _, row_areas_km2 = calcular_densidade_gpw(pop_gpw, lats, dlon, dlat)
    ext = (lons.min() - dlon / 2, lons.max() + dlon / 2, lats.min() - dlat / 2, lats.max() + dlat / 2)

    d_steps = sorted({2000, 1500, 1000, 700, 500, 300, 200, d_opt} | ({100} if d_opt <= 100 else set()), reverse=True)
    d_steps = [d for d in d_steps if d >= d_opt]
    frames = []

    print("[GIF CCA] Generating frames for CCA urban cluster construction...")
    for step_idx, d_val in enumerate(d_steps):
        cidades_grid, pop_cidades, _ = cca_grid_orientador(
            pop_gpw, lons, lats, row_areas_km2[:, None], float(d_val), float(l_km)
        )
        pop_sorted = np.sort(pop_cidades[pop_cidades > 0])[::-1]
        a100, r2_100 = alpha_ols_rank_tamanho(pop_sorted, min(100, len(pop_sorted)))

        # Map each urban pixel to the population of its cluster
        max_cid = int(cidades_grid.max()) + 1
        lut_pop = np.zeros(max_cid, dtype=np.float64)
        if pop_cidades.size > 0:
            lut_pop[1 : len(pop_cidades) + 1] = pop_cidades
        cluster_pop_map = np.where(cidades_grid > 0, lut_pop[cidades_grid], np.nan)

        fig = plt.figure(figsize=(8.8, 3.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46)

        ax_map = fig.add_subplot(gs[0, 0])
        ax_map.set_facecolor("#eef3f8")
        # Subtle continental land background
        land_bg = np.where(pop_gpw > 0, 1.0, np.nan)
        ax_map.imshow(
            land_bg,
            extent=ext,
            origin="upper",
            cmap="Greys",
            vmin=0.0,
            vmax=3.0,
            alpha=0.25,
            interpolation="nearest",
            zorder=1,
        )
        im = ax_map.imshow(
            cluster_pop_map,
            extent=ext,
            origin="upper",
            cmap="plasma",
            norm=LogNorm(vmin=1e3, vmax=2.5e7),
            interpolation="nearest",
            zorder=3,
        )
        format_geo_axes(ax_map, box=zoom_box, xticks=(-120, -105, -90, -75), yticks=(10, 20, 30, 40, 50))
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        is_opt = " [calibrated optimum]" if d_val == d_opt else ""
        ax_map.set_title(f"CCA: $D = {d_val}$ people/km$^2$, $L^* = {l_km:g}$ km{is_opt}", pad=8, fontsize=8.0)
        add_panel_label(ax_map, "a", x=-0.14, y=1.04)
        add_scale_bar(ax_map, km=1000, pos=(0.66, 0.06))
        add_vertical_colorbar(fig, im, ax_map, label="Urban cluster population $S_k$ (people)")
        add_world_inset(
            fig,
            ax_map,
            zoom_box=zoom_box,
            inset_rect=(0.02, 0.03, 0.34, 0.26),
            use_axes_coords=True,
        )

        ax_rank = fig.add_subplot(gs[0, 1])
        if pop_sorted.size > 2:
            ranks = np.arange(1, pop_sorted.size + 1, dtype=np.float64)
            idx_sub = np.unique(np.geomspace(1, pop_sorted.size, num=min(250, pop_sorted.size), dtype=int) - 1)
            sc = ax_rank.scatter(
                ranks[idx_sub],
                pop_sorted[idx_sub],
                c=pop_sorted[idx_sub],
                cmap="plasma",
                norm=LogNorm(vmin=1e3, vmax=2.5e7),
                s=18,
                alpha=0.9,
                edgecolors="none",
                zorder=4,
            )
            add_vertical_colorbar(fig, sc, ax_rank, label="Cluster size $S_r$ (people)")
            r_ref = np.logspace(0, np.log10(min(300, pop_sorted.size)), 50)
            ax_rank.plot(r_ref, pop_sorted[0] * (r_ref ** (-1.0)), color="#111111", ls="--", lw=1.2, label="Exact Zipf law ($S_r \\propto r^{-1}$)")

        ax_rank.set_xscale("log")
        ax_rank.set_yscale("log")
        ax_rank.set_xlim(0.8, 1e4)
        ax_rank.set_ylim(5e2, 5e7)
        ax_rank.set_xlabel("Urban Cluster Rank $r$")
        ax_rank.set_ylabel("Cluster Population $S_r$ (people)")
        ax_rank.set_title(f"Zipf Rank-Size Evolution ($N = {len(pop_sorted):,}$ clusters)", pad=8, fontsize=8.0)
        ax_rank.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
        ax_rank.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=6.0)
        add_panel_label(ax_rank, "b", x=-0.16, y=1.04)

        info_box = (
            f"Step {step_idx + 1}/{len(d_steps)}\n"
            f"$D = {d_val}$ people/km$^2$\n"
            f"Clusters $N = {len(pop_sorted):,}$\n"
            f"OLS $\\alpha_{{100}} = {a100:.2f}$ ($R^2 = {r2_100:.3f}$)"
        )
        ax_rank.text(
            0.96,
            0.95,
            info_box,
            transform=ax_rank.transAxes,
            fontsize=6.2,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92),
            zorder=8,
        )

        frames.append(_fig_to_pil(fig, dpi=150))
        plt.close(fig)

    out_gif = THIS_DIR / "cca_urban_clustering_evolution.gif"
    durations = [650 if d != d_opt else 1800 for d in d_steps]
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"[GIF CCA] Saved: {out_gif}")


def generate_ipba_gif():
    set_scientific_style()
    THIS_DIR.mkdir(parents=True, exist_ok=True)

    land_dir = REPO_ROOT / "notebooks" / "gebco_cluster_dvider" / "datasets" / "processed" / "land"
    bacias_tif = REPO_ROOT / "notebooks" / "results" / "bacias_bloco_americas_ipba.tif"
    escala_csv = REPO_ROOT / "notebooks" / "results" / "escala_pop_lixo_global.csv"

    zoom_box = (-125.0, 7.0, -65.0, 50.0)
    info_d, dem_full = ReadGeoTiff(land_dir / "bloco_americas.tif")
    _, grid_full = ReadGeoTiff(bacias_tif)
    gt = info_d["GeoTransform"]
    lons_f = gt[0] + (np.arange(grid_full.shape[1]) + 0.5) * gt[1]
    lats_f = gt[3] + (np.arange(grid_full.shape[0]) + 0.5) * gt[5]
    cm = (lons_f >= zoom_box[0]) & (lons_f <= zoom_box[2])
    rm = (lats_f >= zoom_box[1]) & (lats_f <= zoom_box[3])
    full_grid = grid_full[np.ix_(rm, cm)]
    dem = dem_full[np.ix_(rm, cm)]
    lons, lats = lons_f[cm], lats_f[rm]
    df_esc = pd.read_csv(escala_csv)
    sub_na = df_esc[
        (df_esc["territorio_id"] == "americas_norte_central_caribe")
        & (df_esc["unidade_slug"] == "nurdle_10min")
    ].copy()

    ext = (lons.min() - gt[1] / 2, lons.max() + gt[1] / 2, lats.min() + gt[5] / 2, lats.max() - gt[5] / 2)
    endo_mask = full_grid < 0
    basin_grid = np.where(full_grid > 0, full_grid, 0)

    row_areas = pixel_area_by_latitude(lats, abs(gt[1]), abs(gt[5]))
    b_ids, _, b_areas_km2 = somar_areas_por_bacia(basin_grid, row_areas)
    max_id = int(b_ids.max()) + 1
    area_lut = np.full(max_id, np.nan, dtype=np.float64)
    area_lut[b_ids] = b_areas_km2
    full_area_map = np.where(basin_grid > 0, area_lut[np.clip(basin_grid, 0, max_id - 1)], np.nan)

    # Elevation invasion thresholds from coast (0 m) up to continental peaks (4,500 m)
    h_thresholds = [25, 75, 150, 250, 400, 650, 1000, 1600, 2500, 5000]
    frames = []

    print("[GIF IPBA] Generating frames for IPBA invasion percolation...")
    land_mask = basin_grid > 0
    total_land_px = max(1, int(land_mask.sum()))

    for step_idx, h_max in enumerate(h_thresholds):
        invaded_mask = land_mask & (dem <= h_max)
        pct_invaded = 100.0 * float(invaded_mask.sum()) / total_land_px
        partial_grid = np.where(invaded_mask, basin_grid, 0)
        _, _, part_areas = somar_areas_por_bacia(partial_grid, row_areas)
        part_sorted = np.sort(part_areas[part_areas >= 150.0])[::-1]

        area_frame = np.where(invaded_mask, full_area_map, np.nan)

        fig = plt.figure(figsize=(8.8, 3.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.46)

        ax_map = fig.add_subplot(gs[0, 0])
        ax_map.set_facecolor("#eef3f8")
        ax_map.imshow(
            np.where(land_mask, 1.0, np.nan),
            extent=ext,
            origin="upper",
            cmap="Greys",
            vmin=0.0,
            vmax=3.0,
            alpha=0.25,
            interpolation="nearest",
            zorder=1,
        )
        ax_map.imshow(np.where(endo_mask, 1.0, np.nan), extent=ext, origin="upper", cmap="Greys",
                      vmin=0.0, vmax=2.0, interpolation="nearest", zorder=2)
        im = ax_map.imshow(
            area_frame,
            extent=ext,
            origin="upper",
            cmap="plasma",
            norm=LogNorm(vmin=50.0, vmax=3.2e6),
            interpolation="nearest",
            zorder=3,
        )
        if not sub_na.empty:
            ax_map.scatter(
                sub_na["foz_lon"],
                sub_na["foz_lat"],
                s=18,
                marker="v",
                c="#00f5d4",
                edgecolors="#111111",
                linewidths=0.45,
                label=f"Matched coastal sinks ($n={len(sub_na)}$)",
                zorder=6,
            )
            ax_map.legend(loc="upper right", frameon=True, facecolor="#222222", edgecolor="none", labelcolor="white", fontsize=5.8)

        format_geo_axes(ax_map, box=zoom_box, xticks=(-120, -105, -90, -75), yticks=(10, 20, 30, 40, 50))
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title(f"IPBA Percolation: $h \\leq {h_max}$ m ({pct_invaded:.1f}% Invaded)", pad=8, fontsize=8.0)
        add_panel_label(ax_map, "a", x=-0.14, y=1.04)
        add_scale_bar(ax_map, km=1000, pos=(0.66, 0.06))
        add_vertical_colorbar(fig, im, ax_map, label="Drainage basin area $A$ (km$^2$)")
        add_world_inset(
            fig,
            ax_map,
            zoom_box=zoom_box,
            inset_rect=(0.02, 0.03, 0.34, 0.26),
            use_axes_coords=True,
        )

        ax_ccdf = fig.add_subplot(gs[0, 1])
        if part_sorted.size > 4:
            ccdf = np.arange(1, part_sorted.size + 1, dtype=np.float64) / part_sorted.size
            idx_sub = np.unique(np.geomspace(1, part_sorted.size, num=min(220, part_sorted.size), dtype=int) - 1)
            sc = ax_ccdf.scatter(
                part_sorted[idx_sub],
                ccdf[idx_sub],
                c=part_sorted[idx_sub],
                cmap="plasma",
                norm=LogNorm(vmin=150.0, vmax=3.2e6),
                s=18,
                alpha=0.9,
                edgecolors="none",
                zorder=4,
            )
            add_vertical_colorbar(fig, sc, ax_ccdf, label="Invaded basin area $A_0$ (km$^2$)")

        x_ref = np.logspace(2.2, 6.4, 50)
        y_ref = 0.85 * (x_ref / x_ref[0]) ** (-0.90)
        ax_ccdf.plot(x_ref, y_ref, color="#111111", ls="--", lw=1.2, label="Universal law $\\tau \\approx 1.90$ (Oliveira et al.)")

        ax_ccdf.set_xscale("log")
        ax_ccdf.set_yscale("log")
        ax_ccdf.set_xlim(120.0, 5.0e6)
        ax_ccdf.set_ylim(2e-4, 1.4)
        ax_ccdf.set_xlabel("Spherical Basin Area $A_0$ (km$^2$)")
        ax_ccdf.set_ylabel("Cumulative Probability $P(A \\geq A_0)$")
        ax_ccdf.set_title("Emergence of Universal Basin Scaling", pad=8, fontsize=8.0)
        ax_ccdf.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
        ax_ccdf.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=5.8)
        add_panel_label(ax_ccdf, "b", x=-0.16, y=1.04)

        info_box = (
            f"Step {step_idx + 1}/{len(h_thresholds)}\n"
            f"Invasion ceiling $h \\leq {h_max}$ m\n"
            f"Invaded land: {pct_invaded:.1f}%\n"
            f"Max basin: {(part_sorted[0] if part_sorted.size else 0):,.0f} km$^2$"
        )
        ax_ccdf.text(
            0.96,
            0.95,
            info_box,
            transform=ax_ccdf.transAxes,
            fontsize=6.1,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92),
            zorder=8,
        )

        frames.append(_fig_to_pil(fig, dpi=150))
        plt.close(fig)

    out_gif = THIS_DIR / "ipba_watershed_percolation_evolution.gif"
    durations = [650] * (len(h_thresholds) - 1) + [1800]
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"[GIF IPBA] Saved: {out_gif}")


if __name__ == "__main__":
    generate_cca_gif()
    generate_ipba_gif()
