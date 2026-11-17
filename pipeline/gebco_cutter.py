"""
pipeline/gebco_cutter.py
-------------------------
Módulo para segmentação do relevo global (GEBCO 2026) em porções de terra (ilhas e massas
continentais) utilizando diretamente os algoritmos originais da pasta `externos/`:
  1. Divisão do globo em porções de terra conectadas (`externos/GebcoCutter`):
     - `ReadGeoTiff` e `WriteGeoTiff` (`externos/GebcoCutter/geotiff/geotiff.py`)
     - `Percolation.site_percolation()` (`externos/GebcoCutter/percolation/percolation.py`)
     - Recorte de cada porção de terra com periodicidade longitudinal (`01_clustering.ipynb`).
  2. Bacias hidrográficas em cada porção de terra (`externos/IPBA`):
     - `Lattice`, `Heap` e `SetDrainageBasins` (`pipeline/ipba_runner.py`).
     - Área esférica exata por Triângulos Esféricos (Teorema de L'Huilier + Haversine).
  3. Aglomerados urbanos em cada porção de terra (`externos/CCA`):
     - `burn` de `externos/CCA/cca/city_clustering.py` calibrado via Lei de Zipf
       sobre o raster GPW v4 alinhado 4x4 à porção de terra.
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_GEBCO_DIR = REPO_ROOT / "externos" / "GebcoCutter"
if str(EXTERNAL_GEBCO_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_GEBCO_DIR))

from percolation.percolation import Percolation  # Algoritmo oficial de externos/GebcoCutter

_spec_gt = importlib.util.spec_from_file_location(
    "gebcocutter_geotiff", EXTERNAL_GEBCO_DIR / "geotiff" / "geotiff.py"
)
_mod_gt = importlib.util.module_from_spec(_spec_gt)
assert _spec_gt is not None and _spec_gt.loader is not None
_spec_gt.loader.exec_module(_mod_gt)
ReadGeoTiff = _mod_gt.ReadGeoTiff
WriteGeoTiff = _mod_gt.WriteGeoTiff

from .cca_calibrator import calibrate_zipf_grid, crop_gpw_for_gebco_cluster, run_calibrated_cca
from .ipba_runner import (
    EARTH_RADIUS_KM,
    extract_coastal_sinks,
    run_ipba,
    spherical_triangle_row_areas,
)
from .spatial_matcher import split_urban_clusters_by_basins

# Referencia geografica auxiliar para rotulagem continental
GRANDES_ILHAS: dict[str, dict[str, Any]] = {
    "North & Central America": {"bbox": (-128.0, 8.0, -65.0, 52.0), "marker": "o", "color": "C0"},
    "South America": {"bbox": (-82.0, -56.0, -34.0, 13.0), "marker": "s", "color": "C2"},
    "Europe & Eurasia": {"bbox": (-15.0, 35.0, 45.0, 72.0), "marker": "^", "color": "C3"},
    "Africa": {"bbox": (-18.0, -35.0, 52.0, 38.0), "marker": "D", "color": "C4"},
    "Australia": {"bbox": (112.0, -45.0, 155.0, -10.0), "marker": "v", "color": "C1"},
}


def executar_percolacao_gebco_cutter(
    gebco_path: str | Path,
    h_star: float = 0.0,
    cache_npz: str | Path | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa a divisão do globo em porções de terra conectadas usando diretamente
    `ReadGeoTiff` e `Percolation.site_percolation()` de `externos/GebcoCutter`.
    Retorna: (_info, _arr, label, vals_sorted_by_size, sizes_sorted)
    """
    _info, _arr = ReadGeoTiff(str(gebco_path))
    p = _arr > h_star

    if cache_npz is not None and Path(cache_npz).is_file():
        data = np.load(cache_npz)
        return _info, _arr, data["label"], data["vals"], data["sizes"]

    percolation = Percolation(p)
    # Evitar overflow de uint8 em grades globais mantendo exatamente a mesma arvore union-find
    percolation.rank = np.zeros(percolation.n, dtype=np.uint32)
    percolation.site_percolation()

    label = percolation.parent.reshape(percolation.nrows, percolation.ncols)
    vals, sizes = np.unique(label[p], return_counts=True)
    order = np.argsort(sizes)[::-1]
    vals = vals[order]
    sizes = sizes[order]

    if cache_npz is not None:
        Path(cache_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, label=label, vals=vals, sizes=sizes)

    return _info, _arr, label, vals, sizes


def recortar_cluster_gebco_cutter(
    _info: dict[str, Any],
    _arr: np.ndarray,
    label: np.ndarray,
    val: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int, int, np.ndarray]:
    """
    Recorta uma porção de terra (`val`) segundo `externos/GebcoCutter/01_clustering.ipynb`,
    incluindo tratamento de periodicidade longitudinal (`empty_cols`, `i0`, `(xs - i0) % ncols`).
    """
    _, ncols = label.shape
    cluster = label == val

    empty_cols = np.flatnonzero(~cluster.any(axis=0))
    i0 = int(empty_cols[0]) if empty_cols.size else 0

    ys, xs = np.nonzero(cluster)
    xs_shifted = (xs - i0) % ncols

    j_min, j_max = int(ys.min()), int(ys.max())
    i_min, i_max = int(xs_shifted.min()), int(xs_shifted.max())
    cols = (np.arange(i_min, i_max + 1) + i0) % ncols

    cluster_cropped = cluster[j_min : j_max + 1, cols]
    _arr_cropped = _arr[j_min : j_max + 1, cols]

    nodata = _info["NoDataValue"] if _info.get("NoDataValue") is not None else -32768
    arr = np.where(cluster_cropped, _arr_cropped, nodata)

    height, width = arr.shape
    _gt = _info["GeoTransform"]
    x_pixel = int(cols[0])

    x_geo = _gt[0] + x_pixel * _gt[1] + j_min * _gt[2]
    y_geo = _gt[3] + x_pixel * _gt[4] + j_min * _gt[5]
    gt = (x_geo, _gt[1], _gt[2], y_geo, _gt[4], _gt[5])

    info = {
        "RasterYSize": height,
        "RasterXSize": width,
        "GeoTransform": gt,
        "Projection": _info["Projection"],
        "NoDataValue": nodata,
    }
    return arr, cluster_cropped, info, j_min, j_max, cols


def nomear_porcao_terra(val: int, n_pixels: int, lat_med: float, lon_med: float) -> str:
    """Atribui nome legível às principais porções de terra segmentadas pelo GebcoCutter."""
    if n_pixels > 5_000_000 and lat_med > 0:
        return "Afro-Eurásia"
    if n_pixels > 5_000_000 and lat_med < -60:
        return "Antártida"
    if n_pixels > 2_000_000 and lon_med < -30:
        return "Américas (Norte, Central e Sul)"
    if n_pixels > 400_000 and lat_med < 0 and lon_med > 100:
        return "Austrália"
    if n_pixels > 400_000 and lat_med > 55 and lon_med < 0:
        return "Groenlândia"
    if 20_000 <= n_pixels <= 35_000 and 50 <= lat_med <= 60 and -10 <= lon_med <= 5:
        return "Grã-Bretanha"
    if 20_000 <= n_pixels <= 30_000 and 30 <= lat_med <= 42 and 128 <= lon_med <= 145:
        return "Honshu (Japão)"
    if 40_000 <= n_pixels <= 52_000 and -28 <= lat_med <= -10 and 42 <= lon_med <= 52:
        return "Madagascar"
    return f"Ilha #{val}"


def classificar_macroregiao(lat: float, lon: float) -> str:
    """Mapeia coordenadas (lat, lon) para macro-região continental para visualização."""
    if lat < -60.0:
        return "Antártida"
    if lon < -30.0:
        return "Am. Norte/Central" if lat >= 12.5 else "Am. do Sul"
    if -30.0 <= lon <= 60.0 and lat >= 35.0:
        return "Europa"
    if -30.0 <= lon <= 60.0 and lat < 35.0:
        return "África"
    if lon > 60.0 and lat < -10.0:
        return "Austrália/Oceania"
    return "Ásia"


def processar_porcao_terra_gebco_cutter(
    val: int,
    arr_c: np.ndarray,
    cl_c: np.ndarray,
    info_c: dict[str, Any],
    j_min: int,
    j_max: int,
    cols_gebco: np.ndarray,
    gpw_path: str | Path,
    out_dir: str | Path = "relatorios/porcoes_terra",
) -> pd.DataFrame:
    """
    Executa o pipeline completo (`externos/IPBA` + Triângulos Esféricos + `externos/CCA`)
    sobre uma única porção de terra segmentada por `externos/GebcoCutter`.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_csv = out_path / f"bacias_porcao_{int(val)}.csv"
    if cache_csv.is_file() and cache_csv.stat().st_size > 10:
        return pd.read_csv(cache_csv)

    t0 = time.time()
    n_land_px = int(cl_c.sum())
    gt_c = info_c["GeoTransform"]

    # 1. Checagem rapida de populacao na porcao de terra (pula Antartida e rochedos desabitados)
    pop, px_area_gpw, lats_1d, lons_1d, gt_gpw, gpw_meta = crop_gpw_for_gebco_cluster(
        j_min, j_max, cols_gebco, cl_c, gpw_path
    )
    tot_pop = gpw_meta["total_population"]
    if tot_pop <= 0.0:
        empty_df = pd.DataFrame(
            columns=[
                "land_cluster_id", "landmass_name", "sink_id", "basin_id", "latitude", "longitude",
                "elevation", "pixel_count", "area_km2", "area_fraction", "cumulative_fraction",
                "in_top_95pct", "urban_population", "total_population", "urban_pixels",
                "total_pixels", "urban_fraction", "d_star_opt", "l_opt_km", "macroregiao",
            ]
        )
        empty_df.to_csv(cache_csv, index=False)
        del pop, px_area_gpw
        return empty_df

    # Para massas continentais grandes, libera `pop` antes do IPBA para minimizar pico de RAM
    if n_land_px > 500_000:
        del pop, px_area_gpw
        gc.collect()
        recarregar_gpw = True
    else:
        recarregar_gpw = False

    # 2. Halo de 1 pixel de oceano ao redor do recorte para que `extract_coastal_sinks` e
    # `externos/IPBA` enxerguem toda a borda litoranea e operem estritamente no interior da grade
    nodata = float(info_c.get("NoDataValue", -32768))
    dem_pad = np.pad(arr_c.astype(np.float32), pad_width=1, mode="constant", constant_values=nodata)
    gt_pad = (gt_c[0] - gt_c[1], gt_c[1], gt_c[2], gt_c[3] - gt_c[5], gt_c[4], gt_c[5])

    sink_grid_pad, sinks_df = extract_coastal_sinks(dem_pad, gt_pad, h_star=0.0)
    basin_grid_pad = run_ipba(dem_pad, sink_grid_pad, h_star=0.0)
    basin_grid = basin_grid_pad[1:-1, 1:-1]
    del dem_pad, sink_grid_pad, basin_grid_pad
    gc.collect()

    if recarregar_gpw:
        pop, px_area_gpw, lats_1d, lons_1d, gt_gpw, gpw_meta = crop_gpw_for_gebco_cluster(
            j_min, j_max, cols_gebco, cl_c, gpw_path
        )

    # 3. Area esferica exata (Triangulos Esfericos de L'Huilier + Haversine) de 100% das bacias
    nrows, ncols = basin_grid.shape
    lats_gebco = gt_c[3] + (np.arange(nrows) + 0.5) * gt_c[5]
    px_area_1d = spherical_triangle_row_areas(lats_gebco, abs(gt_c[5]), gt_c[1], R=EARTH_RADIUS_KM)
    px_area_grid = np.broadcast_to(px_area_1d[:, None], (nrows, ncols))

    land_mask = basin_grid > 0
    total_land_area = float(px_area_grid[land_mask].sum())
    labels, counts = np.unique(basin_grid[land_mask], return_counts=True)
    areas = np.bincount(basin_grid[land_mask], weights=px_area_grid[land_mask])[labels]

    df_areas = (
        pd.DataFrame({"basin_id": labels, "pixel_count": counts, "area_km2": areas})
        .sort_values("area_km2", ascending=False)
        .reset_index(drop=True)
    )
    df_areas["area_fraction"] = df_areas["area_km2"] / total_land_area if total_land_area > 0 else 0.0
    df_areas["cumulative_fraction"] = df_areas["area_fraction"].cumsum()
    cut_idx = int(np.searchsorted(df_areas["cumulative_fraction"], 0.95))
    df_areas["in_top_95pct"] = False
    if len(df_areas) > 0:
        df_areas.loc[: min(cut_idx, len(df_areas) - 1), "in_top_95pct"] = True

    # 4. Calibracao CCA via Zipf (`externos/CCA/cca/city_clustering.py` -> `burn`)
    # Para Afro-Eurasia (> 5M pixels de terra), evita d_star=100-200 (~14M px) usando >= 500 hab/km2
    if n_land_px > 5_000_000:
        d_vals = [500.0, 800.0, 1010.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0]
    else:
        d_vals = [100.0, 200.0, 500.0, 800.0, 1010.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0]
    l_vals = [1.0, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # Pre-checagem rapida: se nao houver nem 20 pixels urbanos na menor densidade, usa fallback direto
    max_dens = float((pop / px_area_gpw).max()) if pop.size else 0.0
    if max_dens <= 100.0:
        best_pair = (1010.0, 2.0)
    else:
        zipf_csv = out_path / f"matriz_zipf_porcao_{int(val)}.csv"
        if zipf_csv.is_file() and zipf_csv.stat().st_size > 10:
            df_z = pd.read_csv(zipf_csv)
            valid_z = df_z[df_z["r2"] >= 0.80]
            if len(valid_z) > 0:
                best_row = valid_z.iloc[(valid_z["alpha"] - 1.0).abs().argsort()[:1]].iloc[0]
                best_pair = (float(best_row["d_star"]), float(best_row["l_cutoff"]))
            else:
                best_pair = (1010.0, 2.0)
        else:
            _, _, _, df_grid, best_pair = calibrate_zipf_grid(
                pop, lons_1d, lats_1d, px_area_gpw, d_values=d_vals, l_values=l_vals,
                top_n=100, min_clusters=20, fallback_pair=(1010.0, 2.0)
            )
            df_grid.to_csv(zipf_csv, index=False)

    grid_clusters, _ = run_calibrated_cca(pop, lons_1d, lats_1d, px_area_gpw, best_pair[0], best_pair[1])
    basin_pop_df, _ = split_urban_clusters_by_basins(grid_clusters, pop, basin_grid, gt_gpw, gt_c)

    # 5. Consolidar sinks + areas + populacao
    df_full = (
        sinks_df[["sink_id", "latitude", "longitude", "elevation"]]
        .merge(df_areas, left_on="sink_id", right_on="basin_id", how="inner")
        .merge(basin_pop_df, on="basin_id", how="left")
    )
    for col in ["urban_population", "total_population", "urban_pixels", "total_pixels", "urban_fraction"]:
        df_full[col] = df_full[col].fillna(0.0)

    lat_med = float(df_full["latitude"].median()) if len(df_full) else 0.0
    lon_med = float(df_full["longitude"].median()) if len(df_full) else 0.0
    land_name = nomear_porcao_terra(int(val), n_land_px, lat_med, lon_med)

    df_full["land_cluster_id"] = int(val)
    df_full["landmass_name"] = land_name
    df_full["d_star_opt"] = float(best_pair[0])
    df_full["l_opt_km"] = float(best_pair[1])
    df_full["macroregiao"] = [
        classificar_macroregiao(la, lo) for la, lo in zip(df_full["latitude"], df_full["longitude"])
    ]

    df_full.to_csv(cache_csv, index=False)
    if n_land_px >= 10_000:
        print(
            f"   [{land_name} | val={val}] {time.time()-t0:.1f}s | {len(df_full)} bacias "
            f"({int(df_full['in_top_95pct'].sum())} Top95%) | pop={gpw_meta['total_population']:,.0f} "
            f"| D*={best_pair[0]:g}, L={best_pair[1]:g} km",
            flush=True,
        )
    del pop, px_area_gpw, grid_clusters, basin_grid
    gc.collect()
    return df_full


def gerar_bacias_globais_gebco_cutter(
    gebco_path: str | Path,
    gpw_path: str | Path,
    target_cluster_vals: set[int] | None = None,
    cache_percolation_npz: str | Path = "relatorios/cache_percolacao_gebco_h0.npz",
    out_dir: str | Path = "relatorios/porcoes_terra",
) -> pd.DataFrame:
    """
    Executa a segmentação global com `externos/GebcoCutter` e roda `externos/IPBA` + `externos/CCA`
    em cada porção de terra conectada do globo (ou nas porções especificadas em `target_cluster_vals`).
    """
    info, arr, label, vals, sizes = executar_percolacao_gebco_cutter(
        gebco_path, h_star=0.0, cache_npz=cache_percolation_npz
    )
    vals_to_run = [int(v) for v in vals if target_cluster_vals is None or int(v) in target_cluster_vals]

    all_dfs = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Primeiro carrega todos os ja cacheados sem recortar
    pending_vals = []
    for v in vals_to_run:
        cache_csv = out_path / f"bacias_porcao_{v}.csv"
        if cache_csv.is_file() and cache_csv.stat().st_size > 10:
            df_c = pd.read_csv(cache_csv)
            if len(df_c) > 0:
                all_dfs.append(df_c)
        else:
            pending_vals.append(v)

    if pending_vals:
        print(f"   Porcoes de terra pendentes para IPBA+CCA: {len(pending_vals)} (de {len(vals_to_run)} ativas)", flush=True)
        for idx, v in enumerate(pending_vals):
            arr_c, cl_c, info_c, jmin, jmax, cols = recortar_cluster_gebco_cutter(info, arr, label, v)
            df_c = processar_porcao_terra_gebco_cutter(
                v, arr_c, cl_c, info_c, jmin, jmax, cols, gpw_path, out_dir=out_path
            )
            if len(df_c) > 0:
                all_dfs.append(df_c)
            if (idx + 1) % 100 == 0:
                print(f"   Progresso ilhas menores: {idx + 1}/{len(pending_vals)} processadas...", flush=True)

    if not all_dfs:
        return pd.DataFrame()
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Marcar tambem o corte global de 95% de area acumulada em todo o globo
    df_all = df_all.sort_values("area_km2", ascending=False).reset_index(drop=True)
    tot_a = float(df_all["area_km2"].sum())
    cum_f = (df_all["area_km2"] / tot_a).cumsum() if tot_a > 0 else pd.Series(0.0, index=df_all.index)
    cut_g = int(np.searchsorted(cum_f, 0.95))
    df_all["in_top_95pct_global"] = False
    if len(df_all) > 0:
        df_all.loc[: min(cut_g, len(df_all) - 1), "in_top_95pct_global"] = True
    return df_all
