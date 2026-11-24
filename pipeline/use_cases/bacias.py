"""Caso de uso: segmentação global (GebcoCutter), bacias (IPBA) e aglomerados (CCA)."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import (
    CACHE_PERC,
    COLUNAS_BACIA,
    GEBCO_TIF,
    GPW_TIF,
    PORCOES_DIR,
    ZIPF_D_GRANDE,
    ZIPF_D_PADRAO,
    ZIPF_FALLBACK,
    ZIPF_L_PADRAO,
)
from ..domain.basins import (
    calcular_areas_bacias,
    classificar_macroregiao,
    extrair_sinks_costeiros,
    mascara_costa,
    nomear_porcao_terra,
    particionar_populacao_por_bacia,
)
from ..domain.geometry import km_para_corda, to_xyz
from ..domain.zipf import calibrar_grade_zipf, rotular_clusters_cca, selecionar_par_otimo_zipf
from ..infra.cache import ler_csv_cache, ler_npz_cache, salvar_csv_cache, salvar_npz_cache
from ..infra.externos import agrupar_cca, executar_ipba, ler_geotiff_gebco, percolar_terra
from ..infra.rasters import recortar_gpw_porcao, recortar_porcao_gebco


def obter_percolacao_global(
    gebco_path: str | Path = GEBCO_TIF,
    h_star: float = 0.0,
    cache_npz: str | Path | None = CACHE_PERC,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lê o relevo GEBCO e obtém (do cache ou calculando) a percolação global de terra."""
    info, arr = ler_geotiff_gebco(gebco_path)
    cached = ler_npz_cache(cache_npz)
    if cached is not None:
        return info, arr, cached["label"], cached["vals"], cached["sizes"]

    label, vals, sizes = percolar_terra(arr > h_star)
    salvar_npz_cache(cache_npz, label=label, vals=vals, sizes=sizes)
    return info, arr, label, vals, sizes


def identificar_porcoes_ativas(
    pontos_df: pd.DataFrame,
    max_r_km: float = 50.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    """Lista as porções de terra (ordenadas por tamanho) a até `max_r_km` de algum ponto."""
    info, arr, label, vals, _ = obter_percolacao_global()
    gt = info["GeoTransform"]
    sr, sc = np.nonzero(mascara_costa(arr > 0))
    slats = gt[3] + (sr + 0.5) * gt[5]
    slons = ((gt[0] + (sc + 0.5) * gt[1] + 180.0) % 360.0) - 180.0

    tree = cKDTree(to_xyz(pontos_df["lat"].to_numpy(), pontos_df["lon"].to_numpy()))
    vizinhos = tree.query_ball_point(to_xyz(slats, slons), r=km_para_corda(max_r_km))
    ativos = {int(v) for v in np.unique(label[sr, sc][np.array([bool(x) for x in vizinhos])])}
    return info, arr, label, [int(v) for v in vals if int(v) in ativos]


def _calibrar_ou_ler_zipf(
    val: int,
    n_land_px: int,
    pop: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    px_area: np.ndarray,
    out_dir: Path,
) -> tuple[float, float]:
    if not pop.size or float((pop / px_area).max()) <= 100.0:
        return ZIPF_FALLBACK

    zipf_csv = out_dir / f"matriz_zipf_porcao_{val}.csv"
    df_cached = ler_csv_cache(zipf_csv)
    if df_cached is not None:
        return selecionar_par_otimo_zipf(df_cached)

    d_vals = ZIPF_D_GRANDE if n_land_px > 5_000_000 else ZIPF_D_PADRAO
    df_grid, par = calibrar_grade_zipf(
        pop, lons_1d, lats_1d, px_area, agrupar_cca,
        d_values=d_vals, l_values=ZIPF_L_PADRAO,
    )
    salvar_csv_cache(df_grid, zipf_csv)
    return par


def _extrair_bacias_com_halo(arr_c: np.ndarray, info_c: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    """Adiciona halo de 1 pixel de oceano para que IPBA enxergue toda a borda litorânea."""
    gt_c = info_c["GeoTransform"]
    nodata = float(info_c.get("NoDataValue", -32768))
    dem_pad = np.pad(arr_c.astype(np.float32), pad_width=1, mode="constant", constant_values=nodata)
    gt_pad = (gt_c[0] - gt_c[1], gt_c[1], gt_c[2], gt_c[3] - gt_c[5], gt_c[4], gt_c[5])

    sink_grid, sinks_df = extrair_sinks_costeiros(dem_pad, gt_pad, h_star=0.0)
    basin_grid = executar_ipba(dem_pad, sink_grid, h_star=0.0)[1:-1, 1:-1]
    return basin_grid, sinks_df


def processar_porcao_terra(
    val: int,
    info: dict[str, Any],
    arr: np.ndarray,
    label: np.ndarray,
    gpw_path: str | Path = GPW_TIF,
    out_dir: str | Path = PORCOES_DIR,
) -> pd.DataFrame:
    """Executa (ou lê do cache) o fluxo GebcoCutter -> IPBA -> Zipf/CCA para uma porção `val`."""
    out_path = Path(out_dir)
    cache_csv = out_path / f"bacias_porcao_{int(val)}.csv"
    cached = ler_csv_cache(cache_csv)
    if cached is not None:
        return cached

    t0 = time.time()
    arr_c, cl_c, info_c, j_min, j_max, cols = recortar_porcao_gebco(info, arr, label, int(val))
    n_land_px = int(cl_c.sum())
    gt_c = info_c["GeoTransform"]

    # 1. Recorte GPW e descarte rápido de porções desabitadas
    pop, px_area, lats_1d, lons_1d, gt_gpw, tot_pop = recortar_gpw_porcao(j_min, j_max, cols, cl_c, gpw_path)
    if tot_pop <= 0.0:
        vazio = pd.DataFrame(columns=COLUNAS_BACIA)
        salvar_csv_cache(vazio, cache_csv)
        return vazio

    # Libera GPW durante IPBA em massas continentais grandes para reduzir pico de RAM
    recarregar_gpw = n_land_px > 500_000
    if recarregar_gpw:
        del pop, px_area
        gc.collect()

    # 2. Bacias via IPBA e áreas esféricas exatas (L'Huilier + Haversine)
    basin_grid, sinks_df = _extrair_bacias_com_halo(arr_c, info_c)
    df_areas = calcular_areas_bacias(basin_grid, gt_c)

    if recarregar_gpw:
        pop, px_area, lats_1d, lons_1d, gt_gpw, tot_pop = recortar_gpw_porcao(j_min, j_max, cols, cl_c, gpw_path)

    # 3. Calibração Zipf + CCA + divisão populacional por bacia
    d_opt, l_opt = _calibrar_ou_ler_zipf(int(val), n_land_px, pop, lons_1d, lats_1d, px_area, out_path)
    grid_clusters, _ = rotular_clusters_cca(pop, lons_1d, lats_1d, px_area, d_opt, l_opt, agrupar_cca)
    df_pop = particionar_populacao_por_bacia(grid_clusters, pop, basin_grid, gt_gpw, gt_c)

    # 4. Consolidação final
    df = (
        sinks_df[["sink_id", "latitude", "longitude", "elevation"]]
        .merge(df_areas, left_on="sink_id", right_on="basin_id", how="inner")
        .merge(df_pop, on="basin_id", how="left")
    )
    for col in ("urban_population", "total_population", "urban_pixels", "total_pixels", "urban_fraction"):
        df[col] = df[col].fillna(0.0)

    lat_med = float(df["latitude"].median()) if len(df) else 0.0
    lon_med = float(df["longitude"].median()) if len(df) else 0.0
    nome = nomear_porcao_terra(int(val), n_land_px, lat_med, lon_med)

    df["land_cluster_id"] = int(val)
    df["landmass_name"] = nome
    df["d_star_opt"] = float(d_opt)
    df["l_opt_km"] = float(l_opt)
    df["macroregiao"] = [classificar_macroregiao(la, lo) for la, lo in zip(df["latitude"], df["longitude"])]

    salvar_csv_cache(df, cache_csv)
    if n_land_px >= 10_000:
        print(f"   [{nome} | val={val}] {time.time()-t0:.1f}s | {len(df)} bacias | pop={tot_pop:,.0f} | D*={d_opt:g}, L={l_opt:g} km", flush=True)
    return df


def carregar_bacias(pontos: dict[str, pd.DataFrame], max_r_km: float = 50.0) -> pd.DataFrame:
    """Retorna 100% das bacias (IPBA + CCA) das porções de terra próximas aos pontos informados."""
    info, arr, label, porcoes = identificar_porcoes_ativas(pd.concat(pontos.values(), ignore_index=True), max_r_km=max_r_km)
    dfs = [processar_porcao_terra(v, info, arr, label) for v in porcoes]
    return pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
