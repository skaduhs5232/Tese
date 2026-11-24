"""Regras de domínio para linha de costa, bacias hidrográficas e porções de terra."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .geometry import area_triangulos_esfericos

_VIZINHOS_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def mascara_costa(terra: np.ndarray) -> np.ndarray:
    """Pixels de terra adjacentes (4-conectividade) a pelo menos um pixel de água."""
    return terra & binary_dilation(~terra, structure=_VIZINHOS_4)


def extrair_sinks_costeiros(
    dem: np.ndarray,
    gt: tuple[float, float, float, float, float, float],
    h_star: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Identifica todos os sinks costeiros (borda terra/água) e retorna grade + tabela."""
    rows, cols = np.nonzero(mascara_costa(dem > h_star))
    sink_ids = np.arange(1, len(rows) + 1, dtype=np.uint32)

    sink_grid = np.zeros(dem.shape, dtype=np.uint32)
    sink_grid[rows, cols] = sink_ids

    sinks_df = pd.DataFrame({
        "sink_id": sink_ids,
        "row": rows,
        "col": cols,
        "longitude": ((gt[0] + (cols + 0.5) * gt[1] + 180.0) % 360.0) - 180.0,
        "latitude": gt[3] + (rows + 0.5) * gt[5],
        "elevation": dem[rows, cols],
    })
    return sink_grid, sinks_df


def calcular_areas_bacias(
    basin_grid: np.ndarray,
    gt: tuple[float, float, float, float, float, float],
    limiar_top: float = 0.95,
) -> pd.DataFrame:
    """Calcula área esférica exata (km²) e fração acumulada para 100% das bacias."""
    nrows, ncols = basin_grid.shape
    lats = gt[3] + (np.arange(nrows) + 0.5) * gt[5]
    area_grid = np.broadcast_to(area_triangulos_esfericos(lats, abs(gt[5]), gt[1])[:, None], (nrows, ncols))

    terra = basin_grid > 0
    total_area = float(area_grid[terra].sum())
    labels, counts = np.unique(basin_grid[terra], return_counts=True)
    areas = np.bincount(basin_grid[terra], weights=area_grid[terra])[labels]

    df = (
        pd.DataFrame({"basin_id": labels, "pixel_count": counts, "area_km2": areas})
        .sort_values("area_km2", ascending=False)
        .reset_index(drop=True)
    )
    df["area_fraction"] = df["area_km2"] / total_area if total_area > 0 else 0.0
    df["cumulative_fraction"] = df["area_fraction"].cumsum()
    df["in_top_95pct"] = False
    if len(df) > 0:
        corte = min(int(np.searchsorted(df["cumulative_fraction"], limiar_top)), len(df) - 1)
        df.loc[:corte, "in_top_95pct"] = True
    return df


def particionar_populacao_por_bacia(
    grid_clusters: np.ndarray,
    pop_grid: np.ndarray,
    basin_grid: np.ndarray,
    gt_gpw: tuple[float, float, float, float, float, float],
    gt_gebco: tuple[float, float, float, float, float, float],
) -> pd.DataFrame:
    """Atribui população total e urbana (CCA) pixel a pixel para cada bacia do IPBA."""
    gpw_h, gpw_w = pop_grid.shape
    gebco_h, gebco_w = basin_grid.shape
    sy = int(round(abs(gt_gebco[5]) / abs(gt_gpw[5])))
    sx = int(round(gt_gebco[1] / gt_gpw[1]))

    r_idx = np.clip(np.arange(gpw_h) // sy, 0, gebco_h - 1)
    c_idx = np.clip(np.arange(gpw_w) // sx, 0, gebco_w - 1)
    bacias_gpw = basin_grid[r_idx[:, None], c_idx[None, :]]

    valid = bacias_gpw > 0
    urban = valid & (grid_clusters > 0)
    ativas = np.unique(bacias_gpw[valid])
    size = int(ativas.max()) + 1 if len(ativas) else 0

    tot_pop = np.bincount(bacias_gpw[valid], weights=pop_grid[valid], minlength=size)[ativas]
    tot_px = np.bincount(bacias_gpw[valid], minlength=size)[ativas]
    urb_pop = np.bincount(bacias_gpw[urban], weights=pop_grid[urban], minlength=size)[ativas]
    urb_px = np.bincount(bacias_gpw[urban], minlength=size)[ativas]

    return pd.DataFrame({
        "basin_id": ativas,
        "urban_population": urb_pop,
        "total_population": tot_pop,
        "urban_pixels": urb_px,
        "total_pixels": tot_px,
        "urban_fraction": np.divide(urb_pop, tot_pop, out=np.zeros_like(urb_pop, dtype=float), where=tot_pop > 0),
    })


def nomear_porcao_terra(val: int, n_pixels: int, lat_med: float, lon_med: float) -> str:
    """Atribui nome legível às principais massas continentais e grandes ilhas."""
    regras = [
        (n_pixels > 5_000_000 and lat_med > 0, "Afro-Eurásia"),
        (n_pixels > 5_000_000 and lat_med < -60, "Antártida"),
        (n_pixels > 2_000_000 and lon_med < -30, "Américas (Norte, Central e Sul)"),
        (n_pixels > 400_000 and lat_med < 0 and lon_med > 100, "Austrália"),
        (n_pixels > 400_000 and lat_med > 55 and lon_med < 0, "Groenlândia"),
        (20_000 <= n_pixels <= 35_000 and 50 <= lat_med <= 60 and -10 <= lon_med <= 5, "Grã-Bretanha"),
        (20_000 <= n_pixels <= 30_000 and 30 <= lat_med <= 42 and 128 <= lon_med <= 145, "Honshu (Japão)"),
        (40_000 <= n_pixels <= 52_000 and -28 <= lat_med <= -10 and 42 <= lon_med <= 52, "Madagascar"),
    ]
    return next((nome for cond, nome in regras if cond), f"Ilha #{val}")


def classificar_macroregiao(lat: float, lon: float) -> str:
    """Mapeia (lat, lon) para macro-região continental."""
    if lat < -60.0:
        return "Antártida"
    if lon < -30.0:
        return "Am. Norte/Central" if lat >= 12.5 else "Am. do Sul"
    if -30.0 <= lon <= 60.0:
        return "Europa" if lat >= 35.0 else "África"
    return "Austrália/Oceania" if (lon > 60.0 and lat < -10.0) else "Ásia"
