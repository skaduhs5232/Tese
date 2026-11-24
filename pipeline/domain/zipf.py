"""Calibração de parâmetros (D*, L) do CCA via Lei de Zipf (domínio puro)."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..config import ZIPF_D_PADRAO, ZIPF_FALLBACK, ZIPF_L_PADRAO

AgruparCCAFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def ajustar_zipf(cluster_pops: np.ndarray, top_n: int = 100, min_clusters: int = 20) -> dict[str, float] | None:
    """Ajusta log10(P) = log10(P1) - alpha * log10(r) nos maiores aglomerados urbanos."""
    valid = np.sort(cluster_pops[cluster_pops > 0])[::-1]
    n_fit = min(top_n, len(valid))
    if n_fit < min_clusters:
        return None
    fit = stats.linregress(np.log10(np.arange(1, n_fit + 1, dtype=float)), np.log10(valid[:n_fit]))
    return {
        "alpha": float(-fit.slope),
        "r2": float(fit.rvalue**2),
        "stderr": float(fit.stderr),
        "p_value": float(fit.pvalue),
    }


def selecionar_par_otimo_zipf(
    df_grid: pd.DataFrame,
    fallback: tuple[float, float] = ZIPF_FALLBACK,
    r2_min: float = 0.80,
) -> tuple[float, float]:
    """Escolhe (D*, L) com R² >= r2_min que minimiza |alpha - 1|."""
    valid = df_grid[df_grid["r2"] >= r2_min] if len(df_grid) else df_grid
    if len(valid) == 0:
        return float(fallback[0]), float(fallback[1])
    best = valid.iloc[int((valid["alpha"] - 1.0).abs().argmin())]
    return float(best["d_star"]), float(best["l_cutoff"])


def calibrar_grade_zipf(
    pop: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    pixel_area: np.ndarray,
    agrupar_fn: AgruparCCAFn,
    d_values: Sequence[float] = ZIPF_D_PADRAO,
    l_values: Sequence[float] = ZIPF_L_PADRAO,
    top_n: int = 100,
    min_clusters: int = 20,
    fallback: tuple[float, float] = ZIPF_FALLBACK,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Varre a grade (D*, L), calcula o ajuste de Zipf e retorna (df_grid, par_otimo)."""
    density = pop / pixel_area
    records = []

    for d_star in d_values:
        rows, cols = np.nonzero(density > d_star)
        if len(rows) == 0:
            continue
        act_lats, act_lons, act_pops = lats_1d[rows], lons_1d[cols], pop[rows, cols]

        for l_cutoff in l_values:
            labels = agrupar_fn(act_lats, act_lons, l_cutoff)
            n_clusters = int(labels.max()) + 1 if labels.size else 0
            if n_clusters < min_clusters:
                records.append({"d_star": d_star, "l_cutoff": l_cutoff, "n_clusters": n_clusters, "alpha": np.nan, "r2": np.nan, "stderr": np.nan, "p_value": np.nan})
                continue

            pops = np.bincount(labels, weights=act_pops, minlength=n_clusters)
            fit = ajustar_zipf(pops, top_n=top_n, min_clusters=min_clusters)
            if fit is not None:
                records.append({"d_star": d_star, "l_cutoff": l_cutoff, "n_clusters": n_clusters, **fit})

    df_grid = pd.DataFrame(records)
    return df_grid, selecionar_par_otimo_zipf(df_grid, fallback=fallback)


def rotular_clusters_cca(
    pop: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    pixel_area: np.ndarray,
    d_cutoff: float,
    l_cutoff: float,
    agrupar_fn: AgruparCCAFn,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Executa o CCA com (D*, L) calibrados e retorna (grid_labels, clusters_df)."""
    rows, cols = np.nonzero((pop / pixel_area) > d_cutoff)
    if len(rows) == 0:
        return np.zeros(pop.shape, dtype=np.uint32), pd.DataFrame()

    act_lats, act_lons = lats_1d[rows], lons_1d[cols]
    act_pops, act_areas = pop[rows, cols], pixel_area[rows, cols]
    labels = agrupar_fn(act_lats, act_lons, l_cutoff)
    c = int(labels.max()) + 1 if labels.size else 0

    grid_labels = np.zeros(pop.shape, dtype=np.uint32)
    grid_labels[rows, cols] = (labels + 1).astype(np.uint32)

    pops = np.bincount(labels, weights=act_pops, minlength=c)
    safe_pops = np.where(pops == 0, 1.0, pops)
    clusters_df = (
        pd.DataFrame({
            "cluster_id": np.arange(1, c + 1, dtype=np.uint32),
            "population": pops,
            "area_km2": np.bincount(labels, weights=act_areas, minlength=c),
            "pixel_count": np.bincount(labels, minlength=c),
            "centroid_latitude": np.bincount(labels, weights=act_lats * act_pops, minlength=c) / safe_pops,
            "centroid_longitude": np.bincount(labels, weights=act_lons * act_pops, minlength=c) / safe_pops,
        })
        .sort_values("population", ascending=False)
        .reset_index(drop=True)
    )
    return grid_labels, clusters_df
