"""
pipeline/cca_calibrator.py
--------------------------
Módulo de calibração de parâmetros (D* x L) do City Clustering Algorithm (CCA)
via Lei de Zipf, em conformidade com:
  - Oliveira et al. (2014), Sci. Rep. 4:4235
  - Oliveira et al. (2018), R. Soc. Open Sci. 5:180468
  - Operti, Oliveira et al. (2017), arXiv:1706.05139
  - Diretrizes do Prof. Erneson Oliveira (áudio 1):
    1. Calibrar D* e L separadamente para cada porção de terra / região.
    2. Grade 10x10: 10 valores de L (ex.: 0 a 9 km) e 10 de D* (50 a 1500 hab/km²).
    3. Ajuste da Lei de Zipf (log10 P ~ -alpha log10 r) nos maiores aglomerados.
    4. Montagem das matrizes 10x10 de expoentes (alpha) e coeficientes de determinação (R²).
    5. Escolha do par ótimo (D*, L) com expoente alpha mais próximo de 1.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from osgeo import gdal
from scipy import stats

from .ipba_runner import EARTH_RADIUS_KM, spherical_triangle_row_areas

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_CCA_DIR = REPO_ROOT / "externos" / "CCA" / "cca"
if str(EXTERNAL_CCA_DIR) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_CCA_DIR))

from city_clustering import burn  # Reutilização do algoritmo oficial do orientador
from geracao_graficos.estilo_erneson import (
    anotar_expoente,
    aplicar_estilo,
    colorbar_direita,
    eixos_log10,
    referencia_inclinacao_1,
    reta_ols,
    rotulo_canto,
    rotulo_painel,
    salvar,
    scatter_vazado,
)


def crop_gpw(
    bbox: tuple[float, float, float, float],
    gpw_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float, float, float], dict[str, Any]]:
    """
    Recorta o raster GPW v4 para a bounding box especificada e calcula
    a área esférica exata de cada pixel em km² pela composição de dois
    triângulos esféricos (Teorema de L'Huilier + Haversine, Oliveira et al., 2019).

    bbox: (lon_min, lat_min, lon_max, lat_max)
    Retorna: (pop, pixel_area, lats_1d, lons_1d, geotransform, metadata)
    """
    gdal.UseExceptions()
    ds = gdal.Open(str(gpw_path), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()

    lon_min, lat_min, lon_max, lat_max = bbox

    col0 = max(0, int(np.floor((lon_min - gt[0]) / gt[1])))
    col1 = min(ds.RasterXSize, int(np.ceil((lon_max - gt[0]) / gt[1])))
    row0 = max(0, int(np.floor((gt[3] - lat_max) / abs(gt[5]))))
    row1 = min(ds.RasterYSize, int(np.ceil((gt[3] - lat_min) / abs(gt[5]))))

    x_size = col1 - col0
    y_size = row1 - row0

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    raw_pop = band.ReadAsArray(col0, row0, x_size, y_size).astype(np.float32)

    # Tratar nodata e valores negativos
    pop = np.where((raw_pop == nodata) | (raw_pop < 0), 0.0, raw_pop)

    new_origin_x = gt[0] + col0 * gt[1]
    new_origin_y = gt[3] + row0 * gt[5]
    new_gt = (new_origin_x, gt[1], gt[2], new_origin_y, gt[4], gt[5])

    # Vetores 1D de coordenadas dos centros dos pixels
    lons_1d = new_gt[0] + (np.arange(x_size) + 0.5) * new_gt[1]
    lats_1d = new_gt[3] + (np.arange(y_size) + 0.5) * new_gt[5]

    # Área exata do sítio em km² via composição de dois triângulos esféricos (L'Huilier + Haversine)
    area_1d = spherical_triangle_row_areas(lats_1d, abs(new_gt[5]), new_gt[1], R=EARTH_RADIUS_KM)
    pixel_area = np.broadcast_to(area_1d[:, None], (y_size, x_size))

    meta = {
        "col0": col0,
        "row0": row0,
        "x_size": x_size,
        "y_size": y_size,
        "total_population": float(pop.sum()),
    }
    del ds
    return pop, pixel_area, lats_1d, lons_1d, new_gt, meta


def calibrate_zipf_grid(
    pop: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    pixel_area: np.ndarray,
    d_values: Sequence[float] | None = None,
    l_values: Sequence[float] | None = None,
    top_n: int = 100,
    min_clusters: int = 20,
    d_max: float = 5000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, tuple[float, float]]:
    """
    Executa a calibração do CCA na grade de parâmetros (D* x L),
    com suporte a ajuste dinâmico com D* até d_max (padrão: 5000 hab/km²),
    calculando o ajuste da Lei de Zipf para cada par.

    Retorna:
      - matrix_alpha: matriz 2D com expoentes de Zipf
      - matrix_r2: matriz 2D com coeficientes de determinação R²
      - matrix_se: matriz 2D com erros-padrão do ajuste
      - df_grid: tabela consolidada com todas as rodadas
      - (best_d, best_l): par ótimo minimizando |alpha - 1| com R² >= 0.80
    """
    if d_values is None:
        # Grade abrangente com ajuste dinâmico até 5000 hab/km², incluindo 1010
        d_values = [100.0, 200.0, 500.0, 800.0, 1010.0, 1500.0, 2000.0, 3000.0, 4000.0, min(5000.0, d_max)]
    if l_values is None:
        # Valores de raio espacial incluindo passos finos e 1.7 km
        l_values = [1.0, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    n_d = len(d_values)
    n_l = len(l_values)

    matrix_alpha = np.full((n_d, n_l), np.nan, dtype=float)
    matrix_r2 = np.full((n_d, n_l), np.nan, dtype=float)
    matrix_se = np.full((n_d, n_l), np.nan, dtype=float)

    density = pop / pixel_area
    records = []

    best_score = float("inf")
    best_pair = (float(d_values[0]), float(l_values[0]))

    for i, d_star in enumerate(d_values):
        rows, cols = np.nonzero(density > d_star)
        n_act = len(rows)
        if n_act == 0:
            continue

        act_lats = lats_1d[rows]
        act_lons = lons_1d[cols]
        act_pops = pop[rows, cols]

        for j, l_cutoff in enumerate(l_values):
            if l_cutoff == 0.0:
                # Caso limite L = 0: sítios isolados sem agregação espacial
                labels = np.arange(n_act, dtype=np.int64)
            else:
                labels = burn(act_lats, act_lons, l_cutoff)

            n_clusters = int(labels.max()) + 1 if labels.size else 0
            if n_clusters < min_clusters:
                records.append(
                    {
                        "d_star": d_star,
                        "l_cutoff": l_cutoff,
                        "n_clusters": n_clusters,
                        "alpha": np.nan,
                        "r2": np.nan,
                        "stderr": np.nan,
                        "p_value": np.nan,
                    }
                )
                continue

            cluster_pops = np.bincount(labels, weights=act_pops, minlength=n_clusters)
            # Ordenar população dos aglomerados em ordem decrescente
            valid_pops = cluster_pops[cluster_pops > 0]
            valid_pops.sort()
            sorted_pops = valid_pops[::-1]

            n_fit = min(top_n, len(sorted_pops))
            if n_fit < min_clusters:
                continue

            p_fit = sorted_pops[:n_fit]
            ranks = np.arange(1, n_fit + 1, dtype=float)

            # Ajuste linear da Lei de Zipf: log10 P = log10 P1 - alpha * log10 r
            log_r = np.log10(ranks)
            log_p = np.log10(p_fit)

            slope, intercept, r_val, p_val, se = stats.linregress(log_r, log_p)
            alpha = float(-slope)  # Expoente positivo de Zipf
            r2 = float(r_val**2)

            matrix_alpha[i, j] = alpha
            matrix_r2[i, j] = r2
            matrix_se[i, j] = float(se)

            records.append(
                {
                    "d_star": d_star,
                    "l_cutoff": l_cutoff,
                    "n_clusters": n_clusters,
                    "alpha": alpha,
                    "r2": r2,
                    "stderr": float(se),
                    "p_value": float(p_val),
                }
            )

            # Critério de seleção: expoente mais próximo de 1 com bom ajuste
            score = abs(alpha - 1.0)
            if r2 >= 0.80 and score < best_score:
                best_score = score
                best_pair = (float(d_star), float(l_cutoff))

    df_grid = pd.DataFrame(records)
    return matrix_alpha, matrix_r2, matrix_se, df_grid, best_pair


def run_calibrated_cca(
    pop: np.ndarray,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
    pixel_area: np.ndarray,
    d_cutoff: float,
    l_cutoff: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Executa o CCA com os parâmetros calibrados (D*, L) e retorna:
      - grid_labels: matriz 2D com IDs de aglomerado (0 = não urbano, 1..C = manchas)
      - clusters_df: tabela com métricas de cada mancha urbana
    """
    density = pop / pixel_area
    rows, cols = np.nonzero(density > d_cutoff)
    n_act = len(rows)

    if n_act == 0:
        return np.zeros(pop.shape, dtype=np.uint32), pd.DataFrame()

    act_lats = lats_1d[rows]
    act_lons = lons_1d[cols]
    act_pops = pop[rows, cols]
    act_areas = pixel_area[rows, cols]

    if l_cutoff == 0.0:
        labels = np.arange(n_act, dtype=np.int64)
    else:
        labels = burn(act_lats, act_lons, l_cutoff)

    c = int(labels.max()) + 1 if labels.size else 0

    grid_labels = np.zeros(pop.shape, dtype=np.uint32)
    grid_labels[rows, cols] = (labels + 1).astype(np.uint32)

    cluster_pops = np.bincount(labels, weights=act_pops, minlength=c)
    cluster_areas = np.bincount(labels, weights=act_areas, minlength=c)
    cluster_counts = np.bincount(labels, minlength=c)

    # Centróides ponderados por população
    sum_lat = np.bincount(labels, weights=act_lats * act_pops, minlength=c)
    sum_lon = np.bincount(labels, weights=act_lons * act_pops, minlength=c)
    safe_pops = np.where(cluster_pops == 0, 1.0, cluster_pops)
    centroid_lat = sum_lat / safe_pops
    centroid_lon = sum_lon / safe_pops

    clusters_df = (
        pd.DataFrame(
            {
                "cluster_id": np.arange(1, c + 1, dtype=np.uint32),
                "population": cluster_pops,
                "area_km2": cluster_areas,
                "pixel_count": cluster_counts,
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
            }
        )
        .sort_values("population", ascending=False)
        .reset_index(drop=True)
    )

    return grid_labels, clusters_df


def plot_zipf_calibration(
    matrix_alpha: np.ndarray,
    matrix_r2: np.ndarray,
    d_values: Sequence[float],
    l_values: Sequence[float],
    opt_d: float,
    opt_l: float,
    clusters_df: pd.DataFrame,
    output_basepath: str | Path,
    region_name: str = "North & Central America",
) -> tuple[Path, Path]:
    """
    Gera o painel científico de calibração D* x L contendo:
      (a) Heatmap do expoente de Zipf alpha (colormap plasma) com contorno em alpha = 1.
      (b) Heatmap do R² (colormap plasma) e estrela preta vazada no par ótimo.
      (c) Curva rank-size da distribuição das manchas urbanas no par ótimo com reta OLS.
    """
    aplicar_estilo()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # Painel (a): Matriz de Alpha
    im_a = ax_a.imshow(
        matrix_alpha,
        cmap="plasma",
        origin="lower",
        norm=Normalize(vmin=0.6, vmax=1.4),
        aspect="auto",
        interpolation="nearest",
    )
    # Contorno em alpha = 1
    ax_a.contour(matrix_alpha, levels=[1.0], colors="black", linewidths=1.2, origin="lower")
    colorbar_direita(fig, ax_a, im_a, r"Zipf exponent $\alpha$")
    ax_a.set_xticks(np.arange(len(l_values)))
    ax_a.set_xticklabels([f"{x:g}" for x in l_values])
    ax_a.set_yticks(np.arange(len(d_values)))
    ax_a.set_yticklabels([f"{int(y)}" for y in d_values])
    ax_a.set_xlabel(r"Cluster distance cutoff $\ell$ (km)")
    ax_a.set_ylabel(r"Density cutoff $D^*$ (inhab./km$^2$)")
    rotulo_painel(ax_a, "a")

    # Painel (b): Matriz de R²
    im_b = ax_b.imshow(
        matrix_r2,
        cmap="plasma",
        origin="lower",
        norm=Normalize(vmin=0.7, vmax=1.0),
        aspect="auto",
        interpolation="nearest",
    )
    colorbar_direita(fig, ax_b, im_b, r"Coefficient of determination $R^2$")
    ax_b.set_xticks(np.arange(len(l_values)))
    ax_b.set_xticklabels([f"{x:g}" for x in l_values])
    ax_b.set_yticks(np.arange(len(d_values)))
    ax_b.set_yticklabels([f"{int(y)}" for y in d_values])
    ax_b.set_xlabel(r"Cluster distance cutoff $\ell$ (km)")
    rotulo_painel(ax_b, "b")

    # Destacar par ótimo com estrela vazada
    opt_j = list(l_values).index(opt_l) if opt_l in l_values else 0
    opt_i = list(d_values).index(opt_d) if opt_d in d_values else 0
    ax_b.scatter(opt_j, opt_i, marker="*", s=140, edgecolors="white", facecolors="black", lw=1.2, zorder=50)

    # Painel (c): Curva Rank-Size no par ótimo
    top_pops = clusters_df["population"].values[:100]
    ranks = np.arange(1, len(top_pops) + 1, dtype=float)
    log_r = np.log10(ranks)
    log_p = np.log10(top_pops)

    scatter_vazado(ax_c, log_r, log_p, c="C0", marker="o", s=25, label="Urban clusters")
    ols_res = reta_ols(ax_c, log_r, log_p, color="black", lw=1.2)
    referencia_inclinacao_1(ax_c, log_r.min(), log_r.max(), log_p.max())

    eixos_log10(ax_c, xlabel=r"$\log_{10} r$ (rank)", ylabel=r"$\log_{10} P$ (population)")
    rotulo_painel(ax_c, "c")
    rotulo_canto(ax_c, region_name, canto="upper right", bbox=False)
    anotar_expoente(
        ax_c,
        f"$\\alpha = {abs(ols_res['slope']):.2f} \\pm {ols_res['stderr']:.2f}$\n$R^2 = {ols_res['r2']:.2f}$\n$D^* = {int(opt_d)}$ inhab./km$^2$\n$\\ell = {opt_l:g}$ km",
        canto="lower left",
    )

    plt.tight_layout()
    return salvar(fig, output_basepath)
