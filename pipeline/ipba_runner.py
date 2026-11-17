"""
pipeline/ipba_runner.py
-----------------------
Módulo de extração simplificada de bacias hidrográficas e desembocaduras (sinks)
via IPBA (Invasion Percolation-Based Algorithm) em conformidade com:
  - Oliveira et al. (2019), Sci. Rep. 9:9845
  - Diretrizes do Prof. Erneson Oliveira (áudio 2):
    1. Altura de corte h* = 0 (terra > 0, oceano <= 0).
    2. Sinks = todos os pixels de terra vizinhos de água (linha de costa).
    3. Rodada única do IPBA (sem 2ª rodada recursiva).
    4. Pós-filtro de área acumulada (90% - 95%) para descarte de microbacias litorâneas.
    5. Apenas bacias não endorreicas (que deságuam no oceano).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import binary_dilation

EARTH_RADIUS_KM = 6378.137


class IPBASite:
    __slots__ = ("height", "sigma", "label", "parent", "status", "status1")

    def __init__(self, height: float, rng_perturbation: float):
        self.height = np.float32(height + rng_perturbation)
        self.sigma = np.uint32(1)
        self.label = np.uint32(0)
        self.parent = np.uint32(0)
        self.status = np.uint32(0)
        self.status1 = np.uint8(0)


class IPBAHeap:
    def __init__(self, n: int):
        self.n = n
        self.m = 0
        self.key = np.empty(self.n, dtype=np.float32)
        self.item = np.empty(self.n, dtype=np.uint32)

    def insert(self, key: float, item: int) -> None:
        self.m += 1
        i = self.m - 1
        self.key[i] = key
        self.item[i] = item

        p = (i - 1) // 2
        while i > 0 and self.key[p] > self.key[i]:
            self.key[i], self.key[p] = self.key[p], self.key[i]
            self.item[i], self.item[p] = self.item[p], self.item[i]
            i = p
            p = (i - 1) // 2

    def extract_min(self) -> tuple[float, int]:
        min_key = self.key[0]
        min_item = int(self.item[0])

        self.key[0] = self.key[self.m - 1]
        self.item[0] = self.item[self.m - 1]
        self.m -= 1

        # Heapify down
        i = 0
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i

            if left < self.m and self.key[left] < self.key[smallest]:
                smallest = left
            if right < self.m and self.key[right] < self.key[smallest]:
                smallest = right

            if smallest != i:
                self.key[i], self.key[smallest] = self.key[smallest], self.key[i]
                self.item[i], self.item[smallest] = self.item[smallest], self.item[i]
                i = smallest
            else:
                break

        return min_key, min_item


def crop_gebco(
    bbox: tuple[float, float, float, float],
    gebco_path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float], dict[str, Any]]:
    """
    Recorta o raster GEBCO para a bounding box especificada:
    bbox: (lon_min, lat_min, lon_max, lat_max)
    Retorna: (dem_array, geotransform, metadata)
    """
    gdal.UseExceptions()
    ds = gdal.Open(str(gebco_path), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()

    lon_min, lat_min, lon_max, lat_max = bbox

    # Conversão de coordenadas geográficas para índices de pixel
    col0 = max(0, int(np.floor((lon_min - gt[0]) / gt[1])))
    col1 = min(ds.RasterXSize, int(np.ceil((lon_max - gt[0]) / gt[1])))
    row0 = max(0, int(np.floor((gt[3] - lat_max) / abs(gt[5]))))
    row1 = min(ds.RasterYSize, int(np.ceil((gt[3] - lat_min) / abs(gt[5]))))

    x_size = col1 - col0
    y_size = row1 - row0

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    dem = band.ReadAsArray(col0, row0, x_size, y_size).astype(np.float32)

    # Novo geotransform para o recorte
    new_origin_x = gt[0] + col0 * gt[1]
    new_origin_y = gt[3] + row0 * gt[5]
    new_gt = (new_origin_x, gt[1], gt[2], new_origin_y, gt[4], gt[5])

    meta = {
        "col0": col0,
        "row0": row0,
        "x_size": x_size,
        "y_size": y_size,
        "nodata": nodata,
        "projection": ds.GetProjection(),
    }
    del ds
    return dem, new_gt, meta


def extract_coastal_sinks(
    dem: np.ndarray,
    gt: tuple[float, float, float, float, float, float],
    h_star: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Identifica os sinks costeiros conforme o protocolo do Prof. Erneson:
    - h* = 0: terra = dem > h*, água/oceano = dem <= h*
    - Sinks são todos os pixels de terra vizinhos imediatos (4-conectividade) de água.
    Retorna (sink_grid com IDs 1..N_sinks, DataFrame com metadados dos sinks).
    """
    nrows, ncols = dem.shape
    land = dem > h_star
    water = ~land

    # Dilatação 4-conectividade da água para encontrar a borda litorânea em terra
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    coast_mask = land & binary_dilation(water, structure=struct4)

    sink_grid = np.zeros((nrows, ncols), dtype=np.uint32)
    sink_rows, sink_cols = np.nonzero(coast_mask)
    n_sinks = len(sink_rows)

    sink_ids = np.arange(1, n_sinks + 1, dtype=np.uint32)
    sink_grid[sink_rows, sink_cols] = sink_ids

    # Coordenadas geográficas dos centros dos pixels de sink
    sink_lons = gt[0] + (sink_cols + 0.5) * gt[1]
    sink_lats = gt[3] + (sink_rows + 0.5) * gt[5]
    sink_elevations = dem[sink_rows, sink_cols]

    sinks_df = pd.DataFrame(
        {
            "sink_id": sink_ids,
            "row": sink_rows,
            "col": sink_cols,
            "longitude": sink_lons,
            "latitude": sink_lats,
            "elevation": sink_elevations,
        }
    )

    return sink_grid, sinks_df


def run_ipba(
    dem: np.ndarray,
    sink_grid: np.ndarray,
    h_star: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Executa a rodada única do Invasion Percolation-Based Algorithm (IPBA) sobre o DEM.
    Cada pixel de terra drena pelo caminho topográfico de menor resistência até atingir um sink litorâneo.
    Retorna a matriz 2D basin_grid, onde cada pixel de terra contém o ID do sink receptor (0 = mar).
    """
    nrows, ncols = dem.shape
    n = nrows * ncols
    land = dem > h_star

    rng = np.random.default_rng(seed)
    perturbations = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    # Inicialização dos sítios na rede regular
    flat_dem = dem.ravel()
    sites = [IPBASite(float(flat_dem[k]), float(perturbations[k])) for k in range(n)]

    # Atribuir nodata / oceano
    for k in range(n):
        if not land.flat[k]:
            sites[k].height = np.float32(-99999.0)
            sites[k].sigma = np.uint32(0)

    # Atribuir rótulos dos sinks
    flat_sinks = sink_grid.ravel()
    for k in range(n):
        if flat_sinks[k] > 0:
            sites[k].label = np.uint32(flat_sinks[k])

    heap = IPBAHeap(n)
    burner: list[int] = []

    # Vizinhança 4-conectada plana (sem PBC horizontal para recortes regionais)
    def _get_neighbors(k: int) -> list[int]:
        r = k // ncols
        c = k % ncols
        nbrs = []
        if r > 0:
            nbrs.append(k - ncols)
        if r < nrows - 1:
            nbrs.append(k + ncols)
        if c > 0:
            nbrs.append(k - 1)
        if c < ncols - 1:
            nbrs.append(k + 1)
        return nbrs

    # Invasão por percolação a partir de cada pixel de terra
    for k in range(n):
        if sites[k].sigma == 1:
            for b in burner:
                sites[b].status1 = np.uint8(0)
            burner.clear()
            heap.m = 0

            burner.append(k)
            sites[k].status1 = np.uint8(1)
            t_old = k
            heap.insert(sites[k].height, k)

            stop = False
            w = 0
            while heap.m > 0 and not stop:
                _, t = heap.extract_min()
                sites[t].status1 = np.uint8(2)
                sites[t].parent = np.uint32(t_old)
                t_old = t

                if sites[t].sigma >= 2:
                    w = int(sites[t].sigma) - 2
                    stop = True
                elif sites[t].label > 0:
                    w = t
                    stop = True

                if not stop:
                    for s in _get_neighbors(t):
                        if sites[s].status1 == 0:
                            burner.append(s)
                            sites[s].status1 = np.uint8(1)
                            heap.insert(sites[s].height, s)

            if stop:
                i = t
                target_status = sites[w].label if sites[w].label > 0 else sites[w].status
                while sites[i].parent != i:
                    sites[i].sigma = np.uint32(w + 2)
                    sites[i].status = target_status
                    i = int(sites[i].parent)
                sites[i].sigma = np.uint32(w + 2)
                sites[i].status = target_status

    basin_grid = np.zeros((nrows, ncols), dtype=np.uint32)
    for k in range(n):
        basin_grid.flat[k] = sites[k].status

    return basin_grid


def spherical_triangle_row_areas(
    lats_center_deg: np.ndarray,
    dlat_deg: float,
    dlon_deg: float,
    R: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """
    Calcula a área exata de cada sítio (pixel) em km² pela composição de dois
    triângulos esféricos (Oliveira et al., 2019, Sci. Rep. 9:9845, Equações 3 e 4).

    Para cada linha centrada em `lats_center_deg` com passo latitudinal `dlat_deg`
    e passo longitudinal `dlon_deg`:
      - As arestas geodésicas (grande círculo) a, b, c são obtidas via fórmula de Haversine (Eq. 4).
      - A área de cada um dos dois triângulos esféricos que compõem o sítio é obtida
        pelo Teorema de L'Huilier (Todhunter, 1863; Eq. 3):
        A = 4 * R² * arctan(sqrt(tan(s/2) * tan(s_a/2) * tan(s_b/2) * tan(s_c/2))).
    """
    lat1 = np.radians(lats_center_deg - 0.5 * abs(dlat_deg))
    lat2 = np.radians(lats_center_deg + 0.5 * abs(dlat_deg))
    dlon = float(np.radians(abs(dlon_deg)))

    def _haversine_unit(phi_i: np.ndarray, phi_j: np.ndarray, dlambda: float) -> np.ndarray:
        dphi = phi_j - phi_i
        h = np.sin(0.5 * dphi) ** 2 + np.cos(phi_i) * np.cos(phi_j) * (np.sin(0.5 * dlambda) ** 2)
        return 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    def _lhuilier_area(a_u: np.ndarray, b_u: np.ndarray, c_u: np.ndarray) -> np.ndarray:
        s = 0.5 * (a_u + b_u + c_u)
        sa = np.maximum(0.0, s - a_u)
        sb = np.maximum(0.0, s - b_u)
        sc = np.maximum(0.0, s - c_u)
        prod = np.tan(0.5 * s) * np.tan(0.5 * sa) * np.tan(0.5 * sb) * np.tan(0.5 * sc)
        return 4.0 * (R ** 2) * np.arctan(np.sqrt(np.maximum(0.0, prod)))

    edge_south = _haversine_unit(lat1, lat1, dlon)
    edge_north = _haversine_unit(lat2, lat2, dlon)
    edge_merid = _haversine_unit(lat1, lat2, 0.0)
    edge_diag = _haversine_unit(lat1, lat2, dlon)

    return _lhuilier_area(edge_south, edge_merid, edge_diag) + _lhuilier_area(edge_north, edge_merid, edge_diag)


def filter_basins_by_area(
    basin_grid: np.ndarray,
    sinks_df: pd.DataFrame,
    gt: tuple[float, float, float, float, float, float],
    cumulative_area_threshold: float = 0.95,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Calcula as áreas das bacias hidrográficas pela composição de dois triângulos
    esféricos (Teorema de L'Huilier + Haversine, Oliveira et al., 2019),
    ordena por tamanho e retém apenas as maiores bacias que somam `cumulative_area_threshold`
    (ex.: 90% ou 95%) da área terrestre acumulada.

    Retorna:
      - filtered_basin_grid: matriz com IDs das bacias mantidas (0 = oceano ou microbacia descartada)
      - basins_summary_df: tabela com métricas de cada bacia retida
      - retained_sinks_df: tabela de sinks correspondentes às bacias retidas
    """
    nrows, ncols = basin_grid.shape
    row_indices = np.arange(nrows)
    lats = gt[3] + (row_indices + 0.5) * gt[5]

    # Área exata do sítio em km² via composição de dois triângulos esféricos (L'Huilier + Haversine)
    pixel_area_1d = spherical_triangle_row_areas(lats, abs(gt[5]), gt[1], R=EARTH_RADIUS_KM)
    pixel_area_grid = np.broadcast_to(pixel_area_1d[:, None], (nrows, ncols))

    land_mask = basin_grid > 0
    total_land_area = pixel_area_grid[land_mask].sum()

    labels, counts = np.unique(basin_grid[land_mask], return_counts=True)
    areas = np.bincount(basin_grid[land_mask], weights=pixel_area_grid[land_mask])[labels]

    df_all = pd.DataFrame(
        {
            "basin_id": labels,
            "pixel_count": counts,
            "area_km2": areas,
        }
    ).sort_values("area_km2", ascending=False).reset_index(drop=True)

    df_all["area_fraction"] = df_all["area_km2"] / total_land_area
    df_all["cumulative_fraction"] = df_all["area_fraction"].cumsum()

    # Filtro de corte acumulado (95% ou 90%)
    retained_mask = df_all["cumulative_fraction"] <= cumulative_area_threshold
    # Garante a inclusão da bacia limítrofe que cruza o limiar
    cut_idx = int(np.searchsorted(df_all["cumulative_fraction"], cumulative_area_threshold))
    retained_mask.iloc[: cut_idx + 1] = True

    basins_summary_df = df_all[retained_mask].copy().reset_index(drop=True)
    retained_ids = set(basins_summary_df["basin_id"])

    # Filtrar matriz de bacias
    filtered_basin_grid = np.where(np.isin(basin_grid, list(retained_ids)), basin_grid, np.uint32(0))

    # Filtrar tabela de sinks
    retained_sinks_df = (
        sinks_df[sinks_df["sink_id"].isin(retained_ids)]
        .merge(basins_summary_df[["basin_id", "area_km2", "area_fraction"]], left_on="sink_id", right_on="basin_id")
        .sort_values("area_km2", ascending=False)
        .reset_index(drop=True)
    )

    return filtered_basin_grid, basins_summary_df, retained_sinks_df
