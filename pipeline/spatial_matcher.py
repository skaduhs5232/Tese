"""
pipeline/spatial_matcher.py
---------------------------
Módulo de integração espacial entre bacias hidrográficas, manchas urbanas (CCA)
e dados de microplásticos costeiros (NOAA Nurdle Patrol):
  - Particionamento exato das manchas urbanas ao longo das linhas de cumeada das bacias.
  - Divisão proporcional da população pixel a pixel por bacia receptora.
  - Ingestão e padronização da base NOAA Nurdle Patrol (pieces/10 mins).
  - Expansão de raio exclusivo dinâmico (Voronoi esférico) sem invasão de foz vizinha
    e com garantia matemática de zero contagem duplicada.
  - Agregação consolidada por bacia hidrográfica para análise alométrica.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6378.137


def _lat_lon_to_unit_vectors(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Converte latitudes e longitudes (graus) para vetores unitários 3D na esfera."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    return np.column_stack(
        [
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ]
    )


def chord_to_haversine_distance(chord_dist: np.ndarray | float) -> np.ndarray | float:
    """Converte distância euclidiana de corda 3D para distância geodésica em km."""
    # corda = 2 * sin(theta / 2) -> theta = 2 * arcsin(corda / 2)
    sin_half = np.clip(chord_dist / 2.0, 0.0, 1.0)
    theta = 2.0 * np.arcsin(sin_half)
    return theta * EARTH_RADIUS_KM


def haversine_to_chord_distance(km_dist: float) -> float:
    """Converte distância geodésica em km para distância de corda 3D."""
    theta = km_dist / EARTH_RADIUS_KM
    return float(2.0 * np.sin(theta / 2.0))


def split_urban_clusters_by_basins(
    grid_clusters: np.ndarray,
    pop_grid: np.ndarray,
    basin_grid: np.ndarray,
    gt_gpw: tuple[float, float, float, float, float, float],
    gt_gebco: tuple[float, float, float, float, float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide as manchas urbanas do CCA que cruzam divisores de bacias hidrográficas:
    - Conta a população de cada pixel urbano e atribui à respectiva bacia do IPBA.
    - Preserva rigorosamente a conservação da população (zero perda ou contagem fantasma).

    Retorna:
      - basin_population_df: [basin_id, urban_population, total_population, urban_pixels, total_pixels]
      - cluster_splits_df: detalhamento de quais manchas foram particionadas e em quais bacias
    """
    gpw_h, gpw_w = pop_grid.shape
    gebco_h, gebco_w = basin_grid.shape

    # Fator de escala exato entre GPW (30 arc-sec) e GEBCO (2 arc-min)
    scale_y = int(round(abs(gt_gebco[5]) / abs(gt_gpw[5])))
    scale_x = int(round(gt_gebco[1] / gt_gpw[1]))

    # Mapear cada linha e coluna de GPW para GEBCO
    gpw_rows = np.arange(gpw_h)
    gpw_cols = np.arange(gpw_w)

    gebco_r = np.clip(gpw_rows // scale_y, 0, gebco_h - 1)
    gebco_c = np.clip(gpw_cols // scale_x, 0, gebco_w - 1)

    # Reamostrar o rótulo de bacia para a grade GPW
    resampled_basins = basin_grid[gebco_r[:, None], gebco_c[None, :]]

    valid_mask = resampled_basins > 0
    urban_mask = valid_mask & (grid_clusters > 0)

    # 1. Agregação por Bacia
    active_basins = np.unique(resampled_basins[valid_mask])
    max_b = int(active_basins.max()) if len(active_basins) else 0

    total_pop_by_basin = np.bincount(
        resampled_basins[valid_mask],
        weights=pop_grid[valid_mask],
        minlength=max_b + 1,
    )
    total_px_by_basin = np.bincount(
        resampled_basins[valid_mask],
        minlength=max_b + 1,
    )

    urban_pop_by_basin = np.bincount(
        resampled_basins[urban_mask],
        weights=pop_grid[urban_mask],
        minlength=max_b + 1,
    )
    urban_px_by_basin = np.bincount(
        resampled_basins[urban_mask],
        minlength=max_b + 1,
    )

    basin_population_df = pd.DataFrame(
        {
            "basin_id": active_basins,
            "urban_population": urban_pop_by_basin[active_basins],
            "total_population": total_pop_by_basin[active_basins],
            "urban_pixels": urban_px_by_basin[active_basins],
            "total_pixels": total_px_by_basin[active_basins],
        }
    )
    basin_population_df["urban_fraction"] = np.where(
        basin_population_df["total_population"] > 0,
        basin_population_df["urban_population"] / basin_population_df["total_population"],
        0.0,
    )

    # 2. Detalhamento de Manchas Particionadas (vetorizado em Pandas)
    urban_cids = grid_clusters[urban_mask]
    urban_bids = resampled_basins[urban_mask]
    urban_pops = pop_grid[urban_mask]

    df_px = pd.DataFrame(
        {
            "cluster_id": urban_cids,
            "basin_id": urban_bids,
            "pop": urban_pops,
        }
    )
    cluster_splits_df = (
        df_px.groupby(["cluster_id", "basin_id"], as_index=False)
        .agg(
            cluster_pop_in_basin=("pop", "sum"),
            pixels_in_basin=("pop", "count"),
        )
    )
    return basin_population_df, cluster_splits_df


def load_noaa_nurdle_patrol(
    geojson_path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """
    Carrega e padroniza as observações de Nurdle Patrol da NOAA NCEI:
    - Filtra estritamente Medium == 'Nurdle Patrol'.
    - Extrai Standardized_Nurdle__Amount (pieces/10 mins), coordenadas e datas.
    - Se informado bbox (lon_min, lat_min, lon_max, lat_max), filtra espacialmente.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    records = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        medium = str(props.get("Medium", "")).strip()
        if medium != "Nurdle Patrol":
            continue

        # Priorizar coordenadas geográficas em graus das propriedades (EPSG:4326)
        lat = props.get("Latitude__degree_")
        lon = props.get("Longitude_degree_")

        if lat is None or lon is None:
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates")
            if coords and len(coords) >= 2:
                # Verificar se estão em graus
                if abs(coords[0]) <= 180.0 and abs(coords[1]) <= 90.0:
                    lon, lat = coords[0], coords[1]

        if lon is None or lat is None:
            continue

        lon = float(lon)
        lat = float(lat)

        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue

        amt = props.get("Standardized_Nurdle__Amount")
        if amt is None:
            amt = props.get("Microplastics_measurement")

        try:
            amt_val = float(amt)
        except (ValueError, TypeError):
            continue

        records.append(
            {
                "unique_id": props.get("UniqueID") or props.get("ObjectId"),
                "latitude": lat,
                "longitude": lon,
                "nurdles_per_10min": amt_val,
                "date": props.get("Date__MM_DD_YYYY_"),
                "volunteers": props.get("Volunteers_Number"),
                "collecting_time_min": props.get("Collecting_Time__min_"),
                "state": props.get("State"),
                "country": props.get("Country"),
            }
        )

    df_nurdles = pd.DataFrame(records)
    return df_nurdles.reset_index(drop=True)


def match_sinks_with_individual_radius(
    sinks_df: pd.DataFrame,
    microplastics_df: pd.DataFrame,
    max_cap_km: float = 50.0,
    step_km: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pareia observações de microplásticos aos sinks com raio adaptativo individualizado por sink:
    1. Para cada sink s_i, localiza a distância mínima até qualquer sink pertencente a OUTRA bacia b_j != b_i.
    2. O limite estrito de não-invasão é R_max(s_i) = min(max_cap_km, 0.5 * dist_min_outra_bacia).
    3. Cada amostra é associada estritamente ao seu sink mais próximo (partição de Voronoi).
    4. O raio de cada sink é expandido progressivamente até R_max(s_i), capturando o maior número
       de pontos possível sem qualquer risco de sobreposição ou dupla contagem com outras fozes.
    5. Cada sink recebe seu raio efetivo final r_i* e sua contagem individual.

    Retorna:
      - matched_mp: tabela de amostras capturadas com coordenadas, bacia e distância à foz
      - basin_summary: métricas agregadas por bacia hidrográfica
      - sinks_with_radii: tabela de sinks enriquecida com o raio individual e amostras capturadas
    """
    if len(sinks_df) == 0 or len(microplastics_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), sinks_df.copy()

    sinks_clean = sinks_df.copy().reset_index(drop=True)
    sink_lats = sinks_clean["latitude"].values
    sink_lons = sinks_clean["longitude"].values
    sink_xyz = _lat_lon_to_unit_vectors(sink_lats, sink_lons)
    sink_tree = cKDTree(sink_xyz)

    # 1. Distância ao sink mais próximo de OUTRA bacia
    k_search = min(100, len(sinks_clean))
    dists_chord, indices = sink_tree.query(sink_xyz, k=k_search)
    basin_ids = sinks_clean["basin_id"].values

    safe_max_radii_km = np.full(len(sinks_clean), max_cap_km, dtype=float)
    for i in range(len(sinks_clean)):
        my_b = basin_ids[i]
        diff_mask = basin_ids[indices[i]] != my_b
        if np.any(diff_mask):
            first_diff_idx = np.where(diff_mask)[0][0]
            dist_km = chord_to_haversine_distance(dists_chord[i, first_diff_idx])
            safe_max_radii_km[i] = min(max_cap_km, 0.5 * dist_km)

    sinks_clean["safe_max_radius_km"] = safe_max_radii_km

    # 2. Atribuir pontos de microplásticos ao sink mais próximo
    mp_lats = microplastics_df["latitude"].values
    mp_lons = microplastics_df["longitude"].values
    mp_xyz = _lat_lon_to_unit_vectors(mp_lats, mp_lons)

    mp_dists_chord, nearest_sink_indices = sink_tree.query(mp_xyz, k=1)
    mp_dists_km = chord_to_haversine_distance(mp_dists_chord)

    # Ponto é capturado se dist <= safe_max_radius_km do seu sink exclusivo
    allowed_radii = safe_max_radii_km[nearest_sink_indices]
    valid_capture_mask = mp_dists_km <= allowed_radii

    matched_mp = microplastics_df[valid_capture_mask].copy().reset_index(drop=True)
    matched_sink_idx = nearest_sink_indices[valid_capture_mask]
    matched_dists = mp_dists_km[valid_capture_mask]

    matched_mp["matched_sink_id"] = sinks_clean["sink_id"].values[matched_sink_idx]
    matched_mp["basin_id"] = sinks_clean["basin_id"].values[matched_sink_idx]
    matched_mp["distance_to_sink_km"] = matched_dists

    # 3. Raio individualizado efetivo alcançado por cada sink
    # O raio r_i* é o raio mínimo em múltiplos de step_km suficiente para capturar seus pontos
    if len(matched_mp) > 0:
        max_dist_per_sink = matched_mp.groupby("matched_sink_id")["distance_to_sink_km"].max()
        # Arredondar para o próximo múltiplo de step_km
        stepped_radii = np.ceil(max_dist_per_sink / step_km) * step_km
        sinks_clean["individual_radius_km"] = sinks_clean["sink_id"].map(stepped_radii).fillna(0.0)
        samples_per_sink = matched_mp.groupby("matched_sink_id")["nurdles_per_10min"].count()
        sinks_clean["n_samples_captured"] = sinks_clean["sink_id"].map(samples_per_sink).fillna(0).astype(int)
    else:
        sinks_clean["individual_radius_km"] = 0.0
        sinks_clean["n_samples_captured"] = 0

    # 4. Agregação por Bacia Hidrográfica
    if len(matched_mp) == 0:
        return matched_mp, pd.DataFrame(), sinks_clean

    basin_summary = (
        matched_mp.groupby("basin_id", as_index=False)
        .agg(
            n_samples=("nurdles_per_10min", "count"),
            nurdles_median=("nurdles_per_10min", "median"),
            nurdles_mean=("nurdles_per_10min", "mean"),
            nurdles_sum=("nurdles_per_10min", "sum"),
            mean_distance_to_sink_km=("distance_to_sink_km", "mean"),
            max_sink_radius_km=("distance_to_sink_km", "max"),
        )
    )

    return matched_mp, basin_summary, sinks_clean


def match_sinks_with_exclusive_radius(
    sinks_df: pd.DataFrame,
    microplastics_df: pd.DataFrame,
    radius_km: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pareia pontos de coleta aos sinks de bacias através de um raio exclusivo dinâmico:
    1. Calcula a distância ao sink vizinho mais próximo para cada desembocadura: d_min(S_k).
    2. O raio efetivo de busca de S_k é limitado a R_eff = min(radius_km, 0.5 * d_min(S_k)),
       impedindo que a busca avance sobre a zona de influência da foz vizinha.
    3. Cada ponto de microplástico é associado exclusivamente ao seu sink mais próximo (Voronoi),
       se e somente se a distância for <= R_eff daquele sink.
    4. Garantia matemática: ZERO duplicidade de contagem de amostras entre bacias.

    Retorna:
      - paired_samples_df: cada amostra associada ao seu respectivo sink e bacia
      - basin_microplastics_summary: métricas agregadas por bacia (mediana, média, soma, n_amostras)
    """
    if len(sinks_df) == 0 or len(microplastics_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    sink_lats = sinks_df["latitude"].values
    sink_lons = sinks_df["longitude"].values
    sink_xyz = _lat_lon_to_unit_vectors(sink_lats, sink_lons)
    sink_tree = cKDTree(sink_xyz)

    # 1. Distância ao vizinho costeiro mais próximo para cada sink
    # k=2 pois o vizinho mais próximo é o próprio ponto (distância 0)
    dists_chord, _ = sink_tree.query(sink_xyz, k=2)
    nearest_neighbor_chord = dists_chord[:, 1]
    nearest_neighbor_km = chord_to_haversine_distance(nearest_neighbor_chord)

    # Raio exclusivo máximo individual: metade da distância à foz mais próxima
    max_safe_radius_km = 0.5 * nearest_neighbor_km
    effective_radii_km = np.minimum(radius_km, max_safe_radius_km)
    effective_radii_chord = np.array([haversine_to_chord_distance(r) for r in effective_radii_km])

    # 2. Atribuir pontos de microplásticos ao sink mais próximo
    mp_lats = microplastics_df["latitude"].values
    mp_lons = microplastics_df["longitude"].values
    mp_xyz = _lat_lon_to_unit_vectors(mp_lats, mp_lons)

    mp_dists_chord, nearest_sink_indices = sink_tree.query(mp_xyz, k=1)
    mp_dists_km = chord_to_haversine_distance(mp_dists_chord)

    # Verificar se o ponto está dentro do raio efetivo exclusivo daquele sink específico
    assigned_sink_effective_chord = effective_radii_chord[nearest_sink_indices]
    valid_capture_mask = mp_dists_chord <= assigned_sink_effective_chord

    matched_mp = microplastics_df[valid_capture_mask].copy().reset_index(drop=True)
    matched_sink_idx = nearest_sink_indices[valid_capture_mask]
    matched_dists = mp_dists_km[valid_capture_mask]

    matched_mp["matched_sink_id"] = sinks_df["sink_id"].values[matched_sink_idx]
    matched_mp["basin_id"] = sinks_df["basin_id"].values[matched_sink_idx]
    matched_mp["distance_to_sink_km"] = matched_dists

    # 3. Agregação por Bacia Hidrográfica
    if len(matched_mp) == 0:
        return matched_mp, pd.DataFrame()

    summary = (
        matched_mp.groupby("basin_id", as_index=False)
        .agg(
            n_samples=("nurdles_per_10min", "count"),
            nurdles_median=("nurdles_per_10min", "median"),
            nurdles_mean=("nurdles_per_10min", "mean"),
            nurdles_sum=("nurdles_per_10min", "sum"),
            mean_distance_to_sink_km=("distance_to_sink_km", "mean"),
        )
    )

    return matched_mp, summary


def evaluate_radius_sensitivity(
    sinks_df: pd.DataFrame,
    microplastics_df: pd.DataFrame,
    basin_pop_df: pd.DataFrame,
    radii_to_test: Sequence[float] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0),
) -> pd.DataFrame:
    """
    Avalia o impacto do raio exclusivo crescente na quantidade de amostras capturadas,
    no número de bacias pareadas e na correlação alométrica (expoente beta e R²).
    Atende diretamente à Diretriz F6 (plotar_evolucao_raio_exclusivo).
    """
    results = []
    for r in radii_to_test:
        matched_mp, mp_summary = match_sinks_with_exclusive_radius(sinks_df, microplastics_df, radius_km=r)
        if len(mp_summary) == 0:
            continue

        merged = mp_summary.merge(basin_pop_df, on="basin_id", how="inner")
        # Filtrar valores positivos para regressão logarítmica
        valid = merged[(merged["urban_population"] > 0) & (merged["nurdles_median"] > 0)].copy()

        n_basins = len(valid)
        n_samples = int(mp_summary["n_samples"].sum())

        beta = np.nan
        r2 = np.nan
        p_val = np.nan
        se = np.nan

        if n_basins >= 5:
            log_p = np.log10(valid["urban_population"].values)
            log_m = np.log10(valid["nurdles_median"].values)
            slope, intercept, r_val, p_value, std_err = stats.linregress(log_p, log_m)
            beta = float(slope)
            r2 = float(r_val**2)
            p_val = float(p_value)
            se = float(std_err)

        results.append(
            {
                "radius_km": float(r),
                "paired_basins": n_basins,
                "captured_samples": n_samples,
                "allometric_beta": beta,
                "r2": r2,
                "stderr": se,
                "p_value": p_val,
            }
        )

    return pd.DataFrame(results)
