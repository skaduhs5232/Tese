"""
pipeline/gebco_cutter.py
-------------------------
Módulo para segmentação do relevo global (GEBCO 2026) em grandes ilhas e massas continentais
utilizando a lógica de percolação de sítios (site percolation) e recorte geográfico
do pacote `externos/GebcoCutter` (Prof. Erneson Oliveira):
  - América do Norte e Central
  - América do Sul
  - África
  - Europa e Eurásia
  - Austrália e Oceania

Para cada grande massa continental / ilha:
  1. Recorta relevo (GEBCO) e densidade populacional (GPW v4).
  2. Executa IPBA em rodada única (h* = 0, corte cumulativo 95%).
  3. Identifica as bacias ativas que possuem estações de coleta de microplásticos (NOAA).
  4. Calibra dinamicamente o CCA (com D* até 5000 hab/km² e L ótimo) via Lei de Zipf.
  5. Particiona as manchas urbanas nos divisores de drenagem da ilha.
  6. Aplica o raio exclusivo adaptativo individualizado por sink.
  7. Consolida os dados pareados de todas as ilhas para a comparação global.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .ipba_runner import crop_gebco, extract_coastal_sinks, run_ipba, filter_basins_by_area
from .cca_calibrator import crop_gpw, calibrate_zipf_grid, run_calibrated_cca
from .spatial_matcher import (
    load_noaa_nurdle_patrol,
    split_urban_clusters_by_basins,
    match_sinks_with_individual_radius,
)

# Definições geográficas das grandes ilhas e massas continentais (lon_min, lat_min, lon_max, lat_max)
GRANDES_ILHAS: dict[str, dict[str, Any]] = {
    "North & Central America": {
        "bbox": (-128.0, 8.0, -65.0, 52.0),
        "marker": "o",
        "color": "C0",
    },
    "South America": {
        "bbox": (-82.0, -56.0, -34.0, 13.0),
        "marker": "s",
        "color": "C2",
    },
    "Europe & Eurasia": {
        "bbox": (-15.0, 35.0, 45.0, 72.0),
        "marker": "^",
        "color": "C3",
    },
    "Africa": {
        "bbox": (-18.0, -35.0, 52.0, 38.0),
        "marker": "D",
        "color": "C4",
    },
    "Australia": {
        "bbox": (112.0, -45.0, 155.0, -10.0),
        "marker": "v",
        "color": "C1",
    },
}


def processar_ilha_continental(
    island_name: str,
    island_info: dict[str, Any],
    gebco_path: str | Path,
    gpw_path: str | Path,
    noaa_path: str | Path,
    output_dir: str | Path = "relatorios/ilhas",
    d_max: float = 5000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Executa o pipeline completo para uma grande ilha / massa continental:
    Retorna (df_paired_island, sinks_with_radii, stats_meta).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = island_name.lower().replace(" & ", "_").replace(" ", "_")

    bbox = island_info["bbox"]
    print(f"\n=======================================================", flush=True)
    print(f"Processando Ilha / Continente: {island_name} {bbox}", flush=True)
    print(f"=======================================================", flush=True)

    t0 = time.time()
    # 1. Carregar pontos NOAA da região para verificar se há dados
    df_nurdles = load_noaa_nurdle_patrol(noaa_path, bbox)
    print(f"[{island_name}] Amostras de microplásticos na região: {len(df_nurdles)}", flush=True)
    if len(df_nurdles) == 0:
        print(f"[{island_name}] Nenhuma amostra na região. Pulando.", flush=True)
        return pd.DataFrame(), pd.DataFrame(), {}

    # 2. Recorte GEBCO e IPBA
    dem, gt_gebco, gebco_meta = crop_gebco(bbox, gebco_path)
    sink_grid, sinks_df = extract_coastal_sinks(dem, gt_gebco, h_star=0.0)
    basin_grid = run_ipba(dem, sink_grid, h_star=0.0)
    filtered_basin_grid, basins_df, ret_sinks = filter_basins_by_area(basin_grid, sinks_df, gt_gebco, 0.95)
    print(f"[{island_name}] Bacias filtradas (95% área): {len(basins_df)}, Sinks costeiros: {len(ret_sinks)}", flush=True)

    # 3. Pareamento espacial com raio individualizado por sink
    matched_mp, basin_mp_summary, sinks_with_radii = match_sinks_with_individual_radius(
        ret_sinks, df_nurdles, max_cap_km=50.0, step_km=1.0
    )
    print(f"[{island_name}] Amostras capturadas exclusivamente: {len(matched_mp)} em {len(basin_mp_summary)} bacias", flush=True)
    if len(basin_mp_summary) == 0:
        return pd.DataFrame(), sinks_with_radii, {}

    # Filtrar bacias que têm dados de microplásticos
    active_basin_ids = set(basin_mp_summary["basin_id"].unique())

    # 4. GPW & CCA Dinâmico
    pop, pixel_area, lats_1d, lons_1d, gt_gpw, gpw_meta = crop_gpw(bbox, gpw_path)
    print(f"[{island_name}] População total da ilha: {gpw_meta['total_population']:,.0f}", flush=True)

    # Calibração dinâmica de Zipf
    d_vals = [100.0, 200.0, 500.0, 800.0, 1010.0, 1500.0, 2000.0, 3000.0, 4000.0, min(5000.0, d_max)]
    l_vals = [1.0, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    m_alpha, m_r2, m_se, df_grid, best_pair = calibrate_zipf_grid(
        pop, lons_1d, lats_1d, pixel_area, d_values=d_vals, l_values=l_vals, top_n=100
    )
    print(f"[{island_name}] Par ótimo calibrado via Zipf: D* = {best_pair[0]} hab/km², L = {best_pair[1]} km", flush=True)

    # Executar CCA com o par ótimo da ilha
    grid_clusters, clusters_df = run_calibrated_cca(
        pop, lons_1d, lats_1d, pixel_area, best_pair[0], best_pair[1]
    )

    # 5. Particionamento de manchas por bacia
    basin_pop_df, splits_df = split_urban_clusters_by_basins(
        grid_clusters, pop, filtered_basin_grid, gt_gpw, gt_gebco
    )

    # 6. Unir microplásticos, população e área das bacias
    paired = pd.merge(basin_mp_summary, basin_pop_df, on="basin_id", how="inner")
    if "area_km2" in basins_df.columns:
        paired = pd.merge(paired, basins_df[["basin_id", "area_km2"]], on="basin_id", how="left")
    paired["island_name"] = island_name
    paired["d_star_opt"] = best_pair[0]
    paired["l_opt_km"] = best_pair[1]

    # Salvar resultados da ilha
    paired.to_csv(out_dir / f"dados_pareados_{slug}.csv", index=False)
    sinks_with_radii.to_csv(out_dir / f"sinks_raios_{slug}.csv", index=False)
    df_grid.to_csv(out_dir / f"matriz_zipf_{slug}.csv", index=False)

    meta = {
        "island": island_name,
        "n_samples": len(matched_mp),
        "n_basins_paired": len(paired),
        "d_star": best_pair[0],
        "l_km": best_pair[1],
        "elapsed_sec": time.time() - t0,
    }
    print(f"[{island_name}] Concluído em {meta['elapsed_sec']:.1f}s. Bacias pareadas: {len(paired)}", flush=True)
    return paired, sinks_with_radii, meta


def executar_pipeline_global_ilhas(
    gebco_path: str | Path,
    gpw_path: str | Path,
    noaa_path: str | Path,
    output_dir: str | Path = "relatorios/ilhas",
    d_max: float = 5000.0,
) -> pd.DataFrame:
    """
    Executa o pipeline sobre todas as grandes ilhas e massas continentais do globo,
    e consolida os dados pareados em uma única tabela global.
    """
    all_paired = []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for island_name, info in GRANDES_ILHAS.items():
        df_p, _, _ = processar_ilha_continental(
            island_name, info, gebco_path, gpw_path, noaa_path, output_dir=out_dir, d_max=d_max
        )
        if len(df_p) > 0:
            all_paired.append(df_p)

    if not all_paired:
        print("Nenhum dado pareado globalmente.")
        return pd.DataFrame()

    df_global = pd.concat(all_paired, ignore_index=True)
    global_csv = Path("relatorios") / "dados_pareados_global_todas_ilhas.csv"
    df_global.to_csv(global_csv, index=False)
    print(f"\n=======================================================", flush=True)
    print(f"PIPELINE GLOBAL CONCLUÍDO! Total bacias pareadas: {len(df_global)}", flush=True)
    print(f"Salvo em: {global_csv}", flush=True)
    print(f"=======================================================", flush=True)
    return df_global
