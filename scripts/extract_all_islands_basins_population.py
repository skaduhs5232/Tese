"""
scripts/extract_all_islands_basins_population.py
------------------------------------------------
Calcula a população urbana (CCA) e população total da bacia para TODAS as bacias
de cada grande ilha continental, utilizando os parâmetros ótimos de Zipf já calibrados.
Salva:
  relatorios/ilhas/populacao_todas_bacias_{slug}.csv
"""

import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.gebco_cutter import GRANDES_ILHAS
from pipeline.ipba_runner import crop_gebco, extract_coastal_sinks, run_ipba, filter_basins_by_area
from pipeline.cca_calibrator import crop_gpw, run_calibrated_cca
from pipeline.spatial_matcher import split_urban_clusters_by_basins

def processar_todas_ilhas():
    gebco_path = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
    gpw_path = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
    out_dir = REPO_ROOT / "relatorios" / "ilhas"

    for nome, info in GRANDES_ILHAS.items():
        slug = nome.lower().replace(" & ", "_").replace(" ", "_")
        csv_out = out_dir / f"populacao_todas_bacias_{slug}.csv"
        zipf_csv = out_dir / f"matriz_zipf_{slug}.csv"

        if csv_out.is_file() and csv_out.stat().st_size > 1000:
            print(f"[{nome}] Já existe {csv_out.name}. Pulando recalculo.")
            continue

        print(f"\n[{nome}] Extraindo população de todas as bacias...")
        t0 = time.time()

        # Obter par ótimo de Zipf já calibrado
        df_zipf = pd.read_csv(zipf_csv)
        best = df_zipf.iloc[(df_zipf["alpha"] - 1.0).abs().argsort()[:1]].iloc[0]
        d_star, l_km = float(best["d_star"]), float(best["l_cutoff"])
        print(f"[{nome}] Par Zipf ótimo: D* = {d_star} hab/km², L = {l_km} km")

        bbox = info["bbox"]
        dem, gt_gebco, _ = crop_gebco(bbox, gebco_path)
        s_grid, s_df = extract_coastal_sinks(dem, gt_gebco, h_star=0.0)
        b_grid = run_ipba(dem, s_grid, h_star=0.0)
        f_bgrid, b_df, ret_sinks = filter_basins_by_area(b_grid, s_df, gt_gebco, 0.95)

        pop, px_area, lats_1d, lons_1d, gt_gpw, _ = crop_gpw(bbox, gpw_path)
        grid_clusters, _ = run_calibrated_cca(pop, lons_1d, lats_1d, px_area, d_star, l_km)
        b_pop_df, _ = split_urban_clusters_by_basins(grid_clusters, pop, f_bgrid, gt_gpw, gt_gebco)

        merged = pd.merge(b_pop_df, b_df[["basin_id", "area_km2"]], on="basin_id", how="left")
        merged["island_name"] = nome
        merged["d_star_opt"] = d_star
        merged["l_opt_km"] = l_km
        merged.to_csv(csv_out, index=False)
        print(f"[{nome}] Concluído em {time.time()-t0:.1f}s. Bacias salvas: {len(merged)}")

if __name__ == "__main__":
    processar_todas_ilhas()
