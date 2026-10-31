import time
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.ipba_runner import crop_gebco, extract_coastal_sinks, run_ipba, filter_basins_by_area
from pipeline.cca_calibrator import crop_gpw, calibrate_zipf_grid, run_calibrated_cca, plot_zipf_calibration
from pipeline.spatial_matcher import (
    split_urban_clusters_by_basins,
    load_noaa_nurdle_patrol,
    match_sinks_with_exclusive_radius,
    evaluate_radius_sensitivity,
)
from geracao_graficos.figuras import (
    plotar_bacias_ipba,
    plotar_mapa_cca,
    plotar_evolucao_raio_exclusivo,
    plotar_alometria_e_controle_area,
)

Path("relatorios/figuras").mkdir(parents=True, exist_ok=True)

bbox = (-125.0, 7.0, -65.0, 50.0)
gebco_path = 'dados/gebco/gebco_2026_sub_ice_geotiff.tif'
gpw_path = 'dados/gpw_data/gpw_v4_population_count_rev11_2020_30_sec.tif'
noaa_path = 'dados/noaa_microplastic/Hub_Microplastics_Replace_2485283804436522212.geojson'

print("=== 1. GEBCO & IPBA ===", flush=True)
t0 = time.time()
dem, gt_gebco, _ = crop_gebco(bbox, gebco_path)
print(f"GEBCO crop shape: {dem.shape} in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
sink_grid, sinks_df = extract_coastal_sinks(dem, gt_gebco, h_star=0.0)
print(f"Coastal sinks: {len(sinks_df)} in {time.time()-t1:.1f}s", flush=True)

t2 = time.time()
basin_grid = run_ipba(dem, sink_grid, h_star=0.0)
print(f"IPBA raw run in {time.time()-t2:.1f}s", flush=True)

t3 = time.time()
filtered_basin_grid, basins_df, ret_sinks = filter_basins_by_area(basin_grid, sinks_df, gt_gebco, 0.95)
print(f"Filtered basins (95% area): {len(basins_df)}, Sinks: {len(ret_sinks)} in {time.time()-t3:.1f}s", flush=True)

print("=== 2. GPW & CCA Calibration ===", flush=True)
t4 = time.time()
pop, pixel_area, lats_1d, lons_1d, gt_gpw, meta = crop_gpw(bbox, gpw_path)
print(f"GPW crop: {pop.shape}, total pop: {meta['total_population']:,.0f} in {time.time()-t4:.1f}s", flush=True)

d_vals = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]
l_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
cache_file = Path("relatorios/cache_calibracao_eua.npz")

if cache_file.is_file():
    print(f"Loading CCA calibration from cache {cache_file}...", flush=True)
    cache_data = np.load(cache_file)
    m_alpha = cache_data["m_alpha"]
    m_r2 = cache_data["m_r2"]
    m_se = cache_data["m_se"]
    best_pair = (float(cache_data["best_d"]), float(cache_data["best_l"]))
    df_grid = pd.read_csv("relatorios/matriz_calibracao_zipf_eua.csv")
    print(f"Loaded from cache: Best pair D*={best_pair[0]}, L={best_pair[1]}", flush=True)
else:
    t5 = time.time()
    m_alpha, m_r2, m_se, df_grid, best_pair = calibrate_zipf_grid(pop, lons_1d, lats_1d, pixel_area, d_vals, l_vals)
    print(f"CCA 10x10 calibration in {time.time()-t5:.1f}s. Best pair: D*={best_pair[0]}, L={best_pair[1]}", flush=True)
    np.savez(cache_file, m_alpha=m_alpha, m_r2=m_r2, m_se=m_se, best_d=best_pair[0], best_l=best_pair[1])
    df_grid.to_csv("relatorios/matriz_calibracao_zipf_eua.csv", index=False)

print("=== 3. Extract Calibrated Urban Clusters ===", flush=True)
t6 = time.time()
grid_clusters, clusters_df = run_calibrated_cca(pop, lons_1d, lats_1d, pixel_area, best_pair[0], best_pair[1])
print(f"Clusters extracted in {time.time()-t6:.1f}s. Total clusters: {len(clusters_df)}", flush=True)

# Plot F5: Calibração Zipf
plot_zipf_calibration(
    m_alpha, m_r2, d_vals, l_vals, best_pair[0], best_pair[1],
    clusters_df, "relatorios/figuras/f5_zipf_calibration_matrices_and_rank_size",
    "North & Central America"
)
print("F5 Zipf calibration plot saved.", flush=True)

print("=== 4. Split Clusters Across Basins ===", flush=True)
t7 = time.time()
basin_pop_df, splits_df = split_urban_clusters_by_basins(grid_clusters, pop, filtered_basin_grid, gt_gpw, gt_gebco)
n_urban = (basin_pop_df["urban_population"] > 0).sum()
print(f"Splits done in {time.time()-t7:.1f}s. Basins with urban pop: {n_urban}, splits rows: {len(splits_df)}", flush=True)

print("=== 5. Load Nurdle Patrol & Match ===", flush=True)
t8 = time.time()
nurdles_df = load_noaa_nurdle_patrol(noaa_path, bbox)
print(f"Loaded {len(nurdles_df)} Nurdle Patrol points in {time.time()-t8:.1f}s", flush=True)

t9 = time.time()
sens_df = evaluate_radius_sensitivity(ret_sinks, nurdles_df, basin_pop_df)
print(f"Radius sensitivity evaluated in {time.time()-t9:.1f}s:\n{sens_df[['radius_km', 'paired_basins', 'captured_samples', 'allometric_beta', 'r2', 'p_value']]}", flush=True)
sens_df.to_csv("relatorios/sensibilidade_raio_exclusivo_eua.csv", index=False)

t10 = time.time()
matched_mp, mp_summary = match_sinks_with_exclusive_radius(ret_sinks, nurdles_df, radius_km=25.0)
paired_df = mp_summary.merge(basin_pop_df, on="basin_id").merge(basins_df[["basin_id", "area_km2"]], on="basin_id")
paired_df.to_csv("relatorios/dados_pareados_bacias_microplasticos_r25km.csv", index=False)
print(f"Matching at R=25km in {time.time()-t10:.1f}s. Paired basins: {len(paired_df)}", flush=True)
print(paired_df[["basin_id", "urban_population", "n_samples", "nurdles_median", "area_km2"]].head(10), flush=True)

print("=== 6. Test Scientific Plots ===", flush=True)
t11 = time.time()
plotar_bacias_ipba(filtered_basin_grid, ret_sinks, bbox, "relatorios/figuras/f4_ipba_drainage_basins_americas")
plotar_mapa_cca(pop, grid_clusters, pixel_area, bbox, best_pair[0], best_pair[1], "relatorios/figuras/f8_cca_urban_clusters_americas")
plotar_evolucao_raio_exclusivo(sens_df, "relatorios/figuras/f6_exclusive_radius_growth_americas")
plotar_alometria_e_controle_area(paired_df, "relatorios/figuras/f7_allometry_urban_pop_vs_microplastics_r25km")
print(f"All figures generated and saved in {time.time()-t11:.1f}s", flush=True)
print(f"TOTAL PIPELINE COMPLETED IN {time.time()-t0:.1f}s!", flush=True)
