# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and autonomous AI agents when working with code in this repository.
Also read **[`AGENTS.md`](AGENTS.md)** and **[`diretrizes/README.md`](diretrizes/README.md)** before modifying code or generating figures.

## Project overview

This is a master's thesis (`Tese`, in Portuguese) research pipeline that tests whether the population of a drainage basin correlates with the coastal microplastic found at its river mouth, worldwide, following **Oliveira, E. A. et al. (2019), *"A universal approach for drainage basins"*, Scientific Reports 9:9845**:

- **GEBCO 2026** (`dados/gebco/gebco_2026_sub_ice_geotiff.tif`) — global relief raster → landmass percolation (`notebooks/gebco_cluster_dvider`) grouped into **2 contiguous hydrological blocks** (Americas; Afro-Eurasia + Oceania), drainage basins via the advisor's IPBA (`notebooks/IPBA`, `run_ipba_grid`) and endorheic basins via priority-flood (`calculos/hidrologia_endorreica/`).
- **GPW v4 Rev11** (`dados/gpw_data/gpw_v4_population_count_rev11_2020_30_sec.tif`) — **population count per pixel** (2020, 30 arc-sec, aggregated to 1') → density by exact 2-spherical-triangle pixel area, dynamic Zipf calibration and the advisor's CCA (`notebooks/CCA`) in 6 disjoint demographic regions; urban blobs spanning several basins are split by summing person counts per pixel inside each basin.
- **NOAA NCEI Microplastics Hub** (`dados/noaa_microplastic/*.geojson` → `dados/ingestao_noaa_geojson.py` → `dados/noaa_microplastics.csv`; 29,076 records, 12,088 coastal after removing `Ocean water`: Nurdle Patrol `pieces/10 mins`, Beach `pieces/m3`, Ocean sediment `pieces kg-1 d.w.`) → paired with the nearest river mouth. The previous FeatureServer export is kept as `dados/noaa_microplastics_legacy.csv` for scenario B.

## Local execution & modular architecture

**Everything runs 100% locally** using the repository virtualenv (`venv\Scripts\python.exe`). Never depend on Google Colab, `gdown`, or runtime `git clone`.

The advisor's algorithms live unmodified in `notebooks/CCA` (`cca_grid`) and `notebooks/IPBA` (`run_ipba_grid`). Everything else is NumPy (+ `numba.njit`) in the root package **`calculos/`**:

1. **`calculos/geometria_esferica/`** — Haversine (`haversine_km`, `haversine_deg`, $R_{\text{earth}} = 6378.137$ km) and pixel area as the sum of **two spherical triangles** via L'Huilier (`pixel_area_by_latitude`, `somar_areas_por_bacia`).
2. **`calculos/percolacao_ilhas/`** — Landmass percolation ($h > 0$ m, 4-connected), `identificar_oceano_e_sinks`, bounding boxes with ocean padding.
3. **`calculos/hidrologia_endorreica/`** — Priority-flood depression filling and `identificar_bacias_endorreicas`: depressions accepted only when they contain a documented terminal lake (`SISTEMAS_ENDORREICOS`), spill restricted by the filled reference surface, forced exorheic outlets for DEM-blocked gorges (`SAIDAS_EXORREICAS_FORCADAS`).
4. **`calculos/alometria_estatistica/`** — GPW density (`calcular_densidade_gpw`), **dynamic Zipf calibration** (`calibrar_zipf_dinamico`: D 100–2000 × L 1000–5000 m, min |α−1| s.t. OLS rank-size R² ≥ 0.98, no fixed D*/L), urban blob splitting (`dividir_manchas_e_somar_populacao_bacias`), river-mouth–NOAA pairing (`parear_fozes_e_sites`), OLS log-log scaling with bootstrap CI and permutation p (`ajuste_escala_log_log`), leave-one-out (`leave_one_out_alometria`).

## End-to-end execution order

1. `python dados/download_dados.py --overview 2 --gerar-tiles-raw` — GEBCO 2026 and GPW v4; ingests the local NOAA GeoJSON (`--noaa-remoto` to query the FeatureServer instead).
2. `python notebooks/gebco_cluster_dvider/executar_divisao_ilhas.py` — Global percolation (4,735 landmasses, 149.49 Mkm²) and export of the 2 hydrological blocks (`catalogo["blocos_hidrologicos"]`).
3. `python notebooks/executar_pipeline_global.py` (~2.5 min; `--reusar-zipf`, `--sem-sensibilidade`) — IPBA per block + 37 endorheic basins (excluded from marine emitters) + dynamic Zipf/CCA per region + basin–city coupling + NOAA pairing at R ∈ {15, 25, 38, 50} km + scenarios A/B/C (combined R ≤ 25 km: n = 81, β = 0.45 [0.23, 0.72], r = 0.38). Writes `notebooks/results/` (block rasters `bacias_bloco_*_ipba.tif` with exorheic > 0 / endorheic < 0, regional crops, `zipf_matriz_<regiao>.npz`, `cca_calibracao_<regiao>.json`, `bacias_global_consolidado.csv`, `escala_pop_lixo_global.csv`, `metricas_alometria_global.json`, `validacao_bacias_literatura.csv`).
4. Figure generators in `geracao_graficos/<theme>/` (shared region boxes in `geracao_graficos/regioes_plot.py`): `gebco/plot_global_islands_clusters.py`, `watherSeads/plot_drainage_basins_ipba.py`, `zipf/plot_zipf_calibration_regions.py`, `populacao/plot_population_cca_regions.py`, `noaa/plot_noaa_global_sampling.py`, `integrado/plot_global_allometry_correlation.py`, `gifs/generate_cca_ipba_gifs.py`.
5. `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_Analise_Integrada_CCA_IPBA_Microplasticos.ipynb`.

## Working conventions

- Communicate with the user in Portuguese (`pt-BR`), while all text inside figures must be **100% in English**.
- **Figure generation rules (`AGENTS.md` & `diretrizes/`)**: English labels only, `cmap='plasma'`, serif fonts, 300 DPI `.png` + vector `.pdf` saved alongside the `.py` script in `geracao_graficos/<theme>/`, zero overlapping elements; maps show the zoomed study area with a gray vector world-map inset (`diretrizes/assets/world_land_110m.geojson`, 2:1 aspect, red zoom box, `add_world_inset`) in a free ocean corner and a vertical colorbar on the right; never `ConnectionPatch` diagonal lines.
