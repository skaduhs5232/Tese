# Diretriz 02: Estrutura e Organização da Pasta de Geração de Gráficos

Esta diretriz estabelece a organização de pastas, entradas e saídas para todos os scripts geradores de figuras F1–F10 do projeto.

---

## 1. Módulo Central de Estilo

- **`geracao_graficos/estilo_erneson.py`**: único local onde `ESTILO_ERNESON` (`rcParams`) e as funções de plotagem padrão Erneson Oliveira são definidos. Todos os scripts F1–F10 importam deste módulo.
- **`geracao_graficos/regioes_plot.py`**: metadados das 2 regiões de estudo nas Américas (`america_do_sul` e `americas_norte_central`).

---

## 2. Catálogo dos Scripts Geradores (F1 a F10)

Cada subpasta temática em `geracao_graficos/<tema>/` contém o script `.py` gerador e os arquivos exportados em `.png` (300 DPI) e `.pdf` vetorial:

| Código | Script Gerador | Entradas Principais | Saídas (`.png` + `.pdf` / `.gif`) |
| :--- | :--- | :--- | :--- |
| **F1** | `geracao_graficos/gebco/gebco_world_map.py` | `dados/gebco/gebco_2026_sub_ice_geotiff.tif`, catálogo do recorte Américas | `gebco_world_map.(png\|pdf)` |
| **F1b** | `geracao_graficos/gebco/plot_global_islands_clusters.py` | Recorte GEBCO Américas + diagnóstico R1 | `gebco_global_islands_clusters.(png\|pdf)` |
| **F2** | `geracao_graficos/populacao/gpw_world_map.py` | `dados/gpw_data/gpw_v4_population_count_rev11_2020_30_sec.tif` | `gpw_world_map.(png\|pdf)` |
| **F3** | `geracao_graficos/noaa/noaa_world_map.py` | `notebooks/results/escala_pop_lixo_global.csv`, `world_land_110m.geojson` | `noaa_world_map.(png\|pdf)` |
| **F3b** | `geracao_graficos/noaa/plot_noaa_global_sampling.py` | `dados/noaa_microplastics.csv`, `escala_pop_lixo_global.csv` | `noaa_global_microplastic_regimes.(png\|pdf)` |
| **F4** | `geracao_graficos/watherSeads/plot_drainage_basins_ipba.py` | `bacias_bloco_americas_ipba.tif`, `bacias_global_consolidado.csv`, DEM Américas | `ipba_drainage_basins_americas.(png\|pdf)`, `ipba_amazon_basin_zoom.(png\|pdf)`, `ipba_drainage_basins_universal_scaling.(png\|pdf)` |
| **F5** | `geracao_graficos/populacao/plot_population_cca_regions.py` | GPW v4, `cca_calibracao_<regiao>.json`, `zipf_matriz_<regiao>.npz` | `population_cca_regions.(png\|pdf)` |
| **F6 & F7** | `geracao_graficos/zipf/plot_zipf_calibration_regions.py` | `zipf_matriz_<regiao>.npz`, `cca_calibracao_<regiao>.json` | `zipf_rank_size_regions.(png\|pdf)`, `zipf_calibration_dstar_l_regions.(png\|pdf)`, `zipf_calibration_<regiao>.(png\|pdf)` |
| **F8** | `geracao_graficos/integrado/plot_global_allometry_correlation.py` | `escala_pop_lixo_global.csv`, `metricas_alometria_global.json`, `bacias_global_consolidado.csv` | `global_allometry_population_microplastics.(png\|pdf)`, `allometry_by_radius_americas.(png\|pdf)` |
| **F9** | `geracao_graficos/integrado/plot_distributions_erneson.py` | `bacias_global_consolidado.csv`, `zipf_matriz_<regiao>.npz`, `escala_pop_lixo_global.csv` | `distributions_basins_cca_noaa.(png\|pdf)` |
| **F10** | `geracao_graficos/gifs/generate_cca_ipba_gifs.py` | DEM Américas, `bacias_bloco_americas_ipba.tif`, GPW v4, `cca_calibracao_*.json` | `cca_urban_clustering_evolution.gif`, `ipba_watershed_percolation_evolution.gif` |
