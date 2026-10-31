# Diretriz 02: Estrutura e Organização da Pasta de Geração de Gráficos

Esta diretriz estabelece a organização modular de `geracao_graficos/`, cujas funções são chamadas diretamente pelo notebook principal `notebooks/pipeline_americas.ipynb`.

---

## 1. Módulo Central de Estilo

- **[`geracao_graficos/estilo_erneson.py`](../geracao_graficos/estilo_erneson.py)**: único local onde `ESTILO_ERNESON` (`rcParams`) e as primitivas de plotagem padrão Erneson Oliveira são definidos (`aplicar_estilo`, `rotulo_painel`, `rotulo_canto`, `anotar_expoente`, `guia_inclinacao`, `scatter_vazado`, `reta_ols`, `curva_nw`, `referencia_inclinacao_1`, `eixos_log10`, `colorbar_direita`, `eixo_mapa`, `cores_bacias_tab20`, `cores_manchas`, `inset_localizador`, `salvar`).
- **[`geracao_graficos/figuras.py`](../geracao_graficos/figuras.py)**: funções de alto nível que geram as figuras científicas da dissertação e exportam simultaneamente `.png` (300 DPI) e `.pdf` vetorial ao lado do módulo.

---

## 2. Catálogo das Funções Geradoras

| Código | Função em `geracao_graficos/figuras.py` | Entradas Principais | Saídas (`.png` + `.pdf`) |
| :--- | :--- | :--- | :--- |
| **F1** | `plotar_mapa_mundi_gebco` | Matriz GEBCO 2026 global | `f1_gebco_world_map.(png\|pdf)` |
| **F2** | `plotar_mapa_mundi_gpw` | `dados/gpw_data/gpw_v4_population_count_rev11_2020_30_sec.tif` | `f2_gpw_world_map.(png\|pdf)` |
| **F3** | `plotar_mapa_mundi_microplasticos` | Estações globais NOAA + LITTERBASE e estações pareadas nas Américas | `f3_microplastics_world_map.(png\|pdf)` |
| **F4** | `plotar_bacias_ipba_americas` | Grade de bacias IPBA (rodada única, corte 95%) e tabela de áreas | `f4_ipba_drainage_basins_americas.(png\|pdf)` |
| **F5** | `plotar_calibracao_zipf_regioes` | Matrizes $10 \times 10$ ($D^* \times \ell$) e curvas rank-size nas 2 regiões | `f5_zipf_calibration_matrices_and_rank_size.(png\|pdf)` |
| **F6** | `plotar_evolucao_raio_exclusivo` | Tabela de raios exclusivos crescentes ($5, 10, \dots, 50$ km) | `f6_exclusive_radius_growth_americas.(png\|pdf)` |
| **F7** | `plotar_alometria_e_controle_area` | Pares bacia-microplástico no raio de referência e resíduos por área | `f7_allometry_urban_pop_vs_microplastics_r25km.(png\|pdf)` |
| **F8** | `plotar_mapa_cca_americas` | Densidade GPW v4 e manchas urbanas do CCA calibradas nas 2 regiões | `f8_cca_urban_clusters_americas.(png\|pdf)` |

