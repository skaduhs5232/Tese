# Repositório Central de Figuras: `geracao_graficos/`

Esta pasta centraliza todas as figuras científicas geradas para a dissertação de mestrado, artigos e apresentações relacionadas.

> [!IMPORTANT]
> Qualquer agente ou pesquisador que for gerar imagens nesta pasta **DEVE SEGUIR AS DIRETRIZES FORMAIS** documentadas em:
> [`diretrizes/README.md`](../diretrizes/README.md) e [`diretrizes/01_PADRAO_VISUAL_E_FIGURAS.md`](../diretrizes/01_PADRAO_VISUAL_E_FIGURAS.md).

---

## Regras Obrigatórias de Organização e Padrão Visual

1. **Subpastas Temáticas**: Toda figura deve residir dentro de uma subpasta temática de acordo com seu escopo científico (`gebco/`, `noaa/`, `watherSeads/`, `zipf/`, `populacao/`, `integrado/`).
2. **Dupla Presença**: Em cada subpasta deve constar obrigatoriamente:
   - O **código-fonte Python** que gerou a imagem (ex: `plot_global_islands_clusters.py`).
   - A **própria imagem gerada** em formato raster de alta resolução (`.png`, 300 DPI) e em formato vetorial (`.pdf` para o LaTeX).
3. **Padrão Científico, Mapa-Múndi e Escala Vertical**:
   - Todo o texto dentro das imagens deve estar **100% em inglês**.
   - O mapa de cores contínuo deve ser estritamente o **`plasma`**.
   - **Mapa-Múndi Localizador na Esquerda em Baixo (`lower left`)**: Imagens que contêm mapas devem ter como foco principal a região de estudo com zoom, e no **canto inferior esquerdo (`lower left`)** um **mapa-múndi localizador (inset) todo em cinza** (`diretrizes/assets/world_land_110m.geojson`), com retângulo vermelho (`#d90429`) indicando a área de zoom.
   - **Escala de Cores em Pé à Direita (`vertical`, `right`)**: Todas as barras de cores (`colorbar`) devem ser posicionadas verticalmente à direita do eixo (`orientation='vertical'`, `location='right'`).
   - É expressamente proibida a sobreposição de linhas sobre letras identificadoras `(a)`, `(b)` ou dados.

---

## Inventário de Figuras e Scripts Geradores (`.png` 300 DPI + `.pdf`)

As caixas das 6 regiões nos mapas são compartilhadas em [`regioes_plot.py`](regioes_plot.py).

| Subpasta | Script Gerador | Figuras Exportadas (`.png` 300 DPI & `.pdf` vetorial) | Descrição |
| :--- | :--- | :--- | :--- |
| `gebco/` | `plot_global_islands_clusters.py` | `gebco_global_islands_clusters`, `gebco_hydrological_blocks`, `gebco_elevation_6regions` | Relevo GEBCO 2026, lei rank-área das 4.735 massas de terra (149,49 Mkm²), os 2 blocos hidrológicos contíguos do IPBA e o relevo das 6 regiões. |
| `watherSeads/` | `plot_drainage_basins_ipba.py` | `ipba_drainage_basins_universal_scaling`, `ipba_drainage_basins_6regions` | Bacias IPBA por área esférica, com as 37 bacias endorreicas em cinza hachurado, e CCDF das grandes bacias exorreicas ($P = 0{,}90$) por região. |
| `noaa/` | `plot_noaa_global_sampling.py` | `noaa_global_microplastic_regimes`, `noaa_sampling_6regions` | Coletas costeiras do GeoJSON NCEI (12.088 amostras sem `Ocean water`) e distribuição por regime. |
| `zipf/` | `plot_zipf_calibration_regions.py` | `zipf_calibration_6regions`, `zipf_calibration_<regiao>` | Matrizes $\alpha(D, L)$ e $R^2(D, L)$ da calibração dinâmica, com o par ótimo $(D^*, L^*)$, o contorno $R^2 = 0{,}98$ e a faixa sub-pixel $L < L_{\min}$ hachurada; curva rank-size da região. |
| `populacao/` | `plot_population_cca_regions.py` | `population_cca_6regions` | Densidade GPW v4 em cada região demográfica disjunta (vizinhas em cinza) com $(D^*, L^*)$ e o número de manchas CCA. |
| `integrado/` | `plot_global_allometry_correlation.py` | `global_allometry_population_microplastics`, `allometry_scenarios_radius_comparison`, `allometry_correlation_6regions`, `allometry_correlation_<regiao>` (regiões com $n \geq 5$) | Alometria $(M\times A) \propto P^\beta$ a $R \le 25$ km (cenário C, $n = 81$), comparação dos cenários A/B/C, sensibilidade ao raio e painéis regionais a $R \le 50$ km. |
| `gifs/` | `generate_cca_ipba_gifs.py` | `cca_urban_clustering_evolution.gif`, `ipba_watershed_percolation_evolution.gif` | CCA até o $(D^*, L^*)$ calibrado da América do Norte e invasão do IPBA no bloco das Américas, com endorreicas em cinza. |

Consulte os utilitários e o script modelo em:
- [`diretrizes/templates/scientific_plot_utils.py`](../diretrizes/templates/scientific_plot_utils.py)
- [`diretrizes/templates/template_mapa_cientifico.py`](../diretrizes/templates/template_mapa_cientifico.py)
