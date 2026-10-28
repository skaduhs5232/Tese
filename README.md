# NOAA + GEBCO 2026 + GPW v4: Bacias Hidrográficas, População e Microplásticos em Escala Global

Este repositório investiga se a **população residente em uma bacia hidrográfica** explica a **concentração e a carga de microplásticos medidas na foz costeira dessa bacia**, integrando três bases geoespaciais globais sob a formulação de **Oliveira, E. A. et al. (2019), *"A universal approach for drainage basins"*, Scientific Reports 9:9845**.

> [!IMPORTANT]
> Antes de modificar códigos ou gerar figuras, consulte as diretrizes obrigatórias em **[`AGENTS.md`](AGENTS.md)** e **[`diretrizes/README.md`](diretrizes/README.md)**.

---

## Estrutura do Repositório

| Pasta / Arquivo | Conteúdo |
| :--- | :--- |
| **[`calculos/`](calculos/)** | Biblioteca em NumPy + Numba: `geometria_esferica/` (área por 2 triângulos esféricos), `percolacao_ilhas/` (percolação $h>0$, oceano e *sinks*), `hidrologia_endorreica/` (*priority-flood* e bacias sem saída ao mar) e `alometria_estatistica/` (densidade GPW, Zipf dinâmico, divisão de manchas entre bacias, pareamento foz–NOAA, ajustes log-log com bootstrap) |
| **[`notebooks/CCA`](notebooks/CCA)** e **[`notebooks/IPBA`](notebooks/IPBA)** | Algoritmos originais do orientador (City Clustering Algorithm e Invasion Percolation-Based Algorithm), importados sem modificação |
| **[`dados/download_dados.py`](dados/download_dados.py)** | Download/validação local do GEBCO 2026 e do GPW v4 Rev11; ingere o GeoJSON NOAA local via [`dados/ingestao_noaa_geojson.py`](dados/ingestao_noaa_geojson.py) |
| **[`notebooks/gebco_cluster_dvider/executar_divisao_ilhas.py`](notebooks/gebco_cluster_dvider/executar_divisao_ilhas.py)** | Percolação global (4.735 massas de terra, 149,49 Mkm²) e exportação dos **2 blocos hidrológicos contíguos** |
| **[`notebooks/executar_pipeline_global.py`](notebooks/executar_pipeline_global.py)** | Pipeline padrão-ouro v2: IPBA em 2 blocos + endorreísmo + Zipf dinâmico/CCA em 6 regiões + acoplamento bacia–cidade + pareamento NOAA + alometria (cenários A/B/C) |
| **[`notebooks/01_Analise_Integrada_CCA_IPBA_Microplasticos.ipynb`](notebooks/01_Analise_Integrada_CCA_IPBA_Microplasticos.ipynb)** | Notebook mestre que lê os resultados, prova a unicidade do Nilo e o isolamento das bacias endorreicas e exibe as figuras |
| **[`geracao_graficos/`](geracao_graficos/README.md)** | Figuras científicas (`.png` 300 DPI + `.pdf` vetorial + script `.py`) em inglês e paleta `'plasma'` |

## Ordem de Execução

```powershell
venv\Scripts\python.exe dados\download_dados.py --overview 2 --gerar-tiles-raw
venv\Scripts\python.exe dados\ingestao_noaa_geojson.py
venv\Scripts\python.exe notebooks\gebco_cluster_dvider\executar_divisao_ilhas.py
venv\Scripts\python.exe notebooks\executar_pipeline_global.py          # ~2,5 min
venv\Scripts\python.exe geracao_graficos\<tema>\<script>.py             # gebco, watherSeads, zipf, populacao, noaa, integrado, gifs
venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_Analise_Integrada_CCA_IPBA_Microplasticos.ipynb
```

---

## 1. Geometria Esférica Exata (Oliveira et al., 2019, Eqs. 3 e 4)

Cada célula da grade é decomposta em **dois triângulos esféricos** ($ABC$ e $ADC$) sobre a esfera de raio $R_{\text{earth}} = 6378{,}137$ km. Os arcos vêm da fórmula de Haversine e o excesso esférico do Teorema de L'Huilier:

$$A_{\triangle} = 4 R_{\text{earth}}^2 \arctan\sqrt{\tan\tfrac{s}{2}\tan\tfrac{s-a}{2}\tan\tfrac{s-b}{2}\tan\tfrac{s-c}{2}}$$

A área de uma bacia, ilha ou mancha urbana é a soma das áreas esféricas dos seus pixels.

## 2. Hidrologia: IPBA em 2 Blocos Contíguos e Bacias Endorreicas

- **Blocos**: as massas de terra são agrupadas pelo centróide em **Américas** (41,89 Mkm²) e **Afro-Eurásia + Oceania** (93,19 Mkm²). Nenhuma bacia é cortada por caixa arbitrária. O Mar Negro, o Mar de Azov e o Mar de Mármara recebem sementes marinhas, porque os estreitos são sub-pixel a 2'.
- **Endorreísmo** (`calculos/hidrologia_endorreica`): *priority-flood* (Barnes, 2014) sobre o DEM. São aceitas apenas depressões ancoradas em lagos terminais documentados, e o transbordamento é restrito à superfície preenchida de referência. O resultado são **37 bacias endorreicas (19,97 Mkm², 408 M hab)**, entre elas Cáspio, Chade, Aral, Eyre, Tarim, Balkhash, Great Basin e Altiplano. Elas recebem rótulo negativo e `is_endorheic=True` e ficam **fora dos emissores marinhos**.
- **Validação**: 21 de 30 grandes rios ficam a até ±20% da área de referência (Amazonas −5%, Congo +3%, Mississippi −7%, Paraná +8%, Nilo +17%). O **Nilo é uma bacia única**: Cairo, Aswan, Cartum, Lago Tana, Juba, Lago Alberto e Lago Vitória caem no mesmo rótulo (`notebooks/results/validacao_bacias_literatura.csv`). Ficam fora da tolerância: Ob/Yenisei (divisor deslocado), Amur, Níger, Zambeze, Indo, Nelson, Mekong e Pearl.
- **Grandes bacias**: as exorreicas que acumulam $P = 0{,}90$ da área ($A \geq 4.500$ km²), 1.442 no total.

## 3. Demografia: Zipf Dinâmico + CCA por Região Disjunta

A densidade GPW (contagem por pixel dividida pela área esférica, agregada a 1') é varrida em $D \in [100, 2000]$ hab/km² × $L \in [1000, 5000]$ m. Escolhe-se $\min|\alpha - 1|$ sujeito a $R^2 \geq 0{,}98$, com $\alpha$ e $R^2$ obtidos pelo OLS rank-size das 100 maiores manchas. **Não há parâmetros fixos.**

| Região | $D^*$ (hab/km²) | $L^*$ (m) | $\alpha$ | $R^2$ | Manchas | Divididas entre bacias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| América do Norte/Central/Caribe | 100 | 4000 | 0,997 | 0,986 | 15.348 | 853 |
| América do Sul | 100 | 5000 | 1,000 | 0,982 | 9.598 | 501 |
| Europa (até os Urais) | 100 | 2500 | 0,999 | 0,989 | 9.712 | 1.389 |
| África e Madagascar | 300 | 3500 | 1,145 | 0,987 | 3.081 | 366 |
| Ásia | 700 | 3500 | 0,999 | 0,988 | 4.001 | 888 |
| Oceania | 700 | 3000 | 1,232 | 0,983 | 352 | 92 |

A Oceania tem $\alpha > 1$ por macrocefalia urbana (5 capitais costeiras australianas). Na África nenhuma célula com $R^2 \geq 0{,}98$ chega a $\alpha \approx 1$. Três regiões escolhem $D^* = 100$, o limite inferior da grade.

## 4. NOAA Costeira e Alometria

O GeoJSON do NCEI Microplastics Hub tem 29.076 registros. Após remover 16.988 de `Ocean water`, restam **12.088 amostras costeiras**: 10.615 Nurdle Patrol, 893 Beach e 580 Ocean sediment. Cada foz de grande bacia exorreica é pareada ao site mais próximo (uma bacia por site).

| Cenário ($R \le 25$ km, $(M\times A)$ normalizado pela mediana vs. $P$) | $n$ | $\beta$ [IC 95%] | Pearson $r$ ($p$) | Spearman $\rho_s$ |
| :--- | :---: | :---: | :---: | :---: |
| A: baseline v1 | 51 | 0,386 [0,159; 0,656] | 0,408 ($2,9\times10^{-3}$) | 0,421 |
| B: hidrologia v2 + NOAA legado | 54 | 0,412 [0,212; 0,678] | 0,400 ($2,8\times10^{-3}$) | 0,466 |
| **C: hidrologia v2 + NOAA nova (oficial)** | **81** | **0,450 [0,235; 0,719]** | **0,383 ($4,2\times10^{-4}$)** | **0,407** |

O cenário C acrescenta 39 bacias com coleta em relação ao baseline. No regime Nurdle Patrol, $r$ fica entre 0,35 e 0,49 para $R$ = 15 a 50 km ($n$ de 46 a 72). Resultados completos em `notebooks/results/metricas_alometria_global.json` e `escala_pop_lixo_global.csv`.
