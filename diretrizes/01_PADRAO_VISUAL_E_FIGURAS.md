# Diretriz 01: Padrão Visual e Científico para Geração de Imagens e Figuras (Estilo Erneson Oliveira)

Esta diretriz estabelece as **regras obrigatórias** para a criação, edição e exportação de qualquer figura, mapa ou gráfico neste repositório. Toda figura da dissertação segue o padrão visual dos artigos do orientador, **Prof. Erneson Oliveira**:

- **Oliveira, E. A. et al. (2019)**, *"A universal approach for drainage basins"*, *Scientific Reports* **9**:9845 (IPBA, referência principal).
- **Oliveira, E. A. et al. (2018)**, *Royal Society Open Science* **5**:180468 (CCA/CLCA e lei de Zipf).
- **Operti, F. G., Oliveira, E. A. et al. (2017)**, *arXiv*:1706.05139 (varredura de parâmetros $D^* \times \ell$).
- **Oliveira, E. A. et al. (2014)**, *Scientific Reports* **4**:4235 (CCA e escala alométrica).

Quando o estilo original dos artigos diverge de uma decisão específica da dissertação confirmada pelo Thiago, **prevalece a regra do Thiago** (indicada com ⚖️).

---

## 1. Observações dos Artigos Originais (Base da Especificação)

| Elemento | Observação nos Artigos | Fonte |
| :--- | :--- | :--- |
| **Fonte** | Serifada **Computer Modern** (LaTeX) em todos os textos, números dos eixos e símbolos matemáticos em itálico ($\log_{10} P(s)$, $\gamma$, $R^2$). | 2019 Figs. 1–6; 2018 Figs. 2–5 |
| **Rótulos de painel** | **"(a)", "(b)"** minúsculas entre parênteses, **sem negrito**, `fontsize=12`, no canto superior esquerdo: **fora** dos eixos nos gráficos e **dentro** do painel nos mapas. | 2018 Figs. 3–4; 2014 Figs. 1, 3, 5; 2017 Fig. 3 |
| **Eixos log** | Plotam $\log_{10}$ dos valores em **eixo linear**, com rótulos `$\log_{10} A$`, `$\log_{10} P(A)$` e ticks inteiros (`0, 1, 2...`). | 2019 Figs. 4, 6; 2018 Fig. 4 |
| **Moldura e ticks** | Caixa completa (4 spines). Ticks **para fora** (`direction='out'`), só embaixo e à esquerda, sem minor ticks nos log-log e **sem grade (`grid=False`)**. | 2019 Figs. 4, 6; 2018 Figs. 4, 5 |
| **Marcadores** | **Vazados** (`facecolors='none'`), cores do ciclo `tab10` (`C0` azul, `C1` laranja, `C2` verde...). Formas variadas por série: `o`, `*`, `s`, `x`, `^`, `+`. | 2019 Figs. 4, 6; 2018 Fig. 4 |
| **Reta de ajuste** | **OLS em preto sólido** (`lw=1.2`), apenas na faixa ajustada. Estimador de Nadaraya–Watson tracejado colorido (`C1`) com faixa sombreada do IC 95% (`alpha=0.3`). Referência $\beta = 1$ tracejada. | 2019 Fig. 6; 2018 Fig. 4; 2014 Fig. 4 |
| **Guia de inclinação** | Colchete em "L" preto (segmento horizontal e vertical) próximo aos dados, com o símbolo do expoente ($\alpha$, $\beta$) ao lado. | 2019 Figs. 4, 6; 2014 Fig. 4 |
| **Anotação de expoente** | Texto matemático dentro do gráfico, num canto livre, em 2–3 linhas sem caixa: `$\gamma = 0.54 \pm 0.01$` / `$R^2 = 0.87$`. Erro com 2 algarismos significativos e valor com as mesmas casas decimais. Ponto decimal (inglês). | 2019 Fig. 6; 2018 Fig. 4 |
| **Nome da série/região** | Texto romano serifado num canto (`"GEBCO 2026"`, `"South America"`, `"Amazon basin"`, `"CCA"`), sem caixa. | 2019 Figs. 2, 5; 2018 Figs. 3, 4 |
| **Legenda** | **Sem moldura** (`frameon=False`), num canto livre ou fora à direita. | 2019 Fig. 4; 2018 Fig. 5; 2017 Fig. 3 |
| **Mapas** | **Sem eixos, ticks, moldura ou grade** (`set_axis_off()`). Projeção **equiretangular (lon/lat, Plate Carrée)**, `aspect='equal'`, `interpolation='nearest'`, fundo branco. | 2019 Figs. 1, 2, 5; 2018 Figs. 2, 3 |
| **Cor fora do domínio** | Bacias (2019): oceano/fora em **cinza-claro `#e5e5e5`**. População e CCA (2018/2017): terra sem população/mancha em **cinza-escuro `#666666`** e oceano branco `#ffffff`. | 2019 Figs. 2D–I; 2018 Fig. 2; 2017 Fig. 4 |
| **Colorbar** | Vertical, **à direita**, fina (`size="3%"`, `pad=0.05`), rótulo girado 90° com unidade entre parênteses (`"Elevation (km)"`, `"Population density (inhab./km$^2$)"`). Em `LogNorm`, ticks em potências de 10 ($10^k$). | 2019 Figs. 2, 5; 2018 Fig. 3a |

---

## 2. Regras do Thiago que Prevalecem (⚖️ Conflitos com o Estilo Original)

1. ⚖️ **Mapas de bacias do IPBA**: utilizam o colormap **`tab20`** com coloração de grafo de adjacência-4 (`cores_bacias_tab20`), garantindo que **bacias vizinhas nunca tenham a mesma cor** (`assert`). O fundo fora das bacias (oceano) permanece **cinza-claro `#e5e5e5`**, como em 2019. Não possuem colorbar (dado categórico).
2. ⚖️ **Colorbar sempre vertical à direita**: todas as barras de cor contínuas ficam em pé à direita (`colorbar_direita`), substituindo as barras horizontais internas de 2018 Fig. 2 e 2014 Fig. 1.
3. ⚖️ **Colormap contínuo: sempre `plasma`**: nos artigos originais usa-se `inferno`, mas nesta dissertação utiliza-se **sempre `plasma`** para todo dado contínuo (relevo, densidade populacional, matrizes de calibração $D^* \times \ell$ e zoom da Bacia Amazônica). A única exceção é `tab20` nos mapas categóricos de bacias do IPBA.
4. **Inset localizador opcional**: usado apenas quando o recorte geográfico não for óbvio (`inset_localizador`), com continentes vetoriais de `diretrizes/assets/world_land_110m.geojson` em cinza `#d0d0d0` e retângulo tracejado preto.
5. **Idioma e formatos**: textos 100% em inglês, exportação simultânea em **PNG (300 DPI)** e **PDF vetorial** ao lado do script `.py` em `geracao_graficos/<tema>/`, sem sobreposição de elementos.
6. **Três mapas-múndi obrigatórios**: GEBCO, GPW e NOAA (este último exibindo todas as estações Beach e Nurdle Patrol do mundo, com as pareadas em destaque).

---

## 3. Módulo Central `geracao_graficos/estilo_erneson.py` e `rcParams`

Nenhum script define estilo por conta própria: todos importam e chamam `aplicar_estilo()` de [`geracao_graficos/estilo_erneson.py`](../geracao_graficos/estilo_erneson.py).

```python
from geracao_graficos.estilo_erneson import (
    ESTILO_ERNESON,
    aplicar_estilo,
    rotulo_painel,
    rotulo_canto,
    anotar_expoente,
    guia_inclinacao,
    scatter_vazado,
    reta_ols,
    curva_nw,
    referencia_inclinacao_1,
    eixos_log10,
    colorbar_direita,
    eixo_mapa,
    cores_bacias_tab20,
    cores_manchas,
    inset_localizador,
    salvar,
)
```

---

## 4. Especificação por Tipo de Figura (F1 a F10)

- **F1. Mapa-múndi GEBCO** (`geracao_graficos/gebco/gebco_world_map.py`): equiretangular global `(7.0, 3.6)`, sem eixos, `plasma` linear em km, colorbar à direita `"Elevation (km)"`, `rotulo_canto(ax, "GEBCO 2026", "lower right")`, contorno fino do recorte Américas.
- **F2. Mapa-múndi GPW** (`geracao_graficos/populacao/gpw_world_map.py`): equiretangular global, densidade em `plasma` + `LogNorm` ($10^0 \dots 10^4$), terra com densidade 0 em `#666666`, oceano branco, colorbar à direita `"Population density (inhab./km$^2$)"`, `rotulo_canto(ax, "GPW v4", "lower right")`.
- **F3. Mapa-múndi NOAA** (`geracao_graficos/noaa/noaa_world_map.py`): continentes vetoriais `#e5e5e5` sem contorno, oceano branco, todas as estações Beach e Nurdle Patrol do mundo como pontos pequenos preenchidos (`alpha = 0.35`) e, por cima, as pareadas a uma foz das Américas dentro do raio $R$ com marcadores vazados (`C0` `o` Nurdle Patrol, `C1` `s` Beach), legenda sem moldura com $n$ de cada grupo, sem colorbar.
- **F4. Mapas de bacias do IPBA e Zoom Amazônia** (`geracao_graficos/watherSeads/plot_drainage_basins_ipba.py`): painéis (a) América do Sul e (b) América do Norte + Central com `cores_bacias_tab20`, oceano `#e5e5e5`, fozes mantidas em pontos pretos `s=2`, sem colorbar; zoom da Bacia Amazônica (modelo 2019 Fig. 5) com relevo em `plasma` + `LogNorm`, divisores de sub-bacias em linhas pretas finas `lw=0.4` e colorbar à direita `"Elevation (m)"`.
- **F5. Mapa das manchas do CCA** (`geracao_graficos/populacao/plot_population_cca_regions.py`): manchas com `cores_manchas`, terra sem mancha `#666666`, oceano branco, parâmetros `$D^*$` e `$\ell$` no canto superior esquerdo (`fontsize=14`), nomes das 5–8 maiores aglomerações em serifa (`fontsize=9`) sem setas, mais painel de densidade GPW regional em `plasma` + `LogNorm`.
- **F6. Lei de Zipf (rank-tamanho) por região** (`geracao_graficos/zipf/plot_zipf_calibration_regions.py`): um painel por região, eixos `$\log_{10} r$` $\times$ `$\log_{10} P$` em escala linear, círculos vazados `C0`, OLS preto sólido nas $N = 100$ maiores manchas, referência $\alpha = 1$ tracejada, `anotar_expoente` no canto inferior esquerdo e nome da região no superior direito.
- **F7. Calibração $D^* \times \ell$** (`geracao_graficos/zipf/plot_zipf_calibration_regions.py`): Linha 1 = heatmap de $\alpha$ em `plasma` com contorno preto em $\alpha = 1$, faixa $\ell < \ell_{\min}$ em `#e5e5e5` hachurada `///`, células compatíveis com Zipf (IC 95% contém 1) contornadas em branco e estrela preta vazada no ponto ótimo; Linha 2 = heatmap de $R^2$; Linha 3 = curvas $\alpha \times \ell$ por $D^*$ (marcadores vazados `tab10`, barras de erro SE com cap, linha horizontal tracejada $\alpha = 1$ e vertical tracejada vermelha `C3` em $\ell^*$).
- **F8. Alometria população $\times$ plástico** (`geracao_graficos/integrado/plot_global_allometry_correlation.py`): modelo 2019 Fig. 6, eixos `$\log_{10} P$` $\times$ `$\log_{10}(M \times A)$` em escala linear, marcadores vazados por regime NOAA, OLS preto sólido, Nadaraya–Watson tracejado `C1` com banda IC 95%, referência $\beta = 1$ tracejada cinza e guia de inclinação em "L".
- **F9. Distribuições** (`geracao_graficos/integrado/plot_distributions_erneson.py`): densidades de probabilidade em bins logarítmicos ($\log_{10} P(A) \times \log_{10} A$ para área das bacias e população das manchas, modelo 2019 Fig. 4) e histograma da distância foz–site com curva normal ajustada (modelo 2018 Fig. 5).
- **F10. Multi-painel e GIFs** (`geracao_graficos/gifs/generate_cca_ipba_gifs.py`): evolução passo a passo do CCA (`cores_manchas`) e do IPBA (`cores_bacias_tab20`) no padrão visual Erneson.
