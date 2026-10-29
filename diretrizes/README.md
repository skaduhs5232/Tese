# Diretrizes de Ação para Agentes (Tese Mestrado)

Bem-vindo ao repositório da pesquisa de mestrado sobre correlação alométrica entre a população de bacias hidrográficas e a concentração/carga de microplásticos costeiros nas fozes dos rios nas Américas (**América do Sul** e **América do Norte + Central**), fundamentado em **Oliveira, E. A. et al. (2019), *"A universal approach for drainage basins"*, Scientific Reports 9:9845**.

---

## 📌 Sumário das Diretrizes

1. **[01. Padrão Visual e Científico para Imagens (Estilo Erneson Oliveira)](01_PADRAO_VISUAL_E_FIGURAS.md)**:
   - Tipografia Computer Modern (`mathtext.fontset="cm"`), gráficos log-log em eixo linear ($\log_{10} X$), caixa completa com ticks para fora, sem grade, marcadores vazados (`tab10`), reta OLS preta sólida, Nadaraya–Watson tracejado com IC 95% e guia de inclinação em "L".
   - **Regras do Thiago (⚖️)**:
     1. Mapas de bacias do IPBA em **`tab20`** com coloração de grafo (`cores_bacias_tab20`, bacias vizinhas sempre com cores diferentes), oceano em `#e5e5e5`, sem colorbar.
     2. Dados contínuos sempre em **`plasma`** (relevo, densidade GPW, matrizes de calibração, zoom Amazônia) com **colorbar vertical à direita**.
     3. Inset localizador **opcional** (apenas quando o recorte não for óbvio).
     4. Três mapas-múndi obrigatórios: GEBCO, GPW e NOAA (todas as estações Beach e Nurdle Patrol do mundo, com as pareadas em destaque).
     5. Textos 100% em inglês, exportação em `.png` (300 DPI) + `.pdf` vetorial ao lado do `.py`.

2. **[02. Estrutura e Organização de Gráficos](02_ESTRUTURA_E_ORGANIZACAO_GRAFICOS.md)**:
   - Módulo central de estilo: **[`geracao_graficos/estilo_erneson.py`](../geracao_graficos/estilo_erneson.py)**.
   - Lista completa dos scripts geradores F1 a F10, entradas e saídas.

3. **[03. Checklist e Protocolo de Ação](03_CHECKLIST_E_PROTOCOLO_DE_ACAO.md)**:
   - Checklist de 13 pontos (Passo H.6) que o agente confere em todas as figuras.

4. **[Templates e Recursos Prontos](templates/)**:
   - [`geracao_graficos/estilo_erneson.py`](../geracao_graficos/estilo_erneson.py): Módulo oficial de estilo e funções de plotagem.
   - [`templates/scientific_plot_utils.py`](templates/scientific_plot_utils.py): Wrapper de compatibilidade que importa de `estilo_erneson.py`.
   - [`templates/template_mapa_cientifico.py`](templates/template_mapa_cientifico.py): Script modelo executável.
   - [`assets/world_land_110m.geojson`](assets/world_land_110m.geojson): Base vetorial global (Natural Earth 1:110m).
