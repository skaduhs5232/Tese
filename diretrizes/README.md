# Diretrizes de Ação para Agentes (Tese Mestrado)

Bem-vindo ao repositório da pesquisa de mestrado sobre correlação entre população de bacias de drenagem e concentração de microplásticos nas fozes dos rios (América do Norte / NOAA / Nurdle Patrol).

Este diretório contém as **diretrizes formais de ação** que regem o comportamento, a padronização e o fluxo de trabalho de qualquer agente de inteligência artificial (Claude, Gemini, Antigravity, Copilot, etc.) ou pesquisador que atue neste repositório.

---

## 📌 Sumário das Diretrizes

1. **[01. Padrão Visual e Científico para Imagens](01_PADRAO_VISUAL_E_FIGURAS.md)**:
   - Padrão científico de periódicos (tipografia serifada, DPI ≥ 300, exportação vetorial PDF + PNG).
   - **Idioma obrigatório**: 100% em **Inglês** (com glossário padronizado de termos da tese).
   - **Mapa de cores obrigatório**: sempre utilizar o colormap **`'plasma'`** para variáveis escalares e densidades.
   - **Composição com Inset**: Imagem principal é a área ampliada (zoom); **mapa-múndi localizador obrigatoriamente no canto inferior direito, todo em cinza**, com retângulo destacando a área de zoom.
   - **Regra da Não-Sobreposição (Zero-Overlap)**: Proibição de linhas diagonais cruzadas (`ConnectionPatch`) sobrepostas a pontos de dados, linhas de costa ou identificadores de painéis `(a)`, `(b)`, `(c)`.
   - **Alta qualidade vetorial do mapa-múndi**: Eliminação de blocos/tijolos pixelizados de baixa resolução (como visto no Anexo 2). Uso de polígonos vetoriais limpos (`assets/world_land_110m.geojson`).

2. **[02. Estrutura e Organização de Gráficos](02_ESTRUTURA_E_ORGANIZACAO_GRAFICOS.md)**:
   - Toda figura deve residir em uma subpasta temática dentro de `geracao_graficos/` (ex.: `gebco/`, `noaa/`, `watherSeads/`, etc.).
   - Em cada subpasta é estritamente obrigatório que constem **o código Python que gerou a figura** E **a própria imagem** nos formatos `.png` e `.pdf`.
   - Reprodutibilidade e caminhos relativos robustos a partir da raiz do repositório.

3. **[03. Checklist e Protocolo de Ação](03_CHECKLIST_E_PROTOCOLO_DE_ACAO.md)**:
   - Passo a passo operacional para o agente antes, durante e após a geração de figuras.
   - Checklist de 8 pontos de auditoria de qualidade antes de concluir qualquer tarefa.

4. **[Templates e Recursos Prontos](templates/)**:
   - `templates/scientific_plot_utils.py`: Funções utilitárias prontas para configurar fontes científicas, desenhar o inset do mapa-múndi em cinza no canto inferior direito, adicionar barra de escala e identificadores sem colisão.
   - `templates/template_mapa_cientifico.py`: Script de exemplo completo, funcional e executável.
   - `assets/world_land_110m.geojson`: Base geográfica vetorial do mapa-múndi em alta definição para renderização offline ultrarrápida.

---

## ⚡ As 7 Regras de Ouro (Cheat-Sheet do Agente)

> [!IMPORTANT]
> Se você for gerar ou alterar qualquer gráfico ou mapa neste repositório, confira estas 7 regras:
> 1. **Idioma**: Apenas **inglês** na imagem (labels, títulos, legendas, unidades).
> 2. **Colormap**: Apenas **`'plasma'`** para dados e gradientes contínuos.
> 3. **Layout do Mapa**: O mapa principal é o recorte de zoom; no **canto inferior direito** fica o **mapa-múndi em cinza** com o retângulo da região.
> 4. **Sem Sobreposições**: Nunca cruze linhas tracejadas sobre painéis, letras `(a)`, `(b)` ou dados.
> 5. **Qualidade do Mapa-Múndi**: Nunca use grades rasterizadas pixeladas (erro do Anexo 2); use polígonos vetoriais suaves.
> 6. **Pasta de Destino**: Cada figura vai em `geracao_graficos/<tema>/`.
> 7. **Dupla Presença**: A pasta do tema deve conter **o código `.py`** e **a imagem (`.png` e `.pdf`)**.
