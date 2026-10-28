# AGENTS.md — Regras de Ação para Agentes Autônomos

Este arquivo define as diretrizes operacionais mandatórias para **qualquer agente de IA ou desenvolvedor** atuando neste repositório.

Antes de executar qualquer modificação de código, criação de notebooks ou geração de gráficos, você **DEVE LER E SEGUIR** a documentação detalhada na pasta:
👉 **[`diretrizes/`](diretrizes/README.md)**

---

## ⚡ Regras de Ouro Obrigatórias para Geração de Imagens e Figuras

1. **Padrão Científico Internacional**:
   - Tipografia serifada acadêmica (`Latin Modern Roman`, `DejaVu Serif` ou `Times New Roman`).
   - Resolução mínima de **300 DPI** em PNG e versão em **PDF vetorial** para inclusão no LaTeX.
   - Detalhes em: [`diretrizes/01_PADRAO_VISUAL_E_FIGURAS.md`](diretrizes/01_PADRAO_VISUAL_E_FIGURAS.md).

2. **Idioma Obrigatório em Inglês**:
   - Todo texto dentro das figuras (títulos, legendas, eixos, unidades e barras de cor) deve estar **100% em inglês**.
   - Consulte a tabela de tradução padronizada no arquivo de diretrizes visuais.

3. **Mapa de Cores: 'plasma'**:
   - O colormap padrão para dados contínuos, escalas de relevo e densidades deve ser **sempre** o **`'plasma'`** (`cmap='plasma'`).
   - Usar `LogNorm` quando os dados cobrirem várias ordens de magnitude.

4. **Composição de Mapas com Inset Localizador à Esquerda em Baixo e Escala Vertical à Direita**:
   - O gráfico principal deve ser a **região com zoom** (área de estudo ampliada).
   - No **canto inferior esquerdo** (`lower left`) da imagem, deve constar obrigatoriamente um **mapa-múndi localizador (inset) todo em cinza** (`#d0d0d0`).
   - Um retângulo nítido (vermelho `#d90429` ou preto `#222222`) deve marcar a área do zoom sobre o mapa-múndi.
   - A **barra de cores / escala (`colorbar`)** deve ficar obrigatoriamente **em pé à direita** (`orientation='vertical'`, `location='right'`).
   - **NUNCA utilize linhas tracejadas diagonais (`ConnectionPatch`) cruzando painéis**.

5. **Regra de Não-Sobreposição (Zero-Overlap)**:
   - Identificadores de painéis `(a)`, `(b)`, `(c)` devem ficar fora da área de plotagem ou protegidos com margem.
   - Nenhuma linha, legenda ou barra de escala pode cobrir pontos de dados, linhas de costa ou letras.

6. **Alta Qualidade Gráfica do Mapa-Múndi (Sem Pixelização)**:
   - O mapa-múndi deve ser gerado a partir de geometrias vetoriais limpas (`diretrizes/assets/world_land_110m.geojson`), evitando blocos ou "tijolos" pixelizados (erro corrigido do Anexo 2).

7. **Organização Centralizada em `geracao_graficos/<tema>/`**:
   - Toda figura deve residir em uma subpasta temática correspondente (ex: `geracao_graficos/gebco/`, `geracao_graficos/noaa/`, `geracao_graficos/watherSeads/`).
   - A pasta deve conter **o código `.py` gerador E a própria imagem** em `.png` e `.pdf`.
   - Detalhes em: [`diretrizes/02_ESTRUTURA_E_ORGANIZACAO_GRAFICOS.md`](diretrizes/02_ESTRUTURA_E_ORGANIZACAO_GRAFICOS.md).

---

## 🛠️ Recursos e Templates Disponíveis

- **Utilitários Prontos**: [`diretrizes/templates/scientific_plot_utils.py`](diretrizes/templates/scientific_plot_utils.py)
- **Script Modelo**: [`diretrizes/templates/template_mapa_cientifico.py`](diretrizes/templates/template_mapa_cientifico.py)
- **Checklist de Auditoria**: [`diretrizes/03_CHECKLIST_E_PROTOCOLO_DE_ACAO.md`](diretrizes/03_CHECKLIST_E_PROTOCOLO_DE_ACAO.md)
