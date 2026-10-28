# Diretriz 02: Estrutura e Organização da Pasta de Geração de Gráficos

Esta diretriz estabelece as regras obrigatórias de governança, organização de pastas e versionamento para qualquer arquivo de código ou gráfico gerado no repositório.

---

## 1. Localização Obrigatória: `geracao_graficos/`

Toda e qualquer figura utilizada na tese, no artigo científico, em apresentações ou em relatórios técnicos **deve residir na pasta**:
`c:\faculdade\mestrado\Tese\geracao_graficos\`

Nenhuma imagem final deve ficar solta na raiz do repositório, perdida em pastas temporárias, oculta em saídas não rastreadas de notebooks ou apenas salva em `notebooks/results/`.

---

## 2. Organização por Subpastas Temáticas

Dentro de `geracao_graficos/`, **cada tema ou conjunto de análises deve possuir sua própria subpasta dedicada**.

### Mapeamento Padronizado de Subpastas Temáticas

| Subpasta | Tema / Escopo Científico | Exemplos de Conteúdo |
| :--- | :--- | :--- |
| `geracao_graficos/gebco/` | Topografia, batimetria e relevo digital | Mapas de elevação GEBCO 2024, modelo digital de elevação (DEM), perfis altimétricos |
| `geracao_graficos/noaa/` | Microplásticos marinhos e Nurdle Patrol | Pontos de coleta NOAA, distribuição espacial de pellets, concentrações, taxas de amostragem |
| `geracao_graficos/watherSeads/` (ou `bacias/`) | Bacias de drenagem e hidrografia (IPBA) | Fozes de rios (coastal sinks), delimitação de bacias por percolação invasiva, áreas de drenagem |
| `geracao_graficos/populacao/` | Densidade e manchas urbanas (GPW v4 + CCA) | Densidade populacional global e regional, clusters urbanos calibrados |
| `geracao_graficos/zipf/` | Calibração de Zipf e leis de potência | Varreduras (D*, ℓ), matrizes de expoentes α, curvas de Hill, ajuste CSN |
| `geracao_graficos/integrado/` | Cruzamentos e alometria final | Relação população da bacia vs microplásticos na foz, leis de escala, mapas integrados |

> [!NOTE]
> Caso seja criada uma nova linha de análise ou um novo modelo, o agente deve criar uma nova subpasta autoexplicativa em inglês ou português coerente (ex: `geracao_graficos/sensibilidade_raio/`).

---

## 3. Conteúdo Obrigatório em Cada Subpasta

Em cada subpasta temática, **é estritamente obrigatório que coexistam**:

1. **O Script de Código Fonte (`.py`)**:
   - O arquivo executável Python responsável pela geração exata da figura (ex: `plot_gebco_recorte.py`, `plot_noaa_sampling_sites.py`).
   - O script deve ser **autônomo e reproduzível**: ao ser executado via terminal (`python plot_<nome>.py`), ele deve processar os dados e salvar a imagem diretamente na pasta correspondente sem intervenção manual.
2. **A Própria Imagem Gerada (em 2 formatos)**:
   - **Formato Raster (`.png`)**: Resolução mínima de **300 DPI**, fundo limpo, visualização rápida e inserção em apresentações ou web.
   - **Formato Vetorial (`.pdf`)**: Curvas vetoriais perfeitas com fontes incorporadas, pronto para inclusão direta no LaTeX da dissertação (`\includegraphics{...}`).

### Exemplo de Estrutura de Arquivos em uma Subpasta

```text
geracao_graficos/
├── gebco/
│   ├── plot_elevation_study_area.py       # Código que gerou a figura
│   ├── elevation_study_area.png           # Imagem raster (300 DPI)
│   └── elevation_study_area.pdf           # Imagem vetorial para LaTeX
├── noaa/
│   ├── plot_microplastics_sampling.py     # Código que gerou a figura
│   ├── microplastics_sampling.png         # Imagem raster (300 DPI)
│   └── microplastics_sampling.pdf         # Imagem vetorial para LaTeX
└── watherSeads/
    ├── plot_drainage_basins.py            # Código que gerou a figura
    ├── drainage_basins.png                # Imagem raster (300 DPI)
    └── drainage_basins.pdf                # Imagem vetorial para LaTeX
```

---

## 4. Requisitos de Código dos Scripts Geradores

Todo script criado dentro de uma subpasta de `geracao_graficos/` deve seguir as seguintes boas práticas de engenharia de software científico:

1. **Caminhos Relativos Robustos**:
   - O script deve calcular dinamicamente a raiz do repositório a partir de `__file__`, garantindo funcionamento em qualquer máquina ou clone:
     ```python
     import os
     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
     REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
     DATA_DIR = os.path.join(REPO_ROOT, "dados")
     ```
2. **Importação do Módulo de Padrão Científico**:
   - Recomenda-se utilizar os utilitários padronizados fornecidos em `diretrizes/templates/scientific_plot_utils.py` para garantir uniformidade tipográfica, uso de `plasma` e geração do mapa-múndi inset.
3. **Determinismo e Reprodutibilidade**:
   - Fixar sementes aleatórias se houver amostragem, bootstrap ou simulação estocástica (`np.random.seed(42)`).
4. **Mensagens Informativas no Terminal**:
   - Imprimir o progresso da execução e o caminho final dos arquivos salvos ao concluir.

---

## 5. Manutenção do Catálogo `geracao_graficos/README.md`

A pasta `geracao_graficos/` deve manter um arquivo `README.md` atualizado contendo:
- Tabela resumo de todas as figuras existentes.
- Subpasta, nome do script gerador, nome da figura de saída.
- Breve descrição científica do que a figura representa e em qual capítulo/seção da tese ela é utilizada.
