# Diretriz 01: Padrão Visual e Científico para Geração de Imagens e Figuras

Esta diretriz estabelece as **regras obrigatórias e inegociáveis** para a criação, edição e exportação de qualquer figura, mapa ou gráfico neste repositório. Qualquer agente (humano ou inteligência artificial) que for atuar sobre a base de código deve seguir rigorosamente estas especificações.

---

## 1. Padrão Científico de Publicação

Todas as figuras devem atender ao padrão visual exigido por periódicos internacionais de alto impacto (Nature, Science, Elsevier, AGU, IEEE):

1. **Tipografia Acadêmica (Serifada)**:
   - Utilizar fontes clássicas serifadas (`Latin Modern Roman`, `Computer Modern Roman`, `DejaVu Serif` ou `Times New Roman`).
   - Hierarquia de tamanhos proporcional e legível:
     - Títulos de painéis / eixos: `8.5 pt` a `9.5 pt`.
     - Rótulos dos eixos (ticks): `7.0 pt` a `8.0 pt`.
     - Legendas e anotações: `6.5 pt` a `7.5 pt`.
     - Identificadores de painéis `(a)`, `(b)`: `9.0 pt`, negrito (`fontweight='bold'`).
2. **Resolução e Exportação**:
   - Toda figura deve ser exportada em dois formatos:
     - **`.png`**: Rasterizado em alta resolução com no mínimo **300 DPI** (`dpi=300`), fundo limpo, sem compressão destrutiva.
     - **`.pdf`**: Gráfico vetorial com curvas perfeitas e fontes embutidas para compilação direta no LaTeX da tese (`Tese/dissertacao.tex`).
   - Usar sempre `bbox_inches='tight'` e `pad_inches=0.03` ao salvar via matplotlib para evitar cortes nas bordas ou excesso de margens em branco.
3. **Dimensões e Proporções**:
   - Ajustadas para a largura padrão do texto da tese:
     - Largura de coluna simples: `~8.5 cm` (~3.35 polegadas).
     - Largura de página inteira / duas colunas: `~17.0 cm` a `17.5 cm` (~6.7 a 6.9 polegadas).

---

## 2. Idioma Obrigatório: 100% em Inglês

**Regra estrita**: Não é permitido nenhum texto em português dentro das figuras (títulos, legendas, eixos, unidades, anotações e barras de cor). As figuras são destinadas a publicações científicas internacionais e à tese com padrão internacional.

### Glossário de Tradução Padronizado

| Termo Original (PT-BR) | Termo Obrigatório em Inglês (EN) | Observação / Exemplo |
| :--- | :--- | :--- |
| Elevação (m) | `Elevation (m)` | Rótulo da barra de cor de relevo |
| Densidade (hab/km²) | `Population Density (people/km²)` | Densidade populacional (GPW v4) |
| Nurdles (pellets por 10 min) | `Nurdles (pellets / 10 min)` | Concentração Nurdle Patrol |
| mediana na grade, média no site | `grid median, site mean` | Descritor de agregação |
| só outros métodos (X amostras) | `Other methods only (X samples)` | Filtro NOAA microplásticos |
| Nurdle Patrol (X amostras) | `Nurdle Patrol (X samples)` | Amostras Nurdle Patrol |
| Sites Nurdle Patrol (X) | `Nurdle Patrol Sites (X)` | Título de legenda |
| zero (X sites) | `Zero (X sites)` | Sites sem detecção |
| > 0 (X sites) | `> 0 (X sites)` | Sites com detecção positiva |
| Bacia de drenagem / Bacias | `Drainage basin` / `Drainage basins` | IPBA bacias hidrográficas |
| Foz / Fozes dos rios | `River mouth` / `River mouths` | Ponto de descarga no oceano |
| Manchas urbanas | `Urban clusters` / `Urban agglomerations` | Blobs derivados do CCA |
| Recorte / Área de estudo | `Study area` / `Analysis domain` | Domínio recortado da América do Norte |
| População acumulada | `Cumulative population` | Distribuições de Zipf |
| Raio de busca | `Search radius (km)` | Varredura de raio foz-site |

---

## 3. Mapa de Cores Padronizado: 'plasma'

**Regra estrita**: O mapa de cor principal para dados contínuos, escalas térmicas, concentrações e densidades deve ser **sempre** o **`plasma`** (`cmap='plasma'`).

```python
cmap = 'plasma'  # Obrigatório para dados escalares contínuos
```

### Por que 'plasma'?
- **Uniformidade Perceptiva**: Gradiente perceptualmente linear no espaço de luminosidade, prevenindo falsos artefatos visuais.
- **Acessibilidade**: Legível para daltônicos (protanopia, deuteranopia e tritanopia).
- **Impressão em Tons de Cinza**: Mantém monotonicidade de brilho se impresso em preto e branco.

### Cores de Fundo e Máscaras
- **Oceano / Água**: Tom neutro azulado muito claro (ex: `#dce8f2` ou `#e8f1f8`) ou branco (`#ffffff`), dependendo do contexto.
- **Continentes / Máscaras de Terra sem dados**: Tom de cinza neutro e suave (ex: `#e5e5e5` ou `#efefef`), com contornos em `#b0b0b0`.
- **Escalas Logarítmicas**: Quando os dados variarem em ordens de magnitude (como densidade populacional de `1` a `10^4` hab/km² ou contagem de nurdles), utilizar sempre `matplotlib.colors.LogNorm`:
  ```python
  from matplotlib.colors import LogNorm
  norm = LogNorm(vmin=1, vmax=1e4)
  ```

---

## 4. Composição de Mapas com Inset do Mapa-Múndi no Canto Inferior Esquerdo e Escala Vertical à Direita

Para qualquer imagem que apresente um mapa geográfico, a diagramação deve obedecer à seguinte regra arquitetural:

1. **Imagem Principal (Main Plot)**:
   - Deve ser a **região ampliada (zoom)** de interesse (ex.: América do Norte até o Canal do Panamá: `BOX = (-125, 7, -65, 50)`).
   - Ocupa o eixo principal da figura com o máximo de detalhe, ticks de coordenadas legíveis (`°N`, `°W`), barra de escala e visualização dos dados.
2. **Mapa-Múndi Localizador (World Inset Map)**:
   - Deve ser posicionado **obrigatoriamente no canto inferior esquerdo** (`lower left`) da imagem.
   - Pode ser desenhado como um sub-eixo sobreposto (`ax_inset = ax_main.inset_axes([0.02, 0.03, 0.30, 0.25])` ou `fig.add_axes([left, bottom, width, height])`).
   - O mapa-múndi deve ser desenhado **todo em cinza** (`fc='#d0d0d0'`, `ec='#909090'`), com fundo do oceano em branco (`#ffffff`) ou azul neutro muito suave (`#f4f7f9`).
   - **Retângulo de Destaque**: A região onde o zoom foi aplicado deve ser evidenciada no mapa-múndi por um retângulo com borda nítida (ex: vermelha `#d90429` ou preta `#222222`, `lw=1.2`), sem preenchimento (`fill=False`).
3. **Barra de Cores / Escala Vertical à Direita**:
   - A barra de cores (`colorbar`) deve estar **sempre em pé à direita** do painel (`orientation='vertical'`, `location='right'`).
4. **Proibição Absoluta de Linhas Conectoras Diagonais Cruzadas**:
   - **NUNCA** utilizar linhas tracejadas diagonais de projeção de zoom que saem de um painel e cruzam a tela até outro painel (`ConnectionPatch`).
   - O mapa localizador inset no canto inferior esquerdo substitui completamente a necessidade de painéis separados de mapa-múndi com linhas conectoras.

---

## 5. Regra da Não-Sobreposição (Zero-Overlap Guarantee)

**Regra estrita**: Itens importantes da imagem **NÃO** podem ficar sobrepostos. O agente deve planejar a geometria e os limites dos elementos para garantir total clareza.

### Casos de Falha Identificados e Soluções

1. **Sobreposição de Identificadores `(a)`, `(b)`, `(c)`**:
   - *Erro anterior (Anexo 1 e 2)*: Linhas de projeção passavam exatamente por cima dos identificadores `(c)` e `(d)`, tornando-os ilegíveis ou cortados.
   - *Correção*:
     - Não usar linhas conectoras cruzadas.
     - Posicionar os identificadores fora dos eixos de dados (`transform=ax.transAxes`, `x=0.0`, `y=1.02`, `va='bottom'`, `ha='left'`).
     - Se o identificador precisar ficar dentro do mapa, deve possuir uma caixa de fundo (bbox) semi-opaca branca (`bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.85)`).
2. **Sobreposição de Dados Geográficos e Costas**:
   - *Erro anterior (Anexo 1)*: As linhas conectoras cortavam o noroeste do Canadá, Alasca e a costa do Golfo do México, sobrepondo pixels de elevação e de densidade.
   - *Correção*: O uso do inset no canto inferior direito mantém a área continental do zoom 100% desobstruída.
3. **Barras de Escala e Legendas**:
   - Devem ser posicionadas em áreas oceânicas ou espaços vazios (ex.: no Oceano Pacífico ou no canto inferior esquerdo), com margem de segurança (`pad`) para nunca encostar ou cobrir linhas de costa, fozes de rios ou pontos de amostragem.
4. **Barras de Cor (Colorbars)**:
   - Posicionadas na base da figura (`location='bottom'`) ou à direita (`location='right'`), com espaçamento adequado (`shrink=0.75`, `pad=0.08`), rótulo claro e unidades explícitas.

---

## 6. Qualidade Gráfica do Mapa-Múndi (Sem Pixelização)

**Regra estrita**: Deve ser mantida a qualidade vetorial do mapa-múndi, evitando artefatos como os do Anexo 2.

### Análise do Problema do Anexo 2
No código anterior de microplásticos, o mapa-múndi do painel superior era desenhado rasterizando os dados da NOAA em uma grade grosseira de 2° (`grade_outros` e `grade_nurdle`), e o relevo GEBCO era reamostrado com redução agressiva de 80x. O resultado foi um mapa-múndi cheio de "tijolos" e blocos quadrados de pixels cinzas e coloridos, com linhas de costa serrilhadas e aspecto de baixa resolução.

### Padrão Obrigatório de Renderização
1. O mapa-múndi do inset deve ser renderizado a partir de **geometrias vetoriais limpas** (polígonos dos continentes).
2. O repositório disponibiliza o arquivo vetorial padrão em:
   `diretrizes/assets/world_land_110m.geojson` (Natural Earth 1:110m resolution).
3. O agente deve carregar este arquivo e desenhar os continentes usando preenchimento vetorial (`ax_inset.fill(xs, ys, fc='#d0d0d0', ec='#909090', lw=0.4)`).
4. Dessa forma, independentemente da escala de zoom ou do DPI de exportação, o mapa-múndi permanecerá perfeitamente suave, nítido e profissional.

---

## 7. Resumo dos Parâmetros Visuais para Matplotlib

```python
import matplotlib.pyplot as plt

SCIENTIFIC_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "mathtext.fontset": "cm",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

plt.rcParams.update(SCIENTIFIC_RC_PARAMS)
```
