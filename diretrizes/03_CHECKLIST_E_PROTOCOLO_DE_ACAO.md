# Diretriz 03: Checklist e Protocolo de Ação para Agentes (Padrão Erneson Oliveira)

Todo agente de IA ou desenvolvedor deve conferir cada figura gerada contra este checklist (Passo H.6) antes de encerrar qualquer tarefa.

---

## Checklist Obrigatório de Auditoria (Passo H.6)

- [ ] **1. Estilo e Tipografia**: Chamou `aplicar_estilo()` de `geracao_graficos/estilo_erneson.py`. Fonte Computer Modern (`LaTeX` ou `mathtext.fontset="cm"`) e símbolos em modo matemático.
- [ ] **2. Idioma e Notação**: Todo texto 100% em inglês. Unidades entre parênteses (ex.: `Elevation (km)`, `Population density (inhab./km$^2$)`). Ponto decimal.
- [ ] **3. Rótulos de Painel**: `"(a)"`, `"(b)"`… em minúsculas entre parênteses, **sem negrito**, `fontsize=12`, no canto superior esquerdo (fora dos eixos nos gráficos, dentro nos mapas via `rotulo_painel`).
- [ ] **4. Eixos e Moldura dos Gráficos**: Caixa completa (4 spines), ticks **para fora** apenas embaixo e à esquerda, **sem grade (`grid=False`)**, sem minor ticks nos gráficos log-log. Valores transformados por $\log_{10}$ em eixo linear com rótulo `$\log_{10}$`.
- [ ] **5. Marcadores e Regressões**: Marcadores **vazados** (`scatter_vazado`), cores `tab10`, ajuste OLS em **preto sólido** apenas na faixa ajustada (`reta_ols`), anotação `"símbolo = valor ± erro"` + `$R^2$` num canto livre (`anotar_expoente`).
- [ ] **6. Legendas**: Sem moldura (`frameon=False`) e posicionadas sem cobrir pontos de dados.
- [ ] **7. Mapas**: Projeção equiretangular (`aspect="equal"`), sem eixos/ticks/moldura (`eixo_mapa`), `interpolation="nearest"`, cores fora do domínio conforme o tipo (`#e5e5e5` nos mapas de bacias; oceano branco `#ffffff` + terra sem dado `#666666` nos mapas de população e CCA).
- [ ] **8. Mapas de Bacias do IPBA**: Colormap `tab20` via `cores_bacias_tab20` com o `assert` de bacias vizinhas com cores diferentes passando. **Sem colorbar** (dado categórico).
- [ ] **9. Dados Contínuos e Colorbar**: Colormap **`plasma`** (+ `LogNorm` quando cobrir várias décadas) e colorbar vertical **à direita** (`colorbar_direita`) com rótulo girado 90° e unidade entre parênteses.
- [ ] **10. Inset Localizador**: Opcional (`inset_localizador`), incluído apenas quando o recorte geográfico não for óbvio.
- [ ] **11. Três Mapas-Múndi**: Existem os 3 mapas-múndi (`gebco_world_map`, `gpw_world_map`, `noaa_world_map`), sendo que o da NOAA exibe todas as estações Beach e Nurdle Patrol do mundo e destaca as pareadas com fozes das Américas.
- [ ] **12. Exportação e Organização**: Arquivos `.png` (300 DPI) e `.pdf` vetorial salvos ao lado do script `.py` gerador em `geracao_graficos/<tema>/` via `salvar()`.
- [ ] **13. Zero Sobreposição**: Nenhum elemento sobreposto (rótulos de painel, legendas, nomes de cidades, anotações de expoente ou barras de cor).
