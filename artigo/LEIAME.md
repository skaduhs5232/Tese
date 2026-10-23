# Artigo — Lei de Zipf nas manchas urbanas do Nordeste brasileiro

Artigo final da disciplina (máx. 3 páginas + elemento gráfico), em pt-BR. Usa dados brasileiros,
como o manual recomenda: o GPW v4 (2020) para as manchas do CCA e o Censo 2022 do IBGE para os
municípios, com foco no Nordeste.

## Conteúdo

| Arquivo | O que é |
|---|---|
| `main.tex` | O artigo, nas três perguntas do manual: *Qual é o problema?*, *O que foi feito?*, *Resultados e conclusões* (com as limitações). |
| `referencias.bib` | 15 referências (Zipf, Moura Jr. e Ribeiro, CCA, CSN, Hill, Vuong, união e busca, `powerlaw`, GPW v4, IBGE). |
| `analise_brasil.py` | A análise: varredura D* × ℓ do CCA no Brasil, manchas do Nordeste, ajustes de Hill e CSN, municípios do Censo 2022. Grava `resultados_brasil/`. |
| `resultados_brasil/` | `resumo.json` (todos os números do texto), matrizes D* × ℓ do Brasil e do Nordeste, tabelas de manchas e de municípios. |
| `gerar_graficos_artigo.py` | Gera `figs/zipf.pdf` e `figs/zipf.png`: (a) mapa das manchas do Nordeste, (b) posto e tamanho de manchas e municípios, (c) α contra D* no Brasil e no Nordeste. |
| `artigo_overleaf.zip` | `main.tex`, `referencias.bib` e `figs/`, pronto para subir no Overleaf. |

## Reproduzir

Entradas, todas em `../dados/`:

- `gpw_data/` — GPW v4.11, contagem 2020, 30″ (o mesmo zip que os notebooks baixam do Drive);
- `ibge/brasil.geojson` e `ibge/nordeste_ufs.geojson` — API de malhas do IBGE
  (`/api/v3/malhas/paises/BR` e `/api/v3/malhas/regioes/2?intrarregiao=UF`, qualidade intermediária);
- `ibge/pop_municipios_censo2022.json` — SIDRA, `/values/t/4709/n6/all/v/93/p/all`.

```
python analise_brasil.py          # ~8 min: varredura completa e ajustes
python analise_brasil.py --complemento   # só o Nordeste no par nacional (segundos)
python gerar_graficos_artigo.py
```

Os números do `main.tex` vêm de `resultados_brasil/resumo.json`; se a análise for refeita com
outra grade ou outro critério, eles precisam ser conferidos no texto.

## Como subir no Overleaf

1. Entre em <https://www.overleaf.com>.
2. **New Project → Upload Project**.
3. Selecione `artigo_overleaf.zip`.
4. No Overleaf, confirme que o compilador é **pdfLaTeX** (Menu → Compiler).
