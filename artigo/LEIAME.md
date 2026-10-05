# Artigo — Lei de Zipf das manchas urbanas dos EUA

Artigo final da disciplina (máx. 3 páginas + elemento gráfico), em pt-BR.

## Conteúdo

| Arquivo | O que é |
|---|---|
| `main.tex` | O artigo. Estruturado nas três perguntas do manual: *Qual é o problema?*, *O que foi feito?*, *Resultados e conclusões* (esta última já inclui as limitações). |
| `referencias.bib` | 7 referências (Zipf/CCA, Clauset–Shalizi–Newman, os dois artigos do grupo UNIFOR/UFC, GPW v4, U.S. Census Bureau). |
| `figs/zipf.png` | A figura, com os painéis (a), (b) e (c). Gerada pelo notebook. |
| `figs/zipf.pdf` | A mesma figura em vetor (melhor qualidade de impressão — ver abaixo). |
| `artigo_overleaf.zip` | Tudo acima, pronto para subir no Overleaf. |

## Como subir no Overleaf

O projeto **não** foi criado no Overleaf a partir daqui (não tenho acesso à sua
conta). O caminho é:

1. Entre em <https://www.overleaf.com>.
2. **New Project → Upload Project**.
3. Selecione `artigo_overleaf.zip`.
4. No Overleaf, confirme que o compilador é **pdfLaTeX** (Menu → Compiler).

A bibliografia usa BibTeX; o Overleaf resolve as passadas automaticamente.

## A logo da Universidade de Fortaleza

**Falta um arquivo, que só você pode fornecer:** `figs/logo_unifor.png`.

Não incluí a logo porque ela não está no repositório e não é apropriado eu
reconstruir ou baixar a marca institucional. Enquanto o arquivo não existir, o
documento **compila normalmente** e mostra no lugar dela uma caixa vermelha
escrita `[inserir figs/logo_unifor.png]` — é o `\IfFileExists` no topo do
`main.tex`. Basta colocar o PNG em `figs/` com esse nome exato e recompilar;
nenhuma outra alteração é necessária.

## Se quiser a figura em vetor

O `main.tex` usa `figs/zipf.png` (300 dpi). Para impressão em vetor, troque no
`\includegraphics` de `figs/zipf.png` para `figs/zipf.pdf` — o arquivo já está
no zip.

## Regerar a figura

A figura foi gerada pela versão antiga do notebook de Zipf (EUA contíguo, painel
de sensibilidade ao ℓ), cujos resultados estão em `../results/texas_legado/`; o
`gerar_graficos_artigo.py` lê de lá. O notebook atual,
`../notebooks/01_Zipf_calibracao_CCA.ipynb`, usa o recorte até o Canal do Panamá
e a matriz D* × ℓ, então não regera esta figura.
