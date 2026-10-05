# NOAA + GEBCO + GPW4: bacias hidrográficas, população e microplásticos

Este README explica o que os notebooks em [`notebooks/`](notebooks/) fazem, **por que** foram construídos dessa forma e quais alternativas foram descartadas. A especificação curta da metodologia está em [`tasks.txt`](tasks.txt).

## Estrutura do repositório

| Pasta / arquivo | Conteúdo |
|---|---|
| `notebooks/01_Zipf_calibracao_CCA.ipynb` | Calibra os parâmetros do CCA ($D^*$, $\ell$) pela lei de Zipf e grava `results/cca_calibracao.json` |
| `notebooks/02_Micro_Plastic.ipynb` | Pipeline principal: bacias (IPBA), manchas urbanas (CCA), microplástico (NOAA) e alometria população × lixo por bacia |
| `dados/` | GPW v4 (`gpw_data/`), `noaa_microplastics.csv` e os zips baixados (`archives/`) |
| `results/` | Saídas dos notebooks; `results/texas_legado/` guarda as saídas da versão antiga (recorte do Texas, raio de 50 km) |
| `IPBA/` | Cópia de referência do [IPBA](https://github.com/erneson/IPBA); os notebooks clonam o repositório em tempo de execução |
| `artigo/` | Manuscrito LaTeX (Zipf dos EUA) e script das figuras |
| `apresentacao/`, `artigos_relacionados/` | Slides e literatura de apoio |

Os notebooks só rodam no **Google Colab**. Ordem: primeiro o `01`, depois o `02` (enviando o `cca_calibracao.json` gerado pelo `01` para a pasta `results/` do `02`).

## 1. A pergunta

> Existe relação entre **quanta gente mora numa bacia hidrográfica** e **quanto plástico aparece na foz dela**?

| Dado | O que é | Papel |
|---|---|---|
| **GEBCO 2024** | Relevo/batimetria global (15″, ~460 m) | Define as bacias hidrográficas |
| **GPW v4** | Contagem de pessoas por pixel (30″, ~1 km), 2020 | População de cada bacia e manchas urbanas |
| **NOAA Marine Microplastics** | Amostras de microplástico; usamos só o *Nurdle Patrol* (`pieces/10 mins`) | Lixo na foz de cada bacia |

A análise é só de **microplástico**. O macroplástico (WAW / MDMAP) e a análise do Brasil foram retirados do projeto.

## 2. Recorte: América do Norte até o Canal do Panamá

A versão anterior usava uma caixa na costa do Texas, que cortava bacias grandes (Rio Grande, e tudo que drena para o Golfo vindo de fora da caixa). O recorte atual é `(-125, 7, -65, 50)`: EUA contíguo, sul do Canadá, México, América Central e Caribe. O que a caixa pega da América do Sul (a leste de 79°W e ao sul de 12,6°N) vira oceano no relevo e tem população zerada; 79°W fica logo a leste do canal, para não partir a Cidade do Panamá. O limite norte em 50°N deixa a bacia do Mississippi–Missouri inteira.

**Resolução do relevo.** O IPBA cria um objeto Python por pixel e grava os rótulos em `uint16` (no máximo 65.535 sinks). Com o GEBCO nativo a caixa teria ~150 milhões de pixels, então o relevo é reamostrado pela média em blocos de 8 × 8 (2′, ~3,7 km). O GPW continua em 30″.

## 3. Bacias hidrográficas: IPBA + sinks costeiros

O IPBA (*Invasion Percolation Based Algorithm*) simula a água invadindo o relevo a partir de pontos de saída (*sinks*) e agrupa cada pixel de terra na bacia do sink para o qual ele drena. Regra dos sinks: **todo pixel de terra 4-vizinho do oceano é um sink**. A bacia de um rio grande emerge sozinha, porque todos os pixels a montante drenam até o sink da foz.

**Oceano** é todo pixel com elevação ≤ 0 conectado à borda do recorte. Depressões interiores abaixo do nível do mar (Salton Sea, Death Valley, fundo dos Grandes Lagos) continuam sendo terra: não são foz e não viram sinks. O oceano entra no IPBA como NoData, para a percolação não vazar para o mar.

**Alternativa descartada:** `pysheds` (direção e acumulação de fluxo com limiar de área). Ele identifica rede de drenagem, não fozes no oceano, e exige calibrar um limiar de área crítica.

## 4. Bacias principais: as que somam 90% da área

Como cada pixel de costa é um sink, o IPBA gera dezenas de milhares de bacias, quase todas micro-bacias de praia. Ordenando por área e mantendo as que somam 90% da área do recorte (`P = 0,90`), sobram as bacias de rio de verdade, sem lista manual de rios. Bacias que tocam a borda da caixa ou o corte do Panamá são marcadas como truncadas (`toca_borda`).

## 5. Manchas urbanas: CCA calibrado pela lei de Zipf

O CCA (*City Clustering Algorithm*) agrupa pixels com densidade acima de $D^*$ que estão a menos de $\ell$ um do outro. $D^*$ e $\ell$ definem o que é "cidade", e o valor certo muda de região para região: os valores do semiárido brasileiro (30 hab/km², 2 km) tornavam 98% da população americana "urbana", e o limiar do U.S. Census (386 hab/km², 3 km) não vale para o México e a América Central, que agora estão no recorte.

**Critério (orientação de 25/09/2026):** varrer $D^* = 10, 20, \dots, 3000$ hab/km² e $\ell = 100, 200, \dots, 5000$ m, calcular o expoente de Zipf $\alpha$ das manchas em cada par ($S(r) \propto r^{-\alpha}$) e montar a matriz $D^* \times \ell$ colorida por $\alpha$. Os pares que seguem a lei de Zipf ficam sobre a curva $\alpha = 1$, exportada inteira em `results/zipf_curva_alpha1_*.csv`.

Como $\alpha$ é medido e como o ponto é escolhido:

- **Estimador.** $\alpha$ é estimado por Hill sobre as $N$ maiores manchas ($\zeta = (N-1)/\sum \ln(S_i/S_N)$, $\alpha = 1/\zeta$), com $N = 50, 100, 200$. A primeira versão ajustava a cauda com $x_{\min}$ escolhido por KS em cada célula. Esse $x_{\min}$ pulava de uma célula para outra e escolhia pares degenerados.
- **Piso de $\ell$.** $\ell \geq L_{\min} = 1400$ m. Abaixo da diagonal entre pixels do GPW (~1,3 km), pixels vizinhos em diagonal não se ligam, e as "manchas" viram faixas leste-oeste.
- **Escolha do ponto.** Fica o par com o menor $\Delta = \max_N |\alpha_N - 1|$, isto é, o que segue Zipf ao mesmo tempo no topo e mais abaixo no ranking.

Resultado no recorte até o Panamá: $D^* = 1010$ hab/km², $\ell = 1700$ m, $\alpha_{50,100,200} = 0{,}93;\ 0{,}94;\ 1{,}08$, com 12.185 manchas e 58% da população urbana. O ajuste de Clauset–Shalizi–Newman (CSN) sobre a cauda toda ($x_{\min} \approx 21$ mil hab, 1325 manchas) dá $\alpha = 1{,}22$: o rank-size é côncavo e $\alpha$ cresce com $N$. Por isso "Zipf" vale para as maiores manchas, não para a distribuição inteira.

O GPW tem um artefato em Monterrey: cerca de 3 milhões de pessoas em uns 20 pixels, com até 810 mil hab/km². A mancha não muda, porque todos esses pixels já estão acima de $D^*$, mas a densidade de pico dela não é real.

A varredura não roda o `cca_grid` 15.000 vezes: para cada $\ell$, os pixels são ativados em ordem decrescente de densidade e unidos (*union-find*) aos vizinhos ativos a menos de $\ell$; cada vez que a densidade cruza um $D^*$ da grade, as manchas presentes são exatamente as do CCA com aquele $D^*$. Num raster de teste, a varredura reproduziu o `cca_grid` em todas as 600 células comparadas, e o `01` repete essa checagem no par escolhido.

## 6. População por bacia e manchas divididas pela cumeada

A população de uma bacia é a soma dos pixels do GPW dentro dela. **Uma mancha urbana entre duas bacias é dividida**: cada pedaço conta para a bacia em que está, porque é para lá que ele drena. A versão anterior atribuía a mancha inteira à bacia com maior fração de área. A tabela `results/manchas_por_bacia_*.csv` lista os pedaços (mancha, bacia, população, área, fração).

As bacias são levadas para a grade do GPW (e não o contrário) porque o GPW é contagem de pessoas: reamostrar contagem inventaria ou duplicaria gente, enquanto o rótulo de bacia é categórico. Como a grade das bacias (2′) é mais grossa, pixels habitados na linha de costa podem cair num pixel de oceano do GEBCO; eles vão para a bacia de terra mais próxima (até ~7 km), e a população que sobra fora das bacias é reportada.

## 7. Microplástico por bacia: o ponto de coleta mais próximo da foz

O raio de 50 km da versão anterior misturava fontes e fazia bacias vizinhas compartilharem as mesmas amostras. Agora:

- amostras repetidas no mesmo local viram um **site** (média das concentrações);
- cada bacia principal recebe **o site mais próximo da sua foz** (o pixel-sink; a distância é até o pixel, então uma amostra dentro dele está a 0 km);
- além da concentração, a bacia recebe **concentração × área da bacia**, um indicador de quantidade.

**Raio de aceitação.** Uma bacia só tem lixo "medido" se o seu site estiver perto da foz. Em vez de fixar 50 km, o raio $R$ cresce de 1 em 1 km até 50 km, registrando a fração dos sites a até $R$ de uma foz principal, o número de bacias e de sites distintos, e o $r$ e o $\beta$ da lei de escala. O scatter usa dois raios: **$R = 1$ km (medição garantida)** e **$R_{\text{maioria}}$**, o menor raio que contém a maioria (≥ 50%) dos sites.

## 8. Alometria por bacia

Cada ponto do scatter é uma bacia: população da bacia × lixo da bacia, em log-log, com a lei de escala $y = c\,x^\beta$ ajustada por mínimos quadrados, IC de 95% por bootstrap e $p$ por permutação. A comparação "top 10 cidades × lixo" da versão anterior foi retirada: as manchas não são observações independentes de lixo (várias deságuam na mesma bacia).

## 9. Limitações

- A escolha de um único $(D^*, \ell)$ sobre a curva $\alpha = 1$ depende do critério $\Delta$ (topo 50/100/200); com outro $N$ o ponto muda.
- Com $R = 1$ km só 3 bacias principais têm site, poucas demais para ajustar uma reta; o resultado usável é o de $R_{\text{maioria}}$ (38 km, 29 bacias).
- Nova York se divide em ~120 bacias costeiras pequenas, nenhuma entre as principais: com um sink por pixel de costa, a foz do Hudson não junta a cidade numa bacia só.
- Com o relevo em 2′, a foz é um pixel de ~3,7 km; "a até 1 km" significa a até 1 km desse pixel.
- A caixa corta as bacias que drenam para a Baía de Hudson (Red River do Norte etc.); o IPBA as absorve nas bacias vizinhas dentro da caixa.
- O *Nurdle Patrol* se concentra na costa do Golfo; costas com poucas coletas contribuem com poucas bacias.
- Contagens zero de nurdles saem do scatter em log-log.
