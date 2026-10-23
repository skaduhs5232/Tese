"""
Gera a figura do artigo (figs/zipf.pdf e figs/zipf.png) a partir das saídas do analise_brasil.py
(resultados_brasil/), na largura do texto em duas colunas (~17,4 cm):

(a) mapa das manchas urbanas do Nordeste no par (D*, l) calibrado para o Brasil;
(b) rank-size das manchas do Nordeste e dos municípios do Censo 2022, com a referência de Zipf;
(c) alpha das 50, 100 e 200 maiores manchas em função de D*, Nordeste e Brasil, com l calibrado.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi
from shapely.geometry import shape

import analise_brasil as ab

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "resultados_brasil")
FIGS_DIR = os.path.join(AQUI, "figs")
N_TOPO = (50, 100, 200)
ESTILO_N = dict(zip(N_TOPO, (":", "-", "--")))
DILATA_VIS = 2   # pixels; só para as manchas pequenas aparecerem no mapa
NOMES_MAPA = ("Recife", "Fortaleza", "Salvador", "São Luís", "Teresina", "Natal", "Maceió",
              "João Pessoa", "Aracaju", "Feira de Santana", "Petrolina", "Juazeiro do Norte", "Imperatriz",
              "Vitória da Conquista", "Campina Grande", "Mossoró")

os.makedirs(FIGS_DIR, exist_ok=True)
resumo = json.load(open(os.path.join(RES, "resumo.json"), encoding="utf-8"))
D_br, l_br = resumo["br"]["D_star"], resumo["br"]["l_metros"]
manchas = pd.read_csv(os.path.join(RES, "manchas_br.csv"))
mun = pd.read_csv(os.path.join(RES, "municipios_censo2022.csv"))
M = {r: np.load(os.path.join(RES, f"zipf_matriz_{r}.npz")) for r in ("br", "ne")}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "mathtext.fontset": "cm",
})
virgula = plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.6), dpi=300, gridspec_kw=dict(width_ratios=(1.25, 1, 1)))

# (a) mapa das manchas do Nordeste
ax = axes[0]
ne_geo = json.load(open(os.path.join(ab.DADOS, "ibge", "nordeste_ufs.geojson"), encoding="utf-8"))
for f in ne_geo["features"]:
    g = shape(f["geometry"])
    for p in getattr(g, "geoms", [g]):
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, fc="#f2f2f2", ec="#a0a0a0", lw=0.35, zorder=0)
tab = manchas[manchas["ne"]].sort_values("pop")
sc = ax.scatter(tab["lon"], tab["lat"], s=0.3 + 45 * tab["pop"] / 3.5e6, c=tab["pop"], cmap="YlOrRd",
                norm=LogNorm(1e4, 4e6), ec="#7a2e0e", lw=0.2, zorder=2)
ROTULOS = {"Fortaleza": (6, 3), "Recife": (7, 0), "Salvador": (7, -2), "São Luís": (-8, 7),
           "Teresina": (-30, -5)}
for _, r in tab.dropna(subset=["nome"]).iterrows():
    if r["nome"] in ROTULOS:
        ax.annotate(r["nome"], (r["lon"], r["lat"]), ROTULOS[r["nome"]], textcoords="offset points",
                    fontsize=5.4, va="center")
ax.set_xlim(-48.9, -31.0)
ax.set_ylim(-18.5, -1.0)
ax.set_aspect("equal")
ax.set_xticks([-46, -40, -34])
ax.set_xticklabels(["46°O", "40°O", "34°O"])
ax.set_yticks([-4, -10, -16])
ax.set_yticklabels(["4°S", "10°S", "16°S"])
ax.set_title(f"(a) {len(tab):,} manchas do Nordeste".replace(",", "."))


# (b) rank-size: manchas x municípios, Nordeste
ax = axes[1]
x = np.sort(manchas.loc[manchas["ne"], "pop"].to_numpy())[::-1]
y = np.sort(mun.loc[mun["ne"], "pop"].to_numpy())[::-1]
ax.loglog(np.arange(1, x.size + 1), x, "o", ms=1.8, mfc="#c2410c", mec="none", alpha=0.6,
          label="manchas")
ax.loglog(np.arange(1, y.size + 1), y, "s", ms=1.6, mfc="#4b5563", mec="none", alpha=0.55,
          label="municípios")
rr = np.logspace(0, 3.3, 30)
ax.loglog(rr, x[0] / rr, "--", c="k", lw=0.8, label="Zipf")
csn = resumo["ne_par_br"]["csn_manchas"] if "ne_par_br" in resumo else None
if csn:
    nt = int(csn["n"])
    rq = np.logspace(0, np.log10(nt), 40)
    ax.loglog(rq, csn["xmin"] * (rq / nt) ** (-csn["alpha"]), "-", c="#1f6fb4", lw=1.0,
              label=f"CSN: {csn['alpha']:.2f}".replace(".", ","))
ax.set_xlim(0.8, 3e3)
ax.set_ylim(5e3, 6e6)
ax.set_xlabel(r"posto $r$")
ax.set_ylabel(r"população $S(r)$")
ax.set_title("(b) posto e tamanho, Nordeste")
ax.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.2, markerscale=2)

# (c) alpha_N contra D*, Nordeste e Brasil, l calibrado
ax = axes[2]
D, L = M["br"]["D_star"], M["br"]["l_metros"]
i = int(np.flatnonzero(L == l_br)[0])
for r, cor in (("br", "#111111"), ("ne", "#c2410c")):
    for n in N_TOPO:
        ax.plot(D, M[r][f"alpha{n}"][i], ls=ESTILO_N[n], c=cor, lw=0.9)
ax.axhline(1.0, c="#888888", lw=0.6)
ax.axvline(D_br, c="#1f6fb4", lw=0.6)
ax.set_xlim(0, 3000)
ax.set_ylim(0.8, 1.5)
ax.set_xlabel(r"$D^*$ (hab/km$^2$)")
ax.set_ylabel(r"$\alpha$ (Hill)")
ax.set_title(rf"(c) $\ell={l_br / 1000:.1f}$ km".replace(".", "{,}"))
from matplotlib.lines import Line2D
alcas = [Line2D([], [], c="#111111", lw=1.2), Line2D([], [], c="#c2410c", lw=1.2)] + \
        [Line2D([], [], c="#555555", ls=ESTILO_N[n], lw=0.9) for n in N_TOPO]
ax.legend(alcas, ["Brasil", "Nordeste"] + [f"$N={n}$" for n in N_TOPO], loc="upper right", ncol=3,
          frameon=False, handlelength=1.3, handletextpad=0.4, columnspacing=0.6, borderaxespad=0.3, fontsize=5.8)
ax.yaxis.set_major_formatter(virgula)

fig.tight_layout(w_pad=0.5)
fig.savefig(os.path.join(FIGS_DIR, "zipf.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS_DIR, "zipf.png"), dpi=300, bbox_inches="tight")
print("figs/zipf.pdf e figs/zipf.png atualizados")
