"""
gerar_graficos_artigo.py
-------------------------
Gera a figura científica oficial do artigo (figs/zipf.pdf, figs/zipf.png, figs/zipf_600dpi.png e figs/zipf.svg)
seguindo o padrão da comunidade científica internacional (Nature / Scientific Reports / PNAS)
e as diretrizes visuais do grupo de pesquisa (Prof. Dr. Erneson A. Oliveira).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from shapely.geometry import shape

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
RES = AQUI / "resultados_brasil"
FIGS_DIR = AQUI / "figs"
DADOS = RAIZ / "dados"
WORLD_GEOJSON = RAIZ / "diretrizes" / "assets" / "world_land_110m.geojson"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Carregar dados consolidados
resumo = json.load(open(RES / "resumo.json", encoding="utf-8"))
manchas = pd.read_csv(RES / "manchas_br.csv")
mun = pd.read_csv(RES / "municipios_censo2022.csv")

# 2. Configurações tipográficas e estéticas globais
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.3,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.0,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "figure.autolayout": False,
})

HALO_WHITE = [pe.withStroke(linewidth=2.0, foreground="white")]


def fmt_pt(val: float, decimals: int = 2) -> str:
    """Formata número decimal com vírgula para modo matemático LaTeX."""
    return f"{val:.{decimals}f}".replace(".", "{,}")


def add_panel_title(ax: plt.Axes, letter: str, title_text: str) -> None:
    """Insere o identificador (a), (b), etc. em negrito seguido do título."""
    ax.text(
        0.0, 1.028, f"({letter})",
        transform=ax.transAxes,
        fontsize=8.4, fontweight="bold", va="bottom", ha="left", family="serif"
    )
    ax.annotate(
        f" {title_text}",
        xy=(0.0, 1.028), xycoords="axes fraction",
        xytext=(14, 0), textcoords="offset points",
        fontsize=8.1, va="bottom", ha="left", family="serif"
    )


def add_scale_bar(ax: plt.Axes, lon0: float, lat0: float, length_km: int, dy_tick: float = 0.45, fontsize: float = 5.5) -> None:
    """Desenha barra de escala cartográfica geodésica na latitude lat0."""
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat0))
    dlon = length_km / km_per_deg_lon
    lon1 = lon0 + dlon
    ax.plot([lon0, lon1], [lat0, lat0], color="#1e293b", lw=1.1, zorder=15, solid_capstyle="butt")
    ax.plot([lon0, lon0], [lat0 - dy_tick * 0.5, lat0 + dy_tick * 0.5], color="#1e293b", lw=0.9, zorder=15)
    ax.plot([lon1, lon1], [lat0 - dy_tick * 0.5, lat0 + dy_tick * 0.5], color="#1e293b", lw=0.9, zorder=15)
    ax.text(
        0.5 * (lon0 + lon1), lat0 + dy_tick * 0.65,
        f"{length_km} km",
        fontsize=fontsize, ha="center", va="bottom", color="#1e293b",
        family="serif", path_effects=HALO_WHITE, zorder=16
    )


def add_north_arrow(ax: plt.Axes, x_frac: float = 0.055, y_frac: float = 0.90) -> None:
    """Insere seta de Norte cartográfica discreta."""
    ax.annotate(
        "N", xy=(x_frac, y_frac + 0.048), xycoords="axes fraction",
        xytext=(x_frac, y_frac - 0.025), textcoords="axes fraction",
        ha="center", va="center", fontsize=6.2, fontweight="bold", color="#334155",
        arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.35", color="#334155", lw=0.8),
        path_effects=HALO_WHITE, zorder=20
    )


# Proporção editorial para duas colunas (largura total = 6.85 in)
fig = plt.figure(figsize=(6.85, 4.78), dpi=300)
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.28, 1.0],
    hspace=0.30,
    wspace=0.22,
    left=0.075,
    right=0.905,
    top=0.945,
    bottom=0.140,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

# Escala logarítmica unificada plasma para os dois mapas
norm_pop = LogNorm(1e4, 2.2e7)
cmap_pop = plt.cm.plasma

# Carregar geometrias IBGE e continente
br_ufs_path = DADOS / "ibge" / "brasil_ufs.geojson"
ne_ufs_path = DADOS / "ibge" / "nordeste_ufs.geojson"
br_geo = json.load(open(br_ufs_path, encoding="utf-8")) if br_ufs_path.exists() else None
ne_geo = json.load(open(ne_ufs_path, encoding="utf-8")) if ne_ufs_path.exists() else None
world_geo = json.load(open(WORLD_GEOJSON, encoding="utf-8")) if WORLD_GEOJSON.exists() else None

ne_cods = {str(f["properties"].get("codarea", "")) for f in (ne_geo["features"] if ne_geo else [])}

# =========================================================================
# PAINEL (a) MAPA DO BRASIL: MANCHAS URBANAS (CCA)
# =========================================================================
ax_a.set_facecolor("#eef6fb")
ax_a.grid(True, linestyle=":", linewidth=0.35, color="#94a3b8", alpha=0.50, zorder=0)

# Contexto sul-americano em cinza neutro suave
if world_geo:
    for feat in world_geo.get("features", []):
        g = shape(feat["geometry"])
        for p in getattr(g, "geoms", [g]):
            xs, ys = p.exterior.xy
            ax_a.fill(xs, ys, fc="#e5e7eb", ec="#cbd5e1", lw=0.25, zorder=0.5)

# Estados do Brasil (não-Nordeste em tom claro neutro, Nordeste em tom marfim suave)
if br_geo:
    for f in br_geo["features"]:
        cod = str(f["properties"].get("codarea", ""))
        is_ne_uf = cod in ne_cods or cod.startswith("2")
        fc_uf = "#fefce8" if is_ne_uf else "#f8fafc"
        ec_uf = "#64748b" if is_ne_uf else "#94a3b8"
        lw_uf = 0.42 if is_ne_uf else 0.32
        g = shape(f["geometry"])
        for p in getattr(g, "geoms", [g]):
            xs, ys = p.exterior.xy
            ax_a.fill(xs, ys, fc=fc_uf, ec=ec_uf, lw=lw_uf, zorder=1.2 if is_ne_uf else 1.0)

# Retângulo tracejado indicando o recorte do Painel (b) (Nordeste)
rect_ne = Rectangle(
    (-48.8, -18.5), 15.0, 17.6,
    facecolor="none", edgecolor="#334155", lw=0.6, linestyle="--", zorder=2.5
)
ax_a.add_patch(rect_ne)
ax_a.text(
    -34.0, -0.55, "Recorte (b)",
    fontsize=4.9, color="#334155", fontstyle="italic",
    ha="right", va="bottom", path_effects=HALO_WHITE, zorder=6
)

# Plotar manchas urbanas ordenadas por população (sem borda preta nas pequenas para preservar o plasma!)
tab_br = manchas.sort_values("pop").copy()
pop_br_vals = tab_br["pop"].to_numpy()
sizes_br = 0.9 + 34.0 * ((pop_br_vals / 2.0e7) ** 0.60)

# Camada 1: manchas pequenas e médias (sem borda escura que ofusque o colormap plasma)
mask_small_br = pop_br_vals < 3.5e5
sc_br = ax_a.scatter(
    tab_br.loc[mask_small_br, "lon"],
    tab_br.loc[mask_small_br, "lat"],
    s=sizes_br[mask_small_br],
    c=pop_br_vals[mask_small_br],
    cmap=cmap_pop,
    norm=norm_pop,
    edgecolors="none",
    alpha=0.86,
    zorder=3,
)

# Camada 2: grandes centros e metrópoles (com contorno fino escuro para nitidez)
mask_big_br = ~mask_small_br
ax_a.scatter(
    tab_br.loc[mask_big_br, "lon"],
    tab_br.loc[mask_big_br, "lat"],
    s=sizes_br[mask_big_br],
    c=pop_br_vals[mask_big_br],
    cmap=cmap_pop,
    norm=norm_pop,
    edgecolors="#1e293b",
    linewidths=0.35,
    alpha=0.95,
    zorder=4,
)

# Rótulos das principais metrópoles nacionais posicionados 100% dentro da moldura e sem colisões
CIDADES_BR_CALLOUTS = [
    ("Manaus", (-60.02, -3.08), (-63.5, -0.5), "center", "bottom"),
    ("Belém", (-48.46, -1.39), (-50.5, 2.3), "center", "bottom"),
    ("Fortaleza", (-38.54, -3.77), (-32.6, -3.2), "left", "center"),
    ("Recife", (-34.90, -8.05), (-32.0, -8.0), "left", "center"),
    ("Salvador", (-38.48, -12.94), (-32.8, -13.2), "left", "center"),
    ("Brasília", (-47.90, -15.82), (-54.2, -14.8), "right", "center"),
    ("Belo Horizonte", (-43.95, -19.92), (-38.0, -19.2), "left", "center"),
    ("Rio de Janeiro", (-43.25, -22.88), (-38.0, -22.5), "left", "center"),
    ("São Paulo", (-46.63, -23.55), (-40.0, -25.4), "left", "center"),
    ("Curitiba", (-49.27, -25.44), (-55.0, -25.8), "right", "center"),
    ("Porto Alegre", (-51.18, -30.00), (-45.5, -30.5), "left", "center"),
]
for cid, (clon, clat), (tlon, tlat), ha_v, va_v in CIDADES_BR_CALLOUTS:
    ax_a.annotate(
        cid,
        xy=(clon, clat),
        xytext=(tlon, tlat),
        fontsize=5.4,
        va=va_v,
        ha=ha_v,
        family="serif",
        color="#0f172a",
        path_effects=HALO_WHITE,
        arrowprops=dict(arrowstyle="-", color="#334155", lw=0.45, shrinkA=0.5, shrinkB=1.2),
        zorder=10,
    )

ax_a.set_xlim(-74.8, -24.2)
ax_a.set_ylim(-34.5, 5.8)
ax_a.set_aspect("equal")
ax_a.set_xticks([-70, -60, -50, -40, -30])
ax_a.set_xticklabels(["70°O", "60°O", "50°O", "40°O", "30°O"])
ax_a.set_yticks([-30, -20, -10, 0])
ax_a.set_yticklabels(["30°S", "20°S", "10°S", "0°"])

add_north_arrow(ax_a, x_frac=0.06, y_frac=0.90)
add_scale_bar(ax_a, lon0=-72.5, lat0=-32.2, length_km=1000, dy_tick=0.75, fontsize=5.3)
add_panel_title(ax_a, "a", "Brasil: manchas urbanas (CCA, $n = 4.702$)")

# =========================================================================
# PAINEL (b) MAPA DO NORDESTE: MANCHAS URBANAS (CCA)
# =========================================================================
ax_b.set_facecolor("#eef6fb")
ax_b.grid(True, linestyle=":", linewidth=0.35, color="#94a3b8", alpha=0.50, zorder=0)

# Estados vizinhos do Brasil em cinza suave para dar continuidade continental
if br_geo:
    for f in br_geo["features"]:
        cod = str(f["properties"].get("codarea", ""))
        is_ne_uf = cod in ne_cods or cod.startswith("2")
        if not is_ne_uf:
            g = shape(f["geometry"])
            for p in getattr(g, "geoms", [g]):
                xs, ys = p.exterior.xy
                ax_b.fill(xs, ys, fc="#e5e7eb", ec="#cbd5e1", lw=0.30, zorder=0.5)

# Estados do Nordeste em destaque claro com divisas estaduais nítidas
if ne_geo:
    for f in ne_geo["features"]:
        g = shape(f["geometry"])
        for p in getattr(g, "geoms", [g]):
            xs, ys = p.exterior.xy
            ax_b.fill(xs, ys, fc="#fcfcfd", ec="#64748b", lw=0.48, zorder=1.0)

tab_ne = manchas[manchas["ne"]].sort_values("pop").copy()
pop_ne_vals = tab_ne["pop"].to_numpy()
sizes_ne = 1.2 + 40.0 * ((pop_ne_vals / 3.5e6) ** 0.58)

mask_small_ne = pop_ne_vals < 2.0e5
sc_ne = ax_b.scatter(
    tab_ne.loc[mask_small_ne, "lon"],
    tab_ne.loc[mask_small_ne, "lat"],
    s=sizes_ne[mask_small_ne],
    c=pop_ne_vals[mask_small_ne],
    cmap=cmap_pop,
    norm=norm_pop,
    edgecolors="none",
    alpha=0.88,
    zorder=3,
)

mask_big_ne = ~mask_small_ne
ax_b.scatter(
    tab_ne.loc[mask_big_ne, "lon"],
    tab_ne.loc[mask_big_ne, "lat"],
    s=sizes_ne[mask_big_ne],
    c=pop_ne_vals[mask_big_ne],
    cmap=cmap_pop,
    norm=norm_pop,
    edgecolors="#1e293b",
    linewidths=0.40,
    alpha=0.96,
    zorder=4,
)

# Rótulos das 9 capitais estaduais (com população em milhões) + 1ª do interior (Feira de Santana)
# Posicionados no oceano/margem livre, 100% dentro da moldura e sem colisões!
CALLOUTS_NE = [
    ("1º Recife (3,47M)", (-34.90, -8.05), (-33.7, -8.05), "left", "center", True),
    ("2º Fortaleza (3,35M)", (-38.54, -3.77), (-36.6, -2.15), "left", "center", True),
    ("3º Salvador (3,10M)", (-38.48, -12.94), (-35.6, -13.05), "left", "center", True),
    ("4º São Luís (1,30M)", (-44.25, -2.54), (-44.8, -0.45), "center", "bottom", True),
    ("5º Natal (1,23M)", (-35.22, -5.80), (-33.8, -5.15), "left", "center", True),
    ("6º Maceió (1,07M)", (-35.73, -9.66), (-33.7, -9.65), "left", "center", True),
    ("7º João Pessoa (1,05M)", (-34.86, -7.12), (-34.1, -6.65), "left", "center", True),
    ("8º Teresina (0,98M)", (-42.79, -5.09), (-40.6, -1.25), "center", "bottom", True),
    ("9º Aracaju (0,83M)", (-37.07, -10.91), (-34.4, -11.25), "left", "center", True),
    ("10º Feira de Santana (0,53M)", (-38.96, -12.26), (-37.1, -14.70), "left", "center", False),
]
for nome, (clon, clat), (tx, ty), ha_val, va_val, is_cap in CALLOUTS_NE:
    ax_b.annotate(
        nome,
        xy=(clon, clat),
        xytext=(tx, ty),
        fontsize=5.15 if is_cap else 4.85,
        fontweight="bold" if is_cap else "normal",
        family="serif",
        color="#0f172a" if is_cap else "#334155",
        ha=ha_val,
        va=va_val,
        path_effects=HALO_WHITE,
        arrowprops=dict(
            arrowstyle="-",
            color="#334155" if is_cap else "#64748b",
            lw=0.45 if is_cap else 0.35,
            shrinkA=0.5,
            shrinkB=1.2,
        ),
        zorder=10,
    )

ax_b.set_xlim(-49.2, -23.8)
ax_b.set_ylim(-18.8, 1.05)
ax_b.set_aspect("equal")
ax_b.set_xticks([-46, -42, -38, -34, -30, -26])
ax_b.set_xticklabels(["46°O", "42°O", "38°O", "34°O", "30°O", "26°O"])
ax_b.set_yticks([-16, -12, -8, -4, 0])
ax_b.set_yticklabels(["16°S", "12°S", "8°S", "4°S", "0°"])

add_north_arrow(ax_b, x_frac=0.06, y_frac=0.10)
add_scale_bar(ax_b, lon0=-33.0, lat0=-17.3, length_km=400, dy_tick=0.38, fontsize=5.3)
add_panel_title(ax_b, "b", "Nordeste: manchas urbanas (CCA, $n = 1.729$)")

# Forçar renderização do canvas para obter as dimensões exatas de ax_b após aspect='equal'
fig.canvas.draw()
pos_b = ax_b.get_position()

# BARRA DE CORES ALINHADA À ALTURA DO MAPA (b)
cax = fig.add_axes([pos_b.x1 + 0.012, pos_b.y0, 0.012, pos_b.height])
cb = fig.colorbar(sc_br, cax=cax, ticks=[1e4, 1e5, 1e6, 1e7])
cb.ax.set_yticklabels([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$"], fontsize=6.6)
cb.set_label(r"População da mancha $S$ (hab.)", fontsize=7.4, labelpad=4)

# =========================================================================
# PAINEL (c) BRASIL: POSTO E TAMANHO (CURVA DE ZIPF)
# =========================================================================
x_br = np.sort(manchas["pop"].to_numpy())[::-1]
y_br = np.sort(mun["pop"].to_numpy())[::-1]
r_br = np.arange(1, x_br.size + 1)
r_mun_br = np.arange(1, y_br.size + 1)

h_manchas = ax_c.loglog(
    r_br, x_br, "o",
    ms=2.3, mfc="none", mec="#1d4ed8", mew=0.6, alpha=0.75,
    label="Manchas urbanas (CCA)", zorder=3
)[0]
h_mun = ax_c.loglog(
    r_mun_br, y_br, "s",
    ms=1.8, mfc="none", mec="#64748b", mew=0.5, alpha=0.55,
    label="Municípios (Censo 2022)", zorder=2
)[0]

rr_br = np.logspace(0, 3.7, 50)
h_zipf = ax_c.loglog(
    rr_br, x_br[0] / rr_br, "--",
    c="#0f172a", lw=0.95,
    label=r"Lei de Zipf ($S_1/r$, $\alpha=1$)", zorder=4
)[0]

csn_br = resumo["br"]["csn_manchas"]
nt_br = int(csn_br["n"])
rq_br = np.logspace(0, np.log10(nt_br), 50)
alpha_br = csn_br["alpha"]
erro_br = csn_br["erro_alpha"]
p_val_br = csn_br["p_valor"]
a100_br = resumo["br"]["alpha_hill"]["100"]

# R² OLS no topo 100 calculado diretamente dos dados (como em pipeline/cca_calibrator.py)
ols_br_100 = stats.linregress(np.log10(r_br[:100]), np.log10(x_br[:100]))
r2_br_100 = float(ols_br_100.rvalue ** 2)

h_csn = ax_c.loglog(
    rq_br, csn_br["xmin"] * (rq_br / nt_br) ** (-alpha_br), "-",
    c="#0284c7", lw=1.4,
    label=r"Ajuste CSN na cauda ($x \geq x_{\min}$)", zorder=5
)[0]

# Demarcação discreta do início da cauda CSN (r = nt_br)
ax_c.axvline(nt_br, color="#0284c7", ls=":", lw=0.65, alpha=0.7, zorder=1)
ax_c.text(
    nt_br * 0.88, 1.3e7, r"$x_{\min}$",
    fontsize=5.8, color="#0284c7", ha="right", va="center", family="serif"
)

# Destaque da maior metrópole nacional
ax_c.annotate(
    "São Paulo (20,0M)", xy=(1, x_br[0]), xytext=(2.3, 1.85e7),
    fontsize=5.5, color="#0f172a", family="serif",
    arrowprops=dict(arrowstyle="-", color="#475569", lw=0.4),
)

txt_br = (
    rf"$\alpha_{{\mathrm{{CSN}}}} = {fmt_pt(alpha_br)} \pm {fmt_pt(erro_br)}\ (p = {fmt_pt(p_val_br)})$" "\n"
    rf"$\hat{{\alpha}}_{{100}} = {fmt_pt(a100_br)}\ (R^2_{{\mathrm{{OLS}}}} = {fmt_pt(r2_br_100)})$" "\n"
    r"Municípios: $\alpha = 0{,}91 \pm 0{,}02\ (p = 0{,}045)$"
)
ax_c.text(
    0.035, 0.055, txt_br,
    transform=ax_c.transAxes,
    fontsize=5.9, va="bottom", ha="left", linespacing=1.25,
    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cbd5e1", lw=0.55, alpha=0.94),
    zorder=10,
)

ax_c.set_xlim(0.75, 7e3)
ax_c.set_ylim(3.5e3, 3.2e7)
ax_c.set_xlabel(r"Posto $r$ (ordem decrescente)", fontsize=8.0)
ax_c.set_ylabel(r"População $S(r)$ (hab.)", fontsize=8.0)

add_panel_title(ax_c, "c", "Brasil: curva de posto e tamanho")

# =========================================================================
# PAINEL (d) NORDESTE: POSTO E TAMANHO (CURVA DE ZIPF)
# =========================================================================
x_ne = np.sort(manchas.loc[manchas["ne"], "pop"].to_numpy())[::-1]
y_ne = np.sort(mun.loc[mun["ne"], "pop"].to_numpy())[::-1]
r_ne = np.arange(1, x_ne.size + 1)
r_mun_ne = np.arange(1, y_ne.size + 1)

ax_d.loglog(r_ne, x_ne, "o", ms=2.3, mfc="none", mec="#1d4ed8", mew=0.6, alpha=0.75, zorder=3)
ax_d.loglog(r_mun_ne, y_ne, "s", ms=1.8, mfc="none", mec="#64748b", mew=0.5, alpha=0.55, zorder=2)

rr_ne = np.logspace(0, 3.35, 40)
ax_d.loglog(rr_ne, x_ne[0] / rr_ne, "--", c="#0f172a", lw=0.95, zorder=4)

csn_ne = resumo.get("ne_par_br", {}).get("csn_manchas")
if csn_ne:
    nt_ne = int(csn_ne["n"])
    rq_ne = np.logspace(0, np.log10(nt_ne), 50)
    alpha_ne = csn_ne["alpha"]
    erro_ne = csn_ne["erro_alpha"]
    p_val_ne = csn_ne["p_valor"]
    ax_d.loglog(
        rq_ne, csn_ne["xmin"] * (rq_ne / nt_ne) ** (-alpha_ne), "-",
        c="#0284c7", lw=1.4, zorder=5
    )
    ax_d.axvline(nt_ne, color="#0284c7", ls=":", lw=0.65, alpha=0.7, zorder=1)
    ax_d.text(
        nt_ne * 0.88, 3.2e6, r"$x_{\min}$",
        fontsize=5.8, color="#0284c7", ha="right", va="center", family="serif"
    )

# Destacar visualmente as 9 capitais estaduais no topo (r = 1..9)
ax_d.loglog(
    r_ne[:9], x_ne[:9], "o",
    ms=3.2, mfc="#f59e0b", mec="#b45309", mew=0.6, alpha=0.95, zorder=6
)
ax_d.annotate(
    "9 capitais estaduais\n(patamar policêntrico)",
    xy=(3.0, x_ne[2]), xytext=(8.5, 2.2e6),
    fontsize=5.4, color="#92400e", fontweight="bold", family="serif",
    arrowprops=dict(arrowstyle="->", color="#b45309", lw=0.55),
    zorder=8,
)

a_hill_ne = resumo["ne_par_br"]["alpha_hill"]
a50_ne = a_hill_ne["50"]
a100_ne = a_hill_ne["100"]
ols_ne_100 = stats.linregress(np.log10(r_ne[:100]), np.log10(x_ne[:100]))
r2_ne_100 = float(ols_ne_100.rvalue ** 2)

txt_ne = (
    rf"$\alpha_{{\mathrm{{cauda}}}} = {fmt_pt(alpha_ne)} \pm {fmt_pt(erro_ne)}\ (p = {fmt_pt(p_val_ne)})$" "\n"
    rf"$\hat{{\alpha}}_{{50}} = {fmt_pt(a50_ne)}$ (degrau das capitais)" "\n"
    rf"$\hat{{\alpha}}_{{100}} = {fmt_pt(a100_ne)}\ (R^2_{{\mathrm{{OLS}}}} = {fmt_pt(r2_ne_100)})$"
)
ax_d.text(
    0.035, 0.055, txt_ne,
    transform=ax_d.transAxes,
    fontsize=5.9, va="bottom", ha="left", linespacing=1.25,
    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cbd5e1", lw=0.55, alpha=0.94),
    zorder=10,
)

ax_d.set_xlim(0.75, 3e3)
ax_d.set_ylim(3.5e3, 6.5e6)
ax_d.set_xlabel(r"Posto $r$ (ordem decrescente)", fontsize=8.0)
ax_d.set_ylabel(r"População $S(r)$ (hab.)", fontsize=8.0)

add_panel_title(ax_d, "d", "Nordeste: curva de posto e tamanho")

# =========================================================================
# LEGENDA ÚNICA COMPARTILHADA (NA BASE INFERIOR)
# =========================================================================
h_cap = Line2D([0], [0], marker="o", color="none", label="9 capitais do Nordeste",
               markerfacecolor="#f59e0b", markeredgecolor="#b45309", markersize=4.5)

fig.legend(
    [h_manchas, h_cap, h_mun, h_zipf, h_csn],
    [
        "Manchas urbanas (CCA)",
        "9 capitais (NE)",
        "Municípios (Censo 2022)",
        r"Lei de Zipf ($S_1/r$, $\alpha=1$)",
        r"Ajuste CSN (cauda, $x \geq x_{\min}$)",
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.008),
    ncol=5,
    frameon=False,
    fontsize=6.5,
    markerscale=1.35,
    handlelength=1.4,
    handletextpad=0.35,
    columnspacing=1.0,
)

# Exportação simultânea: PDF vetorial, PNG 300 DPI, PNG 600 DPI e SVG vetorial
pdf_out = FIGS_DIR / "zipf.pdf"
png_out = FIGS_DIR / "zipf.png"
png_hires_out = FIGS_DIR / "zipf_600dpi.png"
svg_out = FIGS_DIR / "zipf.svg"

fig.savefig(pdf_out)
fig.savefig(png_out, dpi=300)
fig.savefig(png_hires_out, dpi=600)
fig.savefig(svg_out)
plt.close(fig)

print(f"Sucesso: arquivos exportados em {FIGS_DIR}:")
print(f" - {pdf_out.name} (PDF vetorial)")
print(f" - {png_out.name} (PNG 300 DPI)")
print(f" - {png_hires_out.name} (PNG 600 DPI)")
print(f" - {svg_out.name} (SVG vetorial)")
