import matplotlib.patheffects as pe
"""
Gera GIFs de alta qualidade demonstrando o funcionamento do City Clustering Algorithm (CCA)
e a formação dos aglomerados populacionais (manchas urbanas) do Nordeste brasileiro,
como representado na Figura (a) do artigo.

GIFs gerados em artigo/figs/gifs/:
1. 01_cca_nordeste_evolucao_densidade.gif
   Demonstração regional da emergência das manchas urbanas no Nordeste conforme o limiar
   D* decresce de 4.800 até o par calibrado de 1.450 hab/km² (l = 1,8 km fixo),
   acompanhado de ranking dinâmico das capitais e cartões de estatísticas.

2. 02_cca_nordeste_coalescencia_distancia.gif
   Demonstração regional do efeito da distância de ligação l (200 m a 3.500 m) com
   D* = 1.450 hab/km² fixado, mostrando a transição de células fragmentadas para a
   conurbação calibrada (l = 1,8 km) e hiperaglomeração.

3. 03_cca_recife_conurbacao_metropolitana.gif
   Microescala espacial da conurbação célula a célula no raster GPW v4 da maior
   mancha urbana do Nordeste (Recife - 3,47M hab, 404 km²), unindo Recife, Olinda,
   Jaboatão dos Guararapes, Paulista e Camaragibe.

4. 04_cca_fortaleza_conurbacao_metropolitana.gif
   Microescala espacial da conurbação da 2ª maior mancha urbana do Nordeste
   (Fortaleza - 3,35M hab, 370 km²), unindo Fortaleza, Caucaia e Maracanaú.

5. 05_cca_salvador_conurbacao_metropolitana.gif
   Microescala espacial da conurbação da 3ª maior mancha urbana do Nordeste
   (Salvador - 3,10M hab, 258 km²), unindo Salvador e Lauro de Freitas.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.patches import Rectangle
from shapely.geometry import shape
from numba import njit
import imageio.v2 as imageio
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
if AQUI not in sys.path:
    sys.path.insert(0, AQUI)

import analise_brasil as ab

GIFS_DIR = os.path.join(AQUI, "figs", "gifs")
os.makedirs(GIFS_DIR, exist_ok=True)

print("Carregando raster GPW v4 e malhas IBGE...")
pop_px, no_ne, lon, lat, dx, dy, area_px_km2 = ab.carrega()
nrows, ncols = pop_px.shape
dens_grid = pop_px / area_px_km2
area_grid_km2 = np.broadcast_to(area_px_km2, pop_px.shape)
dens = dens_grid.ravel()
popv = pop_px.ravel()
popne = np.where(no_ne, pop_px, 0.0).ravel()
ordem = np.flatnonzero(dens > 10)
ordem = ordem[np.argsort(-dens[ordem], kind="stable")].astype(np.int64)
n_total = dens.size

ne_geo = json.load(open(os.path.join(ab.DADOS, "ibge", "nordeste_ufs.geojson"), encoding="utf-8"))
geoms_ne = []
for f in ne_geo["features"]:
    g = shape(f["geometry"])
    geoms_ne.extend(getattr(g, "geoms", [g]))

ROTULOS_MAPA = {
    "Fortaleza": (7, 4), "Recife": (8, 0), "Salvador": (8, -3), "São Luís": (-9, 8),
    "Teresina": (-38, -6), "Natal": (8, 2), "Maceió": (8, -2), "João Pessoa": (8, 2),
    "Aracaju": (8, -2), "Feira de Santana": (-55, -8), "Campina Grande": (-58, 4),
    "Juazeiro do Norte": (-65, -8), "Petrolina": (-35, -12), "Mossoró": (7, 5)
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Latin Modern Roman"],
    "mathtext.fontset": "cm",
})

# ------------------------------------------------------------------ Numba Union-Find com Centroides
@njit
def ativa_ate_centroids(D, pos, ordem, dens, popv, popne, pai, tam, soma, soma_ne, soma_lat, soma_lon, ativo, di, dj, valido, nrows, ncols, lat, lon):
    """Ativa pixels com densidade > D e propaga centroides ponderados pela população."""
    n = ordem.size
    while pos < n:
        p = ordem[pos]
        if dens[p] <= D:
            break
        ativo[p] = True
        pai[p] = p
        tam[p] = 1
        soma[p] = popv[p]
        soma_ne[p] = popne[p]
        r = p // ncols
        c = p - r * ncols
        soma_lat[p] = lat[r] * popv[p]
        soma_lon[p] = lon[c] * popv[p]
        for k in range(di.size):
            if not valido[r, k]:
                continue
            rr = r + di[k]
            cc = c + dj[k]
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            q = rr * ncols + cc
            if ativo[q]:
                a = ab._raiz(pai, p)
                b = ab._raiz(pai, q)
                if a != b:
                    if tam[a] < tam[b]:
                        a, b = b, a
                    pai[b] = a
                    tam[a] += tam[b]
                    soma[a] += soma[b]
                    soma_ne[a] += soma_ne[b]
                    soma_lat[a] += soma_lat[b]
                    soma_lon[a] += soma_lon[b]
        pos += 1
    return pos

@njit
def extrai_manchas_ne(ordem, pos, pai, soma, soma_ne, soma_lat, soma_lon):
    """Devolve arrays (pop, lat, lon) das manchas com maioria da população no Nordeste."""
    out_pop = np.empty(pos)
    out_lat = np.empty(pos)
    out_lon = np.empty(pos)
    k = 0
    for i in range(pos):
        p = ordem[i]
        if pai[p] == p and soma_ne[p] > 0.5 * soma[p]:
            out_pop[k] = soma[p]
            out_lat[k] = soma_lat[p] / soma[p]
            out_lon[k] = soma_lon[p] / soma[p]
            k += 1
    return out_pop[:k], out_lat[:k], out_lon[:k]

def associa_nomes(lo_arr, la_arr, pop_arr):
    """Associa cada cidade de referência à mancha mais próxima e compatível."""
    nomes = [None] * len(pop_arr)
    if len(pop_arr) == 0:
        return nomes
    for nome, (r_lo, r_la) in ab.REFS.items():
        d2 = (lo_arr - r_lo) ** 2 + (la_arr - r_la) ** 2
        cand = np.where(d2 < 0.28 ** 2)[0]
        if len(cand) > 0:
            melhor = cand[np.argmax(pop_arr[cand])]
            if nomes[melhor] is None:
                nomes[melhor] = nome
    return nomes

def fig_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()

# ==================================================================
# 1. GIF 1: Evolução Regional no Nordeste por Limiar de Densidade D*
# ==================================================================
def gerar_gif_01_varredura_D():
    print("\n[1/5] Gerando GIF 1: Evolução Regional por Limiar de Densidade D*...")
    t0 = time.time()
    di, dj, valido = ab.vizinhanca(1.8, lat, dx, dy)
    
    # 32 passos cobrindo de 4.800 hab/km² até 1.450 hab/km² (calibrado) e até 900 hab/km²
    d_alta = np.linspace(4800, 2600, 11)
    d_media = np.linspace(2400, 1550, 11)
    d_calib = [1450.0]
    d_baixa = np.linspace(1350, 900, 6)
    d_steps = np.concatenate([d_alta, d_media, d_calib, d_baixa])
    
    pai = np.zeros(n_total, dtype=np.int64)
    tam = np.zeros(n_total, dtype=np.int64)
    soma = np.zeros(n_total)
    soma_ne = np.zeros(n_total)
    soma_lat = np.zeros(n_total)
    soma_lon = np.zeros(n_total)
    ativo = np.zeros(n_total, dtype=np.bool_)
    
    frames_rgb = []
    durations = []
    pos = 0
    norm_c = LogNorm(1e4, 4e6)
    cmap_c = plt.cm.YlOrRd
    
    fig = plt.figure(figsize=(9.8, 5.2), dpi=120, facecolor="#ffffff")
    
    for idx_step, D_val in enumerate(d_steps):
        pos = ativa_ate_centroids(float(D_val), pos, ordem, dens, popv, popne, pai, tam,
                                  soma, soma_ne, soma_lat, soma_lon, ativo, di, dj, valido,
                                  nrows, ncols, lat, lon)
        pop_ne_arr, lat_ne_arr, lon_ne_arr = extrai_manchas_ne(ordem, pos, pai, soma, soma_ne, soma_lat, soma_lon)
        
        ord_pop = np.argsort(-pop_ne_arr)
        pop_ne_arr = pop_ne_arr[ord_pop]
        lat_ne_arr = lat_ne_arr[ord_pop]
        lon_ne_arr = lon_ne_arr[ord_pop]
        nomes_ne = associa_nomes(lon_ne_arr, lat_ne_arr, pop_ne_arr)
        
        fig.clf()
        ax_map = fig.add_axes([0.04, 0.08, 0.53, 0.84])
        ax_hud = fig.add_axes([0.61, 0.61, 0.35, 0.31])
        ax_bar = fig.add_axes([0.61, 0.10, 0.35, 0.44])
        
        # Mapa base
        for p in geoms_ne:
            xs, ys = p.exterior.xy
            ax_map.fill(xs, ys, fc="#f4f4f4", ec="#aaaaaa", lw=0.45, zorder=0)
            
        if len(pop_ne_arr) > 0:
            sizes = 0.5 + 46.0 * (pop_ne_arr / 3.5e6)
            sc = ax_map.scatter(lon_ne_arr, lat_ne_arr, s=sizes, c=pop_ne_arr,
                                cmap=cmap_c, norm=norm_c, ec="#7a2e0e", lw=0.25, zorder=3)
            
            for i_nome, nome in enumerate(nomes_ne):
                if nome and nome in ROTULOS_MAPA and pop_ne_arr[i_nome] >= 60000:
                    dx_pts, dy_pts = ROTULOS_MAPA[nome]
                    ax_map.annotate(nome, (lon_ne_arr[i_nome], lat_ne_arr[i_nome]),
                                    (dx_pts, dy_pts), textcoords="offset points",
                                    fontsize=6.2, fontweight="bold", color="#1f2937",
                                    va="center", zorder=5)
        
        ax_map.set_xlim(-48.9, -31.0)
        ax_map.set_ylim(-18.5, -1.0)
        ax_map.set_aspect("equal")
        ax_map.set_xticks([-46, -40, -34])
        ax_map.set_xticklabels(["46°O", "40°O", "34°O"], fontsize=7.5)
        ax_map.set_yticks([-4, -10, -16])
        ax_map.set_yticklabels(["4°S", "10°S", "16°S"], fontsize=7.5)
        
        is_calibrated = abs(D_val - 1450.0) < 1.0
        badge_cor = "#047857" if is_calibrated else "#374151"
        d_str = f"{int(D_val):,}".replace(",", ".")
        calib_sub = " [Configuração da Imagem (a)]" if is_calibrated else ""
        ax_map.set_title(f"Aglomerações Urbanas no Nordeste (CCA){calib_sub}\n"
                         f"Limiar de densidade: $D^* = {d_str}$ hab/km²  |  $\\ell = 1,8$ km",
                         fontsize=8.3, pad=6, color=badge_cor, fontweight="bold" if is_calibrated else "normal")
        
        # HUD
        ax_hud.axis("off")
        n_manchas = len(pop_ne_arr)
        pop_total_ne = pop_ne_arr.sum()
        pct_ne = (pop_total_ne / 59450718.0) * 100.0
        
        box_bg = "#ecfdf5" if is_calibrated else "#f8fafc"
        box_edge = "#059669" if is_calibrated else "#cbd5e1"
        ax_hud.add_patch(Rectangle((0, 0), 1, 1, transform=ax_hud.transAxes,
                                   fc=box_bg, ec=box_edge, lw=1.2, zorder=0))
        
        status_txt = "* PAR CALIBRADO DA IMAGEM (a) *" if is_calibrated else "VARREDURA DE DENSIDADE CCA"
        ax_hud.text(0.5, 0.88, status_txt, transform=ax_hud.transAxes,
                    ha="center", va="center", fontsize=8.0, fontweight="bold",
                    color="#047857" if is_calibrated else "#475569")
        
        ax_hud.text(0.06, 0.65, f"Limiar $D^*$:", transform=ax_hud.transAxes, fontsize=7.8, color="#334155")
        ax_hud.text(0.94, 0.65, f"{int(D_val):,} hab/km²".replace(",", "."), transform=ax_hud.transAxes,
                    fontsize=8.3, fontweight="bold", ha="right", color="#0f172a")
        
        ax_hud.text(0.06, 0.44, f"Total de manchas no NE:", transform=ax_hud.transAxes, fontsize=7.8, color="#334155")
        ax_hud.text(0.94, 0.44, f"{n_manchas:,} manchas".replace(",", "."), transform=ax_hud.transAxes,
                    fontsize=8.3, fontweight="bold", ha="right", color="#0f172a")
        
        ax_hud.text(0.06, 0.23, f"Pop. aglomerada no NE:", transform=ax_hud.transAxes, fontsize=7.8, color="#334155")
        ax_hud.text(0.94, 0.23, f"{pop_total_ne/1e6:.2f} M hab ({pct_ne:.1f}%)".replace(".", ","),
                    transform=ax_hud.transAxes, fontsize=8.3, fontweight="bold", ha="right", color="#0f172a")
        
        # Ranking
        k_top = 6
        if n_manchas > 0:
            top_k_pop = pop_ne_arr[:min(k_top, n_manchas)][::-1] / 1e6
            top_k_nomes = [nomes_ne[i] if nomes_ne[i] else f"Mancha #{i+1}" for i in range(min(k_top, n_manchas))][::-1]
            cores_bar = [cmap_c(norm_c(p * 1e6)) for p in top_k_pop]
            
            y_pos = np.arange(len(top_k_pop))
            bars = ax_bar.barh(y_pos, top_k_pop, color=cores_bar, edgecolor="#7a2e0e", lw=0.4, height=0.62)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(top_k_nomes, fontsize=7.0, fontweight="medium")
            ax_bar.set_xlim(0, 4.0)
            ax_bar.set_xlabel("População (milhões de habitantes)", fontsize=7.0)
            ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}M".replace(".", ",")))
            ax_bar.tick_params(axis="x", labelsize=6.8)
            ax_bar.set_title("Maiores Manchas Urbanas", fontsize=7.6, fontweight="bold", color="#1e293b", pad=4)
            ax_bar.grid(axis="x", ls="--", lw=0.4, alpha=0.6)
            
            for bar, val in zip(bars, top_k_pop):
                if val > 0.05:
                    ax_bar.text(val + 0.08, bar.get_y() + bar.get_height()/2.0,
                                f"{val:.2f}M".replace(".", ","),
                                va="center", ha="left", fontsize=6.3, color="#1e293b", fontweight="bold")
        else:
            ax_bar.axis("off")
            
        rgb = fig_to_rgb(fig)
        
        if is_calibrated:
            for _ in range(16):  # ~1.6s de pausa no ponto calibrado
                frames_rgb.append(rgb)
                durations.append(0.10)
        else:
            frames_rgb.append(rgb)
            durations.append(0.11)
            
    plt.close(fig)
    out_path = os.path.join(GIFS_DIR, "01_cca_nordeste_evolucao_densidade.gif")
    imageio.mimsave(out_path, frames_rgb, duration=durations, loop=0)
    print(f"Salvo: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB, {len(frames_rgb)} quadros em {time.time()-t0:.1f} s)")
    return out_path

# ==================================================================
# 2. GIF 2: Efeito da Distância de Ligação Espacial l (Coalescência)
# ==================================================================
def gerar_gif_02_varredura_l():
    print("\n[2/5] Gerando GIF 2: Efeito da Distância de Ligação Espacial l...")
    t0 = time.time()
    D_fixo = 1450.0
    
    # Lista de l: de 200m até 3.500m
    l_steps = [200, 400, 600, 800, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000, 2200, 2500, 2800, 3200, 3500]
    
    norm_c = LogNorm(1e4, 4e6)
    cmap_c = plt.cm.YlOrRd
    frames_rgb = []
    durations = []
    
    V = ab.Varredura(pop_px, no_ne, lat, dx, dy, area_px_km2)
    fig = plt.figure(figsize=(9.8, 5.2), dpi=120, facecolor="#ffffff")
    
    for l_val in l_steps:
        di, dj, valido = ab.vizinhanca(l_val / 1000.0, lat, dx, dy)
        pai = np.zeros(n_total, dtype=np.int64)
        tam = np.zeros(n_total, dtype=np.int64)
        soma = np.zeros(n_total)
        soma_ne = np.zeros(n_total)
        soma_lat = np.zeros(n_total)
        soma_lon = np.zeros(n_total)
        ativo = np.zeros(n_total, dtype=np.bool_)
        
        pos = ativa_ate_centroids(D_fixo, 0, ordem, dens, popv, popne, pai, tam,
                                  soma, soma_ne, soma_lat, soma_lon, ativo, di, dj, valido,
                                  nrows, ncols, lat, lon)
        pop_ne_arr, lat_ne_arr, lon_ne_arr = extrai_manchas_ne(ordem, pos, pai, soma, soma_ne, soma_lat, soma_lon)
        
        ord_pop = np.argsort(-pop_ne_arr)
        pop_ne_arr = pop_ne_arr[ord_pop]
        lat_ne_arr = lat_ne_arr[ord_pop]
        lon_ne_arr = lon_ne_arr[ord_pop]
        nomes_ne = associa_nomes(lon_ne_arr, lat_ne_arr, pop_ne_arr)
        
        fig.clf()
        ax_map = fig.add_axes([0.04, 0.08, 0.53, 0.84])
        ax_hud = fig.add_axes([0.61, 0.61, 0.35, 0.31])
        ax_bar = fig.add_axes([0.61, 0.10, 0.35, 0.44])
        
        for p in geoms_ne:
            xs, ys = p.exterior.xy
            ax_map.fill(xs, ys, fc="#f4f4f4", ec="#aaaaaa", lw=0.45, zorder=0)
            
        if len(pop_ne_arr) > 0:
            sizes = 0.5 + 46.0 * (pop_ne_arr / 3.5e6)
            sc = ax_map.scatter(lon_ne_arr, lat_ne_arr, s=sizes, c=pop_ne_arr,
                                cmap=cmap_c, norm=norm_c, ec="#7a2e0e", lw=0.25, zorder=3)
            
            for i_nome, nome in enumerate(nomes_ne):
                if nome and nome in ROTULOS_MAPA and pop_ne_arr[i_nome] >= 60000:
                    dx_pts, dy_pts = ROTULOS_MAPA[nome]
                    ax_map.annotate(nome, (lon_ne_arr[i_nome], lat_ne_arr[i_nome]),
                                    (dx_pts, dy_pts), textcoords="offset points",
                                    fontsize=6.2, fontweight="bold", color="#1f2937",
                                    va="center", zorder=5)
                    
        ax_map.set_xlim(-48.9, -31.0)
        ax_map.set_ylim(-18.5, -1.0)
        ax_map.set_aspect("equal")
        ax_map.set_xticks([-46, -40, -34])
        ax_map.set_xticklabels(["46°O", "40°O", "34°O"], fontsize=7.5)
        ax_map.set_yticks([-4, -10, -16])
        ax_map.set_yticklabels(["4°S", "10°S", "16°S"], fontsize=7.5)
        
        is_calibrated = (l_val == 1800)
        badge_cor = "#047857" if is_calibrated else "#374151"
        l_str = f"{l_val/1000.0:.1f}".replace(".", ",")
        calib_sub = " [Configuração da Imagem (a)]" if is_calibrated else ""
        ax_map.set_title(f"Coalescência Espacial por Distância $\\ell$ (CCA){calib_sub}\n"
                         f"Distância de ligação: $\\ell = {l_str}$ km  |  $D^* = 1.450$ hab/km²",
                         fontsize=8.3, pad=6, color=badge_cor, fontweight="bold" if is_calibrated else "normal")
        
        # HUD
        ax_hud.axis("off")
        n_manchas = len(pop_ne_arr)
        pop_total_ne = pop_ne_arr.sum()
        pct_ne = (pop_total_ne / 59450718.0) * 100.0
        
        box_bg = "#ecfdf5" if is_calibrated else "#f8fafc"
        box_edge = "#059669" if is_calibrated else "#cbd5e1"
        ax_hud.add_patch(Rectangle((0, 0), 1, 1, transform=ax_hud.transAxes,
                                   fc=box_bg, ec=box_edge, lw=1.2, zorder=0))
        
        status_txt = "* PAR CALIBRADO DA IMAGEM (a) *" if is_calibrated else "VARREDURA DA DISTÂNCIA DE LIGAÇÃO"
        ax_hud.text(0.5, 0.88, status_txt, transform=ax_hud.transAxes,
                    ha="center", va="center", fontsize=8.0, fontweight="bold",
                    color="#047857" if is_calibrated else "#475569")
        
        ax_hud.text(0.06, 0.65, f"Distância de Ligação $\\ell$:", transform=ax_hud.transAxes, fontsize=7.8, color="#334155")
        ax_hud.text(0.94, 0.65, f"{l_val:,} m ({l_val/1000.0:.1f} km)".replace(",", "."), transform=ax_hud.transAxes,
                    fontsize=8.3, fontweight="bold", ha="right", color="#0f172a")
        
        ax_hud.text(0.06, 0.44, f"Total de manchas no NE:", transform=ax_hud.transAxes, fontsize=7.8, color="#334155")
        ax_hud.text(0.94, 0.44, f"{n_manchas:,} manchas".replace(",", "."), transform=ax_hud.transAxes,
                    fontsize=8.3, fontweight="bold", ha="right", color="#0f172a")
        
        ax_hud.text(0.06, 0.23, f"Maior mancha urbana:", transform=ax_hud.transAxes, fontsize=7.8, color="#334155")
        max_pop = pop_ne_arr[0] if len(pop_ne_arr) > 0 else 0
        nome_top = nomes_ne[0] if len(nomes_ne) > 0 and nomes_ne[0] else "Recife"
        ax_hud.text(0.94, 0.23, f"{nome_top} ({max_pop/1e6:.2f}M)".replace(".", ","),
                    transform=ax_hud.transAxes, fontsize=8.3, fontweight="bold", ha="right", color="#0f172a")
        
        # Ranking
        k_top = 6
        if n_manchas > 0:
            top_k_pop = pop_ne_arr[:min(k_top, n_manchas)][::-1] / 1e6
            top_k_nomes = [nomes_ne[i] if nomes_ne[i] else f"Mancha #{i+1}" for i in range(min(k_top, n_manchas))][::-1]
            cores_bar = [cmap_c(norm_c(p * 1e6)) for p in top_k_pop]
            
            y_pos = np.arange(len(top_k_pop))
            bars = ax_bar.barh(y_pos, top_k_pop, color=cores_bar, edgecolor="#7a2e0e", lw=0.4, height=0.62)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(top_k_nomes, fontsize=7.0, fontweight="medium")
            ax_bar.set_xlim(0, 4.0)
            ax_bar.set_xlabel("População (milhões de habitantes)", fontsize=7.0)
            ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}M".replace(".", ",")))
            ax_bar.tick_params(axis="x", labelsize=6.8)
            ax_bar.set_title("Maiores Manchas Urbanas", fontsize=7.6, fontweight="bold", color="#1e293b", pad=4)
            ax_bar.grid(axis="x", ls="--", lw=0.4, alpha=0.6)
            
            for bar, val in zip(bars, top_k_pop):
                if val > 0.05:
                    ax_bar.text(val + 0.08, bar.get_y() + bar.get_height()/2.0,
                                f"{val:.2f}M".replace(".", ","),
                                va="center", ha="left", fontsize=6.3, color="#1e293b", fontweight="bold")
        else:
            ax_bar.axis("off")
            
        rgb = fig_to_rgb(fig)
        if is_calibrated:
            for _ in range(16):
                frames_rgb.append(rgb)
                durations.append(0.10)
        else:
            frames_rgb.append(rgb)
            durations.append(0.12)
            
    plt.close(fig)
    out_path = os.path.join(GIFS_DIR, "02_cca_nordeste_coalescencia_distancia.gif")
    imageio.mimsave(out_path, frames_rgb, duration=durations, loop=0)
    print(f"Salvo: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB, {len(frames_rgb)} quadros em {time.time()-t0:.1f} s)")
    return out_path

# ==================================================================
# 3. GIF 3: Microescala Célula a Célula - Conurbação do Recife
# ==================================================================
def gerar_gif_03_recife():
    print("\n[3/5] Gerando GIF 3: Conurbação Metropolitana Célula a Célula do Recife...")
    t0 = time.time()
    
    # Bounding box do Recife Metropolitano
    # lat: [-8.35, -7.75], lon: [-35.15, -34.75]
    r_min = int(np.abs(lat - (-7.72)).argmin())
    r_max = int(np.abs(lat - (-8.33)).argmin())
    c_min = int(np.abs(lon - (-35.14)).argmin())
    c_max = int(np.abs(lon - (-34.78)).argmin())
    
    sub_dens = dens_grid[r_min:r_max, c_min:c_max]
    sub_pop = pop_px[r_min:r_max, c_min:c_max]
    sub_area = area_grid_km2[r_min:r_max, c_min:c_max]
    sub_lat = lat[r_min:r_max]
    sub_lon = lon[c_min:c_max]
    
    extent = [sub_lon[0], sub_lon[-1], sub_lat[-1], sub_lat[0]]
    
    # Locais chave na RMR para rotular
    PONTOS_RMR = {
        "Recife Antigo": (-34.871, -8.063),
        "Boa Viagem": (-34.900, -8.125),
        "Olinda": (-34.855, -7.995),
        "Paulista": (-34.873, -7.935),
        "Jaboatão": (-35.015, -8.110),
        "Camaragibe": (-34.980, -8.020),
    }
    
    V = ab.Varredura(pop_px, no_ne, lat, dx, dy, area_px_km2)
    # Identificar a mancha de Recife na calibração
    lab_calib = V.rotulos(1450, 1800)
    id_recife = lab_calib[int(np.abs(lat - (-8.05)).argmin()), int(np.abs(lon - (-34.88)).argmin())]
    
    # Valores de densidade para a animação do processo de aglomeração
    # Começa mostrando a densidade contínua, depois desce D* de 18.000 até 1.450
    d_seq = [
        ("dens_bruta", None),
        ("filtro_inicio", 18000),
        ("nucleacao_1", 14000),
        ("nucleacao_2", 10000),
        ("expansao_1", 7500),
        ("expansao_2", 5500),
        ("conurbacao_1", 4000),
        ("conurbacao_2", 3000),
        ("conurbacao_3", 2200),
        ("conurbacao_4", 1700),
        ("consolidado", 1450),
    ]
    
    frames_rgb = []
    durations = []
    
    fig = plt.figure(figsize=(9.8, 5.2), dpi=120, facecolor="#ffffff")
    
    # Paleta customizada para múltiplos clusters
    cores_clusters = ["#e11d48", "#2563eb", "#059669", "#d97706", "#7c3aed",
                      "#0891b2", "#ea580c", "#4f46e5", "#16a34a", "#db2777"]
    
    for f_idx, (etapa, D_val) in enumerate(d_seq):
        fig.clf()
        ax_map = fig.add_axes([0.05, 0.08, 0.52, 0.84])
        ax_info = fig.add_axes([0.62, 0.08, 0.34, 0.84])
        ax_info.axis("off")
        
        # Fundo do oceano
        ax_map.set_facecolor("#e2e8f0")
        
        if etapa == "dens_bruta":
            # Mostra o raster contínuo do GPW v4
            im = ax_map.imshow(np.where(sub_dens > 0, sub_dens, np.nan),
                               extent=extent, cmap="inferno", norm=LogNorm(100, 30000),
                               origin="upper", zorder=1)
            ax_map.set_title("1. Densidade Populacional Contínua (GPW v4, 2020)\nRegião Metropolitana do Recife",
                             fontsize=8.3, fontweight="bold", color="#0f172a", pad=6)
            
            # Anotações dos centros urbanos
            for nome_p, (p_lo, p_la) in PONTOS_RMR.items():
                ax_map.plot(p_lo, p_la, "o", color="#38bdf8", ms=4.5, mec="#0f172a", mew=0.8, zorder=5)
                ax_map.annotate(nome_p, (p_lo, p_la), (4, 4), textcoords="offset points",
                                fontsize=6.5, fontweight="bold", color="#ffffff",
                                path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")],
                                zorder=6)
                
            ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                       fc="#f8fafc", ec="#cbd5e1", lw=1.2))
            ax_info.text(0.5, 0.92, "RELEVO DEMOGRÁFICO DA RMR", transform=ax_info.transAxes,
                         ha="center", fontsize=8.2, fontweight="bold", color="#0369a1")
            ax_info.text(0.06, 0.78, "O raster GPW v4 (resolução 30\")\nredistribui a contagem censitária\nnuma grade regular de células de ~0,93 km.",
                         transform=ax_info.transAxes, fontsize=7.4, color="#334155", va="top")
            ax_info.text(0.06, 0.58, "Cores térmicas (inferno):\n• Escuro: baixas densidades / mangues / mar\n• Amarelo/Branco: núcleos hiperdensos\n  (Recife, Olinda, Jaboatão, Paulista)",
                         transform=ax_info.transAxes, fontsize=7.4, color="#334155", va="top")
            ax_info.text(0.06, 0.35, "Próximo passo do CCA:\nIdentificar células que atendem ao\ncritério de densidade urbana D* e\nagrupá-las por distância l = 1,8 km.",
                         transform=ax_info.transAxes, fontsize=7.4, color="#1e293b", fontweight="medium", va="top")
            
            # Repete 10 vezes para introdução clara
            rgb = fig_to_rgb(fig)
            for _ in range(12):
                frames_rgb.append(rgb)
                durations.append(0.12)
            continue
            
        # Para etapas com D* ativo
        lab_full = V.rotulos(D_val, 1800)
        sub_lab = lab_full[r_min:r_max, c_min:c_max]
        
        # Fundo do terreno com relevo esmaecido
        ax_map.imshow(np.where(sub_dens > 0, sub_dens, np.nan),
                      extent=extent, cmap="gray", norm=LogNorm(100, 30000),
                      alpha=0.22, origin="upper", zorder=1)
        
        # Identifica componentes ativas na área
        u_clusters = np.unique(sub_lab[sub_lab > 0])
        n_clusters = len(u_clusters)
        
        # Encontra o cluster de maior população no Recife
        recife_curr_id = sub_lab[int(np.abs(sub_lat - (-8.05)).argmin()), int(np.abs(sub_lon - (-34.88)).argmin())]
        
        # Mascara visual colorida para as células urbanas ativas
        rgb_layer = np.zeros((sub_dens.shape[0], sub_dens.shape[1], 4))
        
        # Células ativas que não são de Recife em cinza azulado
        is_active = (sub_dens >= D_val)
        
        if etapa == "consolidado":
            # No consolidado, toda a mancha do Recife é exibida na cor ouro/laranja do artigo
            for r_i in range(sub_lab.shape[0]):
                for c_i in range(sub_lab.shape[1]):
                    cid = sub_lab[r_i, c_i]
                    if cid == id_recife:
                        rgb_layer[r_i, c_i] = [0.95, 0.45, 0.05, 0.95]
                    elif cid > 0:
                        rgb_layer[r_i, c_i] = [0.55, 0.65, 0.75, 0.7]
        else:
            # Em etapas intermediárias, cada cluster recebe uma cor para mostrar a coalescência!
            for idx_c, cid in enumerate(u_clusters):
                c_mask = (sub_lab == cid)
                cor_hex = cores_clusters[idx_c % len(cores_clusters)]
                # converter hex para rgba
                r_c = int(cor_hex[1:3], 16) / 255.0
                g_c = int(cor_hex[3:5], 16) / 255.0
                b_c = int(cor_hex[5:7], 16) / 255.0
                rgb_layer[c_mask] = [r_c, g_c, b_c, 0.90]
                
        ax_map.imshow(rgb_layer, extent=extent, origin="upper", zorder=3)
        
        # Anotações dos centros urbanos
        for nome_p, (p_lo, p_la) in PONTOS_RMR.items():
            ax_map.plot(p_lo, p_la, "o", color="#ffffff", ms=3.5, mec="#0f172a", mew=0.7, zorder=5)
            ax_map.annotate(nome_p, (p_lo, p_la), (4, 4), textcoords="offset points",
                            fontsize=6.2, fontweight="bold", color="#0f172a",
                            path_effects=[pe.withStroke(linewidth=2, foreground="#ffffff")],
                            zorder=6)
            
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
        ax_map.set_xticks([-35.1, -34.9])
        ax_map.set_xticklabels(["35,1°O", "34,9°O"], fontsize=7.2)
        ax_map.set_yticks([-8.3, -8.1, -7.9])
        ax_map.set_yticklabels(["8,3°S", "8,1°S", "7,9°S"], fontsize=7.2)
        
        is_final = (etapa == "consolidado")
        d_fmt = f"{int(D_val):,}".replace(",", ".")
        titulo_mapa = ("* 3. Mancha Metropolitana do Recife Consolidada (CCA) *\n"
                       f"Configuração Calibrada: $D^* = 1.450$ hab/km² | $\\ell = 1,8$ km"
                       if is_final else
                       f"2. Aglomeração Célula a Célula via CCA (RMR)\n"
                       f"Limiar: $D^* = {d_fmt}$ hab/km² | $\\ell = 1,8$ km")
        
        ax_map.set_title(titulo_mapa, fontsize=8.2, fontweight="bold",
                         color="#047857" if is_final else "#0f172a", pad=6)
        
        # Métricas da mancha do Recife
        recife_mask = (sub_lab == (id_recife if is_final else recife_curr_id)) if recife_curr_id > 0 else np.zeros_like(sub_lab, dtype=bool)
        recife_pop = sub_pop[recife_mask].sum()
        recife_area = sub_area[recife_mask].sum()
        recife_n_px = recife_mask.sum()
        
        # Info card
        bg_card = "#ecfdf5" if is_final else "#f8fafc"
        edge_card = "#059669" if is_final else "#cbd5e1"
        ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                    fc=bg_card, ec=edge_card, lw=1.2))
        
        hdr_txt = "CONURBAÇÃO CONSOLIDADA" if is_final else "PROCESSO DE AGLOMERAÇÃO CCA"
        ax_info.text(0.5, 0.93, hdr_txt, transform=ax_info.transAxes,
                     ha="center", fontsize=8.0, fontweight="bold",
                     color="#047857" if is_final else "#0369a1")
        
        ax_info.text(0.06, 0.83, f"Limiar $D^*$:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        ax_info.text(0.94, 0.83, f"{d_fmt} hab/km²", transform=ax_info.transAxes,
                     fontsize=8.0, fontweight="bold", ha="right", color="#0f172a")
        
        ax_info.text(0.06, 0.74, f"Clusters na área:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        clust_txt = "1 única mancha unificada!" if is_final else f"{n_clusters} fragmentos distintos"
        ax_info.text(0.94, 0.74, clust_txt, transform=ax_info.transAxes,
                     fontsize=8.0, fontweight="bold", ha="right",
                     color="#047857" if is_final else "#0f172a")
        
        ax_info.text(0.06, 0.65, f"População Recife CCA:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        ax_info.text(0.94, 0.65, f"{recife_pop/1e6:.2f} M hab".replace(".", ","), transform=ax_info.transAxes,
                     fontsize=8.5, fontweight="bold", ha="right", color="#b91c1c")
        
        ax_info.text(0.06, 0.56, f"Área da mancha:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        ax_info.text(0.94, 0.56, f"{recife_area:.1f} km² ({recife_n_px} células)".replace(".", ","), transform=ax_info.transAxes,
                     fontsize=8.0, fontweight="bold", ha="right", color="#0f172a")
        
        # Comparação administrativa
        ax_info.text(0.06, 0.44, "Comparação Administrativa (IBGE):", transform=ax_info.transAxes,
                     fontsize=7.4, fontweight="bold", color="#1e293b")
        ax_info.text(0.06, 0.36, "• Município de Recife: 1,49 M hab\n• Jaboatão dos Guararapes: 0,64 M\n• Olinda: 0,35 M | Paulista: 0,34 M\n• Camaragibe: 0,15 M",
                     transform=ax_info.transAxes, fontsize=6.8, color="#334155", va="top")
        
        ax_info.text(0.06, 0.17, "* O CCA conecta essas cidades por\ncontinuidade física de densidade,\nsuperando os limites municipais e\ngerando a maior mancha do NE (3,47M).",
                     transform=ax_info.transAxes, fontsize=6.8, fontweight="medium",
                     color="#047857" if is_final else "#475569", va="top")
        
        rgb = fig_to_rgb(fig)
        if is_final:
            for _ in range(18):  # Pausa de 1,8s no resultado consolidado
                frames_rgb.append(rgb)
                durations.append(0.10)
        else:
            frames_rgb.append(rgb)
            durations.append(0.18)
            
    plt.close(fig)
    out_path = os.path.join(GIFS_DIR, "03_cca_recife_conurbacao_metropolitana.gif")
    imageio.mimsave(out_path, frames_rgb, duration=durations, loop=0)
    print(f"Salvo: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB, {len(frames_rgb)} quadros em {time.time()-t0:.1f} s)")
    return out_path

# ==================================================================
# 4. GIF 4: Microescala Célula a Célula - Conurbação de Fortaleza
# ==================================================================
def gerar_gif_04_fortaleza():
    from gerar_fortaleza_lento import gerar_gif_04_fortaleza_lento
    out_path = os.path.join(GIFS_DIR, "04_cca_fortaleza_conurbacao_metropolitana.gif")
    return gerar_gif_04_fortaleza_lento(out_path=out_path)

# ==================================================================
# 5. GIF 5: Microescala Célula a Célula - Conurbação de Salvador
# ==================================================================
def gerar_gif_05_salvador():
    print("\n[5/5] Gerando GIF 5: Conurbação Metropolitana Célula a Célula de Salvador...")
    t0 = time.time()
    
    # Bounding box de Salvador e RMS
    # lat: [-13.05, -12.75], lon: [-38.60, -38.25]
    r_min = int(np.abs(lat - (-12.75)).argmin())
    r_max = int(np.abs(lat - (-13.05)).argmin())
    c_min = int(np.abs(lon - (-38.60)).argmin())
    c_max = int(np.abs(lon - (-38.25)).argmin())
    
    sub_dens = dens_grid[r_min:r_max, c_min:c_max]
    sub_pop = pop_px[r_min:r_max, c_min:c_max]
    sub_area = area_grid_km2[r_min:r_max, c_min:c_max]
    sub_lat = lat[r_min:r_max]
    sub_lon = lon[c_min:c_max]
    extent = [sub_lon[0], sub_lon[-1], sub_lat[-1], sub_lat[0]]
    
    PONTOS_SSA = {
        "Pelourinho / Centro": (-38.510, -12.972),
        "Barra / Vitória": (-38.530, -13.005),
        "Pituba": (-38.460, -12.990),
        "Itapuã": (-38.360, -12.945),
        "Lauro de Freitas": (-38.325, -12.895),
        "Simões Filho": (-38.405, -12.785),
    }
    
    V = ab.Varredura(pop_px, no_ne, lat, dx, dy, area_px_km2)
    lab_calib = V.rotulos(1450, 1800)
    id_salvador = lab_calib[int(np.abs(lat - (-12.97)).argmin()), int(np.abs(lon - (-38.50)).argmin())]
    
    d_seq = [
        ("dens_bruta", None),
        ("nucleacao_1", 16000),
        ("nucleacao_2", 11000),
        ("expansao_1", 7500),
        ("conurbacao_1", 4500),
        ("conurbacao_2", 3000),
        ("conurbacao_3", 2200),
        ("conurbacao_4", 1700),
        ("consolidado", 1450),
    ]
    
    cores_clusters = ["#e11d48", "#2563eb", "#059669", "#d97706", "#7c3aed",
                      "#0891b2", "#ea580c", "#4f46e5", "#16a34a", "#db2777"]
    
    frames_rgb = []
    durations = []
    fig = plt.figure(figsize=(9.8, 5.2), dpi=120, facecolor="#ffffff")
    
    for f_idx, (etapa, D_val) in enumerate(d_seq):
        fig.clf()
        ax_map = fig.add_axes([0.05, 0.08, 0.52, 0.84])
        ax_info = fig.add_axes([0.62, 0.08, 0.34, 0.84])
        ax_info.axis("off")
        ax_map.set_facecolor("#e2e8f0")
        
        if etapa == "dens_bruta":
            im = ax_map.imshow(np.where(sub_dens > 0, sub_dens, np.nan),
                               extent=extent, cmap="inferno", norm=LogNorm(100, 30000),
                               origin="upper", zorder=1)
            ax_map.set_title("1. Densidade Populacional Contínua (GPW v4, 2020)\nSalvador e Baía de Todos os Santos",
                             fontsize=8.3, fontweight="bold", color="#0f172a", pad=6)
            
            for nome_p, (p_lo, p_la) in PONTOS_SSA.items():
                ax_map.plot(p_lo, p_la, "o", color="#38bdf8", ms=4.5, mec="#0f172a", mew=0.8, zorder=5)
                ax_map.annotate(nome_p, (p_lo, p_la), (4, 4), textcoords="offset points",
                                fontsize=6.5, fontweight="bold", color="#ffffff",
                                path_effects=[pe.withStroke(linewidth=2, foreground="#0f172a")],
                                zorder=6)
                
            ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                       fc="#f8fafc", ec="#cbd5e1", lw=1.2))
            ax_info.text(0.5, 0.92, "RELEVO DEMOGRÁFICO DE SALVADOR", transform=ax_info.transAxes,
                         ha="center", fontsize=8.0, fontweight="bold", color="#0369a1")
            ax_info.text(0.06, 0.78, "A geografia peninsular de Salvador impõe um\nforte confinamento espacial entre o Oceano\nAtlântico e a Baía de Todos os Santos.",
                         transform=ax_info.transAxes, fontsize=7.4, color="#334155", va="top")
            ax_info.text(0.06, 0.55, "Terceira Maior Mancha do Nordeste:\n• Densidade máxima no miolo e na orla\n• Conurbação linear contínua para o norte\n  em direção a Lauro de Freitas e Camaçari",
                         transform=ax_info.transAxes, fontsize=7.4, color="#334155", va="top")
            ax_info.text(0.06, 0.30, "A seguir:\nO CCA agrega a península e o litoral norte\nformando a mancha de 3,10M habitantes.",
                         transform=ax_info.transAxes, fontsize=7.4, color="#1e293b", fontweight="medium", va="top")
            
            rgb = fig_to_rgb(fig)
            for _ in range(12):
                frames_rgb.append(rgb)
                durations.append(0.12)
            continue
            
        lab_full = V.rotulos(D_val, 1800)
        sub_lab = lab_full[r_min:r_max, c_min:c_max]
        
        ax_map.imshow(np.where(sub_dens > 0, sub_dens, np.nan),
                      extent=extent, cmap="gray", norm=LogNorm(100, 30000),
                      alpha=0.22, origin="upper", zorder=1)
        
        u_clusters = np.unique(sub_lab[sub_lab > 0])
        n_clusters = len(u_clusters)
        salv_curr_id = sub_lab[int(np.abs(sub_lat - (-12.97)).argmin()), int(np.abs(sub_lon - (-38.50)).argmin())]
        
        rgb_layer = np.zeros((sub_dens.shape[0], sub_dens.shape[1], 4))
        is_final = (etapa == "consolidado")
        
        if is_final:
            for r_i in range(sub_lab.shape[0]):
                for c_i in range(sub_lab.shape[1]):
                    cid = sub_lab[r_i, c_i]
                    if cid == id_salvador:
                        rgb_layer[r_i, c_i] = [0.95, 0.45, 0.05, 0.95]
                    elif cid > 0:
                        rgb_layer[r_i, c_i] = [0.55, 0.65, 0.75, 0.7]
        else:
            for idx_c, cid in enumerate(u_clusters):
                c_mask = (sub_lab == cid)
                cor_hex = cores_clusters[idx_c % len(cores_clusters)]
                r_c = int(cor_hex[1:3], 16) / 255.0
                g_c = int(cor_hex[3:5], 16) / 255.0
                b_c = int(cor_hex[5:7], 16) / 255.0
                rgb_layer[c_mask] = [r_c, g_c, b_c, 0.90]
                
        ax_map.imshow(rgb_layer, extent=extent, origin="upper", zorder=3)
        
        for nome_p, (p_lo, p_la) in PONTOS_SSA.items():
            ax_map.plot(p_lo, p_la, "o", color="#ffffff", ms=3.5, mec="#0f172a", mew=0.7, zorder=5)
            ax_map.annotate(nome_p, (p_lo, p_la), (4, 4), textcoords="offset points",
                            fontsize=6.2, fontweight="bold", color="#0f172a",
                            path_effects=[pe.withStroke(linewidth=2, foreground="#ffffff")],
                            zorder=6)
            
        ax_map.set_xlim(extent[0], extent[1])
        ax_map.set_ylim(extent[2], extent[3])
        ax_map.set_xticks([-38.5, -38.3])
        ax_map.set_xticklabels(["38,5°O", "38,3°O"], fontsize=7.2)
        ax_map.set_yticks([-13.0, -12.8])
        ax_map.set_yticklabels(["13,0°S", "12,8°S"], fontsize=7.2)
        
        d_fmt = f"{int(D_val):,}".replace(",", ".")
        titulo_mapa = ("* 3. Mancha Metropolitana de Salvador Consolidada (CCA) *\n"
                       f"Configuração Calibrada: $D^* = 1.450$ hab/km² | $\\ell = 1,8$ km"
                       if is_final else
                       f"2. Aglomeração Célula a Célula via CCA (RMS)\n"
                       f"Limiar: $D^* = {d_fmt}$ hab/km² | $\\ell = 1,8$ km")
        
        ax_map.set_title(titulo_mapa, fontsize=8.2, fontweight="bold",
                         color="#047857" if is_final else "#0f172a", pad=6)
        
        salv_mask = (sub_lab == (id_salvador if is_final else salv_curr_id)) if salv_curr_id > 0 else np.zeros_like(sub_lab, dtype=bool)
        salv_pop = sub_pop[salv_mask].sum()
        salv_area = sub_area[salv_mask].sum()
        salv_n_px = salv_mask.sum()
        
        bg_card = "#ecfdf5" if is_final else "#f8fafc"
        edge_card = "#059669" if is_final else "#cbd5e1"
        ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                                    fc=bg_card, ec=edge_card, lw=1.2))
        
        hdr_txt = "CONURBAÇÃO CONSOLIDADA" if is_final else "PROCESSO DE AGLOMERAÇÃO CCA"
        ax_info.text(0.5, 0.93, hdr_txt, transform=ax_info.transAxes,
                     ha="center", fontsize=8.0, fontweight="bold",
                     color="#047857" if is_final else "#0369a1")
        
        ax_info.text(0.06, 0.83, f"Limiar $D^*$:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        ax_info.text(0.94, 0.83, f"{d_fmt} hab/km²", transform=ax_info.transAxes,
                     fontsize=8.0, fontweight="bold", ha="right", color="#0f172a")
        
        ax_info.text(0.06, 0.74, f"Clusters na área:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        clust_txt = "1 única mancha unificada!" if is_final else f"{n_clusters} fragmentos distintos"
        ax_info.text(0.94, 0.74, clust_txt, transform=ax_info.transAxes,
                     fontsize=8.0, fontweight="bold", ha="right",
                     color="#047857" if is_final else "#0f172a")
        
        ax_info.text(0.06, 0.65, f"População Salvador CCA:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        ax_info.text(0.94, 0.65, f"{salv_pop/1e6:.2f} M hab".replace(".", ","), transform=ax_info.transAxes,
                     fontsize=8.5, fontweight="bold", ha="right", color="#b91c1c")
        
        ax_info.text(0.06, 0.56, f"Área da mancha:", transform=ax_info.transAxes, fontsize=7.6, color="#475569")
        ax_info.text(0.94, 0.56, f"{salv_area:.1f} km² ({salv_n_px} células)".replace(".", ","), transform=ax_info.transAxes,
                     fontsize=8.0, fontweight="bold", ha="right", color="#0f172a")
        
        ax_info.text(0.06, 0.44, "Comparação Administrativa (IBGE):", transform=ax_info.transAxes,
                     fontsize=7.4, fontweight="bold", color="#1e293b")
        ax_info.text(0.06, 0.36, "• Município de Salvador: 2,42 M hab\n• Lauro de Freitas: 0,20 M\n• Camaçari: 0,30 M (parcial)",
                     transform=ax_info.transAxes, fontsize=6.8, color="#334155", va="top")
        
        ax_info.text(0.06, 0.17, "* O CCA absorve a conurbação linear da\norla norte, delineando a 3ª maior\nmancha urbana do Nordeste (3,10M).",
                     transform=ax_info.transAxes, fontsize=6.8, fontweight="medium",
                     color="#047857" if is_final else "#475569", va="top")
        
        rgb = fig_to_rgb(fig)
        if is_final:
            for _ in range(18):
                frames_rgb.append(rgb)
                durations.append(0.10)
        else:
            frames_rgb.append(rgb)
            durations.append(0.18)
            
    plt.close(fig)
    out_path = os.path.join(GIFS_DIR, "05_cca_salvador_conurbacao_metropolitana.gif")
    imageio.mimsave(out_path, frames_rgb, duration=durations, loop=0)
    print(f"Salvo: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB, {len(frames_rgb)} quadros em {time.time()-t0:.1f} s)")
    return out_path

# ==================================================================
# Execução Principal
# ==================================================================
def main():
    print("=" * 70)
    print("GERADOR DE GIFS: AGLOMERAÇÕES POPULACIONAIS POR MEIO DO CCA")
    print("Baseado na Figura (a) do artigo (GPW v4 + CCA Nordeste)")
    print("=" * 70)
    
    t_inicio = time.time()
    g1 = gerar_gif_01_varredura_D()
    g2 = gerar_gif_02_varredura_l()
    g3 = gerar_gif_03_recife()
    g4 = gerar_gif_04_fortaleza()
    g5 = gerar_gif_05_salvador()
    
    print("\n" + "=" * 70)
    print(f"TODOS OS 5 GIFS FORAM GERADOS COM SUCESSO EM {time.time()-t_inicio:.1f} s!")
    print(f"Diretório de saída: {GIFS_DIR}")
    for p in [g1, g2, g3, g4, g5]:
        print(f" - {os.path.basename(p)} ({os.path.getsize(p)/(1024*1024):.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
