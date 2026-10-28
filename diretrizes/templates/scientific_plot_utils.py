"""
scientific_plot_utils.py
------------------------
Funções utilitárias padronizadas para geração de figuras científicas no repositório.
Implementa as diretrizes formais:
  - Idioma 100% em inglês
  - Mapa de cores 'plasma'
  - Inset do mapa-múndi em cinza no canto inferior esquerdo (lower left)
  - Escala / barra de cores vertical à direita (vertical, right)
  - Garantia de não-sobreposição (zero overlap)
  - Alta qualidade vetorial para os continentes (sem pixelização)
  - Exportação casada em PNG (300 DPI) e PDF vetorial
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import shape

# Caminho para os polígonos vetoriais padrão do mundo (Natural Earth 1:110m)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIRETRIZES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
WORLD_GEOJSON = os.path.join(DIRETRIZES_DIR, "assets", "world_land_110m.geojson")

# Parâmetros de estilo científico para Matplotlib
SCIENTIFIC_RC = {
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


def set_scientific_style():
    """Aplica os parâmetros visuais de publicação científica ao Matplotlib."""
    plt.rcParams.update(SCIENTIFIC_RC)


def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.04):
    """
    Adiciona identificador de painel (a), (b), (c) fora da área de dados e à esquerda do título,
    garantindo que nunca seja sobreposto por títulos, linhas, bordas ou dados.
    """
    clean_label = label.strip("()")
    ax.text(x, y, f"({clean_label})", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", ha="left", va="bottom", zorder=100)


def add_scale_bar(ax, km: float, pos: tuple = (0.72, 0.06)):
    """
    Adiciona barra de escala geográfica em quilômetros com texto em inglês.
    Por padrão posicionada no canto inferior direito (pos=(0.72, 0.06)) para não colidir
    com o mapa-múndi localizador no canto inferior esquerdo.
    """
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    lat_mid = (y0 + y1) / 2.0
    cos_lat = max(0.2, float(np.cos(np.radians(lat_mid))))
    dlon = km / (111.32 * cos_lat)
    xa = x0 + pos[0] * (x1 - x0)
    ya = y0 + pos[1] * (y1 - y0)

    # Caixa suave translúcida atrás da barra de escala para legibilidade perfeita
    pad_x = 0.02 * (x1 - x0)
    pad_y = 0.025 * (y1 - y0)
    bg = Rectangle(
        (xa - pad_x, ya - pad_y * 0.6),
        dlon + 2 * pad_x,
        pad_y * 2.4,
        facecolor="white",
        edgecolor="none",
        alpha=0.80,
        zorder=29,
    )
    ax.add_patch(bg)

    ax.plot([xa, xa + dlon], [ya, ya], c="k", lw=2.2, solid_capstyle="butt", zorder=30)
    ax.text(xa + dlon / 2.0, ya + 0.015 * (y1 - y0), f"{int(km)} km",
            ha="center", va="bottom", fontsize=6.8, fontweight="bold", zorder=30)


def add_vertical_colorbar(fig, im, ax, label: str, extend: str = "neither",
                          shrink: float = 0.86, pad: float = 0.025, fraction: float = 0.046):
    """
    Adiciona a barra de cores / escala obrigatoriamente em pé à direita (vertical, right).
    """
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        location="right",
        shrink=shrink,
        pad=pad,
        fraction=fraction,
        extend=extend,
    )
    cbar.set_label(label, fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)
    return cbar


def add_world_inset(fig, ax_main, zoom_box=(-125.0, 7.0, -65.0, 50.0),
                    inset_rect=None,
                    use_axes_coords: bool = True,
                    corner: str = "auto",
                    max_w: float = 0.27,
                    max_h: float = 0.18,
                    box_edgecolor="#d90429",
                    land_color="#d0d0d0",
                    ocean_color="#ffffff",
                    border_color="#888888"):
    """
    Cria e desenha o mapa-múndi localizador (inset) com proporção 2:1 (360° x 180°)
    estritamente travada (`aspect='equal'`, zero achatamento) e posicionado de forma
    adaptativa no canto oceânico livre do painel principal `ax_main`.
    """
    x0, y0, x1, y1 = zoom_box

    if use_axes_coords and ax_main is not None:
        xlim = ax_main.get_xlim()
        ylim = ax_main.get_ylim()
        dlon = abs(xlim[1] - xlim[0]) if xlim[1] != xlim[0] else abs(x1 - x0)
        dlat = abs(ylim[1] - ylim[0]) if ylim[1] != ylim[0] else abs(y1 - y0)
        r_main = max(0.25, dlon / max(1e-6, dlat))

        # Garante razão física exata 2:1 (360° : 180°) sem qualquer deformação
        w_frac = min(max_w, 2.0 * max_h / r_main)
        h_frac = w_frac * r_main / 2.0

        if corner == "auto":
            # Na América do Sul (lon ~ -83 a -32, lat ~ -56 a 13), o canto inferior direito
            # é o Oceano Atlântico Sul totalmente livre de continente
            if -90.0 <= x0 <= -70.0 and y0 < -20.0:
                corner_use = "lower right"
            else:
                corner_use = "lower left"
        else:
            corner_use = corner

        pad_x, pad_y = 0.02, 0.025
        if corner_use == "lower right":
            rect_calc = (1.0 - w_frac - pad_x, pad_y, w_frac, h_frac)
        elif corner_use == "upper right":
            rect_calc = (1.0 - w_frac - pad_x, 1.0 - h_frac - pad_y, w_frac, h_frac)
        elif corner_use == "upper left":
            rect_calc = (pad_x, 1.0 - h_frac - pad_y, w_frac, h_frac)
        else:
            rect_calc = (pad_x, pad_y, w_frac, h_frac)

        ax_inset = ax_main.inset_axes(rect_calc, zorder=25)
    else:
        rect_use = inset_rect if inset_rect is not None else (0.02, 0.03, 0.26, 0.18)
        ax_inset = fig.add_axes(rect_use)

    ax_inset.set_facecolor(ocean_color)

    if os.path.exists(WORLD_GEOJSON):
        with open(WORLD_GEOJSON, "r", encoding="utf-8") as f:
            world_data = json.load(f)
        for feature in world_data["features"]:
            geom = shape(feature["geometry"])
            geoms = getattr(geom, "geoms", [geom])
            for poly in geoms:
                xs, ys = poly.exterior.xy
                ax_inset.fill(xs, ys, fc=land_color, ec=border_color, lw=0.35, zorder=2)
    else:
        ax_inset.text(0.5, 0.5, "World Map", ha="center", va="center", fontsize=6)

    ax_inset.set_xlim(-180.0, 180.0)
    ax_inset.set_ylim(-90.0, 90.0)
    ax_inset.set_aspect("equal", adjustable="box")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(0.7)

    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                     edgecolor=box_edgecolor, linewidth=1.1, zorder=10)
    ax_inset.add_patch(rect)

    return ax_inset


def format_geo_axes(ax, box=(-125.0, 7.0, -65.0, 50.0),
                    xticks=None,
                    yticks=None):
    """
    Configura limites e rótulos de coordenadas geográficas em inglês (°W/°E, °S/°N).
    """
    x0, y0, x1, y1 = box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")

    if xticks is None:
        xticks = np.linspace(x0, x1, 5)[1:-1] if (x1 - x0) > 20 else np.linspace(x0, x1, 4)
        xticks = np.round(xticks / 5.0) * 5.0
        xticks = [x for x in np.unique(xticks) if x0 <= x <= x1]
    if yticks is None:
        yticks = np.linspace(y0, y1, 5)[1:-1] if (y1 - y0) > 20 else np.linspace(y0, y1, 4)
        yticks = np.round(yticks / 5.0) * 5.0
        yticks = [y for y in np.unique(yticks) if y0 <= y <= y1]

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):g}°W" if x < 0 else (f"{x:g}°E" if x > 0 else "0°") for x in xticks])

    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y):g}°S" if y < 0 else (f"{y:g}°N" if y > 0 else "0°") for y in yticks])

    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#333333")


def save_scientific_figure(fig, output_basepath: str, dpi: int = 300):
    """
    Salva a figura casada em PNG (300 DPI) e PDF vetorial para o LaTeX.
    """
    base = os.path.splitext(output_basepath)[0]
    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    print(f"[OK] Figure saved in high quality:\n    PNG: {png_path}\n    PDF: {pdf_path}")
