"""
scientific_plot_utils.py
------------------------
Camada de compatibilidade que reexporta os utilitários do módulo central
`geracao_graficos/estilo_erneson.py` (padrão visual dos artigos do Prof. Erneson Oliveira).
"""

import os
import sys
from pathlib import Path
import numpy as np
from matplotlib.patches import Rectangle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from geracao_graficos.estilo_erneson import (
    ESTILO_ERNESON as SCIENTIFIC_RC,
    WORLD_GEOJSON,
    aplicar_estilo,
    rotulo_painel,
    rotulo_canto,
    anotar_expoente,
    guia_inclinacao,
    scatter_vazado,
    reta_ols,
    curva_nw,
    referencia_inclinacao_1,
    eixos_log10,
    colorbar_direita,
    eixo_mapa,
    cores_bacias_tab20,
    cores_manchas,
    inset_localizador,
    salvar,
)


def set_scientific_style():
    aplicar_estilo()


def add_panel_label(ax, label: str, x: float = -0.18, y: float = 1.02, dentro: bool = False):
    rotulo_painel(ax, label, dentro=dentro)


def add_scale_bar(ax, km: float, pos: tuple = (0.72, 0.06)):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    lat_mid = (y0 + y1) / 2.0
    cos_lat = max(0.2, float(np.cos(np.radians(lat_mid))))
    dlon = km / (111.32 * cos_lat)
    xa = x0 + pos[0] * (x1 - x0)
    ya = y0 + pos[1] * (y1 - y0)
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
    ax.plot([xa, xa + dlon], [ya, ya], c="k", lw=2.0, solid_capstyle="butt", zorder=30)
    ax.text(
        xa + dlon / 2.0,
        ya + 0.015 * (y1 - y0),
        f"{int(km)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=30,
    )


def add_vertical_colorbar(fig, im, ax, label: str, log: bool = False, **kwargs):
    return colorbar_direita(fig, ax, im, label, log=log)


def add_world_inset(fig, ax_main, zoom_box=(-125.0, 7.0, -65.0, 50.0), corner: str = "lower left", **kwargs):
    canto = "lower left" if corner == "auto" else corner
    return inset_localizador(ax_main, zoom_box, canto=canto)


def format_geo_axes(ax, box=(-125.0, 7.0, -65.0, 50.0), xticks=None, yticks=None):
    x0, y0, x1, y1 = box
    eixo_mapa(ax, extent=(x0, x1, y0, y1))


def save_scientific_figure(fig, output_basepath: str, dpi: int = 300):
    return salvar(fig, output_basepath, dpi=dpi)
