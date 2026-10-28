"""
template_mapa_cientifico.py
----------------------------
Script modelo de referência para criação de mapas científicos no repositório.
Demonstra a aplicação prática de todas as regras:
  1. Tipografia e estilo científico de publicação
  2. Todos os textos, legendas e unidades em inglês
  3. Mapa de cores contínuo 'plasma'
  4. Inset do mapa-múndi no canto inferior esquerdo (lower left) em cinza neutro
  5. Barra de cores / escala em pé à direita (vertical, right)
  6. Sem linhas cruzadas ou sobreposição de identificadores
  7. Qualidade vetorial suave para continentes
  8. Saída sincronizada em .png (300 DPI) e .pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import scientific_plot_utils as spu

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

spu.set_scientific_style()

STUDY_BOX = (-125.0, 7.0, -65.0, 50.0)

fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.set_facecolor("#dce8f2")

spu.format_geo_axes(ax, box=STUDY_BOX, xticks=(-120, -105, -90, -75), yticks=(10, 20, 30, 40, 50))

lon_grid = np.linspace(STUDY_BOX[0], STUDY_BOX[2], 200)
lat_grid = np.linspace(STUDY_BOX[1], STUDY_BOX[3], 200)
X, Y = np.meshgrid(lon_grid, lat_grid)
Z = np.sin(np.radians(X * 2)) * np.cos(np.radians(Y * 2)) + 1.0

norm = Normalize(vmin=0, vmax=2)
im = ax.imshow(Z, extent=(STUDY_BOX[0], STUDY_BOX[2], STUDY_BOX[1], STUDY_BOX[3]),
               origin="lower", cmap="plasma", norm=norm, alpha=0.9, zorder=5)

# Barra de cor em pé à direita (vertical, right)
spu.add_vertical_colorbar(fig, im, ax, label="Synthetic Measurement Index (normalized units)")

# Identificador de painel (a) seguro contra sobreposição
spu.add_panel_label(ax, "a")

# Barra de escala no canto inferior direito (longe do inset à esquerda)
spu.add_scale_bar(ax, km=500, pos=(0.72, 0.06))

# Mapa-múndi localizador INSET no canto inferior esquerdo (lower left)
inset_ax = spu.add_world_inset(
    fig=fig,
    ax_main=ax,
    zoom_box=STUDY_BOX,
    inset_rect=(0.02, 0.03, 0.30, 0.25),
    use_axes_coords=True,
    box_edgecolor="#d90429",
    land_color="#d0d0d0",
    ocean_color="#ffffff",
    border_color="#888888",
)

output_demo = os.path.join(CURRENT_DIR, "exemplo_mapa_cientifico")
spu.save_scientific_figure(fig, output_demo, dpi=300)
plt.close(fig)

print("Exemplo de mapa científico gerado com sucesso!")
