"""
template_mapa_cientifico.py
----------------------------
Script modelo de referência para criação de mapas e gráficos científicos no repositório,
utilizando `geracao_graficos/estilo_erneson.py` (padrão visual dos artigos do Prof. Erneson Oliveira).
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import (
    FIGSIZE_MAPA_MUNDI,
    aplicar_estilo,
    colorbar_direita,
    eixo_mapa,
    inset_localizador,
    rotulo_canto,
    rotulo_painel,
    salvar,
)


def main():
    aplicar_estilo()
    bbox = (-85.0, -56.0, -32.0, 13.0)
    lon = np.linspace(bbox[0], bbox[2], 240)
    lat = np.linspace(bbox[3], bbox[1], 240)
    X, Y = np.meshgrid(lon, lat)
    Z = np.sin(np.radians(X * 2.5)) * np.cos(np.radians(Y * 2.5)) + 1.0

    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
    im = ax.imshow(Z, extent=extent, origin="upper", cmap="plasma", norm=Normalize(vmin=0.0, vmax=2.0), interpolation="nearest")
    eixo_mapa(ax, extent=extent)
    rotulo_painel(ax, "a", dentro=True)
    rotulo_canto(ax, "South America", canto="lower right")
    inset_localizador(ax, bbox=bbox, canto="lower left")
    colorbar_direita(fig, ax, im, "Synthetic index (arb. units)")
    salvar(fig, CURRENT_DIR / "exemplo_mapa_cientifico")


if __name__ == "__main__":
    main()
