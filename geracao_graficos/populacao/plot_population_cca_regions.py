"""
plot_population_cca_regions.py
------------------------------
Generates the scientific GPW v4 population density figures for the 6 disjoint demographic
regions, with the region boundary used for the dynamic Zipf/CCA calibration and the selected
(D*, L*) pair in each title. Strictly follows `AGENTS.md` and `diretrizes/`:
  - 100% English text
  - Colormap: 'plasma'
  - Gray world locator inset with red zoom box
  - Vertical colorbar on the right
  - Exported in both 300 DPI PNG and vector PDF in `geracao_graficos/populacao/`
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for p in (REPO_ROOT, THIS_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diretrizes.templates.scientific_plot_utils import (
    set_scientific_style,
    add_panel_label,
    add_vertical_colorbar,
    add_world_inset,
    format_geo_axes,
    save_scientific_figure,
)
from notebooks.executar_pipeline_global import ler_janela_gpw, codigo_regiao, REGIOES_DEMOGRAFICAS
from calculos import calcular_densidade_gpw
from regioes_plot import REGIOES_PLOT


def generate_population_cca_figures():
    set_scientific_style()
    gpw_path = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
    results_dir = REPO_ROOT / "notebooks" / "results"
    codigo_por_id = {r["id"]: r["codigo"] for r in REGIOES_DEMOGRAFICAS}

    fig6, axes6 = plt.subplots(2, 3, figsize=(11.4, 6.8))
    fig6.subplots_adjust(wspace=0.44, hspace=0.42)
    for idx_t, (ax_r, reg) in enumerate(zip(axes6.ravel(), REGIOES_PLOT)):
        tid, box = reg["id"], reg["box"]
        with open(results_dir / f"cca_calibracao_{tid}.json", "r", encoding="utf-8") as f:
            cal = json.load(f)

        pop_arr, lats, lons, gt = ler_janela_gpw(gpw_path, box, fator_agregacao=4)
        dens_km2, _ = calcular_densidade_gpw(pop_arr, lats, abs(gt[1]), abs(gt[5]))
        lon2d, lat2d = np.meshgrid(lons, lats)
        dentro = codigo_regiao(lat2d, lon2d) == codigo_por_id[tid]
        dens_plot = np.where(dens_km2 > 0.1, dens_km2, np.nan)
        ext = (lons.min() - gt[1] / 2, lons.max() + gt[1] / 2, lats.min() + gt[5] / 2, lats.max() - gt[5] / 2)

        ax_r.set_facecolor("#eef3f8")
        ax_r.imshow(np.where(~dentro & np.isfinite(dens_plot), 1.0, np.nan), extent=ext, origin="upper",
                    cmap=ListedColormap(["#d0d0d0"]), interpolation="nearest", zorder=1)
        im_r = ax_r.imshow(np.where(dentro, dens_plot, np.nan), extent=ext, origin="upper", cmap="plasma",
                           norm=LogNorm(vmin=1.0, vmax=1.0e4), interpolation="nearest", zorder=2)
        format_geo_axes(ax_r, box=box)
        ax_r.set_title(
            f"{reg['label']}\n($N={cal['n_manchas_urbanas']:,}$ clusters, $D^*={int(cal['D_star_hab_km2'])}$, "
            f"$L^*={int(cal['l_metros'])}$ m)",
            fontsize=7.6, pad=5,
        )
        add_panel_label(ax_r, chr(ord("a") + idx_t), x=-0.16, y=1.06)
        add_vertical_colorbar(fig6, im_r, ax_r, label="Density (people / km$^2$)", shrink=0.82, extend="both")
        add_world_inset(fig6, ax_r, zoom_box=box, use_axes_coords=True, corner="auto")

    save_scientific_figure(fig6, str(THIS_DIR / "population_cca_6regions"), dpi=300)
    plt.close(fig6)


if __name__ == "__main__":
    generate_population_cca_figures()
