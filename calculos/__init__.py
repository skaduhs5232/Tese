"""
Pacote `calculos`
-----------------
Implementação manual em NumPy (+ Numba) das formulações matemáticas e estatísticas da dissertação:
  1. `geometria_esferica`   : Área esférica de pixel via 2 triângulos esféricos (L'Huilier / Oliveira et al., 2019) e distância de Haversine.
  2. `percolacao_ilhas`     : Percolação de sítios em clusters de elevação (h > h*) e identificação de oceano conectado à borda + sinks costeiros.
  3. `alometria_estatistica`: Densidade populacional esférica, calibração de Zipf (Hill, CSN e varredura dinâmica OLS/R²), soma de habitantes por pixel urbano em cada bacia e regressão alométrica log-log.
  4. `hidrologia_endorreica`: Priority-flood e identificação de bacias endorreicas (sem foz oceânica) complementar ao IPBA.

Os algoritmos de CCA e IPBA são importados diretamente dos repositórios oficiais do orientador:
  - `notebooks/CCA/cca/city_clustering.py` (`cca_grid`, `cca`, `burn`)
  - `notebooks/IPBA/drainage_basins` (`SetDrainageBasins`, `run_ipba_grid`)
"""

from .geometria_esferica import (
    R_EARTH_KM,
    R_MOON_KM,
    R_MARS_KM,
    haversine_rad,
    haversine_deg,
    spherical_triangle_area,
    pixel_area_by_latitude,
    pixel_area_grid_2d,
    somar_areas_por_bacia,
)
from .percolacao_ilhas import (
    Percolation,
    extrair_recorte_cluster,
    identificar_oceano_e_sinks,
)
from .alometria_estatistica import (
    pearson_r,
    spearman_rho,
    ajuste_escala_log_log,
    leave_one_out_alometria,
    parear_fozes_e_sites,
    calcular_densidade_gpw,
    vizinhanca_haversine,
    escala_minima_l_metros,
    alpha_hill,
    ajuste_csn_zipf,
    calibrar_zipf_territorio,
    reamostrar_bacias_para_gpw,
    dividir_manchas_e_somar_populacao_bacias,
    calibrar_zipf_dinamico,
)
from .hidrologia_endorreica import (
    priority_flood_preenchimento,
    identificar_bacias_endorreicas,
    SISTEMAS_ENDORREICOS,
)

__all__ = [
    "R_EARTH_KM",
    "R_MOON_KM",
    "R_MARS_KM",
    "haversine_rad",
    "haversine_deg",
    "spherical_triangle_area",
    "pixel_area_by_latitude",
    "pixel_area_grid_2d",
    "somar_areas_por_bacia",
    "Percolation",
    "extrair_recorte_cluster",
    "identificar_oceano_e_sinks",
    "pearson_r",
    "spearman_rho",
    "ajuste_escala_log_log",
    "leave_one_out_alometria",
    "parear_fozes_e_sites",
    "calcular_densidade_gpw",
    "vizinhanca_haversine",
    "escala_minima_l_metros",
    "alpha_hill",
    "ajuste_csn_zipf",
    "calibrar_zipf_territorio",
    "reamostrar_bacias_para_gpw",
    "dividir_manchas_e_somar_populacao_bacias",
    "calibrar_zipf_dinamico",
    "priority_flood_preenchimento",
    "identificar_bacias_endorreicas",
    "SISTEMAS_ENDORREICOS",
]
