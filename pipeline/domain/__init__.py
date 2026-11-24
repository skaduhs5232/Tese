"""Camada de domínio puro: geometria esférica, bacias, Zipf e leis de escala."""

from .basins import (
    calcular_areas_bacias,
    classificar_macroregiao,
    extrair_sinks_costeiros,
    mascara_costa,
    nomear_porcao_terra,
    particionar_populacao_por_bacia,
)
from .geometry import area_triangulos_esfericos, km_para_corda, to_xyz
from .scaling import (
    ajustar_lei_potencia,
    filtrar_pop,
    filtrar_redes_superficie,
    media_por_sink_raio,
    medias_bins,
    pareados_por_raio,
    tabela_ajustes,
)
from .zipf import ajustar_zipf, calibrar_grade_zipf, rotular_clusters_cca, selecionar_par_otimo_zipf

__all__ = [
    "to_xyz",
    "km_para_corda",
    "area_triangulos_esfericos",
    "mascara_costa",
    "extrair_sinks_costeiros",
    "calcular_areas_bacias",
    "particionar_populacao_por_bacia",
    "nomear_porcao_terra",
    "classificar_macroregiao",
    "ajustar_zipf",
    "selecionar_par_otimo_zipf",
    "calibrar_grade_zipf",
    "rotular_clusters_cca",
    "filtrar_redes_superficie",
    "filtrar_pop",
    "media_por_sink_raio",
    "pareados_por_raio",
    "medias_bins",
    "ajustar_lei_potencia",
    "tabela_ajustes",
]
