"""
Pacote de acesso e harmonização dos datasets do projeto (GEBCO 2026, GPW v4 Rev11,
AWI LITTERBASE e NOAA NCEI Marine Microplastics).

Todas as funções são chamadas diretamente pelo notebook principal.
"""

from .microplasticos import (
    BBOX_AMERICAS,
    GEBCO_GLOBAL_TIF,
    GPW_GLOBAL_TIF,
    carregar_microplasticos_globais_para_mapa,
    inventariar_e_unificar_microplasticos,
    obter_microplasticos_por_fonte_e_bbox,
    obter_nurdle_patrol_noaa,
    verificar_rasters_base,
)

__all__ = [
    "BBOX_AMERICAS",
    "GEBCO_GLOBAL_TIF",
    "GPW_GLOBAL_TIF",
    "carregar_microplasticos_globais_para_mapa",
    "inventariar_e_unificar_microplasticos",
    "obter_microplasticos_por_fonte_e_bbox",
    "obter_nurdle_patrol_noaa",
    "verificar_rasters_base",
]
