"""
pipeline
--------
Arquitetura limpa em camadas para análise de bacias hidrográficas e microplásticos:
  - config: caminhos e constantes do projeto.
  - domain: geometria esférica, bacias, calibração Zipf e lei de potência (puro, sem I/O).
  - infra: adaptadores para `externos/` (GebcoCutter, IPBA, CCA), GDAL, NOAA e cache.
  - use_cases: orquestração de porções de terra e carregamento de bacias.
  - presentation: gráficos padronizados (mosaicos, varredura de raios e calibração Zipf).
"""

from .domain.scaling import (
    ajustar_lei_potencia,
    filtrar_pop,
    filtrar_redes_superficie,
    media_por_sink_raio,
    pareados_por_raio,
    tabela_ajustes,
)
from .infra.noaa import carregar_noaa_por_tipo
from .presentation.graficos import (
    plot_zipf_calibration,
    plotar_lei_potencia,
    plotar_mosaico,
    plotar_varredura,
)
from .use_cases.bacias import carregar_bacias, identificar_porcoes_ativas, processar_porcao_terra

__all__ = [
    "carregar_noaa_por_tipo",
    "filtrar_redes_superficie",
    "filtrar_pop",
    "identificar_porcoes_ativas",
    "processar_porcao_terra",
    "carregar_bacias",
    "media_por_sink_raio",
    "pareados_por_raio",
    "ajustar_lei_potencia",
    "tabela_ajustes",
    "plotar_lei_potencia",
    "plotar_mosaico",
    "plotar_varredura",
    "plot_zipf_calibration",
]
