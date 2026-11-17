"""
geracao_graficos
----------------
Módulo de geração de gráficos e mapas científicos no padrão Erneson Oliveira.
"""
from .estilo_erneson import (
    ESTILO_ERNESON,
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
from .figuras import (
    plotar_bacias_ipba,
    plotar_mapa_cca,
    plotar_evolucao_raio_exclusivo,
    plotar_alometria_e_controle_area,
    plotar_comparacao_ilhas_global,
)

__all__ = [
    "ESTILO_ERNESON",
    "WORLD_GEOJSON",
    "aplicar_estilo",
    "rotulo_painel",
    "rotulo_canto",
    "anotar_expoente",
    "guia_inclinacao",
    "scatter_vazado",
    "reta_ols",
    "curva_nw",
    "referencia_inclinacao_1",
    "eixos_log10",
    "colorbar_direita",
    "eixo_mapa",
    "cores_bacias_tab20",
    "cores_manchas",
    "inset_localizador",
    "salvar",
    "plotar_bacias_ipba",
    "plotar_mapa_cca",
    "plotar_evolucao_raio_exclusivo",
    "plotar_alometria_e_controle_area",
    "plotar_comparacao_ilhas_global",
]
