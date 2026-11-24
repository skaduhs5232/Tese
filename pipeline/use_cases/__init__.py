"""Casos de uso: orquestração de porções de terra e carregamento de bacias."""

from .bacias import carregar_bacias, identificar_porcoes_ativas, processar_porcao_terra

__all__ = ["identificar_porcoes_ativas", "processar_porcao_terra", "carregar_bacias"]
