"""Camada de infraestrutura: algoritmos externos, rasters GDAL, NOAA e cache em disco."""

from .cache import ler_csv_cache, ler_npz_cache, salvar_csv_cache, salvar_npz_cache
from .externos import agrupar_cca, executar_ipba, ler_geotiff_gebco, percolar_terra
from .noaa import carregar_noaa_por_tipo
from .rasters import amostrar_cota_gebco, recortar_gpw_porcao, recortar_porcao_gebco

__all__ = [
    "ler_csv_cache",
    "salvar_csv_cache",
    "ler_npz_cache",
    "salvar_npz_cache",
    "ler_geotiff_gebco",
    "percolar_terra",
    "executar_ipba",
    "agrupar_cca",
    "amostrar_cota_gebco",
    "recortar_porcao_gebco",
    "recortar_gpw_porcao",
    "carregar_noaa_por_tipo",
]
