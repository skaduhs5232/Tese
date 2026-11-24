"""Configurações, caminhos e constantes compartilhadas do pipeline."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Caminhos de entrada e saída
NOAA_GEOJSON = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
GEBCO_TIF = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
GPW_TIF = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
CACHE_PERC = REPO_ROOT / "relatorios" / "cache_percolacao_gebco_h0.npz"
PORCOES_DIR = REPO_ROOT / "relatorios" / "porcoes_terra"
FIGURAS_DIR = REPO_ROOT / "relatorios" / "figuras"

# Constantes físicas e de grade
EARTH_RADIUS_KM = 6378.137
GEBCO_TO_GPW_SCALE = 4
GEBCO_TOTAL_COLS = 10800

# Padrões de calibração Zipf / CCA
ZIPF_D_PADRAO = (100.0, 200.0, 500.0, 800.0, 1010.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0)
ZIPF_D_GRANDE = (500.0, 800.0, 1010.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0)
ZIPF_L_PADRAO = (1.0, 1.5, 1.7, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
ZIPF_FALLBACK = (1010.0, 2.0)

# Metadados NOAA e visualização
UNIDADES = {
    "Água oceânica": "pieces/m³",
    "Areia de praia": "pieces/m³",
    "Sedimento oceânico": "pieces/kg",
    "Nurdle Patrol": "pieces/10 min",
}
REDES_SUPERFICIE = ("Neuston net", "Manta net", "AVANI net")
EIXOS_X = {"total_population": "População da Bacia", "area_km2": "Área da Bacia (km²)"}
CORES = ("#0284c7", "#d97706", "#7c3aed", "#059669")

COLUNAS_BACIA = (
    "land_cluster_id", "landmass_name", "sink_id", "basin_id", "latitude", "longitude",
    "elevation", "pixel_count", "area_km2", "area_fraction", "cumulative_fraction",
    "in_top_95pct", "urban_population", "total_population", "urban_pixels",
    "total_pixels", "urban_fraction", "d_star_opt", "l_opt_km", "macroregiao",
)
