"""
pipeline
--------
Pacote de processamento geoespacial e alometria:
  - ipba_runner: Extração de bacias e sinks costeiros via IPBA simplificado e área por triângulos esféricos (L'Huilier).
  - cca_calibrator: Varredura 10x10, calibração via Lei de Zipf e extração de manchas CCA.
  - spatial_matcher: Particionamento de manchas por bacia, ingestão NOAA e raio exclusivo.
"""
from .ipba_runner import (
    crop_gebco,
    extract_coastal_sinks,
    run_ipba,
    filter_basins_by_area,
    spherical_triangle_row_areas,
)
from .cca_calibrator import crop_gpw, calibrate_zipf_grid, run_calibrated_cca, plot_zipf_calibration
from .spatial_matcher import (
    split_urban_clusters_by_basins,
    load_noaa_nurdle_patrol,
    match_sinks_with_exclusive_radius,
    match_sinks_with_individual_radius,
    evaluate_radius_sensitivity,
)

__all__ = [
    "crop_gebco",
    "extract_coastal_sinks",
    "run_ipba",
    "filter_basins_by_area",
    "spherical_triangle_row_areas",
    "crop_gpw",
    "calibrate_zipf_grid",
    "run_calibrated_cca",
    "plot_zipf_calibration",
    "split_urban_clusters_by_basins",
    "load_noaa_nurdle_patrol",
    "match_sinks_with_exclusive_radius",
    "match_sinks_with_individual_radius",
    "evaluate_radius_sensitivity",
]
