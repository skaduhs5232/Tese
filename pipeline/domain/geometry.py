"""Geometria esférica: projeção 3D, distância de corda e área por triângulos esféricos."""

from __future__ import annotations

import numpy as np

from ..config import EARTH_RADIUS_KM


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Converte latitudes e longitudes (graus) em vetores unitários 3D na esfera."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def km_para_corda(dist_km: float, raio_km: float = EARTH_RADIUS_KM) -> float:
    """Converte distância geodésica (km) em distância de corda na esfera unitária."""
    return float(2.0 * np.sin((dist_km / raio_km) / 2.0))


def _haversine_unit(phi_i: np.ndarray, phi_j: np.ndarray, dlambda: float) -> np.ndarray:
    h = np.sin(0.5 * (phi_j - phi_i)) ** 2 + np.cos(phi_i) * np.cos(phi_j) * (np.sin(0.5 * dlambda) ** 2)
    return 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _area_lhuilier(a: np.ndarray, b: np.ndarray, c: np.ndarray, raio_km: float) -> np.ndarray:
    s = 0.5 * (a + b + c)
    prod = (
        np.tan(0.5 * s)
        * np.tan(0.5 * np.maximum(0.0, s - a))
        * np.tan(0.5 * np.maximum(0.0, s - b))
        * np.tan(0.5 * np.maximum(0.0, s - c))
    )
    return 4.0 * (raio_km**2) * np.arctan(np.sqrt(np.maximum(0.0, prod)))


def area_triangulos_esfericos(
    lats_centro_deg: np.ndarray,
    dlat_deg: float,
    dlon_deg: float,
    raio_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Área exata de cada pixel (km²) via dois triângulos esféricos (L'Huilier + Haversine)."""
    lat1 = np.radians(lats_centro_deg - 0.5 * abs(dlat_deg))
    lat2 = np.radians(lats_centro_deg + 0.5 * abs(dlat_deg))
    dlon = float(np.radians(abs(dlon_deg)))

    sul = _haversine_unit(lat1, lat1, dlon)
    norte = _haversine_unit(lat2, lat2, dlon)
    merid = _haversine_unit(lat1, lat2, 0.0)
    diag = _haversine_unit(lat1, lat2, dlon)
    return _area_lhuilier(sul, merid, diag, raio_km) + _area_lhuilier(norte, merid, diag, raio_km)
