"""
area_esferica.py
----------------
Implementação manual em NumPy das equações geodésicas do artigo:
Oliveira, E. A. et al. (2019). "A universal approach for drainage basins",
Scientific Reports, 9, 9845. https://doi.org/10.1038/s41598-019-46165-0

Equações implementadas (Seção Methods, págs. 7-8):
  - Eq. (3): Área de um triângulo esférico de lados (a, b, c) pelo Teorema de L'Huilier.
  - Eq. (4): Distância geodésica (grande círculo) entre dois pontos via fórmula de Haversine.
  - Área de cada sítio (pixel) pela composição de dois triângulos esféricos.
"""

import numpy as np

# Raio equatorial terrestre adotado em Oliveira et al. (2019), pág. 7
R_EARTH_KM = 6378.137
R_MOON_KM = 1737.4
R_MARS_KM = 3396.19


def haversine_rad(phi_i, lam_i, phi_j, lam_j, R_k: float = R_EARTH_KM):
    """
    Equação (4) de Oliveira et al. (2019):
    Distância de grande círculo d_ij entre os pontos (phi_i, lam_i) e (phi_j, lam_j)
    medidos em radianos sobre uma esfera de raio R_k (km).

    d_ij = 2 * R_k * arcsin( sqrt( sin^2(dphi/2) + cos(phi_i)*cos(phi_j)*sin^2(dlam/2) ) )
    """
    dphi = phi_j - phi_i
    dlam = lam_j - lam_i
    sin_dphi2 = np.sin(dphi * 0.5)
    sin_dlam2 = np.sin(dlam * 0.5)
    arg = sin_dphi2 * sin_dphi2 + np.cos(phi_i) * np.cos(phi_j) * (sin_dlam2 * sin_dlam2)
    arg = np.clip(arg, 0.0, 1.0)
    return 2.0 * R_k * np.arcsin(np.sqrt(arg))


def haversine_deg(lon_i, lat_i, lon_j, lat_j, R_k: float = R_EARTH_KM):
    """
    Versão em graus decimais da Equação (4) de Haversine.
    Aceita escalares ou arrays NumPy compatíveis via broadcasting.
    """
    phi_i = np.radians(lat_i)
    lam_i = np.radians(lon_i)
    phi_j = np.radians(lat_j)
    lam_j = np.radians(lon_j)
    return haversine_rad(phi_i, lam_i, phi_j, lam_j, R_k=R_k)


def spherical_triangle_area(a, b, c, R_k: float = R_EARTH_KM):
    """
    Equação (3) de Oliveira et al. (2019) — Fórmula de L'Huilier:
    Área de um triângulo esférico com lados geodésicos a, b, c (em km)
    sobre uma esfera de raio R_k (em km).

    A = 4 * R_k^2 * arctan( sqrt( tan(s/2) * tan(s_a/2) * tan(s_b/2) * tan(s_c/2) ) )
    onde:
      s   = (a/R_k + b/R_k + c/R_k) / 2
      s_a = s - a/R_k
      s_b = s - b/R_k
      s_c = s - c/R_k
    """
    a_r = a / R_k
    b_r = b / R_k
    c_r = c / R_k

    s = 0.5 * (a_r + b_r + c_r)
    s_a = np.maximum(0.0, s - a_r)
    s_b = np.maximum(0.0, s - b_r)
    s_c = np.maximum(0.0, s - c_r)

    prod = np.tan(0.5 * s) * np.tan(0.5 * s_a) * np.tan(0.5 * s_b) * np.tan(0.5 * s_c)
    prod = np.maximum(0.0, prod)
    return 4.0 * (R_k * R_k) * np.arctan(np.sqrt(prod))


def pixel_area_by_latitude(lat_centers_deg, dlon_deg: float, dlat_deg: float,
                           R_k: float = R_EARTH_KM) -> np.ndarray:
    """
    Calcula a área (em km^2) de cada pixel de uma grade regular em coordenadas geográficas
    pela composição de DOIS triângulos esféricos (Oliveira et al., 2019, pág. 7):

    Para um sítio centrado em latitude phi com passos (dlon_deg, dlat_deg):
      V1 = (phi_s, 0),    V2 = (phi_s, dlon)
      V3 = (phi_n, 0),    V4 = (phi_n, dlon)

    Triângulo 1 (V1, V2, V4):
      - lado sul     a1 = d(V1, V2)
      - lado leste   b1 = d(V2, V4)
      - diagonal     c  = d(V1, V4)

    Triângulo 2 (V1, V3, V4):
      - lado oeste   a2 = d(V1, V3) = b1
      - lado norte   b2 = d(V3, V4)
      - diagonal     c  = d(V1, V4)

    Retorna um array 1D do mesmo tamanho de `lat_centers_deg` com a área de 1 pixel em cada linha.
    """
    lat_c = np.asarray(lat_centers_deg, dtype=np.float64)
    half_dlat = 0.5 * abs(float(dlat_deg))
    dlon = np.radians(abs(float(dlon_deg)))

    phi_s = np.radians(np.clip(lat_c - half_dlat, -90.0, 90.0))
    phi_n = np.radians(np.clip(lat_c + half_dlat, -90.0, 90.0))
    lam_0 = np.zeros_like(phi_s)
    lam_1 = np.full_like(phi_s, dlon)

    # Lados geodésicos via Haversine (Eq. 4)
    a_south = haversine_rad(phi_s, lam_0, phi_s, lam_1, R_k=R_k)
    a_north = haversine_rad(phi_n, lam_0, phi_n, lam_1, R_k=R_k)
    b_merid = haversine_rad(phi_s, lam_0, phi_n, lam_0, R_k=R_k)
    c_diag = haversine_rad(phi_s, lam_0, phi_n, lam_1, R_k=R_k)

    # Composição de dois triângulos esféricos (Eq. 3)
    area_t1 = spherical_triangle_area(a_south, b_merid, c_diag, R_k=R_k)
    area_t2 = spherical_triangle_area(b_merid, a_north, c_diag, R_k=R_k)
    return area_t1 + area_t2


def pixel_area_grid_2d(lat_centers_deg, ncols: int, dlon_deg: float, dlat_deg: float,
                       R_k: float = R_EARTH_KM) -> np.ndarray:
    """
    Retorna uma visão 2D (nrows, ncols) sem duplicar memória (via np.broadcast_to)
    com a área esférica exata (km^2) de cada pixel da grade.
    """
    row_areas = pixel_area_by_latitude(lat_centers_deg, dlon_deg, dlat_deg, R_k=R_k)
    return np.broadcast_to(row_areas[:, None], (row_areas.size, int(ncols)))


def somar_areas_por_bacia(basin_labels: np.ndarray, row_areas_km2: np.ndarray):
    """
    Calcula a área total (em km^2) e a contagem de pixels de cada bacia (rótulo > 0)
    usando exclusivamente operações vetoriais do NumPy (sem scipy.ndimage).

    Parâmetros:
      basin_labels : array 2D (nrows, ncols) inteiro com rótulos das bacias (0 = oceano/fora).
      row_areas_km2: array 1D (nrows,) ou 2D (nrows, ncols) com a área esférica de cada pixel.

    Retorna:
      basin_ids : array 1D com os IDs únicos de bacia (> 0)
      area_px   : array 1D com o número de pixels em cada bacia
      area_km2  : array 1D com a área esférica total (km^2) de cada bacia
    """
    nrows, ncols = basin_labels.shape
    if row_areas_km2.ndim == 1:
        pesos = np.broadcast_to(row_areas_km2[:, None], (nrows, ncols)).ravel()
    else:
        pesos = row_areas_km2.ravel()

    flat_labels = basin_labels.ravel()
    mask = flat_labels > 0
    ativos = flat_labels[mask]
    pesos_ativos = pesos[mask]

    basin_ids, inv = np.unique(ativos, return_inverse=True)
    area_px = np.bincount(inv)
    area_km2 = np.bincount(inv, weights=pesos_ativos)
    return basin_ids, area_px, area_km2
