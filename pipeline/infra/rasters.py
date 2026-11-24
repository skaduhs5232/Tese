"""Operações de leitura e recorte nos rasters GEBCO e GPW v4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from ..config import GEBCO_TIF, GEBCO_TO_GPW_SCALE, GEBCO_TOTAL_COLS, GPW_TIF
from ..domain.geometry import area_triangulos_esfericos


def amostrar_cota_gebco(
    lats: np.ndarray,
    lons: np.ndarray,
    gebco_path: str | Path = GEBCO_TIF,
) -> np.ndarray:
    """Retorna a elevação GEBCO (m) correspondente a cada coordenada (lat, lon)."""
    gdal.UseExceptions()
    ds = gdal.Open(str(gebco_path), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    dem = ds.GetRasterBand(1).ReadAsArray()
    cols = np.clip(np.floor((lons - gt[0]) / gt[1]).astype(int), 0, ds.RasterXSize - 1)
    rows = np.clip(np.floor((gt[3] - lats) / abs(gt[5])).astype(int), 0, ds.RasterYSize - 1)
    cota = dem[rows, cols].astype(float)
    del ds
    return cota


def recortar_porcao_gebco(
    info: dict[str, Any],
    arr: np.ndarray,
    label: np.ndarray,
    val: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int, int, np.ndarray]:
    """Recorta uma porção de terra (`val`) com periodicidade longitudinal (±180°)."""
    _, ncols = label.shape
    cluster = label == val

    empty_cols = np.flatnonzero(~cluster.any(axis=0))
    i0 = int(empty_cols[0]) if empty_cols.size else 0

    ys, xs = np.nonzero(cluster)
    xs_shifted = (xs - i0) % ncols
    j_min, j_max = int(ys.min()), int(ys.max())
    i_min, i_max = int(xs_shifted.min()), int(xs_shifted.max())
    cols = (np.arange(i_min, i_max + 1) + i0) % ncols

    cluster_crop = cluster[j_min : j_max + 1, cols]
    nodata = info["NoDataValue"] if info.get("NoDataValue") is not None else -32768
    arr_crop = np.where(cluster_crop, arr[j_min : j_max + 1, cols], nodata)

    gt0 = info["GeoTransform"]
    x_px = int(cols[0])
    gt_crop = (
        gt0[0] + x_px * gt0[1] + j_min * gt0[2], gt0[1], gt0[2],
        gt0[3] + x_px * gt0[4] + j_min * gt0[5], gt0[4], gt0[5],
    )
    info_crop = {
        "RasterYSize": arr_crop.shape[0],
        "RasterXSize": arr_crop.shape[1],
        "GeoTransform": gt_crop,
        "Projection": info["Projection"],
        "NoDataValue": nodata,
    }
    return arr_crop, cluster_crop, info_crop, j_min, j_max, cols


def recortar_gpw_porcao(
    j_min: int,
    j_max: int,
    cols_gebco: np.ndarray,
    mask_gebco: np.ndarray,
    gpw_path: str | Path = GPW_TIF,
    gebco_total_cols: int = GEBCO_TOTAL_COLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float, float, float], float]:
    """Recorta o GPW v4 alinhado 4×4 à porção GEBCO (com wrap ±180°) e mascara à massa de terra."""
    gdal.UseExceptions()
    ds = gdal.Open(str(gpw_path), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    s = GEBCO_TO_GPW_SCALE
    r0, r1 = s * int(j_min), s * (int(j_max) + 1)
    y_size = r1 - r0
    c_start, c_end = int(cols_gebco[0]), int(cols_gebco[-1])

    if c_end >= c_start and len(cols_gebco) == (c_end - c_start + 1):
        c0, x_size = s * c_start, s * len(cols_gebco)
        raw = band.ReadAsArray(c0, r0, x_size, y_size).astype(np.float32)
        gpw_cols = np.arange(c0, c0 + x_size)
    else:
        n1, n2 = gebco_total_cols - c_start, c_end + 1
        p1 = band.ReadAsArray(s * c_start, r0, s * n1, y_size).astype(np.float32)
        p2 = band.ReadAsArray(0, r0, s * n2, y_size).astype(np.float32)
        raw = np.concatenate([p1, p2], axis=1)
        x_size = raw.shape[1]
        gpw_cols = np.concatenate([np.arange(s * c_start, s * gebco_total_cols), np.arange(0, s * n2)])
    del ds

    pop = np.where((raw == nodata) | (raw < 0), 0.0, raw).astype(np.float32)
    del raw
    mask_gpw = np.repeat(np.repeat(mask_gebco.astype(bool), s, axis=0), s, axis=1)
    pop[~mask_gpw] = 0.0
    del mask_gpw

    lons_1d = gt[0] + (gpw_cols + 0.5) * gt[1]
    lats_1d = gt[3] + (np.arange(r0, r1) + 0.5) * gt[5]
    area_1d = area_triangulos_esfericos(lats_1d, abs(gt[5]), gt[1]).astype(np.float32)
    pixel_area = np.broadcast_to(area_1d[:, None], (y_size, x_size))
    new_gt = (gt[0] + s * c_start * gt[1], gt[1], gt[2], gt[3] + r0 * gt[5], gt[4], gt[5])

    return pop, pixel_area, lats_1d, lons_1d, new_gt, float(pop.sum())
