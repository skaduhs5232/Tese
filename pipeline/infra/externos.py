"""Adaptadores isolados para os algoritmos oficiais em `externos/` (GebcoCutter, IPBA, CCA)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from ..config import REPO_ROOT

_DIR_GEBCO = REPO_ROOT / "externos" / "GebcoCutter"
_DIR_IPBA = REPO_ROOT / "externos" / "IPBA"
_DIR_CCA = REPO_ROOT / "externos" / "CCA" / "cca"

for _d in (_DIR_GEBCO, _DIR_IPBA, _DIR_CCA):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from city_clustering import burn as _cca_burn
from drainage_basins import SetDrainageBasins as _SetDrainageBasins
from heap import Heap as _Heap
from lattice import Lattice as _Lattice
from percolation.percolation import Percolation as _Percolation


def _carregar_read_geotiff():
    spec = importlib.util.spec_from_file_location("gebcocutter_geotiff", _DIR_GEBCO / "geotiff" / "geotiff.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ReadGeoTiff


_ReadGeoTiff = _carregar_read_geotiff()


def ler_geotiff_gebco(gebco_path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """Lê o GeoTIFF GEBCO usando `ReadGeoTiff` oficial de `externos/GebcoCutter`."""
    gdal.UseExceptions()
    return _ReadGeoTiff(str(gebco_path))


def percolar_terra(mascara_terra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmenta massas conectadas via `Percolation.site_percolation()` (externos/GebcoCutter)."""
    perc = _Percolation(mascara_terra)
    perc.rank = np.zeros(perc.n, dtype=np.uint32)  # evita overflow uint8 na grade global
    perc.site_percolation()

    label = perc.parent.reshape(perc.nrows, perc.ncols)
    vals, sizes = np.unique(label[mascara_terra], return_counts=True)
    ordem = np.argsort(sizes)[::-1]
    return label, vals[ordem], sizes[ordem]


def executar_ipba(
    dem: np.ndarray,
    sink_grid: np.ndarray,
    h_star: float = 0.0,
    seed: int = 42,
    ispbc: bool = False,
) -> np.ndarray:
    """Executa rodada única do IPBA usando `Lattice`, `Heap` e `SetDrainageBasins` (externos/IPBA)."""
    nrows, ncols = dem.shape
    terra = dem > h_star
    nodata = np.float32(-32768.0)

    random.seed(seed)
    arr0 = np.where(terra, dem, nodata).astype(np.float32).ravel()
    arr1 = sink_grid.ravel()

    lattice = _Lattice(nrows, ncols, arr0)
    for k in range(lattice.n):
        if not terra.flat[k]:
            lattice.sites[k].height = nodata
            lattice.sites[k].sigma = np.uint32(0)
        if arr1[k] > 0:
            lattice.sites[k].label = np.uint32(arr1[k])

    _SetDrainageBasins(lattice, _Heap(lattice.n), ispbc)

    basin_grid = np.zeros((nrows, ncols), dtype=np.uint32)
    for k in range(lattice.n):
        basin_grid.flat[k] = lattice.sites[k].status
    return basin_grid


def agrupar_cca(lats: np.ndarray, lons: np.ndarray, l_cutoff: float) -> np.ndarray:
    """Agrupa pixels ativos via `burn` (externos/CCA); L = 0 mantém sítios isolados."""
    if l_cutoff == 0.0:
        return np.arange(len(lats), dtype=np.int64)
    return _cca_burn(lats, lons, l_cutoff)
