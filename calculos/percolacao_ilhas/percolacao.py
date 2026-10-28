"""
percolacao.py
-------------
Implementação em NumPy + Numba da percolação de sítios (Height Clusters h_i > h*)
descrita em Oliveira et al. (2019), "A universal approach for drainage basins" (págs. 2-3),
e adaptada de `notebooks/gebco_cluster_dvider/percolation/percolation.py`.

Correções em relação à versão original:
  1. `rank` em uint16/int32 (corrige o RuntimeWarning: overflow encountered in scalar add em uint8).
  2. `find_set` iterativo com compressão de caminho (evita RecursionError em grades grandes e roda em C via Numba).
  3. Condições de contorno periódicas horizontais (leste-oeste) e polares (norte-sul) conforme pág. 3 do artigo.
  4. Cálculo sem bug do GeoTransform e caixa delimitadora com tratamento de cruzamento do antimeridiano (±180°).
"""

import numpy as np
from numba import njit


@njit
def _find_set(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    curr = x
    while curr != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt
    return root


@njit
def _union(parent, rank, x, y):
    rx = _find_set(parent, x)
    ry = _find_set(parent, y)
    if rx == ry:
        return
    if rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[rx] = ry
        if rank[rx] == rank[ry]:
            rank[ry] += 1


@njit
def _site_percolation_numba(mask_bool, periodic_lon=True, periodic_poles=True):
    nrows, ncols = mask_bool.shape
    n = nrows * ncols
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int32)
    half = ncols // 2

    for j in range(nrows):
        row_offset = j * ncols
        for i in range(ncols):
            if mask_bool[j, i]:
                k = row_offset + i
                # Vizinho à direita (com contorno periódico horizontal opcional)
                if i + 1 < ncols:
                    if mask_bool[j, i + 1]:
                        _union(parent, rank, k, k + 1)
                elif periodic_lon and mask_bool[j, 0]:
                    _union(parent, rank, k, row_offset)

                # Vizinho abaixo
                if j + 1 < nrows and mask_bool[j + 1, i]:
                    _union(parent, rank, k, k + ncols)

    # Condições de contorno não-convencionais nos polos (Oliveira et al., 2019, pág. 3)
    if periodic_poles and half > 0:
        last_offset = (nrows - 1) * ncols
        for i in range(half):
            if mask_bool[0, i] and mask_bool[0, i + half]:
                _union(parent, rank, i, i + half)
            if mask_bool[nrows - 1, i] and mask_bool[nrows - 1, i + half]:
                _union(parent, rank, last_offset + i, last_offset + i + half)

    for k in range(n):
        if mask_bool[k // ncols, k % ncols]:
            parent[k] = _find_set(parent, k)
        else:
            parent[k] = -1

    return parent.reshape(nrows, ncols)


class Percolation:
    """
    Classe compatível com `notebooks/gebco_cluster_dvider/percolation/percolation.py`,
    acelerada com Numba e livre de overflow.
    """

    def __init__(self, arr: np.ndarray, periodic_lon: bool = True, periodic_poles: bool = True):
        self.arr = np.asarray(arr, dtype=np.bool_)
        self.nrows, self.ncols = self.arr.shape
        self.n = self.nrows * self.ncols
        self.periodic_lon = periodic_lon
        self.periodic_poles = periodic_poles
        self.parent = None

    def site_percolation(self):
        labels_2d = _site_percolation_numba(self.arr, self.periodic_lon, self.periodic_poles)
        self.parent = labels_2d.ravel()
        return labels_2d


def extrair_recorte_cluster(arr: np.ndarray, cluster_mask: np.ndarray, geo_info: dict, pad: int = 1):
    """
    Extrai o sub-raster retangular mínimo de um cluster (ou conjunto de clusters)
    tratando corretamente o cruzamento de longitude (±180°) e corrigindo o bug
    do `y_line = j_min` presente no `01_clustering.ipynb` original.

    Inclui uma margem (`pad=1` pixel) de oceano (`NoDataValue`) ao redor da caixa
    delimitadora para garantir que o flood-fill oceânico do IPBA envolva 100% do
    litoral de todas as ilhas e continentes do recorte.

    Parâmetros:
      arr          : array 2D de elevações (nrows, ncols).
      cluster_mask : array 2D booleano indicando os pixels pertencentes ao território.
      geo_info     : dicionário com 'GeoTransform', 'Projection', 'NoDataValue'.
      pad          : margem em pixels de NoData ao redor do recorte (padrão: 1).

    Retorna:
      arr_cropped  : array 2D recortado com `NoDataValue` nos pixels fora do território.
      info_cropped : dicionário metadados GeoTIFF atualizado para o recorte.
      bbox_deg     : tupla (lon_min, lat_min, lon_max, lat_max) em graus decimais.
    """
    nrows, ncols = arr.shape
    empty_cols = np.flatnonzero(~cluster_mask.any(axis=0))
    i0 = int(empty_cols[0]) if empty_cols.size else 0

    ys, xs = np.nonzero(cluster_mask)
    xs_shifted = (xs - i0) % ncols

    j_min, j_max = int(ys.min()), int(ys.max())
    i_min, i_max = int(xs_shifted.min()), int(xs_shifted.max())
    cols = (np.arange(i_min, i_max + 1) + i0) % ncols

    cluster_cropped = cluster_mask[j_min:j_max + 1, cols]
    arr_sub = arr[j_min:j_max + 1, cols]

    nodata = geo_info.get("NoDataValue", -32768)
    if nodata is None:
        nodata = -32768
    arr_core = np.where(cluster_cropped, arr_sub, nodata).astype(arr.dtype)

    if pad > 0:
        arr_cropped = np.full(
            (arr_core.shape[0] + 2 * pad, arr_core.shape[1] + 2 * pad),
            nodata,
            dtype=arr.dtype,
        )
        arr_cropped[pad:-pad, pad:-pad] = arr_core
    else:
        arr_cropped = arr_core

    height, width = arr_cropped.shape
    _gt = geo_info["GeoTransform"]

    # CORREÇÃO DO BUG ORIGINAL: y_line é j_min (índice de linha), x_pixel é cols[0] (índice de coluna)
    y_line = j_min - pad
    x_pixel = int(cols[0]) - pad

    x_geo = _gt[0] + x_pixel * _gt[1] + y_line * _gt[2]
    y_geo = _gt[3] + x_pixel * _gt[4] + y_line * _gt[5]
    gt = (x_geo, _gt[1], _gt[2], y_geo, _gt[4], _gt[5])

    lon_min = x_geo
    lon_max = x_geo + width * _gt[1]
    lat_max = y_geo
    lat_min = y_geo + height * _gt[5]

    info_cropped = {
        "RasterYSize": int(height),
        "RasterXSize": int(width),
        "GeoTransform": gt,
        "Projection": geo_info.get("Projection", ""),
        "NoDataValue": nodata,
    }
    return arr_cropped, info_cropped, (lon_min, lat_min, lon_max, lat_max)


@njit
def identificar_oceano_e_sinks(dem: np.ndarray, nodata: float = -9999.0, h_star: float = 0.0):
    """
    Identifica em NumPy/Numba puro:
      1. `sea`   : booleano 2D (True apenas para pixels com dem == nodata ou pixels com dem <= h_star
                   CONECTADOS por 4-vizinhança à borda oceânica / máscara oceânica externa).
                   Depressões continentais internas com h <= 0 (não conectadas ao mar aberto,
                   ex: Salton Sea, Death Valley, planícies costeiras internas, Mar Cáspio/Morto)
                   permanecem com `sea = False` (terra firme), garantindo que o IPBA drene
                   toda a bacia até a linha de costa real.
      2. `sinks` : int32 2D (1..N_b nos pixels de terra que têm pelo menos um 4-vizinho em `sea`;
                   0 nos demais pixels). Conforme Oliveira et al. (2019), pág. 2:
                   "the sinks S_k (k=1..N_b) are all the N_b border sites of the height clusters,
                    i.e. the sea shore on Earth."
    """
    nrows, ncols = dem.shape
    sea = np.zeros((nrows, ncols), dtype=np.bool_)
    queue_r = np.empty(nrows * ncols, dtype=np.int32)
    queue_c = np.empty(nrows * ncols, dtype=np.int32)
    head = 0
    tail = 0

    for r in range(nrows):
        for c in range(ncols):
            v = dem[r, c]
            is_invalid = np.isnan(v) or (v == nodata)
            is_border_low = (r == 0 or r == nrows - 1 or c == 0 or c == ncols - 1) and (v <= h_star)
            if is_invalid or is_border_low:
                sea[r, c] = True
                queue_r[tail] = r
                queue_c[tail] = c
                tail += 1

    dr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dc = np.array([0, 0, -1, 1], dtype=np.int32)
    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        for d in range(4):
            nr = r + dr[d]
            nc = c + dc[d]
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if not sea[nr, nc] and dem[nr, nc] <= h_star:
                    sea[nr, nc] = True
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    sinks = np.zeros((nrows, ncols), dtype=np.int32)
    sink_id = 0
    for r in range(nrows):
        for c in range(ncols):
            if not sea[r, c]:
                is_coastal = False
                for d in range(4):
                    nr = r + dr[d]
                    nc = c + dc[d]
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        if sea[nr, nc]:
                            is_coastal = True
                            break
                if is_coastal:
                    sink_id += 1
                    sinks[r, c] = sink_id

    return sea, sinks, sink_id

