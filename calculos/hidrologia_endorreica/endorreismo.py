"""
endorreismo.py
--------------
Identificação de bacias endorreicas (sem desembocadura oceânica) sobre um DEM continental,
complementar ao IPBA (Oliveira et al., 2019). O IPBA, por construção, faz toda depressão
transbordar até um sink costeiro; aqui separamos as depressões interiores reais (Cáspio,
Aral, Chade, Great Basin, Eyre, Tarim, Altiplano...) que não devem contar como emissoras
de lixo marinho.

Algoritmo (4-vizinhança, a mesma do IPBA):
  1. Priority-flood (Barnes et al., 2014) semeado na linha de costa -> DEM preenchido F.
  2. Componentes conexos de F - h > eps_m são depressões; candidatas se área >= area_min_km2
     e profundidade máxima >= prof_min_m.
  3. Watershed por priority-flood semeado na costa (rótulo 0) e no ponto mais baixo de cada
     candidata: cada pixel recebe o rótulo da semente que o alcança no menor nível de inundação.
  4. Uma candidata só é aceita como endorreica se a sua área de captação contém um sistema
     endorreico documentado (`SISTEMAS_ENDORREICOS`). A 2' o DEM "fecha" gargantas e represas
     (Congo em Malebo, Níger em Lokoja, Portas de Ferro do Danúbio, Três Gargantas, reservatórios
     do Volga) e lagos com batimetria (Grandes Lagos, Tanganica, Baikal) também viram depressões
     profundas; sem a âncora documental esses rios exorreicos seriam classificados como endorreicos.
  5. Watershed final semeado apenas na costa e nas candidatas aceitas.
"""

import numpy as np
from numba import njit


# Lagos terminais / fundos de depressões endorreicas documentadas (lon, lat), cf. HydroSHEDS
# (Lehner & Grill, 2013) e Wang et al. (2018, Nature Geoscience 11:926) para o inventário global.
SISTEMAS_ENDORREICOS = {
    "Caspian Sea": (50.5, 41.5),
    "Aral Sea": (59.8, 45.0),
    "Sarygamysh Lake": (57.4, 41.9),
    "Lake Balkhash": (74.5, 46.5),
    "Lake Alakol / Ebinur": (82.0, 45.5),
    "Issyk-Kul": (77.3, 42.4),
    "Lake Tengiz": (69.0, 50.4),
    "Turgai / Shalkar-Teniz": (62.5, 47.8),
    "Sarysu terminal lakes": (66.0, 45.3),
    "Chu terminal lakes": (68.6, 44.7),
    "Inner Mongolia Plateau (Dali Nur)": (116.6, 43.3),
    "Tarim / Lop Nur": (90.3, 40.3),
    "Qaidam Basin": (94.0, 37.5),
    "Tibetan Plateau Lakes": (88.5, 31.8),
    "Gobi / Juyan Lake": (101.2, 42.3),
    "Mongolian Great Lakes (Uvs Nuur)": (92.8, 50.3),
    "Dasht-e Kavir": (54.0, 34.6),
    "Dasht-e Lut": (58.5, 30.5),
    "Sistan / Hamun": (61.3, 31.0),
    "Lake Urmia": (45.4, 37.7),
    "Lake Van": (42.8, 38.6),
    "Dead Sea": (35.5, 31.5),
    "Lake Chad": (14.0, 13.0),
    "Bodele Depression": (17.5, 16.8),
    "Qattara Depression": (27.0, 29.8),
    "Chott Melrhir / Chott el Djerid": (6.3, 34.2),
    "Sebkha Mekerrhane (Tidikelt)": (1.3, 26.4),
    "Murzuq Basin (Fezzan)": (13.8, 27.0),
    "Kufra Basin": (22.5, 24.5),
    "Taoudenni Basin": (-4.0, 22.7),
    "Afar / Danakil": (40.5, 14.0),
    "Lake Turkana": (36.0, 3.5),
    "Okavango / Makgadikgadi": (25.5, -20.7),
    "Etosha Pan": (16.3, -18.8),
    "Lake Eyre": (137.3, -28.4),
    "Lake Torrens": (137.9, -31.0),
    "Great Salt Lake / Bonneville": (-112.6, 41.1),
    "Carson Sink / Lahontan": (-118.5, 39.8),
    "Death Valley": (-116.85, 36.25),
    "Bolson de Mapimi": (-103.4, 27.0),
    "Laguna de Guzman (Chihuahua)": (-107.3, 31.2),
    "Altiplano (Poopo / Uyuni)": (-67.2, -19.3),
    "Laguna Mar Chiquita": (-62.6, -30.6),
}


# Gargantas de rios exorreicos que o DEM a 2' "fecha" dentro de uma depressão endorreica
# documentada: recebem uma semente costeira (rótulo 0), equivalente a abrir a garganta.
SAIDAS_EXORREICAS_FORCADAS = {
    "Upper Zambezi above Victoria Falls (Katima Mulilo)": (24.4, -17.5),
}


@njit
def _push(keys, items, m, key, item):
    i = m
    keys[i] = key
    items[i] = item
    while i > 0:
        p = (i - 1) // 2
        if keys[p] <= keys[i]:
            break
        tk = keys[p]
        keys[p] = keys[i]
        keys[i] = tk
        ti = items[p]
        items[p] = items[i]
        items[i] = ti
        i = p
    return m + 1


@njit
def _pop(keys, items, m):
    key = keys[0]
    item = items[0]
    m -= 1
    keys[0] = keys[m]
    items[0] = items[m]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        s = i
        if l < m and keys[l] < keys[s]:
            s = l
        if r < m and keys[r] < keys[s]:
            s = r
        if s == i:
            break
        tk = keys[s]
        keys[s] = keys[i]
        keys[i] = tk
        ti = items[s]
        items[s] = items[i]
        items[i] = ti
        i = s
    return key, item, m


@njit
def _eh_costa(sea, r, c, nrows, ncols):
    if r == 0 or c == 0 or r == nrows - 1 or c == ncols - 1:
        return True
    return sea[r - 1, c] or sea[r + 1, c] or sea[r, c - 1] or sea[r, c + 1]


@njit
def _priority_flood(dem, sea, seed_label, label_out, transbordo, ref):
    """
    Priority-flood genérico. `seed_label[r, c] >= 0` marca sementes (inseridas com chave = dem).
    Retorna o DEM preenchido e preenche `label_out` com o rótulo da semente que inundou cada pixel.

    `transbordo[k]` é o nível de transbordo da semente k (inf para a costa) e `ref` é o DEM
    preenchido a partir da costa: acima do transbordo, k não toma pixels com ref < nível atual,
    pois esses pixels têm uma saída mais baixa do outro lado da sela (o vale vizinho).
    """
    nrows, ncols = dem.shape
    n = nrows * ncols
    filled = np.full((nrows, ncols), np.nan)
    keys = np.empty(n, dtype=np.float64)
    items = np.empty(n, dtype=np.int64)
    m = 0
    for r in range(nrows):
        for c in range(ncols):
            if not sea[r, c] and seed_label[r, c] >= 0:
                filled[r, c] = dem[r, c]
                label_out[r, c] = seed_label[r, c]
                m = _push(keys, items, m, dem[r, c], r * ncols + c)

    dr = (-1, 0, 0, 1)
    dc = (0, -1, 1, 0)
    while m > 0:
        lvl, it, m = _pop(keys, items, m)
        r = it // ncols
        c = it - r * ncols
        k = label_out[r, c]
        for d in range(4):
            nr = r + dr[d]
            nc = c + dc[d]
            if 0 <= nr < nrows and 0 <= nc < ncols and not sea[nr, nc] and np.isnan(filled[nr, nc]):
                if lvl > transbordo[k] and ref[nr, nc] < lvl - 1e-6:
                    continue
                v = dem[nr, nc]
                f = v if v > lvl else lvl
                filled[nr, nc] = f
                label_out[nr, nc] = k
                m = _push(keys, items, m, f, nr * ncols + nc)
    return filled


@njit
def _sementes_costa(sea):
    nrows, ncols = sea.shape
    seeds = np.full((nrows, ncols), -1, dtype=np.int32)
    for r in range(nrows):
        for c in range(ncols):
            if not sea[r, c] and _eh_costa(sea, r, c, nrows, ncols):
                seeds[r, c] = 0
    return seeds


def priority_flood_preenchimento(dem: np.ndarray, sea_mask: np.ndarray) -> np.ndarray:
    """DEM preenchido F (NaN no mar) por priority-flood semeado na linha de costa."""
    dem64 = np.ascontiguousarray(dem, dtype=np.float64)
    sea = np.ascontiguousarray(sea_mask, dtype=np.bool_)
    lab = np.zeros(dem64.shape, dtype=np.int32)
    return _priority_flood(dem64, sea, _sementes_costa(sea), lab, np.array([np.inf]), dem64)


@njit
def _rotular_componentes(mask):
    nrows, ncols = mask.shape
    lab = np.zeros((nrows, ncols), dtype=np.int32)
    q = np.empty(nrows * ncols, dtype=np.int64)
    n_lab = 0
    dr = (-1, 0, 0, 1)
    dc = (0, -1, 1, 0)
    for r0 in range(nrows):
        for c0 in range(ncols):
            if mask[r0, c0] and lab[r0, c0] == 0:
                n_lab += 1
                lab[r0, c0] = n_lab
                head = 0
                tail = 1
                q[0] = r0 * ncols + c0
                while head < tail:
                    it = q[head]
                    head += 1
                    r = it // ncols
                    c = it - r * ncols
                    for d in range(4):
                        nr = r + dr[d]
                        nc = c + dc[d]
                        if 0 <= nr < nrows and 0 <= nc < ncols and mask[nr, nc] and lab[nr, nc] == 0:
                            lab[nr, nc] = n_lab
                            q[tail] = nr * ncols + nc
                            tail += 1
    return lab, n_lab


@njit
def _estatisticas_depressoes(lab, n_lab, dem, filled, row_areas):
    nrows, ncols = lab.shape
    area = np.zeros(n_lab + 1)
    prof = np.zeros(n_lab + 1)
    nivel = np.full(n_lab + 1, -np.inf)
    pit = np.full(n_lab + 1, -1, dtype=np.int64)
    hmin = np.full(n_lab + 1, np.inf)
    for r in range(nrows):
        for c in range(ncols):
            k = lab[r, c]
            if k > 0:
                area[k] += row_areas[r]
                d = filled[r, c] - dem[r, c]
                if d > prof[k]:
                    prof[k] = d
                if filled[r, c] > nivel[k]:
                    nivel[k] = filled[r, c]
                if dem[r, c] < hmin[k]:
                    hmin[k] = dem[r, c]
                    pit[k] = r * ncols + c
    return area, prof, nivel, pit


def _sementes_costa_e_saidas(sea, dem, lats, lons, saidas):
    """Sementes costeiras mais sementes exorreicas forçadas (pixel mais baixo num raio de 3 px)."""
    seeds = _sementes_costa(sea)
    nrows, ncols = dem.shape
    for lon, lat in saidas.values():
        if lon < lons[0] - 1e-9:
            lon += 360.0
        r = int(round((lat - lats[0]) / (lats[1] - lats[0])))
        c = int(round((lon - lons[0]) / (lons[1] - lons[0])))
        if not (3 <= r < nrows - 3 and 3 <= c < ncols - 3):
            continue
        jan = np.where(sea[r - 3:r + 4, c - 3:c + 4], np.inf, dem[r - 3:r + 4, c - 3:c + 4])
        dr, dc = np.unravel_index(np.argmin(jan), jan.shape)
        seeds[r - 3 + dr, c - 3 + dc] = 0
    return seeds


def _rotulo_no_ponto(lab, lats, lons, lon, lat, raio_px=3):
    """Rótulo de watershed predominante (> 0) na vizinhança do ponto (lon, lat); 0 se não houver."""
    if lon < lons[0] - 1e-9:
        lon += 360.0
    r = int(round((lat - lats[0]) / (lats[1] - lats[0])))
    c = int(round((lon - lons[0]) / (lons[1] - lons[0])))
    nrows, ncols = lab.shape
    if not (0 <= r < nrows and 0 <= c < ncols):
        return -1
    if lab[r, c] > 0:
        return int(lab[r, c])
    jan = lab[max(0, r - raio_px): r + raio_px + 1, max(0, c - raio_px): c + raio_px + 1]
    pos = jan[jan > 0]
    if pos.size == 0:
        return 0
    vals, cnts = np.unique(pos, return_counts=True)
    return int(vals[np.argmax(cnts)])


def identificar_bacias_endorreicas(
    dem: np.ndarray,
    sea_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    row_areas_km2: np.ndarray,
    area_min_km2: float = 1000.0,
    prof_min_m: float = 20.0,
    eps_m: float = 0.5,
    sistemas: dict = None,
    saidas_forcadas: dict = None,
):
    """
    Retorna:
      endo_grid    : int32 2D, 0 = drenagem exorreica, k > 0 = bacia endorreica k, -1 = mar.
      depressoes   : lista de dicts (uma por bacia endorreica aceita) com nome do sistema
                     documentado, pit, nível de transbordo, profundidade e áreas.
      diagnostico  : contagens de depressões, candidatas, aceitas e sistemas sem correspondência.
    """
    sistemas = SISTEMAS_ENDORREICOS if sistemas is None else sistemas
    saidas = SAIDAS_EXORREICAS_FORCADAS if saidas_forcadas is None else saidas_forcadas
    dem64 = np.ascontiguousarray(dem, dtype=np.float64)
    sea = np.ascontiguousarray(sea_mask, dtype=np.bool_)
    nrows, ncols = dem64.shape
    row_areas = np.ascontiguousarray(row_areas_km2, dtype=np.float64)
    area_2d = np.broadcast_to(row_areas[:, None], (nrows, ncols))

    filled = priority_flood_preenchimento(dem64, sea)
    mask = (~sea) & ((filled - dem64) > eps_m)
    lab, n_lab = _rotular_componentes(mask)
    area, prof, nivel, pit = _estatisticas_depressoes(lab, n_lab, dem64, filled, row_areas)
    del mask, lab

    candidatas = np.flatnonzero((area >= area_min_km2) & (prof >= prof_min_m))
    seeds = _sementes_costa_e_saidas(sea, dem64, lats, lons, saidas)
    transb = np.concatenate(([np.inf], nivel[candidatas]))
    for i_c, k in enumerate(candidatas, start=1):
        pr, pc = divmod(int(pit[k]), ncols)
        seeds[pr, pc] = i_c
    ws = np.zeros((nrows, ncols), dtype=np.int32)
    _priority_flood(dem64, sea, seeds, ws, transb, filled)

    nomes_por_cand = {}
    sem_correspondencia = []
    for nome, (lon, lat) in sistemas.items():
        i_c = _rotulo_no_ponto(ws, lats, lons, lon, lat)
        if i_c > 0:
            nomes_por_cand.setdefault(i_c, []).append(nome)
        elif i_c == 0:
            sem_correspondencia.append(nome)
    del ws

    aceitas = sorted(nomes_por_cand)
    seeds = _sementes_costa_e_saidas(sea, dem64, lats, lons, saidas)
    transb = np.concatenate(([np.inf], nivel[candidatas[np.array(aceitas, dtype=np.int64) - 1]]))
    for novo_id, i_c in enumerate(aceitas, start=1):
        pr, pc = divmod(int(pit[candidatas[i_c - 1]]), ncols)
        seeds[pr, pc] = novo_id
    endo = np.zeros((nrows, ncols), dtype=np.int32)
    _priority_flood(dem64, sea, seeds, endo, transb, filled)
    endo[sea] = -1

    area_capt = np.bincount(np.where(endo > 0, endo, 0).ravel(), weights=area_2d.ravel(),
                            minlength=len(aceitas) + 1)
    depressoes = []
    for novo_id, i_c in enumerate(aceitas, start=1):
        k = candidatas[i_c - 1]
        pr, pc = divmod(int(pit[k]), ncols)
        lon_pit = float(lons[pc]) - 360.0 if lons[pc] > 180.0 else float(lons[pc])
        depressoes.append(dict(
            endo_id=novo_id,
            nome_sistema=" + ".join(nomes_por_cand[i_c]),
            pit_row=int(pr),
            pit_col=int(pc),
            pit_lon=lon_pit,
            pit_lat=float(lats[pr]),
            pit_elev_m=float(dem64[pr, pc]),
            nivel_transbordo_m=float(nivel[k]),
            profundidade_m=float(prof[k]),
            area_depressao_km2=float(area[k]),
            area_captacao_km2=float(area_capt[novo_id]),
        ))
    diagnostico = dict(
        n_depressoes_total=int(n_lab),
        n_candidatas=int(candidatas.size),
        n_endorreicas_aceitas=len(aceitas),
        n_candidatas_rejeitadas=int(candidatas.size - len(aceitas)),
        sistemas_sem_correspondencia=sem_correspondencia,
        area_min_km2=float(area_min_km2),
        prof_min_m=float(prof_min_m),
        area_endorreica_total_km2=float(area_capt[1:].sum()),
    )
    return endo, depressoes, diagnostico
